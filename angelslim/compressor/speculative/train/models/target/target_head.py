# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from angelslim.utils import decide_device_for_distributed


class TargetHead(nn.Module):
    """
    Target Head for computing logits from hidden states in offline EAGLE3 training.

    This module takes the last hidden states from the target model and projects them
    to vocabulary logits, which are used as training targets for the draft model.
    """

    def __init__(self, lm_head: nn.Module):
        """
        Initialize the TargetHead.

        Args:
            lm_head: Language model head (typically nn.Linear) that projects
                    hidden states to vocabulary logits. This should be loaded
                    from the target model.
        """
        super().__init__()
        self.lm_head = lm_head

        # Freeze the lm_head parameters since we only use it for inference
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from hidden states.

        Args:
            hidden_states: Hidden states from target model with shape
                          (batch_size, seq_length, hidden_size)

        Returns:
            Logits with shape (batch_size, seq_length, vocab_size)
        """
        # Project hidden states to vocabulary logits
        logits = self.lm_head(hidden_states)
        return logits

    def forward_with_vocab_mapping(
        self, hidden_states: torch.Tensor, t2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits only for draft vocabulary tokens, avoiding full vocab projection.

        Instead of computing [B, S, full_vocab] and then gathering with t2d (boolean mask),
        this method first slices the lm_head weight matrix to only include
        draft-vocab rows, then performs a smaller matmul:
        [B, S, hidden_size] x [hidden_size, draft_vocab_size].

        This reduces computation by ~full_vocab/draft_vocab (e.g., 152064/32000 ≈ 4.75x)
        and avoids materializing the full [B, S, 152064] logits tensor.

        Dtype conversion is handled internally: if hidden_states dtype differs from
        the cached weight dtype, hidden_states is cast automatically.

        Args:
            hidden_states: Hidden states from target model with shape
                          (batch_size, seq_length, hidden_size)
            t2d: Boolean mask tensor with shape (full_vocab_size,).
                 t2d[i] = True means target token i is in the draft vocabulary.
                 The True positions, when sorted, give the target token indices
                 for draft tokens 0, 1, 2, ... (matching the original boolean indexing).

        Returns:
            Logits with shape (batch_size, seq_length, draft_vocab_size)
        """
        # Cache the sliced weight matrix after first call for efficiency.
        # t2d is a boolean mask: True positions correspond to draft vocab tokens.
        # t2d.nonzero().squeeze(-1) gives sorted target token indices for draft vocab,
        # which is equivalent to the original `logits[..., t2d]` boolean indexing.
        if not hasattr(self, "_sliced_weight") or self._sliced_weight is None:
            draft_token_indices = t2d.nonzero(as_tuple=False).squeeze(-1)  # [draft_vocab_size]
            # Slice and cache: [draft_vocab_size, hidden_size]
            self._sliced_weight = self.lm_head.weight[draft_token_indices].detach()

        # Cast hidden_states to match the cached weight dtype if needed
        # (e.g. pre-computed hidden states may be float16 while weights are bfloat16)
        if hidden_states.dtype != self._sliced_weight.dtype:
            hidden_states = hidden_states.to(self._sliced_weight.dtype)

        # Compute logits: [B, S, hidden_size] @ [hidden_size, draft_vocab_size]
        logits = torch.nn.functional.linear(hidden_states, self._sliced_weight)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        lm_head_key: str = "lm_head.weight",
        sub_config_name=None,
    ):
        """
        Load TargetHead from a pretrained model efficiently.

        This method only loads the lm_head weights using safetensors index,
        which is more memory-efficient than loading the entire model.

        Args:
            model_name_or_path: Path to pretrained model or model identifier
            **kwargs: Additional arguments for model loading (e.g., torch_dtype,
                     trust_remote_code, device_map)

        Returns:
            TargetHead instance with loaded lm_head
        """
        # Load model config to get architecture info
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if sub_config_name is not None:
            # now sub_config_name is only for VLM models to get language_model config
            if hasattr(config, sub_config_name):
                config = getattr(config, sub_config_name)
            else:
                raise ValueError(f"Config {config} has no sub-config named {sub_config_name}")

        # Get model dimensions
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
            vocab_size = config.text_config.vocab_size
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
            vocab_size = config.vocab_size
        else:
            raise ValueError(
                f"Cannot determine hidden_size from config (model_type={config.model_type}). "
                f"Please specify sub_config_name parameter to locate the text config."
            )

        # Initialize lm_head
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Load lm_head weights from safetensors
        try:
            # Read safetensors index to locate lm_head weights
            try:
                index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")

                if not os.path.exists(index_path):
                    raise FileNotFoundError(
                        "model.safetensors.index.json"
                        f"not found in {model_name_or_path}. "
                        "Please ensure the model is saved in safetensors "
                        "format with sharding."
                    )

                # Model is sharded, use index to find lm_head
                with open(index_path, "r") as f:
                    index_json = json.loads(f.read())
                    head_path = index_json["weight_map"][lm_head_key]

                # Load lm_head weights using safetensors
                safetensors_file = os.path.join(model_name_or_path, head_path)
            except Exception:
                safetensors_file = os.path.join(model_name_or_path, "model.safetensors")

            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice(lm_head_key)
                _, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim]
            lm_head.weight.data = tensor

        except Exception as e:
            raise RuntimeError(
                f"Failed to load lm_head weights from {model_name_or_path}. " f"Error: {str(e)}"
            )

        # Create TargetHead instance
        target_head = cls(lm_head)

        device = decide_device_for_distributed()
        target_head.to(device)
        target_head.eval()

        return target_head
