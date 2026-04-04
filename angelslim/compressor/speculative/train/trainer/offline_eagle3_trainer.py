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

from typing import Dict

import torch
from torch import nn

from .eagle3_trainer import Eagle3Trainer
from .trainer_factory import Eagle3TrainerFactory


@Eagle3TrainerFactory.register("offline", "LLM")
class OfflineEagle3Trainer(Eagle3Trainer):
    """
    Offline EAGLE3 Trainer for speculative decoding training.

    Uses pre-computed hidden states and logits from offline processing,
    avoiding the need for online target model inference.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_head: nn.Module,
        length: int,
        enable_profiler: bool = False,
        **kwargs
    ):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            enable_profiler: Whether to enable PyTorch Profiler
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(
            draft_model=draft_model, length=length, enable_profiler=enable_profiler, **kwargs
        )
        self.target_head = target_head

    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training from offline-generated inputs.

        Args:
            inputs: Dictionary containing:
                - input_ids: Token IDs
                - target_hiddens: Pre-computed last hidden states from target model
                - hidden_states: Pre-computed aux hidden states from target model
                - attention_mask: Attention mask
                - loss_mask: Mask for loss computation
                - position_ids (optional): Position IDs

        Returns:
            Dictionary with prepared data for draft model training
        """
        #
        inputs_fields = [
            "input_ids",
            "target_hiddens",
            "hidden_states",
            "attention_mask",
            "loss_mask",
        ]
        output_fields = [
            "input_ids",
            "target_logits",
            "hidden_states",
            "attention_mask",
            "loss_mask",
            "position_ids",
        ]

        # Optimization: compute logits only for draft vocabulary tokens
        # instead of full vocab projection (e.g., 152064 -> 32000).
        # dtype conversion is handled inside forward_with_vocab_mapping.
        t2d = self.draft_model.t2d
        target_logits = self.target_head.forward_with_vocab_mapping(inputs["target_hiddens"], t2d)
        position_ids = inputs.get("position_ids", None)
        loss_mask = inputs["loss_mask"]
        input_ids = inputs["input_ids"]

        # Move tensors to correct device; padding shifts are now handled
        # in precompute_targets via _shift_left for better efficiency.
        target_logits = target_logits.to(input_ids.device)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        outputs = {k: inputs[k] for k in inputs_fields if k in output_fields}
        outputs["target_logits"] = target_logits
        outputs["position_ids"] = position_ids
        outputs["loss_mask"] = loss_mask
        outputs["input_ids"] = input_ids

        return outputs


@Eagle3TrainerFactory.register("offline", "VLM")
class OfflineVLMEagle3Trainer(Eagle3Trainer):
    """
    Offline EAGLE3 Trainer for speculative decoding training.

    Uses pre-computed hidden states and logits from offline processing,
    avoiding the need for online target model inference.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_head: nn.Module,
        length: int,
        enable_profiler: bool = False,
        **kwargs
    ):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            enable_profiler: Whether to enable PyTorch Profiler
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(
            draft_model=draft_model, length=length, enable_profiler=enable_profiler, **kwargs
        )
        self.target_head = target_head

    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training from offline-generated inputs.

        Args:
            inputs: Dictionary containing:
                - input_ids: Token IDs
                - target_hiddens: Pre-computed last hidden states from target model
                - hidden_states: Pre-computed aux hidden states from target model
                - attention_mask: Attention mask
                - loss_mask: Mask for loss computation
                - position_ids (optional): Position IDs (3D for VLMs mrope)

        Returns:
            Dictionary with prepared data for draft model training
        """
        inputs_fields = [
            "input_ids",
            "target_hiddens",
            "hidden_states",
            "attention_mask",
            "loss_mask",
            "position_ids",
        ]
        output_fields = [
            "input_ids",
            "target_logits",
            "hidden_states",
            "attention_mask",
            "loss_mask",
            "position_ids",
        ]

        # Optimization: compute logits only for draft vocabulary tokens
        # instead of full vocab projection (e.g., 152064 -> 32000).
        # dtype conversion is handled inside forward_with_vocab_mapping.
        t2d = self.draft_model.t2d
        target_logits = self.target_head.forward_with_vocab_mapping(inputs["target_hiddens"], t2d)
        loss_mask = inputs["loss_mask"]
        input_ids = inputs["input_ids"]
        position_ids = inputs.get("position_ids", None)

        # Move tensors to correct device; padding shifts are now handled
        # in precompute_targets via _shift_left for better efficiency.
        target_logits = target_logits.to(input_ids.device)
        loss_mask = loss_mask[..., None].to(input_ids.device)

        outputs = {k: inputs[k] for k in inputs_fields if k in output_fields}
        outputs["target_logits"] = target_logits
        outputs["loss_mask"] = loss_mask
        outputs["input_ids"] = input_ids
        outputs["position_ids"] = position_ids

        return outputs
