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

import argparse
import json
import logging
import os
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers.image_utils import load_image

from angelslim.compressor.speculative import (
    DatasetManager,
    DraftModelConfig,
    create_target_model,
    infer_model_params,
)
from angelslim.compressor.speculative.train.data.data_utils import (
    build_image_processor_kwargs,
    process_token_dict_to_mappings,
)
from angelslim.utils import decide_device_for_distributed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Rank %(rank)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size, local_rank) or (0, 1, 0) if not distributed
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        # Single process mode
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class HiddenStateGenerator:
    """Generator for creating hidden states from target model."""

    def __init__(
        self,
        target_model,
        output_dir: str,
        rank: int = 0,
        draft_vocab_size: int = None,
        target_vocab_size: int = None,
    ):
        """
        Initialize the hidden state generator.

        Args:
            target_model: The target model for generating hidden states
            output_dir: Directory to save generated hidden states
            rank: Process rank for distributed training
            draft_vocab_size: Size of draft model vocabulary (required for vocab mapping)
            target_vocab_size: Size of target model vocabulary (required for vocab mapping)
        """
        self.target_model = target_model
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.draft_vocab_size = draft_vocab_size
        self.target_vocab_size = target_vocab_size
        _max_pixels = os.environ.get("MAX_PIXELS")
        _min_pixels = os.environ.get("MIN_PIXELS", "1024")
        self.max_pixels = int(_max_pixels) if _max_pixels is not None else None
        self.min_pixels = int(_min_pixels) if _min_pixels is not None else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token_dict = Counter()

        # Packed memmap: Tightly concatenate all samples' data (no padding waste)
        # Each field is stored as a 1D (or 2D for hidden dims) packed array.
        # An offsets array records the cumulative token count for each sample.
        self._memmap_dir = self.output_dir / "memmap_data"
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        self._memmap_files = {}  # {field_name: np.memmap}
        self._memmap_initialized = False
        self._sample_count = 0
        self._total_tokens_written = 0  # Total tokens written across all samples
        self._preallocated_token_capacity = 0  # Preallocated total token capacity
        self._preallocated_sample_capacity = 0  # Preallocated sample capacity
        self._field_extra_dims = {}  # {field_name: tuple of extra dims after seq_len}
        self._field_dtypes = {}  # {field_name: numpy dtype}

        # Resolve image_pad token id for vLLM loss_mask rebuilding
        self._image_pad_token_id = None

    def _resolve_image_pad_token_id(self):
        """Lazily resolve the image_pad token id from the target model's tokenizer."""
        if self._image_pad_token_id is not None:
            return self._image_pad_token_id
        try:
            tokenizer = self.target_model.tokenizer
            _image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
            _tok = getattr(tokenizer, "tokenizer", tokenizer)
            _vocab = _tok.get_vocab() if hasattr(_tok, "get_vocab") else {}
            self._image_pad_token_id = _vocab.get(_image_token)
        except Exception:
            pass
        return self._image_pad_token_id

    def _rebuild_ids_and_loss_mask_for_vllm(
        self,
        orig_input_ids: torch.Tensor,
        orig_loss_mask: torch.Tensor,
        target_seq_len: int,
    ) -> tuple:
        """Rebuild input_ids and loss_mask to match vLLM's actual output length.

        When vLLM processes images, it re-expands image_pad tokens internally.
        The number of expanded tokens (M) may differ from the pre-expansion
        count (N) in orig_input_ids.  This method adjusts the sequences:

        Strategy:
        - Identify contiguous runs of image_pad tokens in orig_input_ids.
        - The total length difference (target_seq_len - orig_seq_len) is
          distributed across image_pad runs proportionally.
        - Non-image_pad tokens and their loss_mask values are preserved as-is.
        - New image_pad tokens get loss_mask = 0 (image tokens are never
          part of the training loss).

        Args:
            orig_input_ids: shape [B, N_orig]
            orig_loss_mask: shape [B, N_orig]
            target_seq_len: N_vllm, the actual sequence length from vLLM output

        Returns:
            Tuple of (new_input_ids, new_loss_mask), both shape [B, target_seq_len]
        """
        pad_id = self._resolve_image_pad_token_id()
        batch_size = orig_input_ids.shape[0]

        new_ids_list = []
        new_mask_list = []

        for b in range(batch_size):
            ids = orig_input_ids[b].tolist()
            mask = orig_loss_mask[b].tolist()
            orig_len = len(ids)
            delta = target_seq_len - orig_len

            if pad_id is None or delta == 0:
                # No image_pad token or lengths already match — just pad/truncate
                if delta > 0:
                    ids = ids + [0] * delta
                    mask = mask + [0] * delta
                else:
                    ids = ids[:target_seq_len]
                    mask = mask[:target_seq_len]
                new_ids_list.append(ids)
                new_mask_list.append(mask)
                continue

            # Find image_pad runs: list of (start, length) tuples
            runs = []
            i = 0
            while i < orig_len:
                if ids[i] == pad_id:
                    run_start = i
                    while i < orig_len and ids[i] == pad_id:
                        i += 1
                    runs.append((run_start, i - run_start))
                else:
                    i += 1

            if not runs:
                # No image_pad runs found; truncate/pad at end
                if delta > 0:
                    ids = ids + [0] * delta
                    mask = mask + [0] * delta
                else:
                    ids = ids[:target_seq_len]
                    mask = mask[:target_seq_len]
                new_ids_list.append(ids)
                new_mask_list.append(mask)
                continue

            # Distribute delta across runs proportionally
            total_pad_tokens = sum(length for _, length in runs)
            # New size for each run
            new_run_sizes = []
            remaining_delta = delta
            for idx_r, (_, length) in enumerate(runs):
                if idx_r == len(runs) - 1:
                    # Last run absorbs remaining delta
                    new_size = length + remaining_delta
                else:
                    share = round(delta * length / total_pad_tokens)
                    new_size = length + share
                    remaining_delta -= share
                new_run_sizes.append(max(new_size, 1))  # At least 1 token per run

            # Reconstruct ids and mask
            result_ids = []
            result_mask = []
            run_idx = 0
            i = 0
            while i < orig_len:
                if run_idx < len(runs) and i == runs[run_idx][0]:
                    # This is an image_pad run
                    run_start, run_len = runs[run_idx]
                    new_size = new_run_sizes[run_idx]
                    result_ids.extend([pad_id] * new_size)
                    result_mask.extend([0] * new_size)  # image_pad tokens: loss_mask = 0
                    i += run_len
                    run_idx += 1
                else:
                    result_ids.append(ids[i])
                    result_mask.append(mask[i])
                    i += 1

            # Final length adjustment (rounding safety)
            if len(result_ids) < target_seq_len:
                pad_count = target_seq_len - len(result_ids)
                result_ids.extend([0] * pad_count)
                result_mask.extend([0] * pad_count)
            elif len(result_ids) > target_seq_len:
                result_ids = result_ids[:target_seq_len]
                result_mask = result_mask[:target_seq_len]

            new_ids_list.append(result_ids)
            new_mask_list.append(result_mask)

        new_input_ids = torch.tensor(new_ids_list, dtype=orig_input_ids.dtype)
        new_loss_mask = torch.tensor(new_mask_list, dtype=orig_loss_mask.dtype)
        return new_input_ids, new_loss_mask

    def _init_memmap(self, data_point: Dict[str, torch.Tensor], total_samples: int):
        """Initialize packed memmap files based on the shape info of the first sample.

        Packed layout: each field is stored as a flat concatenation of all samples'
        valid tokens. An offsets array (length = total_samples + 1) records the
        cumulative token boundary so that sample i occupies
        [offsets[i], offsets[i+1]) in the packed array.

        This eliminates all padding waste — storage equals the sum of actual
        sequence lengths, identical to the old torch.save/shard approach.

        Args:
            data_point: First sample data
            total_samples: Estimated total number of samples (for preallocation)
        """
        # Estimate average seq_len from the first sample to preallocate token capacity
        first_seq_len = 0
        for field_name, tensor in data_point.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            per_sample_shape = tensor.shape[1:]  # (N, ...) or (N,)
            first_seq_len = max(first_seq_len, per_sample_shape[0])
            extra_dims = per_sample_shape[1:]  # () or (D,) or (3*D,)

            # Determine numpy dtype
            if tensor.dtype == torch.bfloat16:
                np_dtype = np.float16
            elif tensor.dtype == torch.float16:
                np_dtype = np.float16
            elif tensor.dtype == torch.float32:
                np_dtype = np.float32
            elif tensor.dtype in (torch.int64, torch.long):
                np_dtype = np.int64
            elif tensor.dtype in (torch.int32, torch.int):
                np_dtype = np.int32
            else:
                np_dtype = np.float32

            self._field_dtypes[field_name] = np_dtype
            self._field_extra_dims[field_name] = tuple(extra_dims)

        # Preallocate: assume avg seq_len ≈ first sample's seq_len, with 1.2x margin
        self._preallocated_sample_capacity = int(total_samples * 1.1) + 10
        self._preallocated_token_capacity = int(first_seq_len * total_samples * 1.2) + 1024

        # Create packed memmap files: shape = (total_tokens, *extra_dims)
        for field_name in self._field_dtypes:
            extra_dims = self._field_extra_dims[field_name]
            packed_shape = (self._preallocated_token_capacity,) + extra_dims
            memmap_path = self._memmap_dir / f"{field_name}.npy"
            self._memmap_files[field_name] = np.memmap(
                str(memmap_path),
                dtype=self._field_dtypes[field_name],
                mode="w+",
                shape=packed_shape,
            )

        # Offsets array: offsets[i] = cumulative token count before sample i
        # Length = preallocated_sample_capacity + 1 (offsets[0] = 0)
        self._offsets = np.memmap(
            str(self._memmap_dir / "offsets.npy"),
            dtype=np.int64,
            mode="w+",
            shape=(self._preallocated_sample_capacity + 1,),
        )
        self._offsets[0] = 0

        self._memmap_initialized = True
        logger.info(
            f"Packed memmap initialized: sample_capacity={self._preallocated_sample_capacity}, "
            f"token_capacity={self._preallocated_token_capacity}, "
            f"fields={list(self._field_dtypes.keys())}, "
            f"dir={self._memmap_dir}",
            extra={"rank": self.rank},
        )

    def _expand_token_capacity(self, required_tokens: int):
        """Expand packed memmap token capacity when running out of space."""
        old_capacity = self._preallocated_token_capacity
        self._preallocated_token_capacity = max(int(old_capacity * 1.5), required_tokens + 1024)

        logger.info(
            f"Expanding packed memmap token capacity: {old_capacity} -> "
            f"{self._preallocated_token_capacity}",
            extra={"rank": self.rank},
        )

        for field_name in self._field_dtypes:
            extra_dims = self._field_extra_dims[field_name]
            old_memmap = self._memmap_files[field_name]
            new_shape = (self._preallocated_token_capacity,) + extra_dims

            new_path = self._memmap_dir / f"{field_name}_new.npy"
            new_memmap = np.memmap(
                str(new_path),
                dtype=self._field_dtypes[field_name],
                mode="w+",
                shape=new_shape,
            )

            if self._total_tokens_written > 0:
                new_memmap[: self._total_tokens_written] = old_memmap[: self._total_tokens_written]
                new_memmap.flush()

            del old_memmap
            self._memmap_files[field_name] = None
            old_path = self._memmap_dir / f"{field_name}.npy"
            os.replace(str(new_path), str(old_path))
            self._memmap_files[field_name] = np.memmap(
                str(old_path),
                dtype=self._field_dtypes[field_name],
                mode="r+",
                shape=new_shape,
            )

    def _expand_sample_capacity(self):
        """Expand offsets array when sample count exceeds preallocated capacity."""
        old_capacity = self._preallocated_sample_capacity
        self._preallocated_sample_capacity = int(old_capacity * 1.5) + 10

        logger.info(
            f"Expanding sample capacity: {old_capacity} -> "
            f"{self._preallocated_sample_capacity}",
            extra={"rank": self.rank},
        )

        old_offsets = self._offsets
        new_offsets_path = self._memmap_dir / "offsets_new.npy"
        new_offsets = np.memmap(
            str(new_offsets_path),
            dtype=np.int64,
            mode="w+",
            shape=(self._preallocated_sample_capacity + 1,),
        )
        # Copy existing offsets (sample_count + 1 entries)
        new_offsets[: self._sample_count + 1] = old_offsets[: self._sample_count + 1]
        new_offsets.flush()

        del old_offsets
        self._offsets = None
        old_offsets_path = self._memmap_dir / "offsets.npy"
        os.replace(str(new_offsets_path), str(old_offsets_path))
        self._offsets = np.memmap(
            str(old_offsets_path),
            dtype=np.int64,
            mode="r+",
            shape=(self._preallocated_sample_capacity + 1,),
        )

    def _write_sample_to_memmap(self, data_point: Dict[str, torch.Tensor]):
        """Write a single sample to packed memmap files.

        Data is appended at position self._total_tokens_written in the packed array.
        The offsets array is updated to record the boundary.
        """
        # Get the seq_len of the current sample
        sample_seq_len = 0
        for _, tensor in data_point.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                sample_seq_len = max(sample_seq_len, tensor.shape[1])

        # Check if token capacity expansion is needed
        required_tokens = self._total_tokens_written + sample_seq_len
        if required_tokens > self._preallocated_token_capacity:
            self._expand_token_capacity(required_tokens)

        # Check if sample capacity expansion is needed
        if self._sample_count >= self._preallocated_sample_capacity:
            self._expand_sample_capacity()

        write_offset = self._total_tokens_written

        for field_name, tensor in data_point.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if field_name not in self._memmap_files:
                continue

            # tensor shape: [1, N, ...], remove batch dimension
            arr = tensor.squeeze(0)  # [N, ...]
            if tensor.dtype == torch.bfloat16:
                arr = arr.float().half()  # bfloat16 -> float32 -> float16
            arr_np = arr.contiguous().numpy()

            seq_len = arr_np.shape[0]
            # Write to packed position: [write_offset : write_offset + seq_len]
            self._memmap_files[field_name][write_offset : write_offset + seq_len] = arr_np

        # Update offsets and counters
        self._total_tokens_written += sample_seq_len
        self._sample_count += 1
        self._offsets[self._sample_count] = self._total_tokens_written

        # Flush every 100 samples to ensure data persistence
        if self._sample_count % 100 == 0:
            for mm in self._memmap_files.values():
                if mm is not None:
                    mm.flush()
            self._offsets.flush()

    def _finalize_memmap(self):
        """Finalize packed memmap writing: truncate to actual size and save metadata."""
        if not self._memmap_initialized or self._sample_count == 0:
            return

        total_tokens = self._total_tokens_written

        # Flush all memmap files
        for mm in self._memmap_files.values():
            if mm is not None:
                mm.flush()
        self._offsets.flush()

        # Truncate packed data files to actual token count
        need_truncate_tokens = total_tokens < self._preallocated_token_capacity
        need_truncate_samples = self._sample_count < self._preallocated_sample_capacity

        if need_truncate_tokens:
            logger.info(
                f"Truncating packed memmap: tokens {self._preallocated_token_capacity} -> "
                f"{total_tokens}, samples {self._preallocated_sample_capacity} -> "
                f"{self._sample_count}",
                extra={"rank": self.rank},
            )

            for field_name in self._field_dtypes:
                extra_dims = self._field_extra_dims[field_name]
                old_memmap = self._memmap_files[field_name]
                final_shape = (total_tokens,) + extra_dims

                final_path = self._memmap_dir / f"{field_name}_final.npy"
                final_memmap = np.memmap(
                    str(final_path),
                    dtype=self._field_dtypes[field_name],
                    mode="w+",
                    shape=final_shape,
                )
                final_memmap[:] = old_memmap[:total_tokens]
                final_memmap.flush()

                del old_memmap, final_memmap
                self._memmap_files[field_name] = None
                old_path = self._memmap_dir / f"{field_name}.npy"
                os.replace(str(final_path), str(old_path))

        if need_truncate_samples:
            # Truncate offsets to actual sample count + 1
            old_offsets = self._offsets
            final_offsets_path = self._memmap_dir / "offsets_final.npy"
            final_offsets = np.memmap(
                str(final_offsets_path),
                dtype=np.int64,
                mode="w+",
                shape=(self._sample_count + 1,),
            )
            final_offsets[:] = old_offsets[: self._sample_count + 1]
            final_offsets.flush()
            del old_offsets, final_offsets
            self._offsets = None
            old_offsets_path = self._memmap_dir / "offsets.npy"
            os.replace(str(final_offsets_path), str(old_offsets_path))

        # Save metadata JSON
        metadata = {
            "format": "packed",  # Distinguish from the old rectangular format
            "total_samples": self._sample_count,
            "total_tokens": total_tokens,
            "fields": {},
        }
        for field_name in self._field_dtypes:
            extra_dims = self._field_extra_dims[field_name]
            final_shape = (total_tokens,) + extra_dims
            metadata["fields"][field_name] = {
                "dtype": np.dtype(self._field_dtypes[field_name]).name,
                "shape": list(final_shape),
            }
        metadata["fields"]["offsets"] = {
            "dtype": "int64",
            "shape": [self._sample_count + 1],
        }

        metadata_path = self._memmap_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Packed memmap finalized: {self._sample_count} samples, "
            f"{total_tokens} total tokens, "
            f"dir={self._memmap_dir}",
            extra={"rank": self.rank},
        )

    def _process_single_sample(self, idx: int, row: Dict[str, Any]) -> bool:
        """
        Process a single sample and save its hidden states.

        Args:
            idx: Sample index
            row: Sample data containing input_ids and loss_mask

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Generate aux and target hiddens
            device = decide_device_for_distributed()

            if "image_paths" in row:
                image_paths = json.loads(row.pop("image_paths"))
                if image_paths:
                    images = [load_image(p) for p in image_paths]

                    # Check if using vLLM backend.
                    # vLLM expects raw PIL Images in multi_modal_data and runs
                    # its own image processor internally (which also computes
                    # image_grid_thw automatically).  Passing pre-processed
                    # pixel_values tensors or image_grid_thw as separate
                    # modality keys causes "Unsupported modality" errors.
                    is_vllm_backend = getattr(self.target_model, "backend_name", None) == "vllm"

                    if is_vllm_backend:
                        # Pass raw PIL Images; vLLM handles preprocessing internally
                        row["raw_images"] = [images]  # list of list (one list per batch sample)
                    else:
                        # HF Transformers backend: preprocess images manually
                        processor = self.target_model.tokenizer
                        if hasattr(processor, "image_processor"):
                            kwargs = build_image_processor_kwargs(
                                processor.image_processor, self.max_pixels, self.min_pixels
                            )
                            vision_encoding = processor.image_processor(
                                images=images, return_tensors="pt", **kwargs
                            )
                        else:
                            kwargs = build_image_processor_kwargs(
                                processor, self.max_pixels, self.min_pixels
                            )
                            vision_encoding = processor(
                                images=images, return_tensors="pt", **kwargs
                            )
                        row["pixel_values"] = vision_encoding["pixel_values"].to(device)
                        if "pixel_values_videos" in vision_encoding:
                            row["pixel_values_videos"] = vision_encoding["pixel_values_videos"].to(
                                device
                            )
                        if "image_grid_thw" in vision_encoding:
                            row["image_grid_thw"] = vision_encoding["image_grid_thw"].to(device)
                        if "video_grid_thw" in vision_encoding:
                            row["video_grid_thw"] = vision_encoding["video_grid_thw"].to(device)
                else:
                    row.pop("image_paths", None)

            # Save original input_ids and loss_mask before sending to model,
            # because vLLM backend may change the effective sequence length.
            orig_input_ids = row["input_ids"].clone()  # B, N_orig
            orig_loss_mask = row["loss_mask"].clone()  # B, N_orig

            for k, v in row.items():
                if isinstance(v, torch.Tensor) and v is not None:
                    row[k] = v.to(device)
            results = self.target_model.get_aux_and_target_hiddens(**row)
            # hidden_states: B, N_vllm, 3*D
            # target_hiddens: B, N_vllm, D
            for k, v in results.items():
                results[k] = v.cpu() if isinstance(v, torch.Tensor) else v

            # Prepare data point.
            # When using vLLM backend, the returned hidden_states seq_len (N_vllm)
            # may differ from the preprocessed input_ids length (N_orig), because
            # _process_single_conversation expands image_pad to N tokens based on
            # MAX_PIXELS, while vLLM internally re-expands a single image_pad to M
            # tokens (M may differ from N even with same MAX_PIXELS due to rounding
            # or processor differences).  We must rebuild input_ids and loss_mask to
            # match N_vllm so that the saved .ckpt has consistent dimensions.
            is_vllm_backend = getattr(self.target_model, "backend_name", None) == "vllm"
            vllm_seq_len = results["hidden_states"].shape[1]
            orig_seq_len = orig_input_ids.shape[1]

            if is_vllm_backend and vllm_seq_len != orig_seq_len:
                input_ids_cpu, loss_mask_cpu = self._rebuild_ids_and_loss_mask_for_vllm(
                    orig_input_ids.cpu(), orig_loss_mask.cpu(), vllm_seq_len
                )
            else:
                input_ids_cpu = orig_input_ids.cpu()
                loss_mask_cpu = orig_loss_mask.cpu()

            data_point = {
                "input_ids": input_ids_cpu,
                "loss_mask": loss_mask_cpu,
                **results,
            }

            masked_ids = input_ids_cpu[loss_mask_cpu == 1]
            unique_ids, counts = masked_ids.unique(return_counts=True)
            batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
            self.token_dict.update(batch_token_dict)

            # Initialize memmap on first write
            if not self._memmap_initialized:
                self._init_memmap(data_point, self._total_samples_estimate)

            # Write directly to memmap
            self._write_sample_to_memmap(data_point)
            return True

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}", extra={"rank": self.rank})
            return False

    def generate(self, dataset) -> Tuple[int, int]:
        """
        Generate hidden states for all samples in the dataset.

        Args:
            dataset: Dataset to process

        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0

        # Set estimated total sample count for memmap preallocation
        self._total_samples_estimate = len(dataset)

        # Only show progress bar on rank 0
        iterator = (
            tqdm(
                enumerate(dataset),
                total=len(dataset),
                desc=f"Rank {self.rank} processing",
            )
            if self.rank == 0
            else enumerate(dataset)
        )

        for idx, row in iterator:
            if self._process_single_sample(idx, row):
                successful += 1
            else:
                failed += 1

        # Finalize memmap writing: truncate to actual size and save metadata
        self._finalize_memmap()

        logger.info(
            f"Processing complete. Success: {successful}, Failed: {failed}",
            extra={"rank": self.rank},
        )
        logger.info(
            f"Results saved to {self._memmap_dir} "
            f"({self._sample_count} samples in memmap format)",
            extra={"rank": self.rank},
        )

        return successful, failed

    def save_vocab_mapping(self, output_dir):
        """
        Compute vocab mapping from token_dict and save to $output_dir/vocab_mapping.pt
        for offline training to directly load.

        Requires draft_vocab_size and target_vocab_size to be set.
        """
        if self.draft_vocab_size is None or self.target_vocab_size is None:
            raise ValueError(
                "draft_vocab_size and target_vocab_size must be set to save vocab mapping. "
                "Please pass --draft_model_config_path argument."
            )

        # Gather token_dict from all ranks and merge on rank 0
        if dist.is_initialized():
            all_token_dicts = [None] * dist.get_world_size()
            dist.all_gather_object(all_token_dicts, dict(self.token_dict))
            merged_token_dict = Counter()
            for td in all_token_dicts:
                merged_token_dict.update(td)
        else:
            merged_token_dict = self.token_dict

        # Only rank 0 computes and saves vocab mapping
        if self.rank != 0:
            return

        vocab_mapping_path = Path(output_dir) / "vocab_mapping.pt"
        logger.info(
            f"Computing vocab mapping (draft_vocab_size={self.draft_vocab_size}, "
            f"target_vocab_size={self.target_vocab_size})...",
            extra={"rank": self.rank},
        )

        d2t, t2d = process_token_dict_to_mappings(
            merged_token_dict,
            self.draft_vocab_size,
            self.target_vocab_size,
        )

        vocab_mapping = {"d2t": d2t, "t2d": t2d}
        torch.save(vocab_mapping, vocab_mapping_path)
        logger.info(
            f"Vocab mapping saved to {vocab_mapping_path}",
            extra={"rank": self.rank},
        )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate hidden states for draft model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset range arguments
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Global start index of dataset (applies before distribution to GPUs)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Global end index of dataset (None means use full dataset). "
        "The range [start, end) will be automatically distributed across all GPUs.",
    )

    # Output configuration
    parser.add_argument(
        "--outdir",
        type=str,
        default="outdir0",
        help="Output directory for generated hidden states",
    )

    # Model configuration
    parser.add_argument(
        "--target_model_name_or_path",
        type=str,
        help="Target model name or path (if different from model_name)",
    )
    parser.add_argument(
        "--target_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Backend for target model",
    )
    parser.add_argument(
        "--modal_type",
        type=str,
        default="LLM",
        choices=["LLM", "VLM"],
        help="Modal type: LLM for language models, VLM for vision-language models",
    )
    parser.add_argument(
        "--training_mode",
        type=str,
        default="offline",
        choices=["online", "offline"],
        help="Training mode: online or offline",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path", type=str, nargs="+", required=True, help="Dataset to use"
    )
    parser.add_argument("--max_model_len", type=int, default=2048, help="Maximum token length")
    parser.add_argument(
        "--chat_template_type",
        type=str,
        default=None,
        help="Chat template type (auto-detected from model config if not specified)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display dataset samples (only on rank 0)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes for data preprocessing",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=None,
        help="Number of max samples for data preprocessing",
    )
    parser.add_argument(
        "--shuffle_seed", type=int, default=42, help="Random seed for shuffling dataset"
    )

    # Draft model config for vocab mapping
    parser.add_argument(
        "--draft_model_config_path",
        type=str,
        default=None,
        help="Path to draft model config file, used to read draft_vocab_size and vocab_size "
        "for computing vocab mapping",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation of dtype

    Returns:
        Corresponding torch dtype
    """
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_mapping.get(dtype_str, torch.bfloat16)


def load_dataset(args: argparse.Namespace, tokenizer, rank: int):
    """
    Load and prepare dataset.

    Args:
        args: Parsed command line arguments
        tokenizer: Tokenizer from target model
        rank: Process rank

    Returns:
        Prepared dataset
    """
    logger.info(f"Loading dataset: {args.dataset_path}", extra={"rank": rank})

    # Only display on rank 0
    display = args.display and rank == 0

    args.train_data_path = None
    args.eval_data_path = args.dataset_path
    dataset_manager = DatasetManager(
        data_args=args,
        tokenizer=tokenizer,
        target_model_type=None if args.modal_type in ("LLM", "TTS") else args.target_model_type,
        max_model_len=args.max_model_len,
        chat_template_type=args.chat_template_type,
        display=display,
    )

    _, dataset, _ = dataset_manager.create_online_datasets()
    logger.info(f"Dataset loaded: {len(dataset)} samples", extra={"rank": rank})

    return dataset


def split_dataset_for_rank(dataset, rank: int, world_size: int, start: int = 0, end: int = None):
    """
    Split dataset for distributed processing.

    The dataset is first sliced to [start:end] range (global range),
    then evenly distributed across all ranks.

    Args:
        dataset: Full dataset
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes
        start: Global start index (default: 0)
        end: Global end index (default: None, means len(dataset))

    Returns:
        Dataset slice for current rank

    Example:
        Dataset has 10000 samples, world_size=4, start=1000, end=5000
        - Global range: [1000, 5000) = 4000 samples
        - Rank 0: [1000, 2000) = 1000 samples
        - Rank 1: [2000, 3000) = 1000 samples
        - Rank 2: [3000, 4000) = 1000 samples
        - Rank 3: [4000, 5000) = 1000 samples
    """
    # Determine the global range to process
    if end is None:
        end = len(dataset)

    # Validate range
    if start < 0 or end > len(dataset) or start >= end:
        raise ValueError(f"Invalid range: start={start}, end={end}, dataset_size={len(dataset)}")

    total_samples = end - start
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size

    # Calculate start and end for this rank
    rank_start = start + rank * samples_per_rank + min(rank, remainder)
    rank_end = rank_start + samples_per_rank + (1 if rank < remainder else 0)

    logger.info(
        f"Rank {rank}/{world_size}: Processing global range [{start}, {end}) -> "
        f"assigned range [{rank_start}, {rank_end}) ({rank_end - rank_start} samples)",
        extra={"rank": rank},
    )

    return dataset.select(range(rank_start, rank_end))


def main():
    """Main execution function."""
    # Setup distributed environment
    rank, world_size, local_rank = setup_distributed()
    logger.info(
        f"Distributed environment initialized: pid: {os.getpid()}, rank {rank},"
        "world_size {world_size}, local_rank {local_rank}",
        extra={"rank": rank},
    )

    # Parse arguments
    args = parse_arguments()
    args.train_data_path = None
    args.eval_data_path = args.dataset_path

    try:
        model_path = args.target_model_name_or_path
        logger.info(
            f"backend: {args.target_backend}, modal_type: {args.modal_type}", extra={"rank": rank}
        )

        draft_vocab_size = None
        target_vocab_size = None
        if args.draft_model_config_path is not None:
            draft_config = DraftModelConfig.from_file(args.draft_model_config_path)
            draft_vocab_size = getattr(draft_config, "draft_vocab_size", None)
            target_vocab_size = getattr(draft_config, "vocab_size", None)
            args.target_model_type = getattr(draft_config, "target_model_type", None)
            logger.info(
                f"Loaded from draft model config: draft_vocab_size={draft_vocab_size}, "
                f"target_vocab_size={target_vocab_size}, "
                f"target_model_type={args.target_model_type}",
                extra={"rank": rank},
            )
        else:
            raise ValueError("draft_model_config_path not specified")

        if args.chat_template_type is None:
            _, _, inferred_chat_template_type = infer_model_params(
                model_name_or_path=model_path,
                model_type=args.target_model_type,
            )
            args.chat_template_type = (
                inferred_chat_template_type
                if inferred_chat_template_type is not None
                else "default"
            )
            logger.info(
                f"chat_template_type not specified, auto deduced: {args.chat_template_type}",
                extra={"rank": rank},
            )
        else:
            logger.info(
                f"Using user-specified chat_template_type: {args.chat_template_type}",
                extra={"rank": rank},
            )

        # Load target model
        torch_dtype = get_torch_dtype(args.torch_dtype)
        target_model = create_target_model(
            backend=args.target_backend,
            modal_type=args.modal_type,
            model_path=args.target_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            target_model_type=args.target_model_type,
        )
        logger.info(
            f"Target model loaded: {args.target_model_name_or_path}",
            extra={"rank": rank},
        )
        if rank == 0:
            logger.info(f"tokenizer: {target_model.tokenizer}", extra={"rank": 0})

        # Load dataset
        dataset = load_dataset(args, target_model.tokenizer, rank)
        if len(dataset) == 0:
            logger.warning("No samples to process after loading dataset", extra={"rank": rank})
            return

        # Split dataset for this rank
        dataset_slice = split_dataset_for_rank(dataset, rank, world_size, args.start, args.end)

        # Generate hidden states
        output_dir = f"{args.outdir}/rank_{rank}"
        logger.info(f"writing hidden states to {output_dir}", extra={"rank": rank})

        generator = HiddenStateGenerator(
            target_model,
            output_dir,
            rank=rank,
            draft_vocab_size=draft_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        successful, failed = generator.generate(dataset_slice)

        # save vocab mapping for offline training
        generator.save_vocab_mapping(args.outdir)

        logger.info(
            f"Rank {rank} - Successful: {successful}, Failed: {failed}",
            extra={"rank": rank},
        )

    except Exception as e:
        logger.error(f"Rank {rank} encountered error: {e}", extra={"rank": rank})

    finally:
        # Synchronize all processes
        if world_size > 1:
            logger.info(
                f"Rank {rank} reached barrier, waiting for other ranks...", extra={"rank": rank}
            )
            dist.barrier()
            logger.info(f"Rank {rank} passed barrier.", extra={"rank": rank})

        # Log final statistics (only on rank 0)
        if rank == 0:
            logger.info("=" * 50, extra={"rank": rank})
            logger.info("Generation Complete!", extra={"rank": rank})
            if "dataset" in dir():
                logger.info(
                    f"Total samples processed across all ranks: {len(dataset)}",
                    extra={"rank": rank},
                )
            logger.info("=" * 50, extra={"rank": rank})

        # Cleanup distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main()
