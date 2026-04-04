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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from angelslim.utils import rank0_print

from ..data_utils import (
    DataCollatorWithPadding,
    VLMDataCollatorWithPadding,
    VLMHunyuanDataCollatorWithPadding,
)
from .base_dataset_builder import DatasetBuilder
from .dataset_builder_factory import DatasetBuilderFactory


class OfflineEagle3Dataset(Dataset):
    """
    Offline Dataset for EAGLE3 training.

    Loads pre-computed hidden states, logits, and other data from .ckpt files.
    Each .ckpt file contains a dictionary with keys: input_ids, target_logits,
    hidden_states, and loss_mask.
    """

    def __init__(self, data_dir: str, file_pattern: str = "*.ckpt", cache_in_memory: bool = False):
        """
        Initialize the OfflineEagle3Dataset.

        Args:
            data_dir: Directory containing .ckpt files
                (will search recursively in subdirectories)
            file_pattern: Pattern to match checkpoint files (default: "*.ckpt")
            cache_in_memory: Whether to cache all data in memory (default: False)
        """
        self.data_dir = Path(data_dir)
        self.cache_in_memory = cache_in_memory

        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Recursively find all checkpoint files in subdirectories
        self.ckpt_files = sorted(list(self.data_dir.rglob(file_pattern)))

        if len(self.ckpt_files) == 0:
            raise ValueError(
                f"No checkpoint files found in {data_dir} "
                f"(including subdirectories) with pattern {file_pattern}"
            )

        rank0_print(
            f"Found {len(self.ckpt_files)} checkpoint files "
            f"in {data_dir} (including subdirectories)"
        )

        # Track valid indices (files that can be loaded successfully)
        self.valid_indices = list(range(len(self.ckpt_files)))

        # Cache data in memory if requested
        self.cached_data: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.cache_in_memory:
            rank0_print("Caching all data in memory...")
            self.cached_data = []
            failed_count = 0
            for i in range(len(self.ckpt_files)):
                data = self._load_ckpt(i)
                if data is not None:
                    self.cached_data.append(data)
                else:
                    failed_count += 1

            # Update valid indices based on successful loads
            self.valid_indices = list(range(len(self.cached_data)))

            if failed_count > 0:
                rank0_print(
                    f"Data caching completed. "
                    f"Successfully loaded {len(self.cached_data)} files, "
                    f"failed to load {failed_count} files"
                )
            else:
                rank0_print("Data caching completed")

    def _load_ckpt(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a checkpoint file.

        Args:
            idx: Index of the checkpoint file

        Returns:
            Dictionary containing input_ids, target_hiddens,
                hidden_states, and loss_mask, or None if loading fails
        """
        ckpt_path = self.ckpt_files[idx]

        try:
            data = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            warnings.warn(
                f"Failed to load checkpoint {ckpt_path}: {e}. Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate required keys
        required_keys = [
            "input_ids",  # B, N
            "target_hiddens",  # B, N, D
            "hidden_states",  # B, N, 3*D
            "loss_mask",  # B, N
        ]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            warnings.warn(
                f"Checkpoint {ckpt_path} is missing required keys: {missing_keys}. "
                f"Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate tensor types
        for key in required_keys:
            if not isinstance(data[key], torch.Tensor):
                warnings.warn(
                    f"Value for key '{key}' in {ckpt_path} is not a torch.Tensor. "
                    f"Skipping this file.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

        attention_mask = torch.ones_like(data["input_ids"])
        data["attention_mask"] = attention_mask  # B, N
        return data

    def __len__(self) -> int:
        """Return the number of valid samples in the dataset."""
        if self.cached_data is not None:
            return len(self.cached_data)
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - input_ids: Token IDs (torch.Tensor)
                - target_logits: Pre-computed logits from target
                    model (torch.Tensor)
                - hidden_states: Pre-computed hidden states from
                    target model (torch.Tensor)
                - loss_mask: Mask for loss computation (torch.Tensor)
        """
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            # Try to load the checkpoint, retry with next valid index if fails
            max_retries = len(self.valid_indices)
            for _attempt in range(max_retries):
                actual_idx = self.valid_indices[idx % len(self.valid_indices)]
                data = self._load_ckpt(actual_idx)
                if data is not None:
                    return data
                else:
                    # Remove failed index from valid_indices
                    self.valid_indices.remove(actual_idx)
                    if len(self.valid_indices) == 0:
                        raise RuntimeError(
                            "All checkpoint files failed to load. " "Cannot continue training."
                        )
                    # Try next index
                    idx += 1

            # If all retries failed, raise error
            raise RuntimeError(f"Failed to load any valid checkpoint after {max_retries} attempts")


class MemmapOfflineEagle3Dataset(Dataset):
    """
    Offline Dataset that reads from packed numpy memmap files.

    Packed format: Data is tightly concatenated without padding.
    - metadata.json contains "format": "packed"
    - offsets.npy: int64 array of length (total_samples + 1), cumulative token boundaries
    - Each field .npy has shape (total_tokens, *extra_dims) — no wasted space
    - Sample i occupies [offsets[i], offsets[i+1]) in the packed array

    Supports both single-directory and multi-rank directory layouts:
    - Single: <data_dir>/memmap_data/metadata.json
    - Multi-rank: <data_dir>/rank_*/memmap_data/metadata.json

    Advantages:
    1. Zero deserialization: Direct memory mapping, no pickle deserialization needed
    2. On-demand loading: OS loads 4KB pages on demand, only reading accessed samples
    3. Multi-process sharing: Multiple DataLoader workers share the same mmap page cache
    4. No merge needed: Sampling phase writes directly to memmap, training phase reads directly
    5. Zero padding waste: Storage size equals sum of actual sequence lengths
    """

    def __init__(self, memmap_dirs: list):
        """
        Initialize the MemmapOfflineEagle3Dataset.

        Args:
            memmap_dirs: List of directories containing memmap files and metadata.json
        """
        self.memmap_dirs = memmap_dirs

        # Load memmap files from all directories and build global index
        # global_index[i] = (dir_idx, local_sample_idx)
        self.global_index = []
        self._dir_memmaps = []  # [{field_name: np.memmap}, ...]
        self._dir_offsets = []  # [np.memmap, ...]
        self._dir_metadata = []  # [metadata_dict, ...]

        # Storage optimization attributes
        self._storage_precision = "float32"  # Default precision
        self._int8_quantization = False  # Whether int8 quantization is enabled

        total_samples = 0
        for dir_idx, memmap_dir in enumerate(memmap_dirs):
            metadata_path = os.path.join(memmap_dir, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self._dir_metadata.append(metadata)
            n_samples = metadata["total_samples"]

            # Read storage optimization config
            if "storage_optimization" in metadata:
                storage_opt = metadata["storage_optimization"]
                self._storage_precision = storage_opt.get("precision", "float32")
                self._int8_quantization = storage_opt.get("int8_quantization", False)

            # Open memmap files (read-only mode)
            field_memmaps = {}
            for field_name, field_info in metadata["fields"].items():
                if field_name == "offsets":
                    continue
                memmap_path = os.path.join(memmap_dir, f"{field_name}.npy")
                if not os.path.exists(memmap_path):
                    warnings.warn(
                        f"Memmap file not found: {memmap_path}, skipping field {field_name}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                field_memmaps[field_name] = np.memmap(
                    memmap_path,
                    dtype=field_info["dtype"],
                    mode="r",
                    shape=tuple(field_info["shape"]),
                )
            self._dir_memmaps.append(field_memmaps)

            # Load offsets
            offsets_path = os.path.join(memmap_dir, "offsets.npy")
            offsets_info = metadata["fields"]["offsets"]
            offsets = np.memmap(
                offsets_path,
                dtype=offsets_info["dtype"],
                mode="r",
                shape=tuple(offsets_info["shape"]),
            )
            self._dir_offsets.append(offsets)

            # Build global index
            for local_idx in range(n_samples):
                self.global_index.append((dir_idx, local_idx))

            total_samples += n_samples
            rank0_print(
                f"[MemmapDataset] Dir {dir_idx}: {memmap_dir} "
                f"({n_samples} samples, {metadata.get('total_tokens', 'N/A')} total tokens)"
                f" | Storage Opt: precision={self._storage_precision}, "
                f"int8_quantization={self._int8_quantization}"
            )

        self.total_samples = total_samples

        rank0_print(
            f"[MemmapDataset] Loaded {self.total_samples} samples "
            f"from {len(memmap_dirs)} memmap directories"
        )

    def _convert_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor precision based on storage config."""
        if self._storage_precision == "bfloat16":
            return tensor.to(torch.bfloat16)
        elif self._storage_precision == "float16":
            return tensor.to(torch.float16)
        else:  # float32 or unknown
            return tensor.to(torch.float32)

    @staticmethod
    def _dequantize_per_token_absmax_int8(
        quantized: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.bfloat16
    ):
        """Dequantize per-token absmax int8 back to float.

        Args:
            quantized: int8 tensor of shape [B, N, D]
            scale: float tensor of shape [B, N, 1]
            target_dtype: Target float dtype

        Returns:
            Dequantized float tensor of shape [B, N, D]
        """
        return quantized.to(target_dtype) * scale.to(target_dtype)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dir_idx, local_idx = self.global_index[idx]
        field_memmaps = self._dir_memmaps[dir_idx]
        offsets = self._dir_offsets[dir_idx]
        start = int(offsets[local_idx])
        end = int(offsets[local_idx + 1])

        data = {}
        for field_name, mm in field_memmaps.items():
            # mm shape: (total_tokens, *extra_dims)
            # Read slice [start:end, ...] for this sample
            arr = mm[start:end]  # [seq_len, ...] or [seq_len]

            # np.memmap returns a view; np.ascontiguousarray materializes
            # the data into a contiguous buffer more efficiently than .copy(),
            # and torch.from_numpy can then share that buffer without an
            # additional copy.
            tensor = torch.from_numpy(np.ascontiguousarray(arr))

            # Apply precision conversion for float tensors (skip int8 quantized data)
            if tensor.is_floating_point():
                tensor = self._convert_precision(tensor)

            # Add batch dimension [1, seq_len, ...]
            tensor = tensor.unsqueeze(0)
            data[field_name] = tensor

        # Dequantize int8 quantized hidden_states and target_hiddens
        if self._int8_quantization:
            target_dtype = (
                torch.bfloat16 if self._storage_precision == "bfloat16" else torch.float16
            )
            for base_name in ("hidden_states", "target_hiddens"):
                int8_key = f"{base_name}_int8"
                scales_key = f"{base_name}_scales"
                if int8_key in data and scales_key in data:
                    data[base_name] = self._dequantize_per_token_absmax_int8(
                        data.pop(int8_key), data.pop(scales_key), target_dtype=target_dtype
                    )

        # Generate attention_mask
        if "input_ids" in data:
            data["attention_mask"] = torch.ones_like(data["input_ids"])

        return data


class OfflineVLMEagle3Dataset(OfflineEagle3Dataset):
    def _load_ckpt(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load a checkpoint file.

        Args:
            idx: Index of the checkpoint file

        Returns:
            Dictionary containing input_ids, target_hiddens,
                hidden_states, and loss_mask, or None if loading fails
        """
        ckpt_path = self.ckpt_files[idx]

        try:
            data = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            warnings.warn(
                f"Failed to load checkpoint {ckpt_path}: {e}. Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate required keys
        required_keys = [
            "input_ids",  # B, N
            "target_hiddens",  # B, N, D
            "hidden_states",  # B, N, 3*D
            "loss_mask",  # B, N
        ]
        # position_ids and inputs_embeds are optional keys,
        # - HF Transformers backend saves position_ids (torch.Tensor)
        # - vLLM backend may save position_ids as None (hook not capturing)
        optional_tensor_keys = [
            "position_ids",  # 3, B, N (optional, vLLM backend may be None)
            "inputs_embeds",  # B, N, D (optional)
        ]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            warnings.warn(
                f"Checkpoint {ckpt_path} is missing required keys: {missing_keys}. "
                f"Skipping this file.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

        # Validate tensor types for required keys
        for key in required_keys:
            if not isinstance(data[key], torch.Tensor):
                warnings.warn(
                    f"Value for key '{key}' in {ckpt_path} is not a torch.Tensor. "
                    f"Skipping this file.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None

        # Validate tensor types for optional keys
        for key in optional_tensor_keys:
            if key in data and not isinstance(data[key], torch.Tensor):
                data[key] = None

        attention_mask = torch.ones_like(data["input_ids"])
        data["attention_mask"] = attention_mask  # B, N
        return data


class LengthBucketSampler(Sampler):
    """Sampler that groups samples by sequence length to minimize padding waste.

    Sorts all samples by length, divides them into buckets of ``bucket_size``,
    shuffles the buckets (and samples within each bucket) every epoch, then
    yields indices one by one.  Because samples in the same bucket have
    similar lengths, the DataCollator's padding overhead is greatly reduced.

    Works with both single-GPU and distributed (DDP / DeepSpeed) training:
    - Single-GPU: use directly as ``sampler`` in DataLoader.
    - Distributed: wrap with ``DistributedSampler`` or pass to
      HuggingFace Trainer which handles distribution automatically.

    Args:
        dataset: A ``MemmapOfflineEagle3Dataset`` (must expose
            ``get_all_sample_lengths()``).
        batch_size: Per-device batch size.  Used to decide the default
            ``bucket_size`` when it is not given explicitly.
        bucket_size: Number of samples per bucket.  Larger values give
            better length homogeneity but less randomness.  Defaults to
            ``batch_size * 50``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        bucket_size: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size or max(batch_size * 20, 100)
        self.seed = seed
        self.epoch = 0

        # Pre-compute lengths (O(N) but only reads offsets, no data IO)
        if hasattr(dataset, "get_all_sample_lengths"):
            self._lengths = dataset.get_all_sample_lengths()
        else:
            # Fallback for non-memmap datasets
            self._lengths = np.arange(len(dataset))

        # Sort indices by length once
        self._sorted_indices = np.argsort(self._lengths)

        rank0_print(
            f"[LengthBucketSampler] {len(dataset)} samples, "
            f"bucket_size={self.bucket_size}, "
            f"length range=[{self._lengths.min()}, {self._lengths.max()}], "
            f"median={int(np.median(self._lengths))}"
        )

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # Split sorted indices into buckets
        indices = self._sorted_indices.copy()
        buckets = [
            indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)
        ]

        # Shuffle bucket order
        rng.shuffle(buckets)

        # Shuffle within each bucket
        for bucket in buckets:
            rng.shuffle(bucket)

        # Yield all indices
        for bucket in buckets:
            yield from bucket.tolist()

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling (called by Trainer / DistributedSampler)."""
        self.epoch = epoch


@DatasetBuilderFactory.register("offline", "LLM")
class OfflineLLMDatasetBuilder(DatasetBuilder):
    def __init__(self, file_pattern: str = "*.ckpt", cache_in_memory: bool = False, **kwargs: Any):
        self.file_pattern = file_pattern
        self.cache_in_memory = cache_in_memory

    def build_dataset(self, datapath: str, **kwargs: Any) -> Dataset:
        """
        Create offline datasets from pre-computed files.

        Automatically detects data format in the following order:
        1. memmap format: <datapath>/memmap_data/metadata.json
            or <datapath>/rank_*/memmap_data/metadata.json
        2. Falls back to individual .ckpt files if no memmap format found.
        """
        # Detect memmap format
        memmap_dirs = self._find_memmap_dirs(datapath)
        if memmap_dirs:
            rank0_print(
                f"[IO-Optimized] Found {len(memmap_dirs)} memmap dir(s), "
                f"using MemmapOfflineEagle3Dataset for zero-copy IO"
            )
            return MemmapOfflineEagle3Dataset(memmap_dirs=memmap_dirs)

        # Fall back to individual .ckpt files
        rank0_print(
            f"[IO] No memmap data found under {datapath}, "
            f"falling back to individual .ckpt files."
        )
        return OfflineEagle3Dataset(
            data_dir=datapath,
            file_pattern=self.file_pattern,
            cache_in_memory=self.cache_in_memory,
        )

    @staticmethod
    def _find_memmap_dirs(datapath: str) -> list:
        """
        Find all memmap data directories under datapath.

        Checks:
        1. <datapath>/memmap_data/metadata.json
        2. <datapath>/rank_*/memmap_data/metadata.json

        Returns:
            List of memmap directory paths, or empty list if none found.
        """
        memmap_dirs = []

        # Check single directory layout
        single_dir = os.path.join(datapath, "memmap_data")
        if os.path.exists(os.path.join(single_dir, "metadata.json")):
            memmap_dirs.append(single_dir)
            return memmap_dirs

        # Check multi-rank directory layout
        if os.path.isdir(datapath):
            for entry in sorted(os.listdir(datapath)):
                if entry.startswith("rank_"):
                    rank_memmap = os.path.join(datapath, entry, "memmap_data")
                    if os.path.exists(os.path.join(rank_memmap, "metadata.json")):
                        memmap_dirs.append(rank_memmap)

        return memmap_dirs

    def get_data_collator(self) -> Any:
        return DataCollatorWithPadding()


@DatasetBuilderFactory.register("offline", "VLM", "qwen2_5_vl")
@DatasetBuilderFactory.register("offline", "VLM", "qwen3_vl")
class OfflineVLMDatasetBuilder(DatasetBuilder):
    def __init__(self, file_pattern: str = "*.ckpt", cache_in_memory: bool = False, **kwargs: Any):
        self.file_pattern = file_pattern
        self.cache_in_memory = cache_in_memory

    def build_dataset(self, datapath: str, **kwargs: Any) -> Dataset:
        """
        Create offline datasets from pre-computed files.

        Automatically detects data format: memmap > individual .ckpt.
        """
        memmap_dirs = OfflineLLMDatasetBuilder._find_memmap_dirs(datapath)
        if memmap_dirs:
            rank0_print(
                f"[IO-Optimized] Found {len(memmap_dirs)} memmap dir(s), "
                f"using MemmapOfflineEagle3Dataset for zero-copy IO"
            )
            return MemmapOfflineEagle3Dataset(memmap_dirs=memmap_dirs)

        return OfflineVLMEagle3Dataset(
            data_dir=datapath,
            file_pattern=self.file_pattern,
            cache_in_memory=self.cache_in_memory,
        )

    def get_data_collator(self) -> Any:
        return VLMDataCollatorWithPadding()


@DatasetBuilderFactory.register("offline", "VLM", "hunyuan_vl")
class OfflineVLMHunyuanVLDatasetBuilder(DatasetBuilder):
    def __init__(self, file_pattern: str = "*.ckpt", cache_in_memory: bool = False, **kwargs: Any):
        self.file_pattern = file_pattern
        self.cache_in_memory = cache_in_memory

    def build_dataset(self, datapath: str, **kwargs: Any) -> Dataset:
        """
        Create offline datasets from pre-computed files.

        Automatically detects data format: memmap > individual .ckpt.
        """
        memmap_dirs = OfflineLLMDatasetBuilder._find_memmap_dirs(datapath)
        if memmap_dirs:
            rank0_print(
                f"[IO-Optimized] Found {len(memmap_dirs)} memmap dir(s), "
                f"using MemmapOfflineEagle3Dataset for zero-copy IO"
            )
            return MemmapOfflineEagle3Dataset(memmap_dirs=memmap_dirs)

        return OfflineVLMEagle3Dataset(
            data_dir=datapath,
            file_pattern=self.file_pattern,
            cache_in_memory=self.cache_in_memory,
        )

    def get_data_collator(self) -> Any:
        return VLMHunyuanDataCollatorWithPadding()
