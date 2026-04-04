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

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from transformers import Trainer

from angelslim.utils import rank0_print
from angelslim.utils.lazy_imports import deepspeed


def _shift_left(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """Shift tensor left by n positions along dim=1 with zero-padding on the right.

    Equivalent to calling ``padding(tensor, left=False)`` n times, but avoids
    creating intermediate tensors for each shift step.
    """
    if n <= 0:
        return tensor
    seq_len = tensor.shape[1]
    if n >= seq_len:
        return torch.zeros_like(tensor)
    # Slice off the first n tokens and pad n zeros on the right
    sliced = tensor[:, n:, ...]
    pad_shape = list(tensor.shape)
    pad_shape[1] = n
    zeros = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((sliced, zeros), dim=1)


def _gpu_mem_stats(device=None):
    """Return GPU memory stats in MB for diagnostics."""
    if not torch.cuda.is_available():
        return {}
    if device is None:
        device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
    reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return {
        "allocated_MB": round(allocated, 1),
        "reserved_MB": round(reserved, 1),
        "max_allocated_MB": round(max_allocated, 1),
    }


class Eagle3Trainer(Trainer, ABC):
    """
    EAGLE3 Trainer for speculative decoding training.

    Implements training logic for EAGLE3 model using a draft model to predict
    tokens based on hidden states from a target model.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        length: int,
        enable_profiler: bool = False,
        custom_train_sampler=None,
        **kwargs,
    ):
        """
        Initialize the OnlineEagle3Trainer.

        Args:
            draft_model: Draft model for token prediction
            length: Number of speculative decoding steps
            enable_profiler: Whether to enable PyTorch Profiler for performance analysis
            custom_train_sampler: Optional custom sampler for training DataLoader
                (e.g. LengthBucketSampler for reducing padding waste)
            **kwargs: Additional arguments passed to parent Trainer
        """
        super().__init__(model=draft_model, **kwargs)
        self.length = length
        self._custom_train_sampler = custom_train_sampler
        self._train_start_time = None
        self._pending_log: dict = (
            {}
        )  # cache acc/ploss log for merging with base Trainer's loss log
        self._pending_log_count: int = 0  # accumulated batch count for averaging the cached log

        # Performance profiling counters (tracks micro-steps, i.e. compute_loss calls)
        self._perf_micro_step_count: int = 0
        self._perf_data_prep_time: float = 0.0
        self._perf_forward_time: float = 0.0
        self._perf_total_step_time: float = 0.0
        self._perf_last_step_end: float = 0.0
        self._perf_dataloader_wait_time: float = 0.0
        self._perf_log_interval: int = 20  # Log performance stats every N global steps
        self._perf_last_logged_global_step: int = 0  # Last global step when PERF was logged

        # PyTorch Profiler
        self._enable_profiler = enable_profiler
        self._profiler: Optional[profile] = None
        self._profiler_started = False

    def _get_train_sampler(self, dataset=None):
        """Override to use custom sampler (e.g. LengthBucketSampler) if provided."""
        if self._custom_train_sampler is not None:
            return self._custom_train_sampler
        return super()._get_train_sampler(dataset)

    def _setup_profiler(self):
        """Setup PyTorch Profiler with schedule for performance analysis."""
        if not self._enable_profiler:
            return

        profiler_output_dir = os.path.join(self.args.output_dir, "profiler")
        os.makedirs(profiler_output_dir, exist_ok=True)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Profiler schedule is defined in terms of global steps.
        # Since profiler.step() is called per micro-step (each compute_loss call),
        # we multiply by gradient_accumulation_steps to convert global steps to micro-steps.
        grad_accum = self.args.gradient_accumulation_steps
        wait_global = 5  # Skip first 5 global steps (warmup / JIT compilation)
        warmup_global = 1  # 1 global step warmup
        active_global = 3  # Actively profile 3 global steps

        profiler_schedule = schedule(
            wait=wait_global * grad_accum,
            warmup=warmup_global * grad_accum,
            active=active_global * grad_accum,
            repeat=1,
        )

        # Total profiler.step() calls needed = (wait + warmup + active) * grad_accum
        self._profiler_total_steps = (wait_global + warmup_global + active_global) * grad_accum
        self._profiler_step_count = 0

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self._profiler = profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=tensorboard_trace_handler(
                profiler_output_dir,
                worker_name=f"rank{local_rank}",
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )

        rank0_print("[PROFILER] PyTorch Profiler enabled")
        rank0_print(f"[PROFILER] Output dir: {profiler_output_dir}")
        rank0_print(
            f"[PROFILER] Schedule: wait={wait_global} global steps, "
            f"warmup={warmup_global} global steps, active={active_global} global steps, repeat=1"
        )
        rank0_print(
            f"[PROFILER] Will profile global steps {wait_global + warmup_global + 1}-"
            f"{wait_global + warmup_global + active_global}, then auto-stop"
        )
        rank0_print(
            f"[PROFILER] Total profiler micro-steps: {self._profiler_total_steps} "
            f"(grad_accum={grad_accum})"
        )
        rank0_print(f"[PROFILER] Activities: {[a.name for a in activities]}")
        rank0_print(
            "[PROFILER] Features: record_shapes=True, profile_memory=True, "
            "with_stack=True, with_flops=True"
        )
        rank0_print(f"[PROFILER] View results with: tensorboard --logdir {profiler_output_dir}")

    def train(self, *args, **kwargs):
        """Override train method to record training start time for estimating remaining time."""
        self._train_start_time = time.time()
        self._perf_last_step_end = self._train_start_time

        # Log initial GPU memory and training config summary
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        mem = _gpu_mem_stats()
        rank0_print("[PERF] === Training Performance Profiling Enabled ===")
        rank0_print(f"[PERF] Initial GPU memory: {mem}")
        rank0_print(
            "[PERF] Draft model parameters: "
            f"{sum(p.numel() for p in self.model.parameters()):,} total, "
            f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable"
        )
        rank0_print(f"[PERF] Training length (spec steps): {self.length}")
        rank0_print(f"[PERF] Gradient checkpointing (args): {self.args.gradient_checkpointing}")
        rank0_print(f"[PERF] Per-device batch size: {self.args.per_device_train_batch_size}")
        rank0_print(f"[PERF] Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
        rank0_print(f"[PERF] World size: {self.args.world_size}")
        global_bach_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        rank0_print(f"[PERF] Effective batch size: {global_bach_size}")
        rank0_print(f"[PERF] DeepSpeed enabled: {self.is_deepspeed_enabled}")
        rank0_print(f"[PERF] bf16: {self.args.bf16}, fp16: {self.args.fp16}")
        rank0_print(f"[PERF] Dataloader num_workers: {self.args.dataloader_num_workers}")
        rank0_print(f"[PERF] Dataloader prefetch_factor: {self.args.dataloader_prefetch_factor}")

        # Setup and start PyTorch Profiler
        self._setup_profiler()
        if self._profiler is not None:
            self._profiler.__enter__()
            self._profiler_started = True

        try:
            result = super().train(*args, **kwargs)
        finally:
            # Safety net: stop profiler if it wasn't already stopped during training
            self._stop_profiler()

        return result

    def _stop_profiler(self):
        """Stop the profiler if it is still running."""
        if self._profiler is not None and self._profiler_started:
            self._profiler.__exit__(None, None, None)
            self._profiler_started = False
            rank0_print(
                f"[PROFILER] Profiler stopped. Trace files saved to: "
                f"{os.path.join(self.args.output_dir, 'profiler')}"
            )

    def log(self, logs: dict, start_time: Optional[float] = None) -> None:
        """
        rewrite log method to merge acc/ploss log with base Trainer's loss log.
        """
        if "loss" in logs and self._pending_log:
            # merge cached acc/ploss data (average)
            count = max(self._pending_log_count, 1)
            acc_ploss = {k: v / count for k, v in self._pending_log.items()}
            merged = {}

            # step
            max_steps = 0
            if self.state is not None:
                global_step = self.state.global_step
                max_steps = self.state.max_steps
                merged["step"] = global_step

            # epoch
            if "epoch" in logs:
                merged["epoch"] = logs["epoch"]
            if "loss" in logs:
                merged["loss"] = logs["loss"]
            if "grad_norm" in logs:
                merged["grad_norm"] = logs["grad_norm"]

            if "learning_rate" in logs:
                merged["lr"] = logs["learning_rate"]

            # acc/ploss
            merged.update(acc_ploss)

            # remaining_time
            if (
                self.state is not None
                and self._train_start_time is not None
                and global_step > 0
                and max_steps > 0
            ):
                elapsed = time.time() - self._train_start_time
                time_per_step = elapsed / global_step
                remaining_seconds = int(time_per_step * (max_steps - global_step))
                hours, remainder = divmod(remaining_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                merged["remaining_time"] = f"{hours:02d}h:{minutes:02d}m:{seconds:02d}s"

            self._pending_log.clear()
            self._pending_log_count = 0
            super().log(merged, start_time)
        else:
            super().log(logs, start_time)

    @property
    def draft_model(self) -> nn.Module:
        """Get the draft model."""
        return self.model

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
        return_outputs: bool = False,
    ) -> Tuple[List[torch.Tensor], List, List[float]]:
        """
        Compute the training loss for the model.

        Args:
            model: The model for which to compute the loss
            inputs: Input data dictionary with input_ids, attention_mask,
                loss_mask, position_ids
            num_items_in_batch: Number of items in batch (unused)
            return_outputs: Whether to return model outputs (unused)

        Returns:
            Tuple of (prediction_losses, value_losses, accuracies) for each step
        """
        step_start = time.time()

        # Measure time waiting for data (time between last step end and this step start)
        if self._perf_last_step_end > 0:
            dataloader_wait = step_start - self._perf_last_step_end
            self._perf_dataloader_wait_time += dataloader_wait

        # Measure data preparation time
        t0 = time.time()
        with record_function("data_preparation"):
            data_for_draft_model = self.prepare_data_for_draft_model(inputs)

            attention_mask = data_for_draft_model["attention_mask"]  # Batch x Seq
            position_ids = data_for_draft_model["position_ids"]
            input_ids = data_for_draft_model["input_ids"]  # Batch x Seq
            target_logits = data_for_draft_model["target_logits"]  # Batch x Seq x Vocab
            loss_mask = data_for_draft_model["loss_mask"]  # Batch x Seq x 1
            hidden_states = data_for_draft_model["hidden_states"]  # Batch x Seq x Hidden

            hidden_states = self.down_project_hidden_states(hidden_states)
            attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
                hidden_states, attention_mask, position_ids
            )
        data_prep_time = time.time() - t0
        self._perf_data_prep_time += data_prep_time

        # Measure forward + loss computation time
        t1 = time.time()
        with record_function("draft_model_forward_and_loss"):
            loss = self.draft_model_training_time_test(
                input_ids,
                hidden_states,
                attention_mask,
                position_ids,
                target_logits,
                loss_mask,
                log_prefix="train",
            )
        forward_time = time.time() - t1
        self._perf_forward_time += forward_time

        total_step_time = time.time() - step_start
        self._perf_total_step_time += total_step_time
        self._perf_micro_step_count += 1
        self._perf_last_step_end = time.time()

        # Step the profiler and auto-stop after the scheduled cycle completes
        if self._profiler is not None and self._profiler_started:
            self._profiler.step()
            self._profiler_step_count += 1
            if self._profiler_step_count >= self._profiler_total_steps:
                self._stop_profiler()
                profiler_macro_steps = self._profiler_step_count // max(
                    self.args.gradient_accumulation_steps, 1
                )
                rank0_print(
                    f"[PROFILER] Profiling complete after {self._profiler_step_count} micro-steps"
                    f" (~{profiler_macro_steps} global steps)"
                )

        # Log performance stats periodically based on global steps
        current_global_step = self.state.global_step if self.state is not None else 0
        if (
            current_global_step > 0
            and current_global_step >= self._perf_last_logged_global_step + self._perf_log_interval
            and self._perf_micro_step_count > 0
        ):
            n = self._perf_micro_step_count
            avg_total = self._perf_total_step_time / n * 1000
            avg_data_prep = self._perf_data_prep_time / n * 1000
            avg_forward = self._perf_forward_time / n * 1000
            avg_dl_wait = self._perf_dataloader_wait_time / n * 1000
            mem = _gpu_mem_stats()

            # Log input shape for understanding batch characteristics
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            rank0_print(
                f"[PERF] Global step {current_global_step} (micro-steps: {n}) | "
                f"avg_step={avg_total:.1f}ms | "
                f"avg_data_prep={avg_data_prep:.1f}ms | "
                f"avg_forward={avg_forward:.1f}ms | "
                f"avg_dl_wait={avg_dl_wait:.1f}ms | "
                f"batch={batch_size}x{seq_len} | "
                f"GPU_mem={mem}"
            )

            # Reset counters to get recent averages
            self._perf_last_logged_global_step = current_global_step
            self._perf_micro_step_count = 0
            self._perf_data_prep_time = 0.0
            self._perf_forward_time = 0.0
            self._perf_total_step_time = 0.0
            self._perf_dataloader_wait_time = 0.0

        return loss

    @abstractmethod
    def prepare_data_for_draft_model(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare data for draft model training.
        """
        pass

    def down_project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Down project hidden states for draft model training.
        """
        # Step 4: Prepare hidden states with gradient tracking
        # Cast hidden_states to the model's dtype to avoid dtype mismatch
        # (e.g. pre-computed hidden states may be float16 while model weights are bfloat16)
        model_dtype = next(self.draft_model.parameters()).dtype
        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(model_dtype)
        if not hidden_states.requires_grad:
            hidden_states.requires_grad = True
        hidden_states = self.draft_model.combine_hidden_states(hidden_states)
        return hidden_states

    def prepare_attention_mask_and_position_ids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attention mask for draft model training.

        When all tokens are valid (attention_mask is all-ones), returns None
        so that SDPA can use its built-in efficient causal mask implementation
        (Flash Attention / memory-efficient attention) instead of materializing
        a full [bsz, 1, seq_len, seq_len] mask tensor.
        """
        # Step 5: Prepare attention mask and position IDs
        batch_size, seq_length, _ = hidden_states.shape
        device = hidden_states.device

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            if position_ids.ndim == 3:
                # MRoPE format: (3, batch, seq_len), keep as-is
                position_ids = position_ids.long()
            else:
                position_ids = position_ids.view(-1, seq_length).long()

        # When attention_mask is all-ones (no padding), return None to let
        # SDPA use its built-in causal mask (avoids allocating a
        # [bsz, 1, seq_len, seq_len] tensor).
        if attention_mask is None or attention_mask.all():
            return None, position_ids

        attention_mask = self.draft_model.prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        return attention_mask, position_ids

    def draft_model_training_time_test(
        self,
        input_ids,
        hidden_states,
        attention_mask,
        position_ids,
        target_logits,
        loss_mask,
        log_prefix="",
    ):
        _, seq_length, _ = hidden_states.shape

        # Step 6: Initialize containers for losses, accuracies and cache
        plosses, acces = [], []
        cache_hidden = [[], []]

        # Pre-compute target distribution outside the loop to avoid
        # redundant computation. Since target_logits is already in draft vocab space
        # (from TargetHead.forward_with_vocab_mapping), we skip the t2d gather.
        # Build a list of (target_log_p, position_mask, target_p_argmax) for each iteration,
        # applying the padding shift incrementally.
        precomputed_targets = []
        with record_function("precompute_targets"):
            for idx in range(self.length):
                # Use direct slicing instead of repeated padding() calls.
                # _shift_left(t, n) shifts tensor left by n positions (equivalent to
                # calling padding(t, left=False) n times) without intermediate tensors.
                #
                # target_logits and input_ids need (idx+1) shift because the initial
                # padding(left=False) that was previously in prepare_data_for_draft_model
                # has been folded into this loop.
                # loss_mask needs idx shift (it was never padded in prepare_data).
                _tl = _shift_left(target_logits, idx + 1)
                _ids = _shift_left(input_ids, idx + 1)
                _lm = _shift_left(loss_mask, idx) if idx > 0 else loss_mask
                with torch.no_grad():
                    position_mask = _lm
                    target_log_p = F.log_softmax(_tl.float(), dim=2).detach()
                    target_p_argmax = _tl.argmax(-1)
                precomputed_targets.append(
                    (target_log_p, position_mask, target_p_argmax, _lm, _ids)
                )
        del target_logits

        # Step 7: Iterative speculative decoding training loop
        for idx in range(self.length):
            target_log_p, position_mask, target_p_argmax, cur_loss_mask, cur_input_ids = (
                precomputed_targets[idx]
            )

            # Step 7.1: Get input embeddings with gradient tracking
            with record_function(f"spec_step_{idx}/embed"):
                # cur_input_ids from precomputed_targets already has the correct
                # left-shift applied (idx+1 positions), so use it for all steps.
                inputs_embeds = self.draft_model.embed_input_ids(cur_input_ids)
                if not inputs_embeds.requires_grad:
                    inputs_embeds.requires_grad = True

            # Step 7.2: Encode through draft model layers
            with record_function(f"spec_step_{idx}/encode_layers"):
                if (
                    getattr(self.draft_model, "gradient_checkpointing", False)
                    and self.draft_model.training
                ):
                    hidden_states, cache_hidden = torch.utils.checkpoint.checkpoint(
                        self.draft_model.encode_layers,
                        inputs_embeds,
                        hidden_states,
                        cache_hidden,
                        attention_mask,
                        position_ids,
                        True,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, cache_hidden = self.draft_model.encode_layers(
                        inputs_embeds=inputs_embeds,
                        hidden_states=hidden_states,
                        cache_hidden=cache_hidden,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                    )

            # Step 7.3: Compute logits from hidden states
            with record_function(f"spec_step_{idx}/compute_logits"):
                logits = self.draft_model.compute_logits(hidden_states)

            # Step 7.5: Compute loss (KL divergence between target and draft distributions)
            # Optimization: use F.kl_div with log_target=True for fused CUDA kernel.
            # KL(target || draft) differs from cross-entropy by a constant (target entropy),
            # so gradients are identical. Loss values will differ from the original
            # cross-entropy formulation but training behavior is unchanged.
            with record_function(f"spec_step_{idx}/loss_computation"):
                out_log_p = F.log_softmax(logits, dim=2)
                # Compute per-token KL divergence, then apply position mask
                kl_per_vocab = F.kl_div(
                    out_log_p, target_log_p, log_target=True, reduction="none"
                )  # [B, S, draft_vocab_size]
                # Sum over vocab dim -> [B, S, 1], apply position mask, then mean
                loss = (kl_per_vocab.sum(dim=2, keepdim=True) * position_mask).mean()

            # Step 7.6: Compute accuracy
            with torch.no_grad():
                correct = (logits.argmax(-1) == target_p_argmax) * position_mask.squeeze(-1)
                accuracy = correct.sum().item() / (cur_loss_mask.sum().item() + 1e-6)

            # Step 7.7: Store loss and accuracy
            plosses.append(loss)
            acces.append(accuracy)

        # Step 8: Compute weighted loss
        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])

        log = {f"{log_prefix}/acc_{i}": acces[i] for i in range(len(acces))}
        log.update({f"{log_prefix}/ploss_{i}": plosses[i].item() for i in range(len(plosses))})
        # Cache log for merging with base Trainer's loss log
        for k, v in log.items():
            self._pending_log[k] = self._pending_log.get(k, 0.0) + v
        self._pending_log_count += 1
        # Step 9: Return loss
        return ploss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override save_model to handle DeepSpeed ZeRO-3 model saving.

        Args:
            output_dir: Directory to save the model. If None, uses self.args.output_dir
            _internal_call: Internal flag used by Trainer
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        # Check if using DeepSpeed ZeRO-3
        is_deepspeed_zero3 = (
            self.is_deepspeed_enabled
            and hasattr(self.accelerator.state, "deepspeed_plugin")
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
        )

        if is_deepspeed_zero3:
            # Handle ZeRO-3 model saving
            self._save_zero3_model(output_dir, _internal_call)
        else:
            # Fall back to parent class save_model
            super().save_model(output_dir, _internal_call)

    def _save_zero3_model(self, output_dir: str, _internal_call: bool = False):
        """
        Save model with DeepSpeed ZeRO-3 specific logic.

        Args:
            output_dir: Directory to save the model
            _internal_call: Internal flag used by Trainer
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save with DeepSpeed's state_dict gathering
        # All processes must participate in parameter gathering to avoid deadlock
        with deepspeed.zero.GatheredParameters(self.model.parameters()):
            state_dict = self.model.state_dict()

        draft_model_state_dict = {k: v for k, v in state_dict.items() if "embed" not in k}

        # Only main process saves the model
        if self.args.should_save and self.accelerator.is_main_process:
            self.model.save_pretrained(
                output_dir,
                is_main_process=True,
                state_dict=draft_model_state_dict,
                save_function=torch.save,
            )

            # Save training arguments
            from transformers.trainer import TRAINING_ARGS_NAME

            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Wait for all processes
        self.accelerator.wait_for_everyone()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        """
        data_for_draft_model = self.prepare_data_for_draft_model(inputs)

        attention_mask = data_for_draft_model["attention_mask"]
        # inputs_embeds = data_for_draft_model["inputs_embeds"]
        position_ids = data_for_draft_model.get("position_ids", None)
        input_ids = data_for_draft_model["input_ids"]
        target_logits = data_for_draft_model["target_logits"]
        loss_mask = data_for_draft_model["loss_mask"]
        hidden_states = data_for_draft_model["hidden_states"]

        with torch.no_grad():
            hidden_states = self.down_project_hidden_states(hidden_states)
            attention_mask, position_ids = self.prepare_attention_mask_and_position_ids(
                hidden_states, attention_mask, position_ids
            )
            loss = self.draft_model_training_time_test(
                input_ids,
                hidden_states,
                attention_mask,
                position_ids,
                target_logits,
                loss_mask,
                log_prefix="eval",
            )
        return (loss, None, None)
