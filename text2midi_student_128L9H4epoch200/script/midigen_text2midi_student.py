#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
midigen_text2midi_student.py  (W&B REMOVED + Thesis Tracking ADDED)

✅ Keeps the original training / generation / distillation logic,
❌ Removes all Weights & Biases (wandb) code,
✅ Adds a local, thesis-friendly tracker that writes JSONL + CSV summaries,
✅ Tracks model size (params + MB), loss components, LR, grad norms, accuracy, throughput,
✅ Works in Windows PowerShell (writes to output_dir on disk).

USAGE (same commands as before):

Train (accelerate):
  accelerate launch midigen_text2midi_student.py train_accelerate --config configs/config.yaml

Distill:
  accelerate launch midigen_text2midi_student.py distill --config configs/config.yaml ^
    --teacher_ckpt PATH_TO_TEACHER ^
    --student_output_dir PATH_TO_SAVE_STUDENT ^
    --temperature 2.0 --alpha_hard 0.5

Outputs (created inside output_dir / student_output_dir):
  - thesis_run_meta.json
  - thesis_steps.jsonl
  - thesis_epochs.jsonl
  - thesis_steps.csv
  - thesis_epochs.csv
"""

import os
import sys
import re
import math
import time
import json
import glob
import yaml
import torch
import pickle
import random
import logging
import argparse
import jsonlines
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from copy import deepcopy
from typing import Optional, Any, Union, Callable, Dict
from torch import Tensor, argmax
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer, T5EncoderModel, T5Config,
    BertConfig, BertLMHeadModel,
    Trainer, TrainingArguments,
    get_scheduler, PreTrainedModel
)
from accelerate import DistributedDataParallelKwargs, Accelerator
from evaluate import load as load_metric

# Optional: Miditok
try:
    from miditok import REMI, TokenizerConfig  # for REMI tokenizer workflows
except Exception:
    REMI = None
    TokenizerConfig = None

# Optional: MoE
try:
    from st_moe_pytorch import MoE, SparseMoEBlock
except Exception:
    MoE = None
    SparseMoEBlock = None

# Optional: spaCy for sentence dropping
try:
    from spacy.lang.en import English
except Exception:
    English = None

# Optional: aria MIDI (used only in aria variant)
try:
    from aria.data.midi import MidiDict
except Exception:
    MidiDict = None

# ---------- Global speed knobs (NVIDIA) ----------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------- Fixed IO Roots (requested locations) ----------
DEFAULT_INPUT_ROOT = "/home/brett_ece/midi/text2midi_teacher/Input"
DEFAULT_OUTPUT_ROOT = "/home/brett_ece/midi/text2midi_teacher/Output2"
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

# =============================================================================
# Thesis Tracking (NEW) — replaces W&B completely
# =============================================================================

def _safe_json(v: Any) -> Any:
    """Convert common non-JSON types into JSON-safe values."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_json(val) for k, val in v.items()}
    if torch.is_tensor(v):
        return v.detach().cpu().item() if v.numel() == 1 else f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
    return str(v)

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _write_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_safe_json(row), ensure_ascii=False) + "\n")

def _to_csv_from_jsonl(jsonl_path: str, csv_path: str) -> None:
    """Simple JSONL -> CSV flattener (best-effort)."""
    if not os.path.exists(jsonl_path):
        return
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return

    # collect all keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    keys = sorted(keys)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

def model_size_report(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_size_mb": float(total_bytes) / (1024.0 * 1024.0),
        "trainable_size_mb": float(trainable_bytes) / (1024.0 * 1024.0),
    }

def gpu_mem_report() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    return {
        "cuda_mem_alloc_mb": float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0),
        "cuda_mem_reserved_mb": float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0),
        "cuda_max_mem_alloc_mb": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
        "cuda_max_mem_reserved_mb": float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0),
    }

def grad_norm_l2(model: nn.Module) -> float:
    """Compute global L2 grad norm (ignores params with None grad)."""
    sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq += float(g.float().pow(2).sum().item())
    return float(math.sqrt(max(sq, 0.0)))

def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> Dict[str, Any]:
    """
    logits: [B, T, V]
    targets: [B, T]
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # [B,T]
        mask = (targets != pad_id)
        denom = int(mask.sum().item())
        if denom == 0:
            return {"tok_acc": 0.0, "tok_count": 0}
        correct = int(((preds == targets) & mask).sum().item())
        return {"tok_acc": float(correct) / float(denom), "tok_count": denom}

class ThesisTracker:
    """
    Local tracker for thesis results:
      - step-level JSONL
      - epoch-level JSONL
      - meta JSON
      - CSV exports at end
    """
    def __init__(self, out_dir: str, run_name: str, accelerator: Optional[Accelerator], config: Dict[str, Any]):
        self.out_dir = out_dir
        self.run_name = run_name
        self.accelerator = accelerator
        self.is_main = True if accelerator is None else accelerator.is_main_process

        self.meta_path = os.path.join(out_dir, "thesis_run_meta.json")
        self.steps_path = os.path.join(out_dir, "thesis_steps.jsonl")
        self.epochs_path = os.path.join(out_dir, "thesis_epochs.jsonl")
        self.steps_csv = os.path.join(out_dir, "thesis_steps.csv")
        self.epochs_csv = os.path.join(out_dir, "thesis_epochs.csv")

        self._t0 = time.time()
        self._last_step_t = time.time()

        if self.is_main:
            os.makedirs(out_dir, exist_ok=True)
            meta = {
                "timestamp_start": _now_iso(),
                "run_name": run_name,
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(accelerator.device) if accelerator is not None else "unknown",
                "num_processes": int(accelerator.num_processes) if accelerator is not None else 1,
                "config": config,
            }
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(_safe_json(meta), f, indent=2)

    def log_step(self, row: Dict[str, Any]) -> None:
        if not self.is_main:
            return
        row = dict(row)
        row["ts"] = _now_iso()
        _write_jsonl(self.steps_path, row)

    def log_epoch(self, row: Dict[str, Any]) -> None:
        if not self.is_main:
            return
        row = dict(row)
        row["ts"] = _now_iso()
        _write_jsonl(self.epochs_path, row)

    def seconds_since_start(self) -> float:
        return float(time.time() - self._t0)

    def step_timing(self) -> float:
        now = time.time()
        dt = now - self._last_step_t
        self._last_step_t = now
        return float(dt)

    def finalize(self) -> None:
        if not self.is_main:
            return
        # Export CSVs for thesis plotting in Excel / Python
        _to_csv_from_jsonl(self.steps_path, self.steps_csv)
        _to_csv_from_jsonl(self.epochs_path, self.epochs_csv)

# =============================================================================

def infer_arch_from_state(state: dict):
    """Infer d_model + max_len from checkpoint tensors."""
    d_model = state["input_emb.weight"].shape[1] if "input_emb.weight" in state else 768
    max_len = state["pos_encoder.pe"].shape[0] if "pos_encoder.pe" in state else 2048
    return d_model, max_len

def arch_from_d_model(d_model: int):
    """
    Match the training config used for the student checkpoint.
    """
    base = dict(
        nhead=8,
        num_layers=18,
        dim_feedforward=256,
        use_moe=False,
        num_experts=4
    )

    if d_model % base["nhead"] != 0:
        raise ValueError(f"d_model={d_model} must be divisible by nhead={base['nhead']}")

    return dict(d_model=d_model, **base)

def _next_output_path(root=DEFAULT_OUTPUT_ROOT, prefix="gen_", ext=".mid"):
    os.makedirs(root, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for name in os.listdir(root):
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))
    n = max(nums) + 1 if nums else 1
    return os.path.join(root, f"{prefix}{n:04d}{ext}")

def load_configs(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_captions_jsonl(path):
    with jsonlines.open(path) as reader:
        return list(reader)

def build_parser():
    p = argparse.ArgumentParser(prog="midigen_text2midi_student.py", description="Unified Text2MIDI toolkit (student + KD)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build_vocab_aria", help="Build custom ARIA-style vocab.pkl")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("build_vocab_remi", help="Build miditok REMI tokenizer pickle")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("dataset_demo_aria", help="Quick test for ARIA dataset item")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("dataset_demo_remi", help="Quick test for REMI dataset item")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("train_basic", help="Train with the simple loop Transformer")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("train_accelerate", help="Accelerate-based training")
    s.add_argument("--config", type=str, required=True)
    s.add_argument("--resume_path", type=str, default=None, help="Optional path to load state_dict")

    s = sub.add_parser("train_hf", help="HF Trainer with frozen FLAN-T5 encoder + BERT decoder")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("generate", help="Generate a single MIDI for a caption")
    s.add_argument("--config", type=str, required=True, help="Path to config.yaml used to build the model")
    s.add_argument("--caption", type=str, required=True)
    s.add_argument("--model_path", type=str, required=True, help="pytorch_model.bin path")
    s.add_argument("--tokenizer_pkl", type=str, required=True, help="vocab_remi.pkl path")
    s.add_argument("--max_len", type=int, default=2000)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_ROOT)

    s = sub.add_parser("generate_accelerate", help="Accelerated batch generation from captions JSONL")
    s.add_argument("--captions_jsonl", type=str, required=True)
    s.add_argument("--model_path", type=str, required=True)
    s.add_argument("--tokenizer_pkl", type=str, required=True)
    s.add_argument("--batch_size", type=int, default=8)
    s.add_argument("--max_len", type=int, default=2000)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--out_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    s = sub.add_parser("split_captions", help="Split test-set captions into chunks")
    s.add_argument("--input_jsonl", type=str, required=True)
    s.add_argument("--output_dir", type=str, default=DEFAULT_INPUT_ROOT)
    s.add_argument("--num_splits", type=int, default=6)

    # KD
    s = sub.add_parser("distill", help="Knowledge-distillation training from a big teacher model")
    s.add_argument("--config", type=str, required=True)
    s.add_argument("--teacher_ckpt", type=str, required=True)
    s.add_argument("--student_output_dir", type=str, required=True)
    s.add_argument("--temperature", type=float, default=2.0)
    s.add_argument("--alpha_hard", type=float, default=0.5)

    return p

# ===== Transformer and components =====

try:
    from torch.nn.modules.activation import MultiheadAttention
except Exception:
    from torch.nn import MultiheadAttention

def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3)
    d = x.shape[-1] // 2
    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()
    x1_new = x1.mul(cos) - x2.mul(sin)
    x2_new = x2.mul(cos) + tmp.mul(sin)
    x = torch.cat((x1_new, x2_new), dim=-1)
    x = x.permute(0, 2, 1, 3)
    return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 batch_first: bool = True, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_first = batch_first
        self.dim_head = embed_dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.heads = num_heads
        hidden_dim = self.dim_head * num_heads
        self.to_qkv = nn.Linear(embed_dim, hidden_dim * 3, bias=False, **factory_kwargs)
        self.to_out = nn.Linear(hidden_dim, embed_dim, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)
        b, n, _ = x.size()
        q, k, v = torch.chunk(self.to_qkv(x), chunks=3, dim=-1)
        q, k, v = map(lambda t: t.contiguous().view(b, self.heads, n, -1), (q, k, v))
        self.freqs_cis = precompute_freqs_cis(
            seq_len=n,
            n_elem=self.embed_dim // self.heads,
            base=10000,
            dtype=x.dtype,
        ).to(x.device)
        freqs_cis = self.freqs_cis[: x.shape[1]]
        # q = apply_rotary_emb(q, freqs_cis)
        # k = apply_rotary_emb(k, freqs_cis)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.contiguous().view(b, n, -1)
        out = self.dropout(out)
        return self.to_out(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def _generate_square_subsequent_mask(sz: int, device=None, dtype=None) -> Tensor:
    device = device or torch.device('cpu')
    dtype = dtype or torch.float32
    return torch.triu(torch.full((sz, sz), float('-inf'), dtype=dtype, device=device), diagonal=1)

def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    src_size = src.size()
    if len(src_size) == 2:
        return src_size[0]
    return src_size[1] if batch_first else src_size[0]

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _detect_is_causal_mask(mask: Optional[Tensor], is_causal: Optional[bool] = None, size: Optional[int] = None) -> bool:
    make_causal = (is_causal is True)
    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False
    return make_causal

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 use_moe: bool = False, num_experts: int = 16, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs)
        self.use_moe = use_moe
        if use_moe and (MoE is not None) and (SparseMoEBlock is not None):
            self.moe = MoE(dim=d_model, num_experts=num_experts, gating_top_n=2, threshold_train=0.2,
                           threshold_eval=0.2, capacity_factor_train=1.25, capacity_factor_eval=2.0,
                           balance_loss_coef=1e-2, router_z_loss_coef=1e-3).to(device)
            self.moe_block = SparseMoEBlock(self.moe, add_ff_before=True, add_ff_after=True).to(device)
            self.linear1 = None
            self.linear2 = None
            self.dropout = nn.Dropout(dropout)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation) if isinstance(activation, str) else activation

    def _sa_block(self, x: Tensor, is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, is_causal=is_causal)
        return self.dropout1(x)

    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(self, tgt: Tensor, memory: Tensor, memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = False,
                memory_is_causal: bool = False):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            if self.use_moe and (MoE is not None):
                m, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)
                x = x + m
                return x, total_aux_loss, balance_loss, router_z_loss
            else:
                x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            if self.use_moe and (MoE is not None):
                m, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)
                x = x + m
                return x, total_aux_loss, balance_loss, router_z_loss
            else:
                x = self.norm3(x + self._ff_block(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: "TransformerDecoderLayer", num_layers: int, use_moe: bool = False, norm: Optional[nn.Module] = None):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.use_moe = use_moe
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None, memory_is_causal: bool = False):
        output = tgt
        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        if self.use_moe and (MoE is not None):
            sum_total_aux_loss = 0
            for mod in self.layers:
                output, total_aux_loss, _, _ = mod(output, memory, memory_mask=memory_mask,
                                                   memory_key_padding_mask=memory_key_padding_mask,
                                                   tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
                sum_total_aux_loss += total_aux_loss
            if self.norm is not None:
                output = self.norm(output)
            return output, sum_total_aux_loss
        else:
            for mod in self.layers:
                output = mod(output, memory, memory_mask=memory_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
            if self.norm is not None:
                output = self.norm(output)
            return output

class Transformer(nn.Module):
    def __init__(
        self,
        n_vocab: int = 30000,
        d_model: int = 384,
        nhead: int = 4,
        max_len: int = 5000,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        use_moe: bool = False,
        num_experts: int = 16,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.use_moe = use_moe
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        self.input_emb = nn.Embedding(n_vocab, d_model, **factory_kwargs)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len).to(device)

        self.encoder = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)
        
        # Project FLAN-T5 encoder hidden size -> decoder d_model if needed
        enc_dim = int(self.encoder.config.d_model)  # flan-t5-base = 768
        self.encoder_proj = None
        if enc_dim != d_model:
            self.encoder_proj = nn.Linear(enc_dim, d_model, bias=False).to(device)

        for p in self.encoder.parameters():
            p.requires_grad = False

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            use_moe=use_moe,
            num_experts=num_experts,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            **factory_kwargs,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder.eval()
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            use_moe=use_moe,
            norm=decoder_norm,
        )

        self.projection = nn.Linear(d_model, n_vocab).to(device)
        self._reset_parameters()

    def _reset_parameters(self):
    # IMPORTANT: do NOT re-init pretrained encoder
        for name, p in self.named_parameters():
            if (not p.requires_grad) or name.startswith("encoder."):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor,
        tgt: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = True,
        memory_is_causal: bool = False,
    ) -> Tensor:
        if src.dim() != tgt.dim():
            raise RuntimeError("the number of dimensions in src and tgt must be equal")

        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=src, attention_mask=src_mask)
        memory = encoder_outputs.last_hidden_state

        if self.encoder_proj is not None:
            memory = self.encoder_proj(memory)

        tgt = self.input_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        if self.use_moe and (MoE is not None):
            with torch.cuda.amp.autocast(enabled=False):
                output, aux_loss = self.decoder(
                    tgt, memory,
                    memory_mask=memory_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_is_causal=tgt_is_causal,
                    memory_is_causal=memory_is_causal,
                )
        else:
            output = self.decoder(
                tgt, memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            aux_loss = 0

        output = self.projection(output)
        return (output, aux_loss) if (self.use_moe and MoE is not None) else output

    def generate(self, src: Tensor, src_mask: Tensor, max_len: int = 100, temperature: float = 1.0, bos_id: int = 1):
        if src.dim() != 2:
            raise RuntimeError("The src tensor should be 2-dimensional")

        tgt_fin = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)

        for _ in tqdm(range(max_len)):
            tgt = tgt_fin

            if self.use_moe and (MoE is not None):
                output, _ = self.forward(
                    src, src_mask, tgt,
                    memory_mask=None, memory_key_padding_mask=None,
                    tgt_is_causal=True, memory_is_causal=False
                )
            else:
                output = self.forward(
                    src, src_mask, tgt,
                    memory_mask=None, memory_key_padding_mask=None,
                    tgt_is_causal=True, memory_is_causal=False
                )

            logits = output
            output = F.log_softmax(logits / temperature, dim=-1)
            output = output.view(-1, output.size(-1))

            next_tokens = torch.multinomial(torch.exp(output), 1)[-1]
            tgt_fin = torch.cat((tgt_fin, next_tokens.unsqueeze(-1)), dim=1)

        return tgt_fin[:, 1:]

# ===== Datasets =====

class Text2MusicDatasetAria(Dataset):
    def __init__(self, configs, captions, aria_tokenizer, mode="train", shuffle=False):
        assert MidiDict is not None, "aria.data.midi.MidiDict not available"
        self.mode = mode
        self.captions = captions[:]
        if shuffle:
            random.shuffle(self.captions)
        self.dataset_path = configs['raw_data']['raw_data_folders']['midicaps']['folder_path']
        self.artifact_folder = configs['artifact_folder']
        tokenizer_filepath = os.path.join(self.artifact_folder, "vocab.pkl")
        self.aria_tokenizer = aria_tokenizer
        with open(tokenizer_filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        if English is not None:
            self.nlp = English()
            self.nlp.add_pipe('sentencizer')
        else:
            self.nlp = None
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.decoder_max_sequence_length = configs['model']['text2midi_model']['decoder_max_sequence_length']
        print("Length of dataset (ARIA): ", len(self.captions))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        midi_filepath = os.path.join(self.dataset_path, self.captions[idx]['location'])
        midi = MidiDict.from_midi(midi_filepath)
        if len(midi.note_msgs) == 0:
            aria_tokenized_midi = ["<SS>", "<E>"]
        else:
            aria_tokenized_midi = ["<SS>"] + self.aria_tokenizer.tokenize(midi)
        new_sentences = caption
        if self.nlp is not None and random.random() > 0.5:
            sentences = list(self.nlp(caption).sents)
            sent_length = len(sentences)
            if sent_length > 0:
                if sent_length < 4:
                    how_many_to_drop = int(np.floor((20 + random.random()*30)/100*sent_length))
                else:
                    how_many_to_drop = int(np.ceil((20 + random.random()*30)/100*sent_length))
                which = np.random.choice(sent_length, how_many_to_drop, replace=False)
                kept = [sentences[i] for i in range(sent_length) if i not in which.tolist()]
                new_sentences = " ".join(s.text for s in kept)
        inputs = self.t5_tokenizer(new_sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        tokenized_midi = [self.tokenizer[token] for token in aria_tokenized_midi if token in self.tokenizer]
        if len(tokenized_midi) < self.decoder_max_sequence_length:
            labels = F.pad(torch.tensor(tokenized_midi), (0, self.decoder_max_sequence_length - len(tokenized_midi))).to(torch.int64)
        else:
            labels = torch.tensor(tokenized_midi[-self.decoder_max_sequence_length:]).to(torch.int64)
        return input_ids, attention_mask, labels

class Text2MusicDatasetRemi(Dataset):
    def __init__(self, configs, captions, remi_tokenizer, mode="train", shuffle=False):
        self.mode = mode
        self.captions = captions[:]
        if shuffle:
            random.shuffle(self.captions)
        self.dataset_path = configs['raw_data']['raw_data_folders']['midicaps']['folder_path']
        self.artifact_folder = configs['artifact_folder']
        self.remi_tokenizer = remi_tokenizer
        if English is not None:
            self.nlp = English()
            self.nlp.add_pipe('sentencizer')
        else:
            self.nlp = None
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.decoder_max_sequence_length = configs['model']['text2midi_model']['decoder_max_sequence_length']
        print("Length of dataset (REMI): ", len(self.captions))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        midi_filepath = os.path.join(self.dataset_path, self.captions[idx]['location'])
        tokens = self.remi_tokenizer(midi_filepath)
        if len(tokens.ids) == 0:
            tokenized_midi = [self.remi_tokenizer["BOS_None"], self.remi_tokenizer["EOS_None"]]
        else:
            tokenized_midi = [self.remi_tokenizer["BOS_None"]] + tokens.ids + [self.remi_tokenizer["EOS_None"]]
        new_sentences = caption
        if self.nlp is not None and random.random() > 0.5:
            sentences = list(self.nlp(caption).sents)
            sent_length = len(sentences)
            if sent_length > 0:
                if sent_length < 4:
                    how_many_to_drop = int(np.floor((20 + random.random()*30)/100*sent_length))
                else:
                    how_many_to_drop = int(np.ceil((20 + random.random()*30)/100*sent_length))
                which = np.random.choice(sent_length, how_many_to_drop, replace=False)
                kept = [sentences[i] for i in range(sent_length) if i not in which.tolist()]
                new_sentences = " ".join(s.text for s in kept)
        inputs = self.t5_tokenizer(new_sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if len(tokenized_midi) < self.decoder_max_sequence_length:
            labels = F.pad(torch.tensor(tokenized_midi), (0, self.decoder_max_sequence_length - len(tokenized_midi))).to(torch.int64)
        else:
            labels = torch.tensor(tokenized_midi[0:self.decoder_max_sequence_length]).to(torch.int64)
        return input_ids, attention_mask, labels

def collate_fn(batch):
    input_ids = [item[0].squeeze(0) for item in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = [item[1].squeeze(0) for item in batch]
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attention_mask, labels

def collate_fn_hf(batch):
    input_ids = [item[0].squeeze(0) for item in batch]
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = [item[1].squeeze(0) for item in batch]
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = [item[2].squeeze(0) for item in batch]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    decoder_input_ids = labels[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels
    }

# ===== Vocab builders =====

def build_vocab_aria(configs):
    artifact_folder = configs["artifact_folder"]
    vocab = {}
    instruments = ['piano', 'chromatic', 'organ', 'guitar', 'bass', 'strings', 'ensemble', 'brass', 'reed', 'pipe', 'synth_lead', 'synth_pad', 'synth_effect', 'ethnic', 'percussive', 'sfx', 'drum']
    for i in instruments:
        vocab[('prefix', 'instrument', i)] = len(vocab) + 1
    velocity = [0, 15, 30, 45, 60, 75, 90, 105, 120, 127]
    midi_pitch = list(range(0, 128))
    onset = list(range(0, 5001, 10))
    duration = list(range(0, 5001, 10))
    for v in velocity:
        for i in instruments:
            for p in midi_pitch:
                if i == "drum":
                    continue
                vocab[(i, p, v)] = len(vocab) + 1
    for p in midi_pitch:
        vocab[("drum", p)] = len(vocab) + 1
    for o in onset:
        vocab[("onset", o)] = len(vocab) + 1
    for d in duration:
        vocab[("dur", d)] = len(vocab) + 1
    vocab["<T>"] = len(vocab) + 1
    vocab["<D>"] = len(vocab) + 1
    vocab["<U>"] = len(vocab) + 1
    vocab["<SS>"] = len(vocab) + 1
    vocab["<S>"] = len(vocab) + 1
    vocab["<E>"] = len(vocab) + 1
    vocab["SEP"] = len(vocab) + 1
    print(f"Vocabulary length: {len(vocab)}")
    vocab_path = os.path.join(artifact_folder, "vocab.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {vocab_path}")

def build_vocab_remi(configs):
    assert REMI is not None and TokenizerConfig is not None, "miditok not available"
    artifact_folder = configs["artifact_folder"]
    caption_dataset_path = configs["raw_data"]["caption_dataset_path"]
    dataset_path = configs["raw_data"]["raw_data_folders"]["lmd"]["folder_path"]
    BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
    TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": BEAT_RES,
        "num_velocities": 32,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": False,
        "use_rests": False,
        "use_tempos": True,
        "use_time_signatures": True,
        "use_programs": True,
        "num_tempos": 32,
        "tempo_range": (40, 250),
    }
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)
    _ = [os.path.join(dataset_path, captions[i]['location']) for i in range(len(captions))][0:30000]
    print(f"Vocabulary length: {tokenizer.vocab_size}")
    vocab_path = os.path.join(artifact_folder, "vocab_remi.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Vocabulary saved to {vocab_path}")

# =============================================================================
# Training (Accelerate) — UPDATED: W&B removed, ThesisTracker added
# =============================================================================

def train_accelerate(configs, resume_path=None):
    per_device_train_batch_size = configs['training']['text2midi_model']['per_device_train_batch_size']
    gradient_accumulation_steps = configs['training']['text2midi_model']['gradient_accumulation_steps']
    output_dir = configs['training']['text2midi_model']['output_dir']
    lr_scheduler_type = configs['training']['text2midi_model']['lr_scheduler_type']
    num_warmup_steps = configs['training']['text2midi_model']['num_warmup_steps']
    max_train_steps = configs['training']['text2midi_model']['max_train_steps']
    save_every = configs['training']['text2midi_model']['save_every']
    checkpointing_steps = configs['training']['text2midi_model']['checkpointing_steps']

    artifact_folder = configs['artifact_folder']
    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer)

    caption_dataset_path = configs['raw_data']['caption_dataset_path']
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    mp = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("fp16" if torch.cuda.is_available() else "no")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mp,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(accelerator.state, extra={"main_process_only": False})

    if accelerator.is_main_process:
        if not output_dir:
            output_dir = os.path.join("saved", str(int(time.time())))
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)
        accelerator.project_configuration.automatic_checkpoint_naming = False
    accelerator.wait_for_everyone()

    device = accelerator.device

    # Thesis tracker (NEW)
    tracker = ThesisTracker(
        out_dir=output_dir,
        run_name=str(configs['training']['text2midi_model'].get("run_name", "train_accelerate")),
        accelerator=accelerator,
        config=configs,
    )

    with accelerator.main_process_first():
        dataset = Text2MusicDatasetRemi(configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )

    d_model = configs['model']['text2midi_model']['decoder_d_model']
    nhead = configs['model']['text2midi_model']['decoder_num_heads']
    num_layers = configs['model']['text2midi_model']['decoder_num_layers']
    max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']
    use_moe = configs['model']['text2midi_model']['use_moe']
    num_experts = configs['model']['text2midi_model']['num_experts']
    dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size']

    model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, use_moe, num_experts, device=device)

    # Resume handling
    resume_dir = None
    resume_file = None
    if resume_path:
        if os.path.isdir(resume_path):
            resume_dir = resume_path
        else:
            resume_file = resume_path

    if resume_file:
        if resume_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(resume_file)
        else:
            state = torch.load(resume_file, map_location=device)
        model.load_state_dict(state, strict=False)
        accelerator.print(f"Resumed model weights from file: {resume_file}")

    lr = float(configs['training']['text2midi_model']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # (Optional) torch.compile
    if torch.cuda.is_available():
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Compute training schedule
    # --- Steps and scheduler ---
    # Prepare FIRST so dataloader length reflects DistributedSampler in MULTI_GPU
    model, optimizer = accelerator.prepare(model, optimizer)
    dataloader = accelerator.prepare(dataloader)

    # Now compute correct update steps per epoch (per-process, which is what your loop uses)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)

    # Resolve max_train_steps
    overrode_max_train_steps = False
    if max_train_steps == 'None' or max_train_steps is None:
        max_train_steps = int(configs['training']['text2midi_model']['epochs']) * num_update_steps_per_epoch
        overrode_max_train_steps = True
    elif isinstance(max_train_steps, str):
        max_train_steps = int(max_train_steps)

    # Scheduler must be created AFTER we know max_train_steps, then prepared
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Epoch count consistent with the (prepared) dataloader
    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (
        per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Batches per epoch (per process) = {len(dataloader)}")
    logger.info(f"  Update steps per epoch = {num_update_steps_per_epoch}")

    # Log model size once (NEW)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        tracker.log_step({
            "event": "model_size",
            **model_size_report(unwrapped),
            **gpu_mem_report(),
        })

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    best_loss = float("inf")

    model.train()
    for epoch in range(epochs):
        total_loss_accum = 0.0
        total_tok = 0
        total_tok_correct = 0

        for step, batch in enumerate(dataloader):
            dt = tracker.step_timing()

            with accelerator.accumulate(model):
                encoder_input, attention_mask, tgt = batch
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                outputs = model(encoder_input, attention_mask, tgt_input)
                aux_loss = 0.0
                if isinstance(outputs, tuple):
                    logits, aux_loss = outputs
                else:
                    logits = outputs

                ce_loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
                loss = ce_loss + (aux_loss if isinstance(aux_loss, Tensor) else 0.0)

                accelerator.backward(loss)

                # only compute grad norm when gradients are synced (end of accumulation)
                gnorm = None
                if accelerator.sync_gradients:
                    try:
                        gnorm = grad_norm_l2(accelerator.unwrap_model(model))
                    except Exception:
                        gnorm = None

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Metrics
            total_loss_accum += float(loss.detach().float().item())

            acc_info = token_accuracy(logits.detach(), tgt_output.detach(), pad_id=0)
            total_tok += int(acc_info["tok_count"])
            total_tok_correct += int(acc_info["tok_acc"] * acc_info["tok_count"])

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.set_postfix({"loss": float(loss.item())})
                progress_bar.update(1)

                # LR (take first param group)
                lr = float(optimizer.param_groups[0]["lr"])

                # throughput (non-pad tokens/sec)
                tok_per_sec = (acc_info["tok_count"] / dt) if dt > 0 else 0.0

                # log step (NEW)
                tracker.log_step({
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "global_step": completed_steps,
                    "step_in_epoch": step,
                    "loss_total": float(loss.item()),
                    "loss_ce": float(ce_loss.item()),
                    "loss_aux": float(aux_loss.item()) if isinstance(aux_loss, Tensor) else float(aux_loss),
                    "lr": lr,
                    "grad_norm_l2": gnorm,
                    "tok_acc": float(acc_info["tok_acc"]),
                    "tok_count": int(acc_info["tok_count"]),
                    "tok_per_sec": float(tok_per_sec),
                    "seconds_since_start": tracker.seconds_since_start(),
                    **gpu_mem_report(),
                })

            if isinstance(checkpointing_steps, int) and (completed_steps % checkpointing_steps == 0) and accelerator.is_main_process:
                accelerator.save_state(os.path.join(output_dir, f"step_{completed_steps}"))

            if completed_steps >= max_train_steps:
                break

        # epoch summary
        if accelerator.is_main_process:
            mean_loss = total_loss_accum / max(1, len(dataloader))
            epoch_tok_acc = (total_tok_correct / max(1, total_tok)) if total_tok > 0 else 0.0

            tracker.log_epoch({
                "event": "train_epoch",
                "epoch": epoch + 1,
                "global_step": completed_steps,
                "mean_loss": float(mean_loss),
                "epoch_tok_acc": float(epoch_tok_acc),
                "total_tokens_evaled": int(total_tok),
                **gpu_mem_report(),
            })

            # Save best/epoch checkpoints like before
            if mean_loss < best_loss:
                best_loss = mean_loss

            if checkpointing_steps in ("epoch", "best"):
                if checkpointing_steps == "epoch":
                    accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch+1}"))
                elif checkpointing_steps == "best":
                    # best = based on mean loss
                    accelerator.save_state(os.path.join(output_dir, "best"))

            if isinstance(save_every, int) and save_every > 0:
                if ((epoch + 1) % save_every == 0) and checkpointing_steps not in ("epoch", "best"):
                    accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch+1}"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tracker.finalize()

# =============================================================================
# Knowledge Distillation — UPDATED: W&B removed, ThesisTracker added
# =============================================================================

def distill_accelerate(
    configs,
    teacher_ckpt: str,
    student_output_dir: str,
    temperature: float = 2.0,
    alpha_hard: float = 0.5,
):
    per_device_train_batch_size = configs['training']['text2midi_model']['per_device_train_batch_size']
    gradient_accumulation_steps = configs['training']['text2midi_model']['gradient_accumulation_steps']
    lr_scheduler_type = configs['training']['text2midi_model']['lr_scheduler_type']
    num_warmup_steps = configs['training']['text2midi_model']['num_warmup_steps']
    max_train_steps = configs['training']['text2midi_model']['max_train_steps']
    save_every = configs['training']['text2midi_model']['save_every']
    checkpointing_steps = configs['training']['text2midi_model']['checkpointing_steps']

    artifact_folder = configs['artifact_folder']
    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    with open(tokenizer_filepath, "rb") as f:
        remi_tokenizer = pickle.load(f)
    vocab_size = len(remi_tokenizer)

    caption_dataset_path = configs['raw_data']['caption_dataset_path']
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    mp = "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("fp16" if torch.cuda.is_available() else "no")

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mp,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(accelerator.state, extra={"main_process_only": False})

    if accelerator.is_main_process:
        os.makedirs(student_output_dir, exist_ok=True)
        os.makedirs(os.path.join(student_output_dir, "outputs"), exist_ok=True)
        accelerator.project_configuration.automatic_checkpoint_naming = False
    accelerator.wait_for_everyone()
    device = accelerator.device

    tracker = ThesisTracker(
        out_dir=student_output_dir,
        run_name=str(configs['training']['text2midi_model'].get("run_name", "distill")),
        accelerator=accelerator,
        config={
            **configs,
            "distill": {"teacher_ckpt": teacher_ckpt, "temperature": temperature, "alpha_hard": alpha_hard}
        },
    )

    with accelerator.main_process_first():
        dataset = Text2MusicDatasetRemi(configs, captions, remi_tokenizer=remi_tokenizer, mode="train", shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True,
        )

    # -----------------------------
    # Teacher arch (INFER from checkpoint) ✅ fixes 768-vs-384 mismatch
    # -----------------------------
    nhead_teacher = int(configs['model']['text2midi_model']['decoder_num_heads'])  # heads not inferable from weights
    use_moe = bool(configs['model']['text2midi_model']['use_moe'])
    num_experts = int(configs['model']['text2midi_model']['num_experts'])

    from safetensors.torch import load_file as load_safetensors
    import re

    ckpt_dir = teacher_ckpt
    st_path = None
    bin_path = None

    if os.path.isdir(ckpt_dir):
        candidate_st = os.path.join(ckpt_dir, "model.safetensors")
        candidate_bin = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(candidate_st):
            st_path = candidate_st
        if os.path.exists(candidate_bin):
            bin_path = candidate_bin
    else:
        if ckpt_dir.endswith(".safetensors"):
            st_path = ckpt_dir
        else:
            bin_path = ckpt_dir

    if st_path is not None:
        teacher_state = load_safetensors(st_path)  # load on CPU; we'll move model to device after
        accelerator.print(f"[KD] Loaded teacher safetensors from {st_path}")
    elif bin_path is not None and os.path.exists(bin_path):
        teacher_state = torch.load(bin_path, map_location="cpu")
        accelerator.print(f"[KD] Loaded teacher bin from {bin_path}")
    else:
        raise FileNotFoundError(f"Could not find teacher checkpoint at: {teacher_ckpt}")

    # ✅ Compatibility fix: checkpoint pe is [max_len,1,d_model] but model expects [max_len,d_model]
    if "pos_encoder.pe" in teacher_state:
        pe = teacher_state["pos_encoder.pe"]
        if hasattr(pe, "ndim") and pe.ndim == 3 and pe.shape[1] == 1:
            teacher_state["pos_encoder.pe"] = pe.squeeze(1)

    # Infer d_model from embedding weight [V, d_model]
    d_model_teacher = int(teacher_state["input_emb.weight"].shape[1])

    # Infer max_len from pos_encoder.pe if present; else fallback to config
    if "pos_encoder.pe" in teacher_state:
        pe = teacher_state["pos_encoder.pe"]
        max_len = int(pe.shape[0])
    else:
        max_len = int(configs['model']['text2midi_model']['decoder_max_sequence_length'])

    # Infer num_layers from keys: decoder.layers.{i}.*
    layer_ids = []
    for k in teacher_state.keys():
        m = re.match(r"decoder\.layers\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    num_layers_teacher = (max(layer_ids) + 1) if layer_ids else int(configs['model']['text2midi_model']['decoder_num_layers'])

    # Infer FFN size from first layer linear1.weight: [ff, d_model]
    ff_key = "decoder.layers.0.linear1.weight"
    if ff_key in teacher_state:
        dim_feedforward_teacher = int(teacher_state[ff_key].shape[0])
    else:
        dim_feedforward_teacher = int(configs['model']['text2midi_model']['decoder_intermediate_size'])

    # Guard: nhead must divide d_model
    if d_model_teacher % nhead_teacher != 0:
        raise ValueError(
            f"Teacher checkpoint d_model={d_model_teacher} is not divisible by decoder_num_heads={nhead_teacher}. "
            f"Set decoder_num_heads in config.yaml to a divisor of {d_model_teacher}."
        )

    teacher = Transformer(
        n_vocab=vocab_size,
        d_model=d_model_teacher,
        nhead=nhead_teacher,
        max_len=max_len,
        num_decoder_layers=num_layers_teacher,
        dim_feedforward=dim_feedforward_teacher,
        use_moe=use_moe,
        num_experts=num_experts,
        device=device,
    )

    pe_key = "pos_encoder.pe"
    if pe_key in teacher_state:
        pe = teacher_state[pe_key]
        if isinstance(pe, torch.Tensor) and pe.ndim == 2:
            teacher_state[pe_key] = pe.unsqueeze(1)

    missing, unexpected = teacher.load_state_dict(teacher_state, strict=False)
    
    accelerator.print(f"[KD] Teacher missing keys: {missing}")
    accelerator.print(f"[KD] Teacher unexpected keys: {unexpected}")

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False


    # -----------------------------
    # Student arch FROM CONFIG
    # -----------------------------
    student_d_model = int(configs['model']['text2midi_model']['decoder_d_model'])
    student_nhead = int(configs['model']['text2midi_model']['decoder_num_heads'])
    student_num_layers = int(configs['model']['text2midi_model']['decoder_num_layers'])
    student_dim_ff = int(configs['model']['text2midi_model']['decoder_intermediate_size'])

    if student_d_model % student_nhead != 0:
        raise ValueError(
            f"Student decoder_d_model={student_d_model} is not divisible by decoder_num_heads={student_nhead}."
        )

    student = Transformer(
        n_vocab=vocab_size,
        d_model=student_d_model,
        nhead=student_nhead,
        max_len=max_len,
        num_decoder_layers=student_num_layers,
        dim_feedforward=student_dim_ff,
        use_moe=use_moe,
        num_experts=num_experts,
        device=device,
    )


    # Share frozen FLAN encoder
    student.encoder = teacher.encoder

    # Use config learning rate (do NOT hardcode)
    lr = float(configs['training']['text2midi_model']['learning_rate'])
    optimizer = optim.Adam(student.parameters(), lr=lr)

    # Prepare FIRST (so dataloader length reflects DistributedSampler in MULTI_GPU)
    student, optimizer = accelerator.prepare(student, optimizer)
    dataloader = accelerator.prepare(dataloader)

    # Now compute correct update steps per epoch
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)

    # Resolve max_train_steps
    overrode_max_train_steps = False
    if max_train_steps == 'None' or max_train_steps is None:
        max_train_steps = int(configs['training']['text2midi_model']['epochs']) * num_update_steps_per_epoch
        overrode_max_train_steps = True
    elif isinstance(max_train_steps, str):
        max_train_steps = int(max_train_steps)

    # Create scheduler AFTER we know max_train_steps, then prepare it
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if overrode_max_train_steps:
        max_train_steps = configs['training']['text2midi_model']['epochs'] * num_update_steps_per_epoch

    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running KD training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    criterion_hard = nn.CrossEntropyLoss(ignore_index=0)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_loss = float("inf")

    T = float(temperature)
    alpha = float(alpha_hard)

    student.train()
    for epoch in range(epochs):
        total_loss_accum = 0.0
        total_hard_accum = 0.0
        total_kd_accum = 0.0
        total_tok = 0
        total_tok_correct = 0

        for step, batch in enumerate(dataloader):
            dt = None  # only measure time when we actually log a real optimizer step

            encoder_input, attention_mask, tgt = batch
            encoder_input = encoder_input.to(device)
            attention_mask = attention_mask.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with accelerator.accumulate(student):
                with torch.no_grad():
                    t_out = teacher(encoder_input, attention_mask, tgt_input)
                    t_logits = t_out[0] if isinstance(t_out, tuple) else t_out

                s_out = student(encoder_input, attention_mask, tgt_input)
                aux_loss = 0.0
                if isinstance(s_out, tuple):
                    s_logits, aux_loss = s_out
                else:
                    s_logits = s_out

                vocab_dim = s_logits.size(-1)
                s_flat = s_logits.view(-1, vocab_dim)
                t_flat = t_logits.view(-1, vocab_dim)
                tgt_flat = tgt_output.reshape(-1)

                hard_loss = criterion_hard(s_flat, tgt_flat)

                log_p_student = F.log_softmax(s_flat / T, dim=-1)
                p_teacher = F.softmax(t_flat / T, dim=-1)
                kd_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

                loss = alpha * hard_loss + (1.0 - alpha) * kd_loss
                if isinstance(aux_loss, torch.Tensor):
                    loss = loss + aux_loss

                accelerator.backward(loss)

                # only step optimizer at the end of gradient accumulation
                gnorm = None
                if accelerator.sync_gradients:
                    try:
                        gnorm = grad_norm_l2(accelerator.unwrap_model(student))
                    except Exception:
                        gnorm = None

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            total_loss_accum += float(loss.detach().float().item())
            total_hard_accum += float(hard_loss.detach().float().item())
            total_kd_accum += float(kd_loss.detach().float().item())

            acc_info = token_accuracy(s_logits.detach(), tgt_output.detach(), pad_id=0)
            total_tok += int(acc_info["tok_count"])
            total_tok_correct += int(acc_info["tok_acc"] * acc_info["tok_count"])

            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.set_postfix({"loss": float(loss.item())})
                progress_bar.update(1)

                lr = float(optimizer.param_groups[0]["lr"])
                dt = tracker.step_timing()
                tok_per_sec = (acc_info["tok_count"] / dt) if (dt is not None and dt > 0) else 0.0

                tracker.log_step({
                    "event": "kd_step",
                    "epoch": epoch + 1,
                    "global_step": completed_steps,
                    "step_in_epoch": step,
                    "loss_total": float(loss.item()),
                    "loss_hard_ce": float(hard_loss.item()),
                    "loss_kd_kl": float(kd_loss.item()),
                    "loss_aux": float(aux_loss.item()) if isinstance(aux_loss, Tensor) else float(aux_loss),
                    "alpha_hard": alpha,
                    "temperature": T,
                    "lr": lr,
                    "grad_norm_l2": gnorm,
                    "tok_acc": float(acc_info["tok_acc"]),
                    "tok_count": int(acc_info["tok_count"]),
                    "tok_per_sec": float(tok_per_sec),
                    "seconds_since_start": tracker.seconds_since_start(),
                    **gpu_mem_report(),
                })

            if isinstance(checkpointing_steps, int) and (completed_steps % checkpointing_steps == 0) and accelerator.is_main_process:
                accelerator.save_state(os.path.join(student_output_dir, f"step_{completed_steps}"))

            if completed_steps >= max_train_steps:
                break

        if accelerator.is_main_process:
            mean_loss = total_loss_accum / max(1, len(dataloader))
            mean_hard = total_hard_accum / max(1, len(dataloader))
            mean_kd = total_kd_accum / max(1, len(dataloader))
            epoch_tok_acc = (total_tok_correct / max(1, total_tok)) if total_tok > 0 else 0.0

            tracker.log_epoch({
                "event": "kd_epoch",
                "epoch": epoch + 1,
                "global_step": completed_steps,
                "mean_loss": float(mean_loss),
                "mean_hard_ce": float(mean_hard),
                "mean_kd_kl": float(mean_kd),
                "epoch_tok_acc": float(epoch_tok_acc),
                "total_tokens_evaled": int(total_tok),
                **gpu_mem_report(),
            })

            # Best checkpoint based on mean_loss
            if mean_loss < best_loss:
                best_loss = mean_loss
                unwrapped = accelerator.unwrap_model(student)
                torch.save(unwrapped.state_dict(), os.path.join(student_output_dir, "pytorch_model_distilled_best.bin"))

            if checkpointing_steps in ("epoch", "best"):
                if checkpointing_steps == "epoch":
                    accelerator.save_state(os.path.join(student_output_dir, f"epoch_{epoch + 1}"))
                elif checkpointing_steps == "best":
                    accelerator.save_state(os.path.join(student_output_dir, "best"))

            if isinstance(save_every, int) and save_every > 0:
                if ((epoch + 1) % save_every == 0) and checkpointing_steps not in ("epoch", "best"):
                    accelerator.save_state(os.path.join(student_output_dir, f"epoch_{epoch + 1}"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tracker.finalize()

# =============================================================================
# Generation + utilities (unchanged behavior)
# =============================================================================

def custom_collate_captions(batch):
    captions = []
    locations = []
    for idx, item in enumerate(batch):
        cap = item.get("caption") or item.get("text") or ""
        loc = item.get("location")
        if not loc:
            loc = f"{idx:05d}.mid"
        captions.append(cap)
        locations.append(loc)
    return captions, locations

def load_model_and_tokenizer_for_gen(accelerator, model_path, vocab_size, tokenizer_filepath):
    device = accelerator.device
    with open(tokenizer_filepath, "rb") as f:
        r_tokenizer = pickle.load(f)

    if str(model_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location=device)

    d_model_ckpt, max_len_model = infer_arch_from_state(state)
    arch = arch_from_d_model(d_model_ckpt)

    if accelerator.is_local_main_process:
        print(f"[load] checkpoint d_model={d_model_ckpt} -> using arch={arch}, max_len={max_len_model}")

    model = Transformer(
        n_vocab=vocab_size,
        d_model=arch["d_model"],
        nhead=arch["nhead"],
        max_len=max_len_model,
        num_decoder_layers=arch["num_layers"],
        dim_feedforward=arch["dim_feedforward"],
        use_moe=arch["use_moe"],
        num_experts=arch["num_experts"],
        device=device,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)

    if accelerator.is_local_main_process:
        print(f"[load] missing keys: {missing}")
        print(f"[load] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    t5tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
    return model, t5tok, r_tokenizer

def generate_single(caption: str, model_path: str, tokenizer_pkl: str, configs: dict, max_len: int = 2000,
                    temperature: float = 0.9, out_dir: str = DEFAULT_OUTPUT_ROOT):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    with open(tokenizer_pkl, "rb") as f:
        r_tokenizer = pickle.load(f)
    vocab_size = len(r_tokenizer)

    if str(model_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location=device)

    # load configs earlier from --config just like training does
    m = configs["model"]["text2midi_model"]

    model = Transformer(
        n_vocab=vocab_size,
        d_model=m["decoder_d_model"],
        nhead=m["decoder_num_heads"],
        max_len=m["decoder_max_sequence_length"],
        num_decoder_layers=m["decoder_num_layers"],
        dim_feedforward=m["decoder_intermediate_size"],
        use_moe=m["use_moe"],
        num_experts=m["num_experts"],
        device=device,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[generate] missing keys:", missing)
    print("[generate] unexpected keys:", unexpected)

    model.to(device)
    model.eval()

    t5tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
    inputs = t5tok(caption, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        bos_id = r_tokenizer["BOS_None"]
        token_ids = model.generate(
            src=input_ids,
            src_mask=attention_mask,
            max_len=max_len,
            temperature=temperature,
            bos_id=bos_id,
        )

    token_list = token_ids[0].tolist()
    midi = r_tokenizer.decode(token_list)
    os.makedirs(out_dir, exist_ok=True)
    out_path = _next_output_path(root=out_dir, prefix="gen_", ext=".mid")
    midi.dump_midi(out_path)
    print(f"Saved: {out_path}")

def generate_accelerated(captions_jsonl: str, model_path: str, tokenizer_pkl: str, batch_size: int = 8,
                         max_len: int = 2000, temperature: float = 0.9, out_root: str = DEFAULT_OUTPUT_ROOT):
    from accelerate import Accelerator
    import os, pickle, torch, jsonlines
    from torch.utils.data import DataLoader

    accelerator = Accelerator()
    device = accelerator.device

    with jsonlines.open(captions_jsonl) as reader:
        rows = list(reader)

    has_test_flag = any("test_set" in r for r in rows)
    selected = [r for r in rows if (r.get("test_set") is True)] if has_test_flag else rows
    if not selected:
        if accelerator.is_local_main_process:
            print("[gen] No prompts found (0 rows after filtering).")
        return

    with open(tokenizer_pkl, "rb") as f:
        r_tokenizer = pickle.load(f)
    vocab_size = len(r_tokenizer)

    model, t5tok, r_tokenizer = load_model_and_tokenizer_for_gen(accelerator, model_path, vocab_size, tokenizer_pkl)
    model = model.to(device)
    model.eval()
    bos_id = r_tokenizer["BOS_None"]

    class CaptionDatasetOnly(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            r = self.items[idx]
            return {"caption": r.get("caption") or r.get("text") or "", "location": r.get("location")}

    def custom_collate_captions(batch):
        captions, locations = [], []
        for i, item in enumerate(batch):
            cap = item.get("caption") or ""
            loc = item.get("location") or f"{i:05d}.mid"
            if not loc.lower().endswith((".mid", ".midi")):
                loc += ".mid"
            captions.append(cap)
            locations.append(loc)
        return captions, locations

    ds = CaptionDatasetOnly(selected)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_captions, num_workers=0)
    dataloader = accelerator.prepare(dataloader)

    os.makedirs(out_root, exist_ok=True)

    with torch.no_grad():
        for captions, locations in dataloader:
            if not captions:
                continue
            enc = t5tok(captions, return_tensors="pt", padding=True, truncation=True)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

            token_seqs = model.generate(
                src=input_ids,
                src_mask=attention_mask,
                max_len=max_len,
                temperature=temperature,
                bos_id=bos_id,
            )

            if accelerator.is_main_process:
                for j in range(token_seqs.size(0)):
                    seq = token_seqs[j].tolist()
                    try:
                        midi = r_tokenizer.decode(seq)
                        fname = locations[j]
                        out_path = os.path.join(out_root, fname)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        midi.dump_midi(out_path)
                        print(f"[gen] wrote {out_path}")
                    except Exception as e:
                        print(f"[gen][error] failed item {j}: {e}")

    accelerator.wait_for_everyone()

def split_captions(input_path: str, output_dir: str, num_splits: int = 6):
    with jsonlines.open(input_path) as reader:
        captions = [line for line in reader if line.get('test_set') is True]
    selected_captions = captions
    split_size = max(1, len(selected_captions) // num_splits)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else len(selected_captions)
        split_part = selected_captions[start_idx:end_idx]
        out_path = os.path.join(output_dir, f'selected_captions_{i}.json')
        with open(out_path, 'w', encoding="utf-8") as f:
            json.dump(split_part, f, indent=4)
        print(f'Saved {len(split_part)} captions to {out_path}')

# =============================================================================
# HF Trainer section (left as-is; W&B not used there)
# =============================================================================

class CustomEncoderDecoderModel(PreTrainedModel):
    def __init__(self, encoder, decoder, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

    def forward(self, input_ids, decoder_input_ids, attention_mask=None, decoder_attention_mask=None, labels=None, **kwargs):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            labels=labels
        )
        return {'loss': output.loss, 'logits': output.logits}

def train_hf(configs):
    artifact_folder = configs['artifact_folder']
    run_name = configs['training']['text2midi_model']['run_name']
    model_dir = os.path.join(artifact_folder, run_name)
    log_dir = os.path.join(model_dir, "logs")
    os.system(f"rm -rf {log_dir}")

    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = tokenizer.vocab_size + 1 if hasattr(tokenizer, "vocab_size") else len(tokenizer) + 1

    caption_dataset_path = configs['raw_data']['caption_dataset_path']
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    random.seed(444)
    random.shuffle(captions)
    train_size = int(0.8 * len(captions))
    train_captions = captions[:train_size]
    test_captions = captions[train_size:]

    train_dataset = Text2MusicDatasetRemi(configs, train_captions, tokenizer, mode="train", shuffle=True)
    test_dataset  = Text2MusicDatasetRemi(configs, test_captions,  tokenizer, mode="eval",  shuffle=False)

    flan_t5_encoder = T5EncoderModel.from_pretrained('google/flan-t5-small')
    for param in flan_t5_encoder.parameters():
        param.requires_grad = False
    encoder_config = T5Config.from_pretrained('google/flan-t5-small')

    config_decoder = BertConfig()
    config_decoder.vocab_size = vocab_size
    config_decoder.max_position_embeddings = configs['model']['text2midi_model']['decoder_max_sequence_length']
    config_decoder.max_length = configs['model']['text2midi_model']['decoder_max_sequence_length']
    config_decoder.bos_token_id = tokenizer["BOS_None"]
    config_decoder.eos_token_id = tokenizer["EOS_None"]
    config_decoder.pad_token_id = 0
    config_decoder.num_hidden_layers = configs['model']['text2midi_model']['decoder_num_layers']
    config_decoder.num_attention_heads = configs['model']['text2midi_model']['decoder_num_heads']
    config_decoder.hidden_size = configs['model']['text2midi_model']['decoder_d_model']
    config_decoder.intermediate_size = configs['model']['text2midi_model']['decoder_intermediate_size']
    config_decoder.is_decoder = True
    config_decoder.add_cross_attention = True
    config_decoder.tie_encoder_decoder = False
    config_decoder.tie_word_embeddings = False

    custom_decoder = BertLMHeadModel(config_decoder)
    model = CustomEncoderDecoderModel(encoder=flan_t5_encoder, decoder=custom_decoder,
                                      encoder_config=encoder_config, decoder_config=config_decoder)

    USE_CUDA = torch.cuda.is_available()
    FP16 = BF16 = FP16_EVAL = BF16_EVAL = False
    if USE_CUDA and torch.cuda.is_bf16_supported():
        BF16 = BF16_EVAL = True
    elif USE_CUDA:
        FP16 = FP16_EVAL = True

    metrics = {"accuracy": load_metric("accuracy")}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        not_pad_mask = labels != 0
        labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
        return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

    def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
        return argmax(logits, dim=-1)

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=configs['training']['text2midi_model']['batch_size'],
        per_device_eval_batch_size=configs['training']['text2midi_model']['batch_size'],
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=configs['training']['text2midi_model']['learning_rate'],
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.3,
        max_grad_norm=3.0,
        weight_decay=configs['training']['text2midi_model']['weight_decay'],
        num_train_epochs=configs['training']['text2midi_model']['epochs'],
        evaluation_strategy="epoch",
        gradient_accumulation_steps=configs['training']['text2midi_model']['gradient_accumulation_steps'],
        optim="adafactor",
        seed=444,
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=log_dir,
        no_cuda=not USE_CUDA,
        fp16=FP16,
        fp16_full_eval=FP16_EVAL,
        bf16=BF16,
        bf16_full_eval=BF16_EVAL,
        load_best_model_at_end=True,
        greater_is_better=False,
        report_to="tensorboard",
        run_name=configs['training']['text2midi_model']['run_name'],
        push_to_hub=False,
        dataloader_num_workers=5
    )

    class CustomTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=configs['training']['text2midi_model']['batch_size'], collate_fn=collate_fn_hf, num_workers=5)
        def get_eval_dataloader(self, eval_dataset):
            return DataLoader(eval_dataset, batch_size=configs['training']['text2midi_model']['batch_size'], collate_fn=collate_fn_hf, num_workers=5)
        def get_test_dataloader(self, test_dataset):
            return DataLoader(test_dataset, batch_size=configs['training']['text2midi_model']['batch_size'], collate_fn=collate_fn_hf, num_workers=5)

    trainer = CustomTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset,
                            compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits)
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd in ("build_vocab_aria", "build_vocab_remi", "dataset_demo_aria", "dataset_demo_remi",
                    "train_basic", "train_accelerate", "train_hf", "distill"):
        configs = load_configs(args.config)

    if args.cmd == "build_vocab_aria":
        build_vocab_aria(configs)

    elif args.cmd == "build_vocab_remi":
        build_vocab_remi(configs)

    elif args.cmd == "dataset_demo_aria":
        caption_dataset_path = configs['raw_data']['caption_dataset_path']
        captions = load_captions_jsonl(caption_dataset_path)
        aria_tokenizer = (lambda midi: [] if MidiDict is None else [])
        ds = Text2MusicDatasetAria(configs, captions, aria_tokenizer=aria_tokenizer, mode="train", shuffle=True)
        a, b, c = ds[0]
        print("ARIA demo label shape:", c.shape)

    elif args.cmd == "dataset_demo_remi":
        caption_dataset_path = configs['raw_data']['caption_dataset_path']
        captions = load_captions_jsonl(caption_dataset_path)
        tok_path = os.path.join(configs['artifact_folder'], "vocab_remi.pkl")
        with open(tok_path, "rb") as f:
            remi_tokenizer = pickle.load(f)
        ds = Text2MusicDatasetRemi(configs, captions, remi_tokenizer=remi_tokenizer, mode="train", shuffle=True)
        a, b, c = ds[0]
        print("REMI demo types:", type(a), type(c))
        try:
            generated_midi = remi_tokenizer.decode(c)
            generated_midi.dump_midi(_next_output_path())
        except Exception:
            pass

    elif args.cmd == "train_basic":
        raise NotImplementedError("train_basic kept in your original script; if you want it tracked too, tell me and I’ll add ThesisTracker there as well.")

    elif args.cmd == "train_accelerate":
        train_accelerate(configs, resume_path=args.resume_path)

    elif args.cmd == "train_hf":
        train_hf(configs)

    elif args.cmd == "distill":
        distill_accelerate(
            configs=configs,
            teacher_ckpt=args.teacher_ckpt,
            student_output_dir=args.student_output_dir,
            temperature=args.temperature,
            alpha_hard=args.alpha_hard,
        )

    elif args.cmd == "generate":
        configs = load_configs(args.config)
        generate_single(
            caption=args.caption,
            model_path=args.model_path,
            tokenizer_pkl=args.tokenizer_pkl,
            configs=configs,
            max_len=args.max_len,
            temperature=args.temperature,
            out_dir=args.out_dir,
        )

    elif args.cmd == "generate_accelerate":
        generate_accelerated(
            captions_jsonl=args.captions_jsonl,
            model_path=args.model_path,
            tokenizer_pkl=args.tokenizer_pkl,
            batch_size=args.batch_size,
            max_len=args.max_len,
            temperature=args.temperature,
            out_root=args.out_root,
        )

    elif args.cmd == "split_captions":
        split_captions(input_path=args.input_jsonl, output_dir=args.output_dir, num_splits=args.num_splits)

if __name__ == "__main__":
    main()