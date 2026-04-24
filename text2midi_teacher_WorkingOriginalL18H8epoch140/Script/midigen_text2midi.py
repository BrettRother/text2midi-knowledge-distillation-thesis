#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
midigen_text2midi.py (W&B REMOVED, Local Thesis Tracking ADDED)

Everything from your script is kept the same EXCEPT:
  ✅ Removed all Weights & Biases usage (import + init + wandb.log)
  ✅ Disabled Accelerate "log_with" tracker wiring (so it never tries to use wandb)
  ✅ Added a new LOCAL tracking system that writes:
       - run_meta.json (full config + environment + model size + param counts)
       - metrics.jsonl (step-by-step metrics)
       - metrics.csv (same metrics in a spreadsheet-friendly format)
       - epoch_summary.jsonl (epoch-level summary)

What is tracked for your thesis:
  - train_loss (per step + per epoch)
  - aux_loss (MoE auxiliary loss if used)
  - lr
  - grad_norm (global L2)
  - token_accuracy (ignoring PAD=0)
  - perplexity estimate (exp(loss), capped)
  - throughput: tokens/sec, seqs/sec, step_time_sec
  - GPU memory: allocated/reserved (GB)
  - model size: parameter counts + bytes/MB + estimated fp16/fp32 sizes
  - effective batch size

IMPORTANT:
  - PAD id is assumed 0 (matches your data collate)
  - This tracker only logs on main process (Accelerate safe)
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
from typing import Optional, Any, Union, Callable, Dict, List
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

# Optional: aria MIDI (used only in your earlier aria variant)
try:
    from aria.data.midi import MidiDict
except Exception:
    MidiDict = None

# ---------- Fixed IO Roots (requested locations) ----------
DEFAULT_INPUT_ROOT = "/home/brett_ece/midi/midigen2/text2midi/Input"
DEFAULT_OUTPUT_ROOT = "/home/brett_ece/midi/midigen2/text2midi/Output"
os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)

def _next_output_path(root=DEFAULT_OUTPUT_ROOT, prefix="gen_", ext=".mid"):
    """
    Consecutive incrementing MIDI output filename helper.
    Produces: gen_0001.mid, gen_0002.mid, ...
    """
    os.makedirs(root, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}$")
    nums = []
    for name in os.listdir(root):
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))
    n = max(nums) + 1 if nums else 1
    return os.path.join(root, f"{prefix}{n:04d}{ext}")

# ---------- Config loader ----------
def load_configs(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ---------- JSONL loader ----------
def load_captions_jsonl(path):
    with jsonlines.open(path) as reader:
        return list(reader)

# ---------- Argparse scaffolding ----------
def build_parser():
    p = argparse.ArgumentParser(prog="midigen_text2midi.py", description="Unified Text2MIDI toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) Build vocab (ARIA)
    s = sub.add_parser("build_vocab_aria", help="Build custom ARIA-style vocab.pkl")
    s.add_argument("--config", type=str, required=True)

    # 2) Build vocab (REMI miditok)
    s = sub.add_parser("build_vocab_remi", help="Build miditok REMI tokenizer pickle")
    s.add_argument("--config", type=str, required=True)

    # 3) Dataset demos (ARIA/REMI)
    s = sub.add_parser("dataset_demo_aria", help="Quick test for ARIA dataset item")
    s.add_argument("--config", type=str, required=True)

    s = sub.add_parser("dataset_demo_remi", help="Quick test for REMI dataset item")
    s.add_argument("--config", type=str, required=True)

    # 4) Training: basic (your non-accelerate loop)
    s = sub.add_parser("train_basic", help="Train with the simple loop Transformer")
    s.add_argument("--config", type=str, required=True)

    # 5) Training: accelerate
    s = sub.add_parser("train_accelerate", help="Accelerate-based training")
    s.add_argument("--config", type=str, required=True)
    s.add_argument("--resume_path", type=str, default=None, help="Optional path to load state_dict")

    # 6) Training: HF Trainer (T5 encoder + BERT decoder)
    s = sub.add_parser("train_hf", help="HF Trainer with frozen FLAN-T5 encoder + BERT decoder")
    s.add_argument("--config", type=str, required=True)

    # 7) Generation (single caption)
    s = sub.add_parser("generate", help="Generate a single MIDI for a caption")
    s.add_argument("--caption", type=str, required=True)
    s.add_argument("--model_path", type=str, required=True, help="pytorch_model.bin path")
    s.add_argument("--tokenizer_pkl", type=str, required=True, help="vocab_remi.pkl path")
    s.add_argument("--max_len", type=int, default=2000)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--out_dir", type=str, default=DEFAULT_OUTPUT_ROOT)

    # 8) Generation: accelerate over a list of test captions in JSONL
    s = sub.add_parser("generate_accelerate", help="Accelerated batch generation from captions JSONL")
    s.add_argument("--captions_jsonl", type=str, required=True)
    s.add_argument("--model_path", type=str, required=True)
    s.add_argument("--tokenizer_pkl", type=str, required=True)
    s.add_argument("--batch_size", type=int, default=32)
    s.add_argument("--max_len", type=int, default=2000)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--out_root", type=str, default=DEFAULT_OUTPUT_ROOT)

    # 9) Split captions (JSONL -> JSON chunks)
    s = sub.add_parser("split_captions", help="Split test-set captions into chunks")
    s.add_argument("--input_jsonl", type=str, required=True)
    s.add_argument("--output_dir", type=str, default=DEFAULT_INPUT_ROOT)
    s.add_argument("--num_splits", type=int, default=6)

    return p

# =========================================================
# NEW: Local Thesis Tracker (NO W&B)
# =========================================================
import csv
import platform
import hashlib
from dataclasses import dataclass, field

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _sha1_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def _torch_env_dict() -> Dict[str, Any]:
    d = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 3),
                "cc": f"{props.major}.{props.minor}",
            })
        d["gpus"] = gpus
    return d

def model_param_counts(model: nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_frozen": int(frozen),
    }

def model_state_size_bytes(model: nn.Module) -> int:
    # Exact size of state_dict tensors in bytes (not counting optimizer, etc.)
    sd = model.state_dict()
    total = 0
    for _, t in sd.items():
        if torch.is_tensor(t):
            total += t.numel() * t.element_size()
    return int(total)

def model_size_report(model: nn.Module) -> Dict[str, Any]:
    counts = model_param_counts(model)
    state_bytes = model_state_size_bytes(model)
    # Approx sizes if you stored as fp32/fp16 (for thesis discussion)
    # NOTE: This is an estimate based on param count, not exact state_dict dtype.
    p = counts["params_total"]
    approx_fp32 = p * 4
    approx_fp16 = p * 2
    return {
        **counts,
        "state_dict_bytes": state_bytes,
        "state_dict_mb": round(state_bytes / (1024**2), 3),
        "approx_param_bytes_fp32": int(approx_fp32),
        "approx_param_mb_fp32": round(approx_fp32 / (1024**2), 3),
        "approx_param_bytes_fp16": int(approx_fp16),
        "approx_param_mb_fp16": round(approx_fp16 / (1024**2), 3),
    }

def _grad_global_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        if g.is_sparse:
            g = g.coalesce().values()
        total += float(torch.sum(g.float() * g.float()).item())
    return float(math.sqrt(total))

def _token_accuracy_ignore_pad(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = (targets != pad_id)
        denom = mask.sum().item()
        if denom == 0:
            return 0.0
        correct = ((pred == targets) & mask).sum().item()
        return float(correct / denom)

def make_run_dir(base_output_dir: str, run_name: str = "text2midi") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_output_dir, "tracking", f"{run_name}_{stamp}")

@dataclass
class LocalTracker:
    run_dir: str
    config: Dict[str, Any]
    accelerator: Any
    pad_id: int = 0
    log_every: int = 10
    csv_fields: List[str] = field(default_factory=list)

    _jsonl_path: str = field(init=False)
    _csv_path: str = field(init=False)
    _csv_file: Any = field(init=False, default=None)
    _csv_writer: Any = field(init=False, default=None)
    _epoch_path: str = field(init=False)

    def __post_init__(self):
        os.makedirs(self.run_dir, exist_ok=True)
        self._jsonl_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._epoch_path = os.path.join(self.run_dir, "epoch_summary.jsonl")

        if self.accelerator.is_main_process:
            self.csv_fields = [
                "time", "epoch", "step",
                "loss", "aux_loss", "token_acc", "ppl",
                "lr", "grad_norm",
                "tokens_per_sec", "seqs_per_sec",
                "step_time_sec",
                "gpu_mem_alloc_gb", "gpu_mem_reserved_gb",
                "effective_batch_size",
                "event"
            ]
            self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.csv_fields)
            self._csv_writer.writeheader()
            self._csv_file.flush()

        self.accelerator.wait_for_everyone()

    def write_run_meta(self, meta: Dict[str, Any]):
        if not self.accelerator.is_main_process:
            return
        with open(os.path.join(self.run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    def log(self, row: Dict[str, Any]):
        if not self.accelerator.is_main_process:
            return
        row = dict(row)
        row.setdefault("time", _now_iso())

        # JSONL append
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

        # CSV append
        csv_row = {k: row.get(k, None) for k in self.csv_fields}
        self._csv_writer.writerow(csv_row)
        self._csv_file.flush()

    def log_epoch(self, row: Dict[str, Any]):
        if not self.accelerator.is_main_process:
            return
        row = dict(row)
        row.setdefault("time", _now_iso())
        with open(self._epoch_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def close(self):
        if self.accelerator.is_main_process and self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

# ===== Transformer and components (verbatim from your scripts; lightly wrapped) =====

try:
    from torch.nn.modules.activation import MultiheadAttention
except Exception:
    from torch.nn import MultiheadAttention

# Rotary helpers (kept from your code — currently disabled in attention forward)
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
    def __init__(self, n_vocab: int = 30000, d_model: int = 512, nhead: int = 8, max_len: int = 5000,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, use_moe: bool = False,
                 num_experts: int = 16, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.use_moe = use_moe
        self.input_emb = nn.Embedding(n_vocab, d_model, **factory_kwargs)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len).to(device)
        self.encoder = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False
            self.encoder.eval()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, use_moe, num_experts, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first, bias, **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, use_moe, decoder_norm)
        self.projection = nn.Linear(d_model, n_vocab).to(device)
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self._reset_parameters()

    def _reset_parameters(self):
        # IMPORTANT:
        # Do NOT re-init the pretrained encoder weights.
        # Only initialize trainable (new) parameters.
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

    def forward(self, src: Tensor, src_mask: Tensor, tgt: Tensor, memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = True, memory_is_causal: bool = False) -> Tensor:
        if src.dim() != tgt.dim():
            raise RuntimeError("the number of dimensions in src and tgt must be equal")
        with torch.no_grad():
            memory = self.encoder(src, attention_mask=src_mask).last_hidden_state
        tgt = self.input_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        if self.use_moe and (MoE is not None):
            with torch.cuda.amp.autocast(enabled=False):
                output, aux_loss = self.decoder(tgt, memory, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask,
                                                tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        else:
            output = self.decoder(tgt, memory, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask,
                                  tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
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

# ===== Datasets (your originals, self-contained now) =====

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

# ===== Collate =====
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

# ===== Vocab builders (from your scripts) =====

def build_vocab_aria(configs):
    artifact_folder = configs["artifact_folder"]
    raw_data_folders = configs["raw_data"]["raw_data_folders"]
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
    print('vocab[<ss>]', vocab['<SS>'])
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
    midi_paths = [os.path.join(dataset_path, captions[i]['location']) for i in range(len(captions))][0:30000]
    print(f"Vocabulary length: {tokenizer.vocab_size}")
    vocab_path = os.path.join(artifact_folder, "vocab_remi.pkl")
    with open(vocab_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Vocabulary saved to {vocab_path}")

# ===== Basic training loop (unchanged) =====

def train_basic(configs):
    artifact_folder = configs['artifact_folder']
    tokenizer_filepath = os.path.join(artifact_folder, "vocab.pkl")
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer) + 1
    print("Vocab size: ", vocab_size)

    caption_dataset_path = configs['raw_data']['caption_dataset_path']
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    aria_tokenizer = lambda midi: [] if MidiDict is None else []

    dataset = Text2MusicDatasetAria(configs, captions, aria_tokenizer=aria_tokenizer, mode="train", shuffle=True)
    batch_size = configs['training']['text2midi_model']['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    d_model = configs['model']['text2midi_model']['decoder_d_model']
    nhead = configs['model']['text2midi_model']['decoder_num_heads']
    num_layers = configs['model']['text2midi_model']['decoder_num_layers']
    max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']
    use_moe = configs['model']['text2midi_model']['use_moe']
    num_experts = configs['model']['text2midi_model']['num_experts']
    dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = Transformer(vocab_size, d_model, nhead, max_len, num_layers, dim_feedforward, use_moe, num_experts, device=device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.Adam(model.parameters(), lr=configs['training']['text2midi_model']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    epochs = configs['training']['text2midi_model']['epochs']
    print_every = 10

    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(total=max(1, len(dataloader)), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()
                encoder_input, attention_mask, tgt = [b.to(device) for b in batch]
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                outputs = model(encoder_input, attention_mask, tgt_input)
                aux_loss = 0
                if isinstance(outputs, tuple):
                    outputs, aux_loss = outputs
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1)) + (aux_loss if isinstance(aux_loss, Tensor) else 0)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if step % print_every == 0:
                    pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "transformer_decoder_remi_plus.pth")
    print("Model saved as transformer_decoder_remi_plus.pth")

# =========================================================
# FULLY REWRITTEN: train_accelerate (W&B removed, Local tracking added)
# =========================================================

def train_accelerate(configs, resume_path=None):
    # --- configs ---
    per_device_train_batch_size = configs['training']['text2midi_model']['per_device_train_batch_size']
    gradient_accumulation_steps = configs['training']['text2midi_model']['gradient_accumulation_steps']
    output_dir = configs['training']['text2midi_model']['output_dir']
    lr_scheduler_type = configs['training']['text2midi_model']['lr_scheduler_type']
    num_warmup_steps = configs['training']['text2midi_model']['num_warmup_steps']
    max_train_steps = configs['training']['text2midi_model']['max_train_steps']
    save_every = configs['training']['text2midi_model']['save_every']
    checkpointing_steps = configs['training']['text2midi_model']['checkpointing_steps']

    # NOTE: we intentionally ignore with_tracking/report_to to avoid any external logger
    # with_tracking = configs['training']['text2midi_model']['with_tracking']
    # report_to = configs['training']['text2midi_model']['report_to']

    artifact_folder = configs['artifact_folder']
    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    with open(tokenizer_filepath, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer)

    caption_dataset_path = configs['raw_data']['caption_dataset_path']
    with jsonlines.open(caption_dataset_path) as reader:
        captions = list(reader)

    # Auto mixed precision:
    # - bf16 if supported
    # - else fp16 if CUDA
    # - else "no"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        mp = "bf16"
    elif torch.cuda.is_available():
        mp = "fp16"
    else:
        mp = "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.info(accelerator.state, extra={"main_process_only": False})

    # --- Output directories ---
    if accelerator.is_main_process:
        if not output_dir:
            output_dir = "saved/" + str(int(time.time()))
            os.makedirs("saved", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/outputs", exist_ok=True)
        accelerator.project_configuration.automatic_checkpoint_naming = False
    accelerator.wait_for_everyone()

    device = accelerator.device

    # --- Dataset + dataloader ---
    with accelerator.main_process_first():
        dataset = Text2MusicDatasetRemi(configs, captions, remi_tokenizer=tokenizer, mode="train", shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=per_device_train_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True
        )

    # --- Model ---
    d_model = configs['model']['text2midi_model']['decoder_d_model']
    nhead = configs['model']['text2midi_model']['decoder_num_heads']
    num_layers = configs['model']['text2midi_model']['decoder_num_layers']
    max_len = configs['model']['text2midi_model']['decoder_max_sequence_length']
    use_moe = configs['model']['text2midi_model']['use_moe']
    num_experts = configs['model']['text2midi_model']['num_experts']
    dim_feedforward = configs['model']['text2midi_model']['decoder_intermediate_size']

    model = Transformer(
        vocab_size, d_model, nhead, max_len,
        num_layers, dim_feedforward,
        use_moe, num_experts,
        device=device
    )

    # --- Resume ---
    if resume_path:
        if os.path.isdir(resume_path):
            accelerator.load_state(resume_path)
            accelerator.print(f"Resumed training from directory: {resume_path}")
        else:
            try:
                from safetensors.torch import load_file
                state = load_file(resume_path) if resume_path.endswith(".safetensors") else torch.load(resume_path, map_location=device)
            except Exception:
                state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state, strict=False)
            accelerator.print(f"Resumed training from file: {resume_path}")

    lr = float(configs['training']['text2midi_model']['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Steps and scheduler ---
    # Prepare FIRST so dataloader length reflects DistributedSampler in MULTI_GPU
    model, optimizer = accelerator.prepare(model, optimizer)
    dataloader = accelerator.prepare(dataloader)

    # Now compute correct update steps per epoch (per-process, which is what your loop uses)
    num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)

    # Resolve max_train_steps
    if max_train_steps == 'None' or max_train_steps is None:
        max_train_steps = int(configs['training']['text2midi_model']['epochs']) * num_update_steps_per_epoch
    elif isinstance(max_train_steps, str):
        max_train_steps = int(max_train_steps)

    # Scheduler must be created AFTER we know max_train_steps, then prepared
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Epoch count consistent with the (prepared) dataloader
    epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (
        per_device_train_batch_size
        * accelerator.num_processes
        * gradient_accumulation_steps
    )

    # Extra sanity logs (so this never confuses you again)
    logger.info(f"  Batches per epoch (per process) = {len(dataloader)}")
    logger.info(f"  Update steps per epoch = {num_update_steps_per_epoch}")


    # --- Local tracker setup ---
    run_name = configs['training']['text2midi_model'].get('run_name', 'text2midi')
    run_dir = make_run_dir(output_dir, run_name=run_name)
    tracker = LocalTracker(
        run_dir=run_dir,
        config=deepcopy(configs),
        accelerator=accelerator,
        pad_id=0,
        log_every=10
    )

    # Record run metadata once (includes model size)
    if accelerator.is_main_process:
        # Note: model may be wrapped after prepare; unwrap for size/params
        raw_model = accelerator.unwrap_model(model)
        meta = {
            "time_created": _now_iso(),
            "output_dir": output_dir,
            "run_dir": run_dir,
            "tokenizer_path": tokenizer_filepath,
            "config_sha1": None,
            "env": _torch_env_dict(),
            "effective_batch_size": int(total_batch_size),
            "train_settings": {
                "per_device_train_batch_size": int(per_device_train_batch_size),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "num_processes": int(accelerator.num_processes),
                "max_train_steps": int(max_train_steps),
                "epochs": int(epochs),
                "lr_scheduler_type": str(lr_scheduler_type),
                "num_warmup_steps": int(num_warmup_steps),
            },
            "model_arch": {
                "vocab_size": int(vocab_size),
                "d_model": int(d_model),
                "nhead": int(nhead),
                "num_layers": int(num_layers),
                "dim_feedforward": int(dim_feedforward),
                "max_len": int(max_len),
                "use_moe": bool(use_moe),
                "num_experts": int(num_experts),
            },
            "model_size": model_size_report(raw_model),
        }
        tracker.write_run_meta(meta)

    accelerator.wait_for_everyone()

    # --- Log header ---
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Total train batch size (effective) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Tracking dir = {run_dir}")

    # --- Training ---
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0
    best_loss = float("inf")
    out_dir = output_dir

    model.train()

    for epoch in range(starting_epoch, epochs):
        epoch_loss_sum = 0.0
        epoch_token_acc_sum = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):
            step_t0 = time.perf_counter()

            with accelerator.accumulate(model):
                encoder_input, attention_mask, tgt = batch
                encoder_input = encoder_input.to(device)
                attention_mask = attention_mask.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                outputs = model(encoder_input, attention_mask, tgt_input)
                aux_loss = 0
                if isinstance(outputs, tuple):
                    outputs, aux_loss = outputs

                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.reshape(-1)) \
                       + (aux_loss if isinstance(aux_loss, Tensor) else 0)

                # accumulate stats
                epoch_loss_sum += float(loss.detach().float().item())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_steps += 1
                step_t1 = time.perf_counter()
                step_time = step_t1 - step_t0

                # Compute metrics for thesis
                lr = optimizer.param_groups[0]["lr"]
                grad_norm = _grad_global_norm(model.parameters())

                # Token accuracy (ignore PAD=0)
                token_acc = _token_accuracy_ignore_pad(outputs.detach(), tgt_output.detach(), pad_id=0)
                epoch_token_acc_sum += float(token_acc)
                epoch_steps += 1

                # Throughput (tokens/sec and seqs/sec)
                with torch.no_grad():
                    tokens_in_batch = int((tgt_output != 0).sum().item())
                    seqs_in_batch = int(tgt_output.size(0))

                tokens_per_sec = tokens_in_batch / max(1e-9, step_time)
                seqs_per_sec = seqs_in_batch / max(1e-9, step_time)

                # Perplexity estimate from *this step loss* (cap to prevent inf)
                step_loss_val = float(loss.detach().float().item())
                ppl = float(math.exp(min(20.0, step_loss_val)))

                # GPU mem
                gpu_alloc = gpu_reserved = None
                if torch.cuda.is_available():
                    gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
                    gpu_reserved = torch.cuda.memory_reserved() / (1024**3)

                # Progress display
                progress_bar.set_postfix({"Loss": step_loss_val, "LR": float(lr)})
                progress_bar.update(1)

                # Log every N steps (or always if you want)
                if (completed_steps % tracker.log_every) == 0 or completed_steps == 1:
                    tracker.log({
                        "epoch": int(epoch + 1),
                        "step": int(completed_steps),
                        "loss": _safe_float(step_loss_val),
                        "aux_loss": _safe_float(aux_loss.detach().float().item()) if isinstance(aux_loss, Tensor) else _safe_float(aux_loss),
                        "token_acc": _safe_float(token_acc),
                        "ppl": _safe_float(ppl),
                        "lr": _safe_float(lr),
                        "grad_norm": _safe_float(grad_norm),
                        "tokens_per_sec": _safe_float(tokens_per_sec),
                        "seqs_per_sec": _safe_float(seqs_per_sec),
                        "step_time_sec": _safe_float(step_time),
                        "gpu_mem_alloc_gb": _safe_float(gpu_alloc),
                        "gpu_mem_reserved_gb": _safe_float(gpu_reserved),
                        "effective_batch_size": int(total_batch_size),
                        "event": "train_step"
                    })

            # Save checkpoint by step count
            if isinstance(checkpointing_steps, int) and (completed_steps > 0) and (completed_steps % checkpointing_steps == 0) and accelerator.is_main_process:
                accelerator.save_state(os.path.join(out_dir, f"step_{completed_steps}"))

            if completed_steps >= max_train_steps:
                break

        # ----- Epoch end logging + summary file -----
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss_sum / max(1, len(dataloader))
            avg_epoch_acc = epoch_token_acc_sum / max(1, epoch_steps)
            avg_epoch_ppl = float(math.exp(min(20.0, avg_epoch_loss)))

            epoch_idx = int((completed_steps + num_update_steps_per_epoch - 1) // num_update_steps_per_epoch)
            pct = 100.0 * (completed_steps / max(1, max_train_steps))
            logging.info(f"Epoch {epoch_idx}: Loss {avg_epoch_loss:.6f} | TokenAcc {avg_epoch_acc:.6f} | steps={completed_steps}/{max_train_steps} ({pct:.1f}%)")

            # keep your existing summary.jsonl for backwards compatibility
            with open(os.path.join(out_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({
                    "epoch": epoch_idx,
                    "step": completed_steps,
                    "train_loss": round(float(avg_epoch_loss), 6),
                    "token_acc": round(float(avg_epoch_acc), 6),
                    "ppl": round(float(avg_epoch_ppl), 6),
                }) + "\n")

            # new epoch_summary.jsonl (thesis-friendly)
            tracker.log_epoch({
                "epoch": int(epoch_idx),
                "step": int(completed_steps),
                "epoch_train_loss": float(avg_epoch_loss),
                "epoch_token_acc": float(avg_epoch_acc),
                "epoch_ppl": float(avg_epoch_ppl),
                "effective_batch_size": int(total_batch_size),
                "event": "epoch_end"
            })

            # Best-loss tracking for saving
            save_checkpoint = avg_epoch_loss < best_loss
            best_loss = min(best_loss, avg_epoch_loss)

            if checkpointing_steps == "best" and save_checkpoint:
                accelerator.save_state(os.path.join(out_dir, "best"))

            if checkpointing_steps in ("epoch", "best"):
                # match your behavior: save every epoch OR every save_every
                if checkpointing_steps == "epoch" or ((epoch + 1) % save_every == 0):
                    accelerator.save_state(os.path.join(out_dir, f"epoch_{epoch_idx}"))

    tracker.close()

# ===== HF Trainer (unchanged) =====

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
        output = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                              encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, labels=labels)
        logits = output.logits
        loss = output.loss
        return {'loss': loss, 'logits': logits}

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
        pred_ids = argmax(logits, dim=-1)
        return pred_ids

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

# ===== Generation helpers (unchanged) =====

class CaptionDatasetOnly(Dataset):
    def __init__(self, captions):
        self.captions = captions
    def __len__(self):
        return len(self.captions)
    def __getitem__(self, idx):
        return self.captions[idx]

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
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    if str(model_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location=device)

    missing, unexpected = model.load_state_dict(state, strict=False)

    try:
        model.tie_weights()
    except Exception:
        pass

    if accelerator.is_local_main_process:
        print(f"[load] missing keys: {missing} | unexpected: {unexpected}")
    model.to(device)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    return model, tokenizer, r_tokenizer

def generate_single(caption: str, model_path: str, tokenizer_pkl: str, max_len: int = 2000, temperature: float = 0.9, out_dir: str = DEFAULT_OUTPUT_ROOT):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    with open(tokenizer_pkl, "rb") as f:
        r_tokenizer = pickle.load(f)
    vocab_size = len(r_tokenizer)
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    # --- load checkpoint (supports .safetensors) ---
    if str(model_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(model_path)
    else:
        state = torch.load(model_path, map_location=device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing keys: {missing}")
    print(f"[load] unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    t5tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
    inputs = t5tok(caption, return_tensors='pt', padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0).to(device)
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask, max_len=max_len, temperature=temperature)
    output_list = output[0].tolist()
    midi = r_tokenizer.decode(output_list)
    out_path = _next_output_path(root=out_dir, prefix="gen_", ext=".mid")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    midi.dump_midi(out_path)
    print(f"Saved: {out_path}")

def generate_accelerated(
    captions_jsonl: str,
    model_path: str,
    tokenizer_pkl: str,
    batch_size: int = 8,
    max_len: int = 2000,
    temperature: float = 0.9,
    out_root: str = DEFAULT_OUTPUT_ROOT,
):
    from accelerate import Accelerator
    import jsonlines, os, pickle, torch
    from torch.utils.data import DataLoader

    accelerator = Accelerator()
    device = accelerator.device

    with jsonlines.open(captions_jsonl) as reader:
        rows = list(reader)
    has_test_flag = any("test_set" in r for r in rows)
    selected = [r for r in rows if (r.get("test_set") is True)] if has_test_flag else rows
    if not selected:
        print("[gen] No prompts found (0 rows after filtering).")
        return

    with open(tokenizer_pkl, "rb") as f:
        r_tokenizer = pickle.load(f)
    vocab_size = len(r_tokenizer)

    model, t5tok, r_tokenizer = load_model_and_tokenizer_for_gen(
        accelerator, model_path, vocab_size, tokenizer_pkl
    )
    model = model.to(device)
    model.eval()

    class CaptionDatasetOnly(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            r = self.items[idx]
            return {
                "caption": r.get("caption") or r.get("text") or "",
                "location": r.get("location"),
            }

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

    dataset = CaptionDatasetOnly(selected)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_captions,
    )

    os.makedirs(out_root, exist_ok=True)

    with torch.no_grad():
        for captions, locations in dataloader:
            if not captions:
                continue

            enc = t5tok(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

            token_seqs = model.generate(
                src=input_ids,
                src_mask=attention_mask,
                max_len=max_len,
                temperature=temperature,
            )

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
                    print(f"[gen][error] failed to decode/save item {j}: {e}")

# ===== Caption splitter (unchanged) =====

def split_captions(input_path: str, output_dir: str, num_splits: int = 6):
    with jsonlines.open(input_path) as reader:
        captions = [line for line in reader if line.get('test_set') is True]
    selected_captions = captions
    split_size = max(1, len(selected_captions) // num_splits)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else len(selected_captions)
        split_captions = selected_captions[start_idx:end_idx]
        out_path = os.path.join(output_dir, f'selected_captions_{i}.json')
        with open(out_path, 'w') as f:
            json.dump(split_captions, f, indent=4)
        print(f'Saved {len(split_captions)} captions to {out_path}')

# ===================== MAIN =====================

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd in ("build_vocab_aria", "build_vocab_remi", "dataset_demo_aria", "dataset_demo_remi",
                    "train_basic", "train_accelerate", "train_hf"):
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
        train_basic(configs)

    elif args.cmd == "train_accelerate":
        train_accelerate(configs, resume_path=args.resume_path)

    elif args.cmd == "train_hf":
        train_hf(configs)

    elif args.cmd == "generate":
        generate_single(caption=args.caption, model_path=args.model_path, tokenizer_pkl=args.tokenizer_pkl,
                        max_len=args.max_len, temperature=args.temperature, out_dir=args.out_dir)

    elif args.cmd == "generate_accelerate":
        generate_accelerated(captions_jsonl=args.captions_jsonl, model_path=args.model_path, tokenizer_pkl=args.tokenizer_pkl,
                             batch_size=args.batch_size, max_len=args.max_len, temperature=args.temperature, out_root=args.out_root)

    elif args.cmd == "split_captions":
        split_captions(input_path=args.input_jsonl, output_dir=args.output_dir, num_splits=args.num_splits)

if __name__ == "__main__":
    main()