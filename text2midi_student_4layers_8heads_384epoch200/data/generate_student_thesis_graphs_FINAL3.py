#!/usr/bin/env python3
"""
generate_student_thesis_graphs.py

Creates:
  - Student-only thesis graphs
  - Teacher vs Student comparison graphs
  - summary.json with key metrics + meta

Fixes included:
  ✅ LR dip fix: thesis_steps.csv logs MANY lr values per step (optimizer param groups).
     Using mean creates fake dips. Using max can create an *upper-envelope* artifact.
     We now aggregate LR per step using MODE(nonzero) by default (fallback: median(nonzero)).
  ✅ Gradient plot fix: gradient needs unique, increasing x (steps). We dedupe steps first.
  ✅ Progress-aligned compare plots: the two compare loss plots use the same progress mapping.

Notes:
  - Student run folder contains: thesis_steps.csv, thesis_run_meta.json
  - Teacher run folder may contain: metrics.csv + run_meta.json (or thesis_steps.csv + thesis_run_meta.json)
"""

import os
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
TEACHER_RUN_DIR = r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\Data"
STUDENT_RUN_DIR = r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_4layers_8heads_384epoch200\data"


OUT_DIR = os.path.join(STUDENT_RUN_DIR, "thesis_graphs")
COMPARE_DIR = os.path.join(OUT_DIR, "compare")

# Plot controls
MAX_POINTS = 1200
SMOOTH_WIN = 200

# ✅ NEW default for LR: reconstruct the true schedule from param-group logs
LR_AGG = "mode_nonzero"   # options: mode_nonzero, median_nonzero, mean, max, min


# ============================================================
# Small utilities
# ============================================================
def to_progress(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    x0, x1 = float(x[0]), float(x[-1])
    if x1 == x0:
        return np.zeros_like(x)
    return (x - x0) / (x1 - x0) * 100.0


def adaptive_smooth(x: np.ndarray, y: np.ndarray, preferred=200):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        return x, y, 1

    window = int(preferred)
    if len(y) < window:
        window = max(5, len(y) // 10)

    sm = pd.Series(y).rolling(window=window, min_periods=window).mean().to_numpy()
    valid = np.isfinite(sm)
    return x[valid], sm[valid], window


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 1200):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def normalize_01(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    lo = float(np.nanmin(y))
    hi = float(np.nanmax(y))
    return (y - lo) / max(1e-12, (hi - lo))


def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def pick_column(df: pd.DataFrame, candidates, contains_any=None):
    cols = list(df.columns)
    norm_map = {_norm_col(c): c for c in cols}

    for cand in candidates:
        cand_n = _norm_col(cand)
        if cand_n in norm_map:
            return norm_map[cand_n]

    if contains_any:
        tokens = [t.lower() for t in contains_any]
        for c in cols:
            nc = _norm_col(c)
            if all(t in nc for t in tokens):
                return c

    return None


def ensure_numeric(df: pd.DataFrame, col: str):
    if col is None:
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_get(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def plot_save(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def legend_outside_right():
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
    )


def run_folder_name(run_dir: str) -> str:
    try:
        norm = os.path.normpath(str(run_dir))
        base = os.path.basename(norm)
        if base.lower() in ("data", "tracking", "output", "outputs"):
            parent = os.path.basename(os.path.dirname(norm))
            return parent or base or str(run_dir)
        return base or str(run_dir)
    except Exception:
        return str(run_dir)


def set_pretty_axes():
    ax = plt.gca()
    ax.grid(True, alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)


# ============================================================
# Core series preparation
# ============================================================
def prepare_xy(df: pd.DataFrame, x_col: str, y_col: str, agg: str = "mean"):
    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.dropna(subset=[x_col, y_col])
    if len(tmp) == 0:
        return np.array([]), np.array([])

    if agg == "mean":
        tmp = tmp.groupby(x_col, as_index=False)[y_col].mean()
    elif agg == "median":
        tmp = tmp.groupby(x_col, as_index=False)[y_col].median()
    elif agg == "max":
        tmp = tmp.groupby(x_col, as_index=False)[y_col].max()
    elif agg == "min":
        tmp = tmp.groupby(x_col, as_index=False)[y_col].min()
    else:
        raise ValueError(f"Unknown agg={agg!r}")

    tmp = tmp.sort_values(x_col)
    return tmp[x_col].to_numpy(), tmp[y_col].to_numpy()


# ✅ NEW: LR reconstruction from param-group logs
def prepare_lr_xy(df: pd.DataFrame, step_col: str, lr_col: str, agg: str = "mode_nonzero"):
    tmp = df[[step_col, lr_col]].copy()
    tmp[step_col] = pd.to_numeric(tmp[step_col], errors="coerce")
    tmp[lr_col] = pd.to_numeric(tmp[lr_col], errors="coerce")
    tmp = tmp.dropna(subset=[step_col, lr_col])
    if len(tmp) == 0:
        return np.array([]), np.array([])

    def _agg_one(group: pd.Series) -> float:
        vals = group.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        # drop zeros/negatives (common from unused param groups or logging quirks)
        nz = vals[vals > 0]
        if len(nz) == 0:
            return float(np.nan)

        if agg == "mode_nonzero":
            # Use a rounded mode to avoid float-equality issues
            r = np.round(nz, 12)
            uniq, counts = np.unique(r, return_counts=True)
            # pick most frequent value; if tie, pick median of tied
            mx = counts.max()
            tied = uniq[counts == mx]
            if len(tied) == 1:
                return float(tied[0])
            return float(np.median(tied))

        if agg == "median_nonzero":
            return float(np.median(nz))

        if agg == "mean":
            return float(np.mean(nz))
        if agg == "max":
            return float(np.max(nz))
        if agg == "min":
            return float(np.min(nz))

        raise ValueError(f"Unknown LR agg={agg!r}")

    out = tmp.groupby(step_col)[lr_col].apply(_agg_one).reset_index(name="lr_agg")
    out = out.dropna(subset=["lr_agg"]).sort_values(step_col)
    return out[step_col].to_numpy(), out["lr_agg"].to_numpy()


def warn_if_many_duplicates(df: pd.DataFrame, step_col: str, value_col: str, label: str):
    if step_col not in df.columns or value_col not in df.columns:
        return
    counts = df.groupby(step_col)[value_col].count()
    mx = int(counts.max()) if len(counts) else 0
    if mx > 1:
        print(f"[{label}] NOTE: '{value_col}' has duplicates per '{step_col}'. max count per step = {mx}.")
        if _norm_col(value_col) in ("lr", "learning_rate", "optimizer_lr", "train_lr"):
            print(f"[{label}]      Using LR_AGG={LR_AGG!r} to reconstruct the true scheduler LR per step.")


# ============================================================
# Metrics/meta loading
# ============================================================
def resolve_metrics_path(run_dir: str, run_name: str) -> str:
    candidates = [
        os.path.join(run_dir, "metrics.csv"),
        os.path.join(run_dir, "thesis_steps.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"[{run_name}] Missing metrics file. Tried: {candidates}")


def resolve_meta_path(run_dir: str, run_name: str):
    candidates = [
        os.path.join(run_dir, "run_meta.json"),
        os.path.join(run_dir, "thesis_run_meta.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    warnings.warn(f"[{run_name}] No meta JSON found in {run_dir} (skipping params/memory plots).")
    return None


def load_run(run_dir: str, run_name: str):
    metrics_path = resolve_metrics_path(run_dir, run_name)
    meta_path = resolve_meta_path(run_dir, run_name)

    metrics = pd.read_csv(metrics_path)

    meta = {}
    if meta_path is not None and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    step_col = pick_column(
        metrics,
        candidates=["global_step", "step", "trainer_step", "train_step", "iter", "iteration"],
        contains_any=["step"],
    )

    epoch_col = pick_column(
        metrics,
        candidates=["epoch", "train_epoch", "trainer_epoch"],
        contains_any=["epoch"],
    )

    loss_col = pick_column(
        metrics,
        candidates=[
            "loss_total",
            "train_loss",
            "loss",
            "training_loss",
            "train/loss",
            "train-loss",
            "loss_ce",
            "loss_kd_kl",
            "loss_hard_ce",
        ],
        contains_any=["loss"],
    )

    if loss_col is None:
        loss_candidates = [
            c for c in metrics.columns
            if "loss" in _norm_col(c) and _norm_col(c) not in ("loss_aux",)
        ]
        if loss_candidates:
            loss_col = loss_candidates[0]

    lr_col = pick_column(
        metrics,
        candidates=["lr", "learning_rate", "train/lr", "optimizer_lr"],
        contains_any=["lr"],
    )

    if step_col is None:
        metrics["__step__"] = np.arange(len(metrics))
        step_col = "__step__"
        print(f"[{run_name}] ⚠️ No step column detected — using row index as step.")

    if loss_col is None:
        raise KeyError(f"[{run_name}] Could not detect a loss column. Columns: {list(metrics.columns)}")

    metrics = ensure_numeric(metrics, step_col)
    metrics = ensure_numeric(metrics, loss_col)
    if epoch_col is not None:
        metrics = ensure_numeric(metrics, epoch_col)
    if lr_col is not None:
        metrics = ensure_numeric(metrics, lr_col)

    metrics = metrics.dropna(subset=[step_col, loss_col]).copy()
    metrics = metrics.sort_values(step_col)

    if lr_col is not None and lr_col in metrics.columns:
        warn_if_many_duplicates(metrics, step_col, lr_col, run_name)

    return {
        "name": run_name,
        "run_dir": run_dir,
        "metrics_path": metrics_path,
        "meta_path": meta_path,
        "metrics": metrics,
        "meta": meta,
        "step_col": step_col,
        "loss_col": loss_col,
        "epoch_col": epoch_col,
        "lr_col": lr_col,
    }


# ============================================================
# Plot functions
# ============================================================
def plot_raw_and_smooth(x_raw, y_raw, x_smooth, y_smooth, win: int,
                        xlabel: str, ylabel: str, title: str, out_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(x_raw, y_raw, alpha=0.20, linewidth=1.0, label="Raw")
    if len(x_smooth) and len(y_smooth):
        plt.plot(x_smooth, y_smooth, linewidth=2.0, label=f"Smoothed (win={win})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    set_pretty_axes()
    plt.legend()
    plot_save(out_path)


def plot_lr_logged(df: pd.DataFrame, step_col: str, lr_col: str, title: str, out_path: str,
                   normalize: bool = False, max_points: int = 1200, lr_agg: str = "mode_nonzero"):
    if lr_col is None or lr_col not in df.columns:
        return

    # ✅ use LR reconstruction (mode/median) instead of max envelope
    x, y = prepare_lr_xy(df, step_col, lr_col, agg=lr_agg)
    if len(x) == 0:
        return

    if normalize:
        y = y / max(1e-12, float(np.nanmax(y)))  # max-normalize (stable for cosine plots)

    x, y = downsample_xy(x, y, max_points=max_points)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2.2)
    plt.xlabel("Step")
    plt.ylabel("Normalized Learning Rate (LR / max LR)" if normalize else "Learning Rate")
    if not normalize:
        ax = plt.gca()
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.title(title)
    set_pretty_axes()
    plot_save(out_path)


def plot_loss_gradient(df: pd.DataFrame, step_col: str, loss_col: str, title: str, out_path: str,
                       smooth_win: int = 200, max_points: int = 1200):
    x, y = prepare_xy(df, step_col, loss_col, agg="mean")
    if len(x) < 3:
        plt.figure(figsize=(10, 5))
        plt.title(title + " (insufficient points)")
        plot_save(out_path)
        return

    if smooth_win and smooth_win > 1:
        x, y, _ = adaptive_smooth(x, y, preferred=smooth_win)

    grad = np.gradient(y, x)

    if len(grad) > 2:
        x = x[1:-1]
        grad = grad[1:-1]

    good = np.isfinite(grad) & np.isfinite(x)
    x = x[good]
    grad = grad[good]

    x, grad = downsample_xy(x, grad, max_points=max_points)

    plt.figure(figsize=(10, 5))
    plt.plot(x, grad)
    plt.xlabel("Step")
    plt.ylabel("d(Loss)/d(Step)")
    plt.title(title)
    set_pretty_axes()
    plot_save(out_path)


# ============================================================
# Summary
# ============================================================
def summarize(run):
    mm = run["metrics"]
    sc = run["step_col"]
    lc = run["loss_col"]

    x, y = prepare_xy(mm, sc, lc, agg="mean")
    return {
        "run": run["name"],
        "run_dir": run["run_dir"],
        "metrics_path": run["metrics_path"],
        "meta_path": run["meta_path"],
        "num_points_raw": int(len(mm)),
        "num_points_unique_step": int(len(x)),
        "first_step": float(x[0]) if len(x) else None,
        "last_step": float(x[-1]) if len(x) else None,
        "final_loss": float(y[-1]) if len(y) else None,
        "best_loss": float(np.nanmin(y)) if len(y) else None,
        "params_total": safe_get(run["meta"], "model_size.params_total"),
        "params_trainable": safe_get(run["meta"], "model_size.params_trainable"),
        "params_frozen": safe_get(run["meta"], "model_size.params_frozen"),
        "approx_param_mb_fp16": safe_get(run["meta"], "model_size.approx_param_mb_fp16"),
        "approx_param_mb_fp32": safe_get(run["meta"], "model_size.approx_param_mb_fp32"),
        "loss_column_used": run["loss_col"],
        "step_column_used": run["step_col"],
        "lr_column_used": run["lr_col"],
        "epoch_column_used": run["epoch_col"],
        "lr_agg_used": LR_AGG,
    }


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COMPARE_DIR, exist_ok=True)

    student = load_run(STUDENT_RUN_DIR, "student")
    teacher = load_run(TEACHER_RUN_DIR, "teacher")

    student_label = run_folder_name(student["run_dir"])
    teacher_label = run_folder_name(teacher["run_dir"])

    print("\nDetected columns:")
    print(f"  student: step={student['step_col']!r}, loss={student['loss_col']!r}, epoch={student['epoch_col']!r}, lr={student['lr_col']!r}")
    print(f"  teacher: step={teacher['step_col']!r}, loss={teacher['loss_col']!r}, epoch={teacher['epoch_col']!r}, lr={teacher['lr_col']!r}")

    # ------------------------------------------------------------
    # 1) STUDENT-ONLY GRAPHS
    # ------------------------------------------------------------
    m = student["metrics"]
    s_step = student["step_col"]
    s_loss = student["loss_col"]

    x_raw = m[s_step].to_numpy()
    y_raw = m[s_loss].to_numpy()
    x_u, y_u = prepare_xy(m, s_step, s_loss, agg="mean")
    x_s, y_s, win = adaptive_smooth(x_u, y_u, preferred=SMOOTH_WIN)

    plot_raw_and_smooth(
        x_raw, y_raw,
        x_s, y_s, win,
        xlabel="Step",
        ylabel="Training Loss",
        title=f"{student_label} - Training Loss vs Step",
        out_path=os.path.join(OUT_DIR, "student_loss_vs_step.png"),
    )

    plt.figure(figsize=(10, 5))
    plt.plot(x_s, y_s)
    plt.xlabel("Step")
    plt.ylabel("Smoothed Loss")
    plt.title(f"{student_label} - Smoothed Training Loss (window={win})")
    set_pretty_axes()
    plot_save(os.path.join(OUT_DIR, "student_loss_smoothed.png"))

    plt.figure(figsize=(10, 5))
    vals = pd.to_numeric(m[s_loss], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    plt.hist(vals, bins=50)
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.title(f"{student_label} - Loss Distribution")
    set_pretty_axes()
    plot_save(os.path.join(OUT_DIR, "student_loss_histogram.png"))

    # ✅ Student LR schedule (this is where your plot was wrong)
    plot_lr_logged(
        df=m,
        step_col=s_step,
        lr_col=student["lr_col"],
        title=f"{student_label} - Learning Rate Schedule (logged, agg={LR_AGG})",
        out_path=os.path.join(OUT_DIR, "student_lr_schedule.png"),
        normalize=False,
        max_points=MAX_POINTS,
        lr_agg=LR_AGG,
    )

    params_train = safe_get(student["meta"], "model_size.params_trainable")
    params_frozen = safe_get(student["meta"], "model_size.params_frozen")
    if params_train is not None and params_frozen is not None:
        plt.figure(figsize=(10, 5))
        plt.pie([params_train, params_frozen], labels=["Trainable", "Frozen"], autopct="%1.1f%%")
        plt.title(f"{student_label} - Parameter Distribution")
        plot_save(os.path.join(OUT_DIR, "student_parameter_pie.png"))

    fp32 = safe_get(student["meta"], "model_size.approx_param_mb_fp32")
    fp16 = safe_get(student["meta"], "model_size.approx_param_mb_fp16")
    if fp32 is not None and fp16 is not None:
        plt.figure(figsize=(10, 5))
        plt.bar(["FP32", "FP16"], [fp32, fp16])
        plt.ylabel("Memory (MB)")
        plt.title(f"{student_label} - Model Parameter Memory Usage")
        set_pretty_axes()
        plot_save(os.path.join(OUT_DIR, "student_memory_fp32_vs_fp16.png"))

    if student["epoch_col"] is not None and student["epoch_col"] in m.columns:
        tmp = m.dropna(subset=[student["epoch_col"]]).copy()
        if len(tmp) > 0:
            epoch_var = tmp.groupby(student["epoch_col"])[s_loss].var()
            plt.figure(figsize=(10, 5))
            plt.plot(epoch_var.index, epoch_var.values)
            plt.xlabel("Epoch")
            plt.ylabel("Loss Variance")
            plt.title(f"{student_label} - Loss Variance per Epoch")
            set_pretty_axes()
            plot_save(os.path.join(OUT_DIR, "student_loss_variance_epoch.png"))

    plot_loss_gradient(
        df=m,
        step_col=s_step,
        loss_col=s_loss,
        title=f"{student_label} - Loss Gradient (Convergence Speed)",
        out_path=os.path.join(OUT_DIR, "student_loss_gradient.png"),
        smooth_win=SMOOTH_WIN,
        max_points=MAX_POINTS,
    )

    # ------------------------------------------------------------
    # 2) COMPARISON GRAPHS (Teacher vs Student)
    # ------------------------------------------------------------
    tm = teacher["metrics"]
    t_step = teacher["step_col"]
    t_loss = teacher["loss_col"]

    tx_u, ty_u = prepare_xy(tm, t_step, t_loss, agg="mean")
    sx_u, sy_u = prepare_xy(m,  s_step, s_loss, agg="mean")

    tx_s, ty_s, t_win = adaptive_smooth(tx_u, ty_u, preferred=SMOOTH_WIN)
    sx_s, sy_s, s_win = adaptive_smooth(sx_u, sy_u, preferred=SMOOTH_WIN)

    # progress-aligned loss plots (unchanged)
    t_raw_x = pd.to_numeric(tm[t_step], errors="coerce").to_numpy()
    s_raw_x = pd.to_numeric(m[s_step], errors="coerce").to_numpy()
    t_raw_y = pd.to_numeric(tm[t_loss], errors="coerce").to_numpy()
    s_raw_y = pd.to_numeric(m[s_loss], errors="coerce").to_numpy()

    t_good = np.isfinite(t_raw_x) & np.isfinite(t_raw_y)
    s_good = np.isfinite(s_raw_x) & np.isfinite(s_raw_y)
    t_raw_x, t_raw_y = t_raw_x[t_good], t_raw_y[t_good]
    s_raw_x, s_raw_y = s_raw_x[s_good], s_raw_y[s_good]

    t_raw_x, t_raw_y = downsample_xy(t_raw_x, t_raw_y, max_points=MAX_POINTS)
    s_raw_x, s_raw_y = downsample_xy(s_raw_x, s_raw_y, max_points=MAX_POINTS)

    t_raw_p = to_progress(t_raw_x)
    s_raw_p = to_progress(s_raw_x)
    t_s_p = to_progress(tx_s)
    s_s_p = to_progress(sx_s)

    plt.figure(figsize=(10, 5))
    plt.plot(t_raw_p, t_raw_y, alpha=0.20, linewidth=1.0, label=f"{teacher_label} Raw")
    plt.plot(s_raw_p, s_raw_y, alpha=0.20, linewidth=1.0, label=f"{student_label} Raw")
    plt.plot(t_s_p, ty_s, linewidth=2.0, label=f"{teacher_label} Smooth (win={t_win})")
    plt.plot(s_s_p, sy_s, linewidth=2.0, label=f"{student_label} Smooth (win={s_win})")
    plt.xlabel("Training Progress (%)")
    plt.ylabel("Training Loss")
    plt.title(f"{teacher_label} vs {student_label} - Training Loss (Progress-Aligned)")
    set_pretty_axes()
    legend_outside_right()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_vs_step.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(t_s_p, ty_s, label=f"Teacher (win={t_win})")
    plt.plot(s_s_p, sy_s, label=f"Student (win={s_win})")
    plt.xlabel("Training Progress (%)")
    plt.ylabel("Smoothed Loss")
    plt.title(f"{teacher_label} vs {student_label} - Smoothed Loss (Progress-Aligned)")
    set_pretty_axes()
    legend_outside_right()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_smoothed.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(t_s_p, ty_s, label=f"Teacher (win={t_win})")
    plt.plot(s_s_p, sy_s, label=f"Student (win={s_win})")
    plt.xlabel("Training Progress (%)")
    plt.ylabel("Smoothed Loss")
    plt.title(f"{teacher_label} vs {student_label} - Smoothed Loss (Progress-Aligned)")
    set_pretty_axes()
    legend_outside_right()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_smoothed_progress.png"))

    # histogram (unchanged)
    plt.figure(figsize=(10, 5))
    t_vals = pd.to_numeric(tm[t_loss], errors="coerce").to_numpy()
    s_vals = pd.to_numeric(m[s_loss], errors="coerce").to_numpy()
    t_vals = t_vals[np.isfinite(t_vals)]
    s_vals = s_vals[np.isfinite(s_vals)]

    if len(t_vals) and len(s_vals):
        all_vals = np.concatenate([t_vals, s_vals])
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanpercentile(all_vals, 99.5))
        if not np.isfinite(hi) or hi <= lo:
            hi = float(np.nanmax(all_vals))

        bins = 80
        plt.hist(
            s_vals, bins=bins, range=(lo, hi),
            density=True, histtype="stepfilled",
            alpha=0.25, label="Student", zorder=1
        )
        plt.hist(
            t_vals, bins=bins, range=(lo, hi),
            density=True, histtype="step",
            linewidth=3.0, label="Teacher", zorder=3
        )
        plt.xlim(lo, hi)
    else:
        plt.hist(t_vals, bins=50, density=True, histtype="step", linewidth=3.0, label="Teacher")
        plt.hist(s_vals, bins=50, density=True, histtype="stepfilled", alpha=0.25, label="Student")

    plt.xlabel("Loss")
    plt.ylabel("Density")
    plt.title(f"{teacher_label} vs {student_label} - Loss Distribution")
    set_pretty_axes()
    legend_outside_right()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_histogram.png"))

    # ✅ LR schedule overlay (logged, normalized) — now correct & should match if both are cosine
    if teacher["lr_col"] and student["lr_col"]:
        if teacher["lr_col"] in tm.columns and student["lr_col"] in m.columns:
            tx, ty = prepare_lr_xy(tm, t_step, teacher["lr_col"], agg=LR_AGG)
            sx, sy = prepare_lr_xy(m,  s_step, student["lr_col"],  agg=LR_AGG)

            plt.figure(figsize=(10, 5))
            if len(tx):
                txp, typ = downsample_xy(to_progress(tx), (ty / max(1e-12, float(np.nanmax(ty)))), max_points=MAX_POINTS)
                plt.plot(txp, typ, linewidth=2.2, label=f"Teacher (cosine, max-norm, agg={LR_AGG})")
            if len(sx):
                sxp, syp = downsample_xy(to_progress(sx), (sy / max(1e-12, float(np.nanmax(sy)))), max_points=MAX_POINTS)
                plt.plot(sxp, syp, linewidth=2.2, label=f"Student (cosine, max-norm, agg={LR_AGG})")

            plt.xlabel("Training Progress (%)")
            plt.ylabel("Normalized Learning Rate (LR / max LR)")
            plt.title(f"{teacher_label} vs {student_label} - LR Schedule (Cosine, Progress-Aligned)")
            set_pretty_axes()
            legend_outside_right()
            plot_save(os.path.join(COMPARE_DIR, "compare_lr_schedule.png"))

    # params/memory compares (unchanged)
    t_params_total = safe_get(teacher["meta"], "model_size.params_total")
    s_params_total = safe_get(student["meta"], "model_size.params_total")
    if t_params_total is not None and s_params_total is not None:
        plt.figure(figsize=(10, 5))
        plt.bar(["Teacher", "Student"], [t_params_total, s_params_total])
        plt.ylabel("Total Parameters")
        plt.title(f"{teacher_label} vs {student_label} - Total Parameters")
        set_pretty_axes()
        plot_save(os.path.join(COMPARE_DIR, "compare_params_total.png"))

    t_fp16 = safe_get(teacher["meta"], "model_size.approx_param_mb_fp16")
    s_fp16 = safe_get(student["meta"], "model_size.approx_param_mb_fp16")
    if t_fp16 is not None and s_fp16 is not None:
        plt.figure(figsize=(10, 5))
        plt.bar(["Teacher FP16", "Student FP16"], [t_fp16, s_fp16])
        plt.ylabel("Memory (MB)")
        plt.title(f"{teacher_label} vs {student_label} - Parameter Memory FP16")
        set_pretty_axes()
        plot_save(os.path.join(COMPARE_DIR, "compare_memory_fp16.png"))

    summary = {
        "student": summarize(student),
        "teacher": summarize(teacher),
        "teacher_run_dir": TEACHER_RUN_DIR,
        "student_run_dir": STUDENT_RUN_DIR,
    }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Student graphs saved to: {OUT_DIR}")
    print(f"✅ Comparison graphs saved to: {COMPARE_DIR}")
    print(f"✅ Summary written to: {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"Used student metrics: {student['metrics_path']}")
    print(f"Used teacher metrics: {teacher['metrics_path']}")
    print(f"LR aggregation used: {LR_AGG} (reconstructs scheduler LR from per-step param-group logs)")


if __name__ == "__main__":
    main()