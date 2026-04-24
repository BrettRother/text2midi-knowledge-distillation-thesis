#!/usr/bin/env python3
"""
generate_student_thesis_graphs.py

Creates:
  - Student-only thesis graphs
  - Teacher vs Student comparison graphs
  - summary.json with key metrics + meta

IMPORTANT (your setup):
  - Student run folder contains: thesis_steps.csv, thesis_run_meta.json (NOT metrics.csv/run_meta.json)
  - Teacher run folder may contain: metrics.csv + run_meta.json (or also thesis_steps.csv/thesis_run_meta.json)

Default behavior:
  - Student run dir = current working directory
  - Teacher run dir = TEACHER_RUN_DIR below
  - Output dir = <student_run_dir>/thesis_graphs
"""

import os
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ============================================================
# CONFIG
# ============================================================
TEACHER_RUN_DIR = "/home/brett_ece/midi/text2midi_teacher/Output/tracking/text2midi_teacher_20260219_151050"

# Student run directory is where you execute the script from:
# e.g. /home/brett_ece/midi/text2midi_student/Output2
STUDENT_RUN_DIR = os.getcwd()

OUT_DIR = os.path.join(STUDENT_RUN_DIR, "thesis_graphs")
COMPARE_DIR = os.path.join(OUT_DIR, "compare")


# ------------------------------
# Helpers
# ------------------------------
def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def pick_column(df: pd.DataFrame, candidates, contains_any=None):
    cols = list(df.columns)
    norm_map = {_norm_col(c): c for c in cols}

    # exact matches (normalized)
    for cand in candidates:
        cand_n = _norm_col(cand)
        if cand_n in norm_map:
            return norm_map[cand_n]

    # contains token matches
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


def adaptive_smooth(y: np.ndarray, preferred=200):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y, 1
    window = preferred
    if len(y) < window:
        window = max(5, len(y) // 10)
    return uniform_filter1d(y, size=window), window


def resolve_metrics_path(run_dir: str, run_name: str) -> str:
    """
    Supports both naming conventions:
      - metrics.csv
      - thesis_steps.csv  (your student Output2 folder)
    """
    candidates = [
        os.path.join(run_dir, "metrics.csv"),
        os.path.join(run_dir, "thesis_steps.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"[{run_name}] Missing metrics file. Tried: {candidates}"
    )


def resolve_meta_path(run_dir: str, run_name: str) -> str:
    """
    Supports both naming conventions:
      - run_meta.json
      - thesis_run_meta.json  (your student Output2 folder)
    """
    candidates = [
        os.path.join(run_dir, "run_meta.json"),
        os.path.join(run_dir, "thesis_run_meta.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # meta is optional; return None
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

    # detect columns
    step_col = pick_column(
        metrics,
        candidates=["step", "global_step", "trainer_step", "train_step", "iter", "iteration"],
        contains_any=["step"],
    )
    loss_col = pick_column(
        metrics,
        candidates=[
            "loss_total",      # <-- prefer this for your thesis_steps.csv
            "train_loss",
            "loss",
            "training_loss",
            "train/loss",
            "train-loss",
            "loss_ce",         # fallback options if total isn't available
            "loss_kd_kl",
            "loss_hard_ce",
        ],
        contains_any=["loss_total"],  # fallback if column name is slightly different
    )

    # if still not found, do a safer fallback that avoids loss_aux
    if loss_col is None:
        # pick any 'loss' column except known aux placeholders
        loss_candidates = [c for c in metrics.columns if "loss" in _norm_col(c) and _norm_col(c) not in ("loss_aux",)]
        if loss_candidates:
            loss_col = loss_candidates[0]

    epoch_col = pick_column(
        metrics,
        candidates=["epoch", "train_epoch", "trainer_epoch"],
        contains_any=["epoch"],
    )
    lr_col = pick_column(
        metrics,
        candidates=["lr", "learning_rate", "train/lr", "optimizer_lr"],
        contains_any=["lr"],
    )

    # fallbacks
    if step_col is None:
        metrics["__step__"] = np.arange(len(metrics))
        step_col = "__step__"
        print(f"[{run_name}] ⚠️ No step column detected — using row index as step.")

    if loss_col is None:
        raise KeyError(f"[{run_name}] Could not detect a loss column. Columns: {list(metrics.columns)}")

    # numeric + cleanup
    metrics = ensure_numeric(metrics, step_col)
    metrics = ensure_numeric(metrics, loss_col)
    if epoch_col is not None:
        metrics = ensure_numeric(metrics, epoch_col)
    if lr_col is not None:
        metrics = ensure_numeric(metrics, lr_col)

    metrics = metrics.dropna(subset=[step_col, loss_col]).copy()
    metrics = metrics.sort_values(step_col)

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


def plot_save(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def summarize(run):
    mm = run["metrics"]
    sc = run["step_col"]
    lc = run["loss_col"]
    steps = mm[sc].to_numpy()
    loss = mm[lc].to_numpy()
    return {
        "run": run["name"],
        "run_dir": run["run_dir"],
        "metrics_path": run["metrics_path"],
        "meta_path": run["meta_path"],
        "num_points": int(len(mm)),
        "first_step": float(steps[0]) if len(steps) else None,
        "last_step": float(steps[-1]) if len(steps) else None,
        "final_loss": float(loss[-1]) if len(loss) else None,
        "best_loss": float(np.nanmin(loss)) if len(loss) else None,
        "params_total": safe_get(run["meta"], "model_size.params_total"),
        "params_trainable": safe_get(run["meta"], "model_size.params_trainable"),
        "params_frozen": safe_get(run["meta"], "model_size.params_frozen"),
        "approx_param_mb_fp16": safe_get(run["meta"], "model_size.approx_param_mb_fp16"),
        "approx_param_mb_fp32": safe_get(run["meta"], "model_size.approx_param_mb_fp32"),
        "loss_column_used": run["loss_col"],
        "step_column_used": run["step_col"],
        "lr_column_used": run["lr_col"],
        "epoch_column_used": run["epoch_col"],
    }


# ------------------------------
# Main
# ------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(COMPARE_DIR, exist_ok=True)

    student = load_run(STUDENT_RUN_DIR, "student")
    teacher = load_run(TEACHER_RUN_DIR, "teacher")

    print("\nDetected columns:")
    print(f"  student: step={student['step_col']!r}, loss={student['loss_col']!r}, epoch={student['epoch_col']!r}, lr={student['lr_col']!r}")
    print(f"  teacher: step={teacher['step_col']!r}, loss={teacher['loss_col']!r}, epoch={teacher['epoch_col']!r}, lr={teacher['lr_col']!r}")

    # ============================================================
    # 1) STUDENT-ONLY GRAPHS
    # ============================================================
    m = student["metrics"]
    s_step = student["step_col"]
    s_loss = student["loss_col"]

    # 1. Loss vs Step
    plt.figure()
    plt.plot(m[s_step], m[s_loss])
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Student Training Loss vs Step")
    plot_save(os.path.join(OUT_DIR, "student_loss_vs_step.png"))

    # 2. Smoothed Loss
    s_loss_vals = m[s_loss].to_numpy()
    s_smooth_vals, s_win = adaptive_smooth(s_loss_vals, preferred=200)

    plt.figure()
    plt.plot(m[s_step], s_smooth_vals)
    plt.xlabel("Step")
    plt.ylabel("Smoothed Loss")
    plt.title(f"Student Smoothed Training Loss (window={s_win})")
    plot_save(os.path.join(OUT_DIR, "student_loss_smoothed.png"))

    # 3. Loss Histogram
    plt.figure()
    plt.hist(s_loss_vals[np.isfinite(s_loss_vals)], bins=50)
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.title("Student Loss Distribution")
    plot_save(os.path.join(OUT_DIR, "student_loss_histogram.png"))

    # 4. Learning Rate Curve (if present)
    if student["lr_col"] is not None and student["lr_col"] in m.columns:
        lr_vals = m[student["lr_col"]].dropna()
        if len(lr_vals) > 0:
            plt.figure()
            plt.plot(m.loc[lr_vals.index, s_step], lr_vals)
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Student Learning Rate Schedule")
            plot_save(os.path.join(OUT_DIR, "student_lr_schedule.png"))

    # 5. Parameter Breakdown (meta)
    params_train = safe_get(student["meta"], "model_size.params_trainable")
    params_frozen = safe_get(student["meta"], "model_size.params_frozen")
    if params_train is not None and params_frozen is not None:
        plt.figure()
        plt.pie([params_train, params_frozen], labels=["Trainable", "Frozen"], autopct="%1.1f%%")
        plt.title("Student Parameter Distribution")
        plot_save(os.path.join(OUT_DIR, "student_parameter_pie.png"))

    # 6. Memory Footprint (meta)
    fp32 = safe_get(student["meta"], "model_size.approx_param_mb_fp32")
    fp16 = safe_get(student["meta"], "model_size.approx_param_mb_fp16")
    if fp32 is not None and fp16 is not None:
        plt.figure()
        plt.bar(["FP32", "FP16"], [fp32, fp16])
        plt.ylabel("Memory (MB)")
        plt.title("Student Model Parameter Memory Usage")
        plot_save(os.path.join(OUT_DIR, "student_memory_fp32_vs_fp16.png"))

    # 7. Loss Variance per Epoch
    if student["epoch_col"] is not None and student["epoch_col"] in m.columns:
        tmp = m.dropna(subset=[student["epoch_col"]]).copy()
        if len(tmp) > 0:
            epoch_var = tmp.groupby(student["epoch_col"])[s_loss].var()
            plt.figure()
            plt.plot(epoch_var.index, epoch_var.values)
            plt.xlabel("Epoch")
            plt.ylabel("Loss Variance")
            plt.title("Student Loss Variance per Epoch")
            plot_save(os.path.join(OUT_DIR, "student_loss_variance_epoch.png"))

    # 8. Convergence Rate (dLoss/dStep)
    step_vals = m[s_step].to_numpy()
    loss_vals = m[s_loss].to_numpy()
    if len(loss_vals) >= 3:
        grad = np.gradient(loss_vals, step_vals)
        plt.figure()
        plt.plot(m[s_step], grad)
        plt.xlabel("Step")
        plt.ylabel("d(Loss)/d(Step)")
        plt.title("Student Loss Gradient (Convergence Speed)")
        plot_save(os.path.join(OUT_DIR, "student_loss_gradient.png"))

    # ============================================================
    # 2) COMPARISON GRAPHS (Teacher vs Student)
    # ============================================================
    tm = teacher["metrics"]
    t_step = teacher["step_col"]
    t_loss = teacher["loss_col"]

    # A) Loss vs Step (overlay)
    plt.figure()
    plt.plot(tm[t_step], tm[t_loss], label="Teacher")
    plt.plot(m[s_step], m[s_loss], label="Student")
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Step (Teacher vs Student)")
    plt.legend()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_vs_step.png"))

    # B) Smoothed Loss (overlay)
    t_loss_vals = tm[t_loss].to_numpy()
    t_smooth, t_win = adaptive_smooth(t_loss_vals, preferred=200)

    plt.figure()
    plt.plot(tm[t_step], t_smooth, label=f"Teacher (win={t_win})")
    plt.plot(m[s_step], s_smooth_vals, label=f"Student (win={s_win})")
    plt.xlabel("Step")
    plt.ylabel("Smoothed Loss")
    plt.title("Smoothed Loss (Teacher vs Student)")
    plt.legend()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_smoothed.png"))

    # C) Loss histogram overlay
    plt.figure()
    plt.hist(t_loss_vals[np.isfinite(t_loss_vals)], bins=50, alpha=0.5, label="Teacher")
    plt.hist(s_loss_vals[np.isfinite(s_loss_vals)], bins=50, alpha=0.5, label="Student")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.title("Loss Distribution (Teacher vs Student)")
    plt.legend()
    plot_save(os.path.join(COMPARE_DIR, "compare_loss_histogram.png"))

    # D) LR schedule overlay (only if both have LR)
    if teacher["lr_col"] is not None and student["lr_col"] is not None:
        t_lr = pd.to_numeric(tm[teacher["lr_col"]], errors="coerce")
        s_lr = pd.to_numeric(m[student["lr_col"]], errors="coerce")
        if t_lr.notna().any() and s_lr.notna().any():
            plt.figure()
            plt.plot(tm.loc[t_lr.dropna().index, t_step], t_lr.dropna(), label="Teacher")
            plt.plot(m.loc[s_lr.dropna().index, s_step], s_lr.dropna(), label="Student")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule (Teacher vs Student)")
            plt.legend()
            plot_save(os.path.join(COMPARE_DIR, "compare_lr_schedule.png"))

    # E) Params + Memory compare bars (if present in both metas)
    t_params_total = safe_get(teacher["meta"], "model_size.params_total")
    s_params_total = safe_get(student["meta"], "model_size.params_total")
    if t_params_total is not None and s_params_total is not None:
        plt.figure()
        plt.bar(["Teacher", "Student"], [t_params_total, s_params_total])
        plt.ylabel("Total Parameters")
        plt.title("Total Parameters (Teacher vs Student)")
        plot_save(os.path.join(COMPARE_DIR, "compare_params_total.png"))

    t_fp16 = safe_get(teacher["meta"], "model_size.approx_param_mb_fp16")
    s_fp16 = safe_get(student["meta"], "model_size.approx_param_mb_fp16")
    if t_fp16 is not None and s_fp16 is not None:
        plt.figure()
        plt.bar(["Teacher FP16", "Student FP16"], [t_fp16, s_fp16])
        plt.ylabel("Memory (MB)")
        plt.title("Parameter Memory FP16 (Teacher vs Student)")
        plot_save(os.path.join(COMPARE_DIR, "compare_memory_fp16.png"))

    # ============================================================
    # 3) Summary JSON
    # ============================================================
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


if __name__ == "__main__":
    main()