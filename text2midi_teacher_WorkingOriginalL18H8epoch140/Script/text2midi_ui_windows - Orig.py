# python text2midi_ui_windows.py

import os
import sys
import time
import threading
import traceback
import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ============================================================
# CONFIGURE THESE PATHS FOR YOUR MACHINE
# ============================================================

SCRIPT_DIR = r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\Script"
MIDIGEN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "midigen_text2midi.py")

MODEL_PATH = r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\model\epoch_140\pytorch_model.bin"
TOKENIZER_PKL = os.path.join(SCRIPT_DIR, "vocab_remi.pkl")

MIDI_OUTPUT_DIR = r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\music\UImusic\midi"

# Good practical default for about ~1 minute, depending on density
MAX_LEN = 2048
TEMPERATURE = 0.9

# Local cache folders to avoid broken paths like E:\
LOCAL_CACHE_ROOT = os.path.join(SCRIPT_DIR, "_local_cache")
HF_HOME_DIR = os.path.join(LOCAL_CACHE_ROOT, "hf_home")
HF_HUB_CACHE_DIR = os.path.join(LOCAL_CACHE_ROOT, "hub")
TRANSFORMERS_CACHE_DIR = os.path.join(LOCAL_CACHE_ROOT, "transformers")
TORCH_HOME_DIR = os.path.join(LOCAL_CACHE_ROOT, "torch")


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_local_cache_dirs() -> None:
    """
    Force Hugging Face / transformers / torch caches to local folders
    so the code does not try to use a missing drive like E:\\.
    """
    ensure_dir(LOCAL_CACHE_ROOT)
    ensure_dir(HF_HOME_DIR)
    ensure_dir(HF_HUB_CACHE_DIR)
    ensure_dir(TRANSFORMERS_CACHE_DIR)
    ensure_dir(TORCH_HOME_DIR)

    os.environ["HF_HOME"] = HF_HOME_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE_DIR
    os.environ["TORCH_HOME"] = TORCH_HOME_DIR
    os.environ["XDG_CACHE_HOME"] = LOCAL_CACHE_ROOT


def import_midigen_module(script_path: str):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"midigen_text2midi.py not found: {script_path}")

    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)

    setup_local_cache_dirs()

    spec = importlib.util.spec_from_file_location("midigen_text2midi_local", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["midigen_text2midi_local"] = module
    spec.loader.exec_module(module)
    return module


def generate_midi_with_model(prompt: str) -> str:
    """
    Uses the thesis generation code directly and returns the created MIDI path.
    """
    if not prompt.strip():
        raise ValueError("Prompt is empty.")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PKL):
        raise FileNotFoundError(f"Tokenizer pickle not found: {TOKENIZER_PKL}")

    ensure_dir(MIDI_OUTPUT_DIR)

    before = set(os.listdir(MIDI_OUTPUT_DIR))
    module = import_midigen_module(MIDIGEN_SCRIPT_PATH)

    if not hasattr(module, "generate_single"):
        raise AttributeError("generate_single() was not found in midigen_text2midi.py")

    module.generate_single(
        caption=prompt,
        model_path=MODEL_PATH,
        tokenizer_pkl=TOKENIZER_PKL,
        max_len=MAX_LEN,
        temperature=TEMPERATURE,
        out_dir=MIDI_OUTPUT_DIR,
    )

    after = set(os.listdir(MIDI_OUTPUT_DIR))
    new_files = sorted(
        [f for f in (after - before) if f.lower().endswith((".mid", ".midi"))]
    )

    if new_files:
        return os.path.join(MIDI_OUTPUT_DIR, new_files[-1])

    midi_files = [
        os.path.join(MIDI_OUTPUT_DIR, f)
        for f in os.listdir(MIDI_OUTPUT_DIR)
        if f.lower().endswith((".mid", ".midi"))
    ]
    if not midi_files:
        raise RuntimeError("Model finished but no MIDI file was found in the output folder.")

    return max(midi_files, key=os.path.getmtime)


# ============================================================
# TKINTER UI
# ============================================================

class Text2MidiUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Text-to-MIDI Generator")
        self.root.geometry("820x520")

        self.status_var = tk.StringVar(value="Ready")
        self.midi_var = tk.StringVar(value="")

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="Text-to-MIDI Demo UI",
            font=("Segoe UI", 16, "bold"),
        )
        title.pack(anchor="w", pady=(0, 12))

        desc = ttk.Label(
            outer,
            text=(
                "Enter a text prompt, generate a MIDI file with the student model."
            ),
            wraplength=760,
            justify="left",
        )
        desc.pack(anchor="w", pady=(0, 12))

        prompt_label = ttk.Label(outer, text="Text prompt:")
        prompt_label.pack(anchor="w")

        self.prompt_text = tk.Text(outer, height=7, wrap="word")
        self.prompt_text.pack(fill="x", pady=(6, 12))
        self.prompt_text.insert("1.0", "upbeat electric rock song with highs and lows")

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 12))

        self.generate_button = ttk.Button(
            controls,
            text="Generate Music",
            command=self.on_generate_clicked,
        )
        self.generate_button.pack(side="left")

        ttk.Button(
            controls,
            text="Play MIDI",
            command=self.play_latest_midi,
        ).pack(side="left", padx=(8, 0))

        ttk.Button(
            controls,
            text="Open MIDI Folder",
            command=lambda: self.open_folder(MIDI_OUTPUT_DIR),
        ).pack(side="left", padx=(8, 0))

        status_frame = ttk.LabelFrame(outer, text="Status", padding=12)
        status_frame.pack(fill="x", pady=(0, 12))

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=760
        ).pack(anchor="w")

        outputs = ttk.LabelFrame(outer, text="Latest Output", padding=12)
        outputs.pack(fill="both", expand=True)

        ttk.Label(outputs, text="MIDI:").grid(row=0, column=0, sticky="nw", padx=(0, 8), pady=(0, 6))
        ttk.Label(outputs, textvariable=self.midi_var, wraplength=640, justify="left").grid(
            row=0, column=1, sticky="nw", pady=(0, 6)
        )

        outputs.columnconfigure(1, weight=1)

    def set_status(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    def open_folder(self, folder: str):
        ensure_dir(folder)
        os.startfile(folder)

    def play_latest_midi(self):
        midi_path = self.midi_var.get().strip()
        if not midi_path:
            messagebox.showerror("No MIDI file", "No MIDI file has been generated yet.")
            return
        if not os.path.exists(midi_path):
            messagebox.showerror("Missing MIDI file", f"MIDI file not found:\n{midi_path}")
            return

        try:
            os.startfile(midi_path)
            self.set_status("Opened MIDI file in default player.")
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def on_generate_clicked(self):
        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showerror("Missing prompt", "Please enter a text prompt first.")
            return

        self.generate_button.config(state="disabled")
        self.midi_var.set("")
        self.set_status("Generating MIDI from text prompt...")

        thread = threading.Thread(
            target=self.run_generation_pipeline,
            args=(prompt,),
            daemon=True
        )
        thread.start()

    def run_generation_pipeline(self, prompt: str):
        try:
            t0 = time.time()

            midi_path = generate_midi_with_model(prompt)
            elapsed = time.time() - t0

            self.root.after(0, lambda: self.midi_var.set(midi_path))
            self.root.after(
                0,
                lambda: self.set_status(
                    f"Done. MIDI created successfully in {elapsed:.1f} seconds."
                ),
            )
        except Exception as e:
            err = f"{e}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self.set_status("Generation failed. See error dialog."))
            self.root.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            self.root.after(0, lambda: self.generate_button.config(state="normal"))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    setup_local_cache_dirs()
    ensure_dir(MIDI_OUTPUT_DIR)

    root = tk.Tk()
    app = Text2MidiUI(root)
    root.mainloop()