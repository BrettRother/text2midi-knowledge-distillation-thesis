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
# MODEL SELECTIONS
# Teacher setup kept exactly as-is.
# Output folder stays the same for all models.
# ============================================================

MODEL_OPTIONS = {
    "Teacher (L18 H8 Epoch 140)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\Script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\Script\midigen_text2midi.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_teacher_WorkingOriginalL18H8epoch140\model\epoch_140\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
    },

    "Student 384 (L18 H8 Epoch 140)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch140\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch140\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch140\model\epoch_140\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch140\script\config.yaml",
    },

    "Student 384-16 (L18 H16 Epoch 198)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_18layers_16heads_384epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_18layers_16heads_384epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_18layers_16heads_384epoch200\model\epoch_198\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_18layers_16heads_384epoch200\script\config.yaml",
    },

    "Student 384 (L18 H8 Epoch 200)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch200\model\epoch_200\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original384L18H8epoch200\script\config.yaml",
    },

    "Student 384 (L9 H8 Epoch 198)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_9ayers_8heads_384epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_9ayers_8heads_384epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_9ayers_8heads_384epoch200\model\epoch_198\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_9ayers_8heads_384epoch200\script\config.yaml",
    },

    "Student 384 (L4 H8 Epoch 198)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_4layers_8heads_384epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_4layers_8heads_384epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_4layers_8heads_384epoch200\model\epoch_198\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_4layers_8heads_384epoch200\script\config.yaml",
    },

    "Student 384 (L9 H4 Epoch 198)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L9H4epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L9H4epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L9H4epoch200\model\epoch_198\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L9H4epoch200\script\config.yaml",
    },

    "Student 384 (L4 H2 Epoch 197)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L4H2epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L4H2epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L4H2epoch200\model\epoch_197\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_384L4H2epoch200\script\config.yaml",
    },

    "Student 192 (L18 H8 Epoch 140)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original192L18H8epoch140\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original192L18H8epoch140\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original192L18H8epoch140\model\epoch_140\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original192L18H8epoch140\script\config.yaml",
    },

    "Student 192 (L9 H4 Epoch 200)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L9H4epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L9H4epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L9H4epoch200\model\epoch_200\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L9H4epoch200\script\config.yaml",
    },

    "Student 192 (L4 H2 Epoch 200)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L4H2epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L4H2epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L4H2epoch200\model\epoch_200\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_192L4H2epoch200\script\config.yaml",
    },

    "Student 128 (L18 H8 Epoch 140)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original128L18H8epoch140\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original128L18H8epoch140\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original128L18H8epoch140\model\epoch_140\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_original128L18H8epoch140\script\config.yaml",
    },

    "Student 128 (L9 H4 Epoch 200)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L9H4epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L9H4epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L9H4epoch200\model\epoch_200\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L9H4epoch200\script\config.yaml",
    },

    "Student 128 (L4 H2 Epoch 200)": {
        "script_dir": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L4H2epoch200\script",
        "midigen_script_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L4H2epoch200\script\midigen_text2midi_student.py",
        "model_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L4H2epoch200\model\epoch_200\pytorch_model.bin",
        "tokenizer_pkl": TOKENIZER_PKL,
        "config_path": r"C:\Users\brett\Downloads\Text2midi_Thesis_Project\text2midi_student_128L4H2epoch200\script\config.yaml",
    },
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_selected_model_config(model_name: str) -> dict:
    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model selection: {model_name}")
    return MODEL_OPTIONS[model_name]


def setup_local_cache_dirs(script_dir: str) -> None:
    """
    Force Hugging Face / transformers / torch caches to local folders
    so the code does not try to use a missing drive like E:\.
    """
    local_cache_root = os.path.join(script_dir, "_local_cache")
    hf_home_dir = os.path.join(local_cache_root, "hf_home")
    hf_hub_cache_dir = os.path.join(local_cache_root, "hub")
    transformers_cache_dir = os.path.join(local_cache_root, "transformers")
    torch_home_dir = os.path.join(local_cache_root, "torch")

    ensure_dir(local_cache_root)
    ensure_dir(hf_home_dir)
    ensure_dir(hf_hub_cache_dir)
    ensure_dir(transformers_cache_dir)
    ensure_dir(torch_home_dir)

    os.environ["HF_HOME"] = hf_home_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache_dir
    os.environ["HF_HUB_CACHE"] = hf_hub_cache_dir
    os.environ["TORCH_HOME"] = torch_home_dir
    os.environ["XDG_CACHE_HOME"] = local_cache_root


def import_midigen_module(script_dir: str, script_path: str):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"midigen_text2midi.py not found: {script_path}")

    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    setup_local_cache_dirs(script_dir)

    spec = importlib.util.spec_from_file_location("midigen_text2midi_local", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["midigen_text2midi_local"] = module
    spec.loader.exec_module(module)
    return module


def generate_midi_with_model(prompt: str, model_name: str) -> str:
    """
    Uses the selected model generation code directly and returns the created MIDI path.
    """
    if not prompt.strip():
        raise ValueError("Prompt is empty.")

    cfg = get_selected_model_config(model_name)

    script_dir = cfg["script_dir"]
    midigen_script_path = cfg["midigen_script_path"]
    model_path = cfg["model_path"]
    tokenizer_pkl = cfg["tokenizer_pkl"]
    config_path = cfg.get("config_path", None)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(tokenizer_pkl):
        raise FileNotFoundError(f"Tokenizer pickle not found: {tokenizer_pkl}")

    ensure_dir(MIDI_OUTPUT_DIR)

    before = set(os.listdir(MIDI_OUTPUT_DIR))
    module = import_midigen_module(script_dir, midigen_script_path)

    if not hasattr(module, "generate_single"):
        raise AttributeError(f"generate_single() was not found in: {midigen_script_path}")

    # Student script requires configs
    if "student" in os.path.basename(midigen_script_path).lower():
        if not config_path:
            raise ValueError(f"No config_path provided for student model: {model_name}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if not hasattr(module, "load_configs"):
            raise AttributeError(f"load_configs() was not found in: {midigen_script_path}")

        configs = module.load_configs(config_path)

        module.generate_single(
            caption=prompt,
            model_path=model_path,
            tokenizer_pkl=tokenizer_pkl,
            configs=configs,
            max_len=MAX_LEN,
            temperature=TEMPERATURE,
            out_dir=MIDI_OUTPUT_DIR,
        )
    else:
        # Teacher script stays exactly how it already worked
        module.generate_single(
            caption=prompt,
            model_path=model_path,
            tokenizer_pkl=tokenizer_pkl,
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
        self.model_var = tk.StringVar(value="Teacher (L18 H8 Epoch 140)")

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
                "Enter a text prompt, choose a model, and generate a MIDI file."
            ),
            wraplength=760,
            justify="left",
        )
        desc.pack(anchor="w", pady=(0, 12))

        model_label = ttk.Label(outer, text="Select model:")
        model_label.pack(anchor="w")

        self.model_dropdown = ttk.Combobox(
            outer,
            textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
            width=50,
        )
        self.model_dropdown.pack(fill="x", pady=(6, 12))
        self.model_dropdown.current(0)

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
        selected_model = self.model_var.get().strip()

        if not prompt:
            messagebox.showerror("Missing prompt", "Please enter a text prompt first.")
            return

        if selected_model not in MODEL_OPTIONS:
            messagebox.showerror("Missing model", "Please select a valid model first.")
            return

        self.generate_button.config(state="disabled")
        self.midi_var.set("")
        self.set_status(f"Generating MIDI from text prompt using {selected_model}...")

        thread = threading.Thread(
            target=self.run_generation_pipeline,
            args=(prompt, selected_model),
            daemon=True
        )
        thread.start()

    def run_generation_pipeline(self, prompt: str, selected_model: str):
        try:
            t0 = time.time()

            midi_path = generate_midi_with_model(prompt, selected_model)
            elapsed = time.time() - t0

            self.root.after(0, lambda: self.midi_var.set(midi_path))
            self.root.after(
                0,
                lambda: self.set_status(
                    f"Done. {selected_model} created MIDI successfully in {elapsed:.1f} seconds."
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
    setup_local_cache_dirs(SCRIPT_DIR)
    ensure_dir(MIDI_OUTPUT_DIR)

    root = tk.Tk()
    app = Text2MidiUI(root)
    root.mainloop()