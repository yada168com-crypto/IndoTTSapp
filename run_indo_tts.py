#!/usr/bin/env python
# coding=utf-8
"""
Standalone runner for Indonesian text-to-speech, using the community
Indonesian fine-tune of Qwen3-TTS:
https://huggingface.co/alkhrzmy/qwen3-tts-0.6b-indonesian

Quick edit-and-run: put the text you want spoken into text.txt (next to this
script), then just run this file (or double-click run_indo_tts.bat). Other
settings live in the CONFIG block below. Everything can also be overridden
from the command line, e.g.:

    python run_indo_tts.py --text "Halo, apa kabar?"
    python run_indo_tts.py --text_file naskah.txt --period_pause 0.4

Note: this checkpoint only has a single fixed speaker ("indonesian_speaker")
- there's no speaker choice or voice cloning here.

Requires the "qwen3-tts" conda environment (see run_indo_tts.bat). This is
the same Qwen3-TTS package used by the Chinese QwenTTS tool, just pointed at
a different (community, not official) model checkpoint.
"""
import argparse
import os
import re
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_FILE = os.path.join(SCRIPT_DIR, "text.txt")

# ============================== CONFIG ==================================
# Edit these for a quick manual run; all can be overridden via CLI flags.

MODEL_PATH = "alkhrzmy/qwen3-tts-0.6b-indonesian"
SPEAKER = "indonesian_speaker"   # this checkpoint only has this one speaker
LANGUAGE = "Auto"

DEFAULT_TEXT = "Halo, ini adalah contoh teks dalam bahasa Indonesia."  # used only if text.txt is missing/empty

# Seconds of silence to insert after every sentence-ending period "." (0 disables this).
PERIOD_PAUSE_SEC = 0.0

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "indo_tts_output")
# ==========================================================================


def next_output_number(output_dir):
    max_n = 0
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            m = re.fullmatch(r"(\d+)\.wav", name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def load_text_file(path, fallback):
    if not os.path.isfile(path):
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content or fallback


def split_by_period(text):
    """Split text into sentences on '.', keeping it attached to each sentence."""
    parts = re.split(r"(?<=\.)\s*", text)
    return [p.strip() for p in parts if p.strip()]


def insert_pauses(wavs, sr, pause_sec):
    """Concatenate a list of wav arrays, inserting `pause_sec` of silence between each."""
    if len(wavs) == 1:
        return wavs[0]
    silence = np.zeros(int(sr * pause_sec), dtype=wavs[0].dtype)
    merged = [wavs[0]]
    for w in wavs[1:]:
        merged.append(silence)
        merged.append(w)
    return np.concatenate(merged)


def pick_device_and_dtype(requested_device):
    import torch
    if requested_device:
        device = requested_device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] '{device}' requested but CUDA is not available in this environment "
              f"(torch={torch.__version__}). Falling back to CPU — generation will be slow.")
        device = "cpu"

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return device, dtype


def pick_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def build_parser():
    p = argparse.ArgumentParser(description="Run the Indonesian Qwen3-TTS fine-tune")
    p.add_argument("--model_path", default=MODEL_PATH, help="Override the HF model id / local path")
    p.add_argument("--text", default=None, help="Text to synthesize (overrides --text_file)")
    p.add_argument("--text_file", default=TEXT_FILE, help="Path to a .txt file containing the text to synthesize")
    p.add_argument("--output", default=None, help="Output .wav path")
    p.add_argument("--device", default=None, help='e.g. "cuda:0" or "cpu" (default: auto-detect)')
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--period_pause", type=float, default=None,
                   help="Seconds of silence to insert after every '.' (default: PERIOD_PAUSE_SEC in CONFIG, 0 disables)")
    p.add_argument("--list_speakers", action="store_true", help="Load the model, print supported speakers/languages, exit")
    return p


def main():
    args = build_parser().parse_args()

    text = args.text if args.text is not None else load_text_file(args.text_file, DEFAULT_TEXT)
    print(f"[INFO] text ({len(text)} chars): {text[:80]}{'...' if len(text) > 80 else ''}")

    period_pause = args.period_pause if args.period_pause is not None else PERIOD_PAUSE_SEC
    sentences = split_by_period(text) if period_pause > 0 else [text]
    gen_text = sentences if len(sentences) > 1 else text
    if len(sentences) > 1:
        print(f"[INFO] split into {len(sentences)} sentence(s) on '.', "
              f"inserting {period_pause:.2f}s of silence between each")

    device, dtype = pick_device_and_dtype(args.device)
    attn_implementation = pick_attn_implementation()

    print(f"[INFO] model={args.model_path} device={device} dtype={dtype} attn={attn_implementation}")

    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    if args.list_speakers:
        print("Supported speakers:", tts.get_supported_speakers())
        print("Supported languages:", tts.get_supported_languages())
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(
        text=gen_text,
        language=LANGUAGE,
        speaker=SPEAKER,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0
    print(f"[INFO] Generated {len(wavs)} wav(s) in {elapsed:.2f}s")

    if len(sentences) > 1:
        wavs = [insert_pauses(wavs, sr, period_pause)]

    out_path = args.output
    if out_path and len(wavs) == 1:
        sf.write(out_path, wavs[0], sr)
        print(f"[INFO] Saved: {out_path}")
    else:
        next_n = next_output_number(OUTPUT_DIR)
        for i, w in enumerate(wavs):
            path = os.path.join(OUTPUT_DIR, f"{next_n + i}.wav")
            sf.write(path, w, sr)
            print(f"[INFO] Saved: {path}")


if __name__ == "__main__":
    main()
