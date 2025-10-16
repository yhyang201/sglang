import argparse
import os
import sys
from pathlib import Path

import torch

STEP_AUDIO_DIR = Path(__file__).resolve().parent.parent / "python" / "sglang" / "srt" / "tts" / "step_audio2"
if STEP_AUDIO_DIR.exists():
    sys.path.insert(0, str(STEP_AUDIO_DIR))

from sglang.srt.tts.step_audio2.step_audio2_tts import StepAudio2TTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for StepAudio2 TTS")
    parser.add_argument(
        "--model-path",
        default="/data/cache/huggingface/hub/Step-Audio-2-mini/token2wav",
        help="Directory containing Token2wav weights",
    )
    parser.add_argument(
        "--prompt-wav",
        default="assets/prompt.wav",
        help="Reference audio for voice cloning",
    )
    parser.add_argument(
        "--output",
        default="temp/stepaudio_tts_test.wav",
        help="Path to write synthesized audio",
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        help="Enable fp16 inference",
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StepAudio2 TTS")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model path: {model_path}")

    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Missing prompt wav: {prompt_wav}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tokens copied from Step-Audio2 reference script for quick validation.
    tokens = [
        1493, 4299, 4218, 2049, 528, 2752, 4850, 4569, 4575, 6372,
        2127, 4068, 2312, 4993, 4769, 2300, 226, 2175, 2160, 2152,
        6311, 6065, 4859, 5102, 4615, 6534, 6426, 1763, 2249, 2209,
        5938, 1725, 6048, 3816, 6058, 958, 63, 4460, 5914, 2379,
        735, 5319, 4593, 2328, 890, 35, 751, 1483, 1484, 1483,
        2112, 303, 4753, 2301, 5507, 5588, 5261, 5744, 5501, 2341,
        2001, 2252, 2344, 1860, 2031, 414, 4366, 4366, 6059, 5300,
        4814, 5092, 5100, 1923, 3054, 4320, 4296, 2148, 4371, 5831,
        5084, 5027, 4946, 4946, 2678, 575, 575, 521, 518, 638,
        1367, 2804, 3402, 4299,
    ]

    engine = StepAudio2TTS(str(model_path), float16=args.float16)
    audio_bytes = engine.generate(tokens, str(prompt_wav))

    with output_path.open("wb") as f:
        f.write(audio_bytes)

    print(f"Wrote {output_path} ({len(audio_bytes)} bytes)")


if __name__ == "__main__":
    main()
