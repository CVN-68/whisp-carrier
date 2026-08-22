"""
whisp_vad_patch.py
Redirects parts of faster-whisper's built-in VAD without patching the package.

Two separate things live here.

1. install_model() swaps the ONNX weights that the built-in VAD loads.
   faster-whisper 1.2.1 hardcodes assets/silero_vad_v6.onnx inside
   get_vad_model(), so --vad_method silero_v4_fw and silero_v5_fw both end up
   running v6. This makes those names mean something again, and it is also the
   only way to try a different silero release without editing site-packages.

2. external_segments() feeds an external VAD's result through the built-in code
   path instead of through clip_timestamps. This is the part that actually
   changes transcription quality, and it needs explaining.

Why the code path matters more than the VAD model
-------------------------------------------------
The two ways of restricting Whisper to speech regions are not equivalent.

Built-in path (transcribe.py: `if vad_filter and clip_timestamps == "0"`):
    get_speech_timestamps() -> collect_chunks() physically deletes the silence
    from the waveform -> np.concatenate -> every 30s window is packed with
    speech -> restore_speech_timestamps() maps the times back. Whisper barely
    sees silence at all.

clip_timestamps path (what this project used for every external VAD):
    the waveform is untouched. generate_segments() builds seek_clips and takes
        segment_size = min(nb_max_frames, content_frames - seek,
                           seek_clip_end - seek)
    followed by pad_or_trim(). A one second clip therefore reaches the encoder
    as one second of audio plus twenty-nine seconds of zeros, which is the
    input shape Whisper hallucinates on the most. On top of that all_tokens and
    prompt_reset_since carry across clip boundaries, so a loop that starts in
    one clip is fed to the next.

With the default speech_pad_ms=900 and min_speech_duration_ms=250 an external
VAD produces a lot of short clips, so this is not a corner case. Routing those
same segments through collect_chunks puts every backend in vad.py on the same
footing as the built-in one.

How the patching works
----------------------
There is an asymmetry worth knowing about.

get_speech_timestamps() looks up `get_vad_model()` in its own module namespace
at call time, so replacing faster_whisper.vad.get_vad_model is enough and
transcribe.py never has to be touched. It is lru_cache'd, so order matters.

get_speech_timestamps itself is bound by transcribe.py at import time
(`from faster_whisper.vad import get_speech_timestamps, ...`), so replacing
faster_whisper.vad.get_speech_timestamps has no effect at all. The name that
has to be replaced is faster_whisper.transcribe.get_speech_timestamps. Both
WhisperModel.transcribe and BatchedInferencePipeline.transcribe read that one
attribute, so a single patch covers both engines.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Input names of the ONNX graph that SileroVADModel.__call__ is written against.
# v5 and v6 share this signature. v4 and earlier take an extra `sr` input and
# use differently shaped h/c states, so swapping the file alone is not enough
# for them: __call__ would have to be replaced as well.
EXPECTED_ONNX_INPUTS = frozenset({"input", "h", "c"})

# Below this, volumedetect-style measurement would call it silence anyway. Used
# only to describe a degenerate result, never to change one.
SAMPLE_RATE = 16000


class VadPatchError(Exception):
    """Raised when a VAD substitution cannot be made safely."""


# ─────────────────────────────────────────────
# 1. Swapping the built-in ONNX model
# ─────────────────────────────────────────────

def resolve_onnx(path_like: str, base: Optional[Path] = None) -> Path:
    """Resolve a --vad_onnx value.

    A relative path is taken against the folder holding the script or the exe,
    matching how the config file is discovered, so the same value works in both
    run modes.
    """
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        if base is None:
            import whisp_config

            base = whisp_config.base_dir()
        path = Path(base) / path
    path = path.resolve()

    if not path.is_file():
        raise VadPatchError(f"VAD model file not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise VadPatchError(
            f"--vad_onnx expects an ONNX file, got '{path.name}'. The built-in "
            "VAD is run through onnxruntime, not torch."
        )
    return path


def install_model(path_like: str, base: Optional[Path] = None) -> List[str]:
    """Make the built-in VAD load a different ONNX file.

    Returns log lines. Raises VadPatchError when the graph does not match what
    SileroVADModel.__call__ expects, rather than letting onnxruntime fail later
    with a message about missing inputs.
    """
    path = resolve_onnx(path_like, base)

    try:
        import faster_whisper.vad as fw_vad
    except ImportError as e:  # pragma: no cover - faster-whisper is required
        raise VadPatchError(f"cannot import faster_whisper.vad: {e}") from e

    try:
        model = fw_vad.SileroVADModel(str(path))
    except Exception as e:
        raise VadPatchError(f"cannot load {path.name} as a silero VAD model: {e}") from e

    _check_graph(model, path)

    # Drop anything the real function already cached, then take over the name.
    # The replacement has no cache_clear of its own, which is why this has to
    # happen against the original.
    clear = getattr(fw_vad.get_vad_model, "cache_clear", None)
    if callable(clear):
        clear()
    fw_vad.get_vad_model = lambda: model

    return [
        f"[VAD] built-in VAD model replaced: {path}",
        "[VAD]   the segmentation logic around it is still the one tuned for "
        "silero v6 (512 sample window, neg_threshold hysteresis)",
    ]


def _check_graph(model: Any, path: Path) -> None:
    session = getattr(model, "session", None)
    if session is None:  # pragma: no cover - shape of SileroVADModel changed
        return

    try:
        names = {i.name for i in session.get_inputs()}
    except Exception:
        return

    if names == set(EXPECTED_ONNX_INPUTS):
        return

    if "sr" in names:
        raise VadPatchError(
            f"{path.name} looks like a silero v4 or older graph (inputs: "
            f"{sorted(names)}). faster-whisper's SileroVADModel.__call__ is "
            "written for the v5/v6 signature (input, h, c with 1x1x128 states) "
            "and does not pass 'sr', so this file cannot be used by swapping it "
            "in. It needs a replacement model class, which is not implemented."
        )

    raise VadPatchError(
        f"{path.name} does not have the input signature faster-whisper expects. "
        f"found {sorted(names)}, expected {sorted(EXPECTED_ONNX_INPUTS)}."
    )


def describe_builtin_model() -> str:
    """One line naming the ONNX file the built-in VAD will use as things stand."""
    try:
        import faster_whisper.vad as fw_vad
        from faster_whisper.utils import get_assets_path
    except ImportError as e:
        return f"[VAD] cannot inspect the built-in VAD: {e}"

    if getattr(fw_vad.get_vad_model, "cache_clear", None) is None:
        return "[VAD] built-in VAD model: replaced by --vad_onnx"
    default = Path(get_assets_path()) / "silero_vad_v6.onnx"
    return f"[VAD] built-in VAD model: {default}"


# ─────────────────────────────────────────────
# 2. Routing an external VAD through the built-in path
# ─────────────────────────────────────────────

def normalize_segments(
    segments_sec: Iterable[Tuple[float, float]],
    n_samples: int,
    sampling_rate: int = SAMPLE_RATE,
) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
    """Turn (start, end) second pairs into the sample dicts collect_chunks wants.

    Sorting and merging are not optional here. SpeechTimestampsMap accumulates
    `chunk["start"] - previous_end` as the silence seen so far, so a pair of
    overlapping chunks would push that total backwards and shift every timestamp
    after it. External backends do overlap: speech_pad_ms is applied per region
    in vad.py, so neighbouring regions closer together than twice the padding
    come back overlapping.
    """
    stats = {"received": 0, "dropped": 0, "clamped": 0, "merged": 0}

    raw: List[Tuple[int, int]] = []
    for pair in segments_sec:
        stats["received"] += 1
        try:
            start_sec, end_sec = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            stats["dropped"] += 1
            continue

        start = int(round(start_sec * sampling_rate))
        end = int(round(end_sec * sampling_rate))

        clipped = False
        if start < 0:
            start, clipped = 0, True
        if end > n_samples:
            end, clipped = n_samples, True
        if clipped:
            stats["clamped"] += 1

        if end <= start:
            stats["dropped"] += 1
            continue
        raw.append((start, end))

    raw.sort()
    merged: List[List[int]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            stats["merged"] += 1
        else:
            merged.append([start, end])

    return [{"start": a, "end": b} for a, b in merged], stats


@contextlib.contextmanager
def external_segments(
    segments_sec: Sequence[Tuple[float, float]],
    sampling_rate: int = SAMPLE_RATE,
):
    """Make the built-in VAD hook return a fixed set of segments.

    Used with vad_filter=True and no clip_timestamps, so faster-whisper trims
    the waveform and restores the timestamps exactly as it does for its own VAD.

    Yields a dict that is filled in when the hook fires. `calls` staying at 0
    means faster-whisper did not take the built-in VAD branch, so the caller
    should say so instead of reporting a result that never happened.
    """
    try:
        import faster_whisper.transcribe as fw_tr
    except ImportError as e:  # pragma: no cover - faster-whisper is required
        raise VadPatchError(f"cannot import faster_whisper.transcribe: {e}") from e

    info: Dict[str, Any] = {
        "calls": 0,
        "chunks": 0,
        "speech_seconds": 0.0,
        "received": len(segments_sec),
        "dropped": 0,
        "clamped": 0,
        "merged": 0,
    }

    original = fw_tr.get_speech_timestamps

    def replacement(audio, vad_options=None, **kwargs) -> List[Dict[str, int]]:
        # Signature mirrors faster_whisper.vad.get_speech_timestamps. vad_options
        # is accepted and ignored: the thresholds were already applied by the
        # external backend that produced these segments.
        chunks, stats = normalize_segments(segments_sec, len(audio), sampling_rate)
        info["calls"] += 1
        info["chunks"] = len(chunks)
        info["speech_seconds"] = sum(c["end"] - c["start"] for c in chunks) / sampling_rate
        info.update({k: stats[k] for k in ("dropped", "clamped", "merged")})
        return chunks

    fw_tr.get_speech_timestamps = replacement
    try:
        yield info
    finally:
        fw_tr.get_speech_timestamps = original


def describe_external(method: str, info: Optional[Dict[str, Any]]) -> List[str]:
    """Log lines for what external_segments() ended up doing."""
    if info is None:
        return []

    if not info["calls"]:
        return [
            f"[VAD] WARNING: {method} segments were prepared but faster-whisper "
            "never called the VAD hook, so they were not used. Check that "
            "vad_filter is on and clip_timestamps is unset.",
        ]

    lines = [
        f"[VAD] {method}: {info['chunks']} chunks | "
        f"{info['speech_seconds']:.1f}s of speech routed through collect_chunks"
    ]
    detail = []
    if info["merged"]:
        detail.append(f"{info['merged']} overlapping merged")
    if info["clamped"]:
        detail.append(f"{info['clamped']} clamped to the audio length")
    if info["dropped"]:
        detail.append(f"{info['dropped']} empty dropped")
    if detail:
        lines.append(f"[VAD]   {', '.join(detail)}")
    return lines
