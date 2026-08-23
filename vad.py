"""
vad.py
Voice Activity Detection methods for whisp-carrier.
Supports: silero_v4_fw, silero_v5_fw, silero_v3, silero_v4, silero_v5,
          pyannote_onnx_v3, auditok, webrtc
"""

from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

# torchaudio 2.8 warns that load() will switch to TorchCodec in 2.9. All three
# loaders below hit it, so it lands in the Amatsukaze log, where an unexplained
# warning reads as a failure. Nothing here can act on it either: the call is
# inside torchaudio and the switch is upstream's to make. Matched by message so
# that a torchaudio warning about something real still gets through.
warnings.filterwarnings(
    "ignore",
    message=r".*load_with_torchcodec.*",
    category=UserWarning,
)

# Segment = (start_sec, end_sec)
Segment = Tuple[float, float]


def _frozen() -> bool:
    """True when running from a PyInstaller build."""
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")


def _ffmpeg_exe() -> str:
    """Path to ffmpeg, preferring the copy bundled beside the exe.

    Every call site here used to pass the bare string "ffmpeg", which only
    resolves when the machine happens to have one on PATH. The exe bundles
    ffmpeg and README states it is not needed separately, so on a clean machine
    the *default* VAD path raised FileNotFoundError for any input that was not
    already a 16 kHz mono WAV -- that is, for the intermediate AAC Amatsukaze
    actually passes.

    It went unnoticed because this development machine has ffmpeg on PATH
    (WinGet) and because the regression fixture, test_speech.wav, is a WAV and
    therefore never reaches the conversion branch.

    audio_filter owns the lookup; imported here rather than duplicated so there
    is one answer to "where is ffmpeg".
    """
    from audio_filter import get_ffmpeg_path
    return get_ffmpeg_path()


def _missing_backend(method: str, package: str, pip_name: str) -> ImportError:
    """Error for an unavailable VAD backend, worded for the current run mode."""
    if _frozen():
        return ImportError(
            f"--vad_method {method} is not available in this exe build "
            f"({package} is not bundled). Use the built-in silero VAD "
            f"(--vad_method silero_v5_fw), or run the script version with "
            f"{package} installed."
        )
    return ImportError(f"{package} not installed. Run: pip install {pip_name}")


def get_speech_segments_silero(
    audio_path: str,
    version: str = "silero_v5_fw",
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
    window_size: int = 1536,
    device: str = "cpu",
    neg_threshold: Optional[float] = None,
) -> List[Segment]:
    """Silero VAD - supports v3, v4, v5, v4_fw, v5_fw, v6, v6_fw variants.

    neg_threshold is the hysteresis silero uses to decide a speech run has
    *ended*: the probability has to fall below it, not merely below `threshold`.
    Left at None silero derives it as max(threshold - 0.15, 0.01), so the
    default configuration runs 0.45/0.30. Lowering it makes the VAD hold on
    through a quiet passage instead of cutting there, which is the one knob
    between "keep the VAD" and "switch it off" (HANDOVER 測定結果 #17).
    """
    # torch is not bundled in the exe: it was 4.3 GB of the payload and the
    # default path does not use it (CTranslate2 for inference, a native library
    # for TEN VAD). The silero backends are the only feature that needs it, and
    # they lost to TEN on all fifteen references, so they are script-version
    # only now. Reported through _missing_backend so the message says that
    # rather than surfacing a bare ImportError.
    try:
        import torch
        import torchaudio
    except ImportError:
        raise _missing_backend(version, "torch/torchaudio", "torch torchaudio")

    # Only forwarded when set, so a silero build whose get_speech_timestamps
    # predates the argument keeps working. None is silero's own default anyway.
    neg_kwargs = {} if neg_threshold is None else {"neg_threshold": neg_threshold}

    # v6系はsilero-vad 6.x パッケージを直接使う
    v6_versions = {"silero_v6", "silero_v6_fw"}

    if version in v6_versions:
        try:
            import tempfile
            from silero_vad import load_silero_vad, get_speech_timestamps as gst
            model = load_silero_vad(onnx=False)
            model = model.to(device)

            # AACなど非WAV形式はtorchaudioが読めない場合があるのでffmpegで変換
            work_path = audio_path
            tmp_wav = None
            if not audio_path.lower().endswith(".wav"):
                import subprocess, tempfile as tf
                tmp_wav = tf.NamedTemporaryFile(suffix=".wav", delete=False).name
                subprocess.run([
                    _ffmpeg_exe(), "-y", "-i", audio_path,
                    "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav
                ], capture_output=True, check=True)
                work_path = tmp_wav

            waveform, sr = torchaudio.load(work_path)
            if tmp_wav:
                os.unlink(tmp_wav)

            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            wav = waveform.squeeze().to(device)

            timestamps = gst(
                wav,
                model,
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=True,
                **neg_kwargs,
            )
            return [(t["start"], t["end"]) for t in timestamps]
        except Exception as e:
            raise RuntimeError(f"silero_v6 requires silero-vad>=6.0: {e}")

    model_map = {
        "silero_v3":    ("snakers4/silero-vad", "silero_vad"),
        "silero_v4":    ("snakers4/silero-vad", "silero_vad"),
        "silero_v5":    ("snakers4/silero-vad", "silero_vad"),
        "silero_v4_fw": ("snakers4/silero-vad", "silero_vad"),
        "silero_v5_fw": ("snakers4/silero-vad", "silero_vad"),
    }

    if version not in model_map:
        raise ValueError(f"Unknown silero version: {version}")

    try:
        from silero_vad import load_silero_vad, get_speech_timestamps as gst
        model = load_silero_vad(onnx=False)
        model = model.to(device)
        utils = None
    except Exception:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        model = model.to(device)

    # AACなど非WAV形式はffmpegで変換
    import subprocess, tempfile as tf2
    work_path = audio_path
    tmp_wav = None
    if not audio_path.lower().endswith(".wav"):
        tmp_wav = tf2.NamedTemporaryFile(suffix=".wav", delete=False).name
        subprocess.run([
            _ffmpeg_exe(), "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav
        ], capture_output=True, check=True)
        work_path = tmp_wav

    waveform, sr = torchaudio.load(work_path)
    if tmp_wav:
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    wav = waveform.squeeze().to(device)

    if utils is not None:
        get_speech_timestamps = utils[0]
        timestamps = get_speech_timestamps(
            wav, model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            window_size_samples=window_size,
            return_seconds=True,
            **neg_kwargs,
        )
    else:
        timestamps = gst(
            wav, model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
            **neg_kwargs,
        )

    return [(t["start"], t["end"]) for t in timestamps]


def _load_wav_16k_mono(audio_path: str):
    """16 kHz mono float32 waveform, decoding through ffmpeg when needed.

    Only used by the TEN backend. The silero paths keep their own copy of this
    so that adding a backend cannot perturb the recorded measurements.

    Reads through soundfile rather than torchaudio. Nothing else on the default
    path needs torch -- inference is CTranslate2 and TEN VAD is a native library
    reached with ctypes -- so torchaudio here was the single import holding
    4.3 GB of torch and bundled CUDA kernels in the exe payload. soundfile is
    already bundled for the writers.

    Resampling is delegated to ffmpeg instead of being done in-process, which
    keeps this function free of a resampler dependency. The two paths that the
    recorded numbers come from are unaffected: production input is Amatsukaze's
    AAC, which was already converted by ffmpeg here, and the reference corpus is
    16 kHz mono WAV from eval/prep.py, which is read directly and byte for byte
    as before (soundfile and torchaudio both scale PCM_16 by 1/32768). Only an
    odd-rate WAV changes resampler, and it now uses the same one prep.py used to
    build the corpus in the first place.
    """
    import numpy as np
    import soundfile as sf

    # Convert unless it is already exactly what TenVad wants. Probing beats
    # branching on the extension: a .wav at 48 kHz stereo used to be read and
    # then silently resampled, and a non-.wav container was converted even when
    # it held 16 kHz mono.
    needs_convert = True
    try:
        info = sf.info(audio_path)
        needs_convert = not (info.samplerate == 16000 and info.channels == 1)
    except Exception:
        # Not something soundfile can open (AAC, MKV, ...). ffmpeg handles it.
        needs_convert = True

    work_path = audio_path
    tmp_wav = None
    if needs_convert:
        import subprocess
        import tempfile
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        subprocess.run([
            _ffmpeg_exe(), "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav
        ], capture_output=True, check=True)
        work_path = tmp_wav

    try:
        audio, _ = sf.read(work_path, dtype="float32", always_2d=False)
    finally:
        if tmp_wav:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def segments_from_probabilities(
    probabilities,
    frame_samples: int,
    total_samples: int,
    threshold: float = 0.45,
    neg_threshold: Optional[float] = None,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
    sampling_rate: int = 16000,
) -> List[Segment]:
    """Turn a frame-level speech probability track into segments.

    This is silero's own aggregation, reimplemented so that a different model
    can be compared against it without the comparison measuring the difference
    in post-processing instead of the difference in the models. Same trigger,
    same hysteresis, same minimum-silence confirmation, same minimum-speech
    filter, and the same padding rule that splits a gap narrower than twice the
    padding rather than letting the two segments overlap.

    Kept separate from get_speech_segments_silero(), which still calls the
    silero package, so the recorded numbers cannot move.
    """
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    min_speech_samples = sampling_rate * min_speech_ms // 1000
    min_silence_samples = sampling_rate * min_silence_ms // 1000
    pad_samples = sampling_rate * speech_pad_ms // 1000

    speeches: List[dict] = []
    current: dict = {}
    triggered = False
    temp_end = 0

    for index, probability in enumerate(probabilities):
        position = frame_samples * index
        if probability >= threshold and temp_end:
            temp_end = 0
        if probability >= threshold and not triggered:
            triggered = True
            current = {"start": position}
            continue
        if probability < neg_threshold and triggered:
            if not temp_end:
                temp_end = position
            if position - temp_end < min_silence_samples:
                continue
            current["end"] = temp_end
            if current["end"] - current["start"] > min_speech_samples:
                speeches.append(current)
            current = {}
            temp_end = 0
            triggered = False

    if triggered and current:
        current["end"] = total_samples
        if current["end"] - current["start"] > min_speech_samples:
            speeches.append(current)

    # Padding, as silero applies it: halve a gap that is narrower than twice the
    # padding so the result stays non-overlapping.
    for index, speech in enumerate(speeches):
        if index == 0:
            speech["start"] = max(0, speech["start"] - pad_samples)
        if index != len(speeches) - 1:
            gap = speeches[index + 1]["start"] - speech["end"]
            if gap < 2 * pad_samples:
                speech["end"] += gap // 2
                speeches[index + 1]["start"] = max(0, speeches[index + 1]["start"] - gap // 2)
            else:
                speech["end"] = min(total_samples, speech["end"] + pad_samples)
                speeches[index + 1]["start"] = max(0, speeches[index + 1]["start"] - pad_samples)
        else:
            speech["end"] = min(total_samples, speech["end"] + pad_samples)

    return [(s["start"] / sampling_rate, s["end"] / sampling_rate) for s in speeches]


def get_speech_segments_ten(
    audio_path: str,
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
    neg_threshold: Optional[float] = None,
    return_probabilities: bool = False,
):
    """TEN VAD (TEN Framework, Apache-2.0).

    A different model family from silero, which is the point: HANDOVER 測定結果
    #17 established that the speech this project misses on 死亡遊戯 sits below
    silero's probability floor, so no silero-side parameter can reach it. TEN VAD
    reports its own frame probability, and the aggregation above is silero's, so
    the only thing that differs is the model.

    The package ships a prebuilt native library and exposes a frame-level API
    (16 ms hops at 16 kHz), so segment building has to happen here.
    """
    try:
        from ten_vad import TenVad
    except ImportError:
        raise _missing_backend("ten", "ten-vad", "ten-vad")

    import numpy as np

    audio = _load_wav_16k_mono(audio_path)
    total_samples = int(audio.shape[0])
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

    hop = 256
    # The handler carries state across frames, so the threshold given here only
    # affects the flag we ignore; segmentation uses the probability track.
    detector = TenVad(hop_size=hop, threshold=threshold)
    frames = total_samples // hop
    probabilities = np.empty(frames, dtype=np.float32)
    for index in range(frames):
        chunk = pcm[index * hop:(index + 1) * hop]
        probabilities[index], _ = detector.process(chunk)

    segments = segments_from_probabilities(
        probabilities,
        frame_samples=hop,
        total_samples=total_samples,
        threshold=threshold,
        neg_threshold=neg_threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    if return_probabilities:
        return segments, probabilities
    return segments


def get_speech_segments_precomputed(
    audio_path: str,
    segments_path: str,
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
    neg_threshold: Optional[float] = None,
    frame_samples: int = 256,
) -> List[Segment]:
    """Speech regions produced elsewhere, put through the same aggregation.

    Some segmenters cannot be installed alongside this project: inaSpeechSegmenter
    wants tensorflow[and-cuda] plus onnxruntime-gpu, and funasr-onnx pins
    numpy<=1.26.4, either of which would disturb the environment every recorded
    number was measured in. So they run in their own virtualenv, write their
    regions to JSON, and this reads them back.

    The regions are rasterised onto the same 16 ms frame grid the TEN backend
    uses and pushed through segments_from_probabilities(), so min_silence,
    speech_pad and min_speech apply identically no matter which model decided
    which frames are speech. Without that, a comparison would be measuring each
    project's smoothing choices rather than its detection.

    JSON shape: {"<wav stem>": [[start_seconds, end_seconds], ...], ...}
    """
    import json

    import numpy as np

    path = Path(segments_path)
    if not path.is_file():
        raise FileNotFoundError(f"--vad_segments_json not found: {path}")
    table = json.loads(path.read_text(encoding="utf-8"))

    stem = Path(audio_path).stem
    if stem not in table:
        raise KeyError(
            f"{path.name} has no entry for {stem!r}; "
            f"it holds {len(table)} entr(ies)"
        )
    regions = table[stem]

    audio = _load_wav_16k_mono(audio_path)
    total_samples = int(audio.shape[0])
    frames = total_samples // frame_samples
    # 1.0 inside a region, 0.0 outside. Any threshold below 1 and above 0 gives
    # the same trigger on this track, so the caller's --vad_threshold does not
    # silently change what the external model decided.
    track = np.zeros(frames, dtype=np.float32)
    for start, end in regions:
        first = max(0, int(float(start) * 16000) // frame_samples)
        last = min(frames, int(float(end) * 16000) // frame_samples + 1)
        if last > first:
            track[first:last] = 1.0

    return segments_from_probabilities(
        track,
        frame_samples=frame_samples,
        total_samples=total_samples,
        threshold=0.5,
        neg_threshold=0.5,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )


def get_speech_segments_pyannote(
    audio_path: str,
    device: str = "cuda",
    use_onnx: bool = True,
) -> List[Segment]:
    """Pyannote VAD v3 (onnx or torch)."""
    # pyannote.audio is deliberately excluded from exe builds: it pulls in
    # pytorch-lightning and speechbrain for a backend that tested worse than
    # the built-in silero VAD.
    try:
        from pyannote.audio import Pipeline as _P
    except ImportError:
        raise _missing_backend("pyannote_v3", "pyannote.audio", "pyannote.audio")

    if use_onnx:
        # Use pyannote's built-in ONNX pipeline
        pipeline = _P.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=False,
        )
    else:
        pipeline = _P.from_pretrained("pyannote/voice-activity-detection")

    import torch
    pipeline = pipeline.to(torch.device(device))
    output = pipeline(audio_path)
    return [(seg.start, seg.end) for seg, _, _ in output.itertracks(yield_label=True)]


def get_speech_segments_auditok(
    audio_path: str,
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
) -> List[Segment]:
    """auditok - audio activity detection (energy-based, no model)."""
    try:
        import auditok
    except ImportError:
        raise _missing_backend("auditok", "auditok", "auditok")

    energy_threshold = 50 + threshold * 10  # rough mapping
    regions = auditok.split(
        audio_path,
        min_dur=min_speech_ms / 1000,
        max_silence=min_silence_ms / 1000,
        energy_threshold=energy_threshold,
    )
    pad = speech_pad_ms / 1000
    segments = []
    for r in regions:
        start = max(0.0, r.meta.start - pad)
        end = r.meta.end + pad
        segments.append((start, end))
    return segments


def get_speech_segments_webrtc(
    audio_path: str,
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
) -> List[Segment]:
    """WebRTC VAD."""
    try:
        import webrtcvad
        import wave
        import struct
    except ImportError:
        raise _missing_backend("webrtc", "webrtcvad", "webrtcvad-wheels")

    aggressiveness = min(3, int(threshold * 4))
    vad = webrtcvad.Vad(aggressiveness)

    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())

    # WebRTC works on 10/20/30ms frames at 8/16/32/48kHz
    frame_ms = 20
    frame_samples = int(sample_rate * frame_ms / 1000)
    frame_bytes = frame_samples * 2  # 16-bit

    frames = [
        pcm_data[i: i + frame_bytes]
        for i in range(0, len(pcm_data) - frame_bytes, frame_bytes)
    ]

    speech_flags = [
        vad.is_speech(f, sample_rate) if len(f) == frame_bytes else False
        for f in frames
    ]

    pad_frames = int(speech_pad_ms / frame_ms)
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    segments = []
    in_speech = False
    start_frame = 0
    silence_count = 0

    for i, is_speech in enumerate(speech_flags):
        if is_speech:
            if not in_speech:
                start_frame = max(0, i - pad_frames)
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    end_frame = min(len(speech_flags), i + pad_frames)
                    duration = (end_frame - start_frame)
                    if duration >= min_speech_frames:
                        segments.append((
                            start_frame * frame_ms / 1000,
                            end_frame * frame_ms / 1000,
                        ))
                    in_speech = False
                    silence_count = 0

    if in_speech:
        end_frame = min(len(speech_flags), len(speech_flags) + pad_frames)
        segments.append((start_frame * frame_ms / 1000, end_frame * frame_ms / 1000))

    return segments


def get_speech_segments(
    audio_path: str,
    method: str = "silero_v5_fw",
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    max_speech_s: float = None,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
    window_size: int = 1536,
    vad_device: str = "cpu",
    neg_threshold: Optional[float] = None,
    segments_json: Optional[str] = None,
) -> List[Segment]:
    """Unified VAD dispatcher.

    neg_threshold only reaches the silero and TEN backends. auditok maps
    `threshold` to an energy level and webrtc to an aggressiveness step, so
    neither has a separate end-of-speech test to set.
    """

    silero_methods = {"silero_v3", "silero_v4", "silero_v5", "silero_v4_fw", "silero_v5_fw", "silero_v6", "silero_v6_fw"}

    if method == "precomputed":
        if not segments_json:
            raise ValueError(
                "--vad_method precomputed needs --vad_segments_json PATH"
            )
        segments = get_speech_segments_precomputed(
            audio_path,
            segments_path=segments_json,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
    elif method == "ten":
        segments = get_speech_segments_ten(
            audio_path,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            neg_threshold=neg_threshold,
        )
    elif method in silero_methods:
        segments = get_speech_segments_silero(
            audio_path,
            version=method,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            window_size=window_size,
            device=vad_device,
            neg_threshold=neg_threshold,
        )
    elif method in {"pyannote_v3", "pyannote_onnx_v3"}:
        use_onnx = (method == "pyannote_onnx_v3")
        segments = get_speech_segments_pyannote(
            audio_path,
            device=vad_device,
            use_onnx=use_onnx,
        )
    elif method == "auditok":
        segments = get_speech_segments_auditok(
            audio_path,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
    elif method == "webrtc":
        segments = get_speech_segments_webrtc(
            audio_path,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
    else:
        raise ValueError(f"Unknown VAD method: {method}")

    # Apply max_speech_s splitting
    if max_speech_s:
        split = []
        for start, end in segments:
            while end - start > max_speech_s:
                split.append((start, start + max_speech_s))
                start += max_speech_s
            split.append((start, end))
        segments = split

    return segments
