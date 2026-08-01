"""
vad.py
Voice Activity Detection methods for whisp-carier.
Supports: silero_v4_fw, silero_v5_fw, silero_v3, silero_v4, silero_v5,
          pyannote_onnx_v3, auditok, webrtc
"""

from __future__ import annotations
import os
from typing import List, Tuple

# Segment = (start_sec, end_sec)
Segment = Tuple[float, float]


def get_speech_segments_silero(
    audio_path: str,
    version: str = "silero_v5_fw",
    threshold: float = 0.45,
    min_speech_ms: int = 250,
    min_silence_ms: int = 3000,
    speech_pad_ms: int = 900,
    window_size: int = 1536,
    device: str = "cpu",
) -> List[Segment]:
    """Silero VAD - supports v3, v4, v5, v4_fw, v5_fw, v6, v6_fw variants."""
    import torch
    import torchaudio

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
                    "ffmpeg", "-y", "-i", audio_path,
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
            "ffmpeg", "-y", "-i", audio_path,
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
        )
    else:
        timestamps = gst(
            wav, model,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )

    return [(t["start"], t["end"]) for t in timestamps]


def get_speech_segments_pyannote(
    audio_path: str,
    device: str = "cuda",
    use_onnx: bool = True,
) -> List[Segment]:
    """Pyannote VAD v3 (onnx or torch)."""
    if use_onnx:
        # Use pyannote's built-in ONNX pipeline
        from pyannote.audio import Pipeline as _P
        pipeline = _P.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=False,
        )
    else:
        from pyannote.audio import Pipeline as _P
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
        raise ImportError("auditok not installed. Run: pip install auditok")

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
        raise ImportError("webrtcvad not installed. Run: pip install webrtcvad-wheels")

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
) -> List[Segment]:
    """Unified VAD dispatcher."""

    silero_methods = {"silero_v3", "silero_v4", "silero_v5", "silero_v4_fw", "silero_v5_fw", "silero_v6", "silero_v6_fw"}

    if method in silero_methods:
        segments = get_speech_segments_silero(
            audio_path,
            version=method,
            threshold=threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            window_size=window_size,
            device=vad_device,
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
