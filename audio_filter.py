"""
audio_filter.py
Audio preprocessing utilities for whisp-carier.
Wraps ffmpeg for noise reduction, vocal extraction, normalization, etc.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def get_ffmpeg_path() -> str:
    """Locate ffmpeg: bundled next to this script, or system PATH."""
    # When packaged as exe, _MEIPASS contains bundled files
    base = getattr(__import__("sys"), "_MEIPASS", None)
    if base:
        candidate = os.path.join(base, "ffmpeg.exe")
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"


FFMPEG = get_ffmpeg_path()


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr.decode(errors='replace')}")


def extract_audio(input_path: str, output_path: str, track: int = 1) -> None:
    """Extract audio track from any media file to 16kHz mono WAV."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-map", f"0:a:{track - 1}",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path,
    ])


def apply_rnndn_sh(input_path: str, output_path: str) -> None:
    """Suppress non-speech using RNNoise SH model (GregorR)."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "arnndn=m=sh.rnnn",
        output_path,
    ])


def apply_rnndn_xiph(input_path: str, output_path: str) -> None:
    """Suppress non-speech using RNNoise Xiph model."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "arnndn=m=xiph.rnnn",
        output_path,
    ])


def apply_fftdn(input_path: str, output_path: str, strength: int = 12) -> None:
    """General FFT-based denoising. strength 1-97, 12=normal."""
    if strength <= 0:
        import shutil
        shutil.copy2(input_path, output_path)
        return
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", f"afftdn=nr={strength}:nf=-25",
        output_path,
    ])


def apply_loudnorm(input_path: str, output_path: str) -> None:
    """EBU R128 loudness normalization."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        output_path,
    ])


def apply_speechnorm(input_path: str, output_path: str) -> None:
    """Extreme speech amplification / normalization."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "speechnorm=e=50:r=0.0001:l=1",
        output_path,
    ])


def apply_lowhighpass(input_path: str, output_path: str) -> None:
    """Band-pass filter 50Hz - 7800Hz."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "highpass=f=50,lowpass=f=7800",
        output_path,
    ])


def apply_gate(input_path: str, output_path: str) -> None:
    """Reduce lower parts of the signal (noise gate)."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "agate=threshold=0.02:ratio=4:attack=10:release=200",
        output_path,
    ])


def apply_tempo(input_path: str, output_path: str, tempo: float = 1.0) -> None:
    """Adjust audio tempo. 1.0 = no change."""
    if tempo == 1.0:
        import shutil
        shutil.copy2(input_path, output_path)
        return
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", f"atempo={tempo}",
        output_path,
    ])


def apply_silence_suppress(input_path: str, output_path: str,
                            noise_db: float = 0, min_duration: float = 3.0) -> None:
    """Suppress quiet parts. noise_db=0 disables."""
    if noise_db == 0:
        import shutil
        shutil.copy2(input_path, output_path)
        return
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", f"silenceremove=stop_periods=-1:stop_duration={min_duration}:stop_threshold={noise_db}dB",
        output_path,
    ])


def select_channel_fc(input_path: str, output_path: str) -> None:
    """Select only front-center (FC) channel."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "pan=mono|c0=FC",
        output_path,
    ])


def select_channel_left(input_path: str, output_path: str) -> None:
    """Select only left channel."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "pan=mono|c0=FL",
        output_path,
    ])


def invert_mix(input_path: str, output_path: str) -> None:
    """Invert left channel polarity and mix to mono."""
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-af", "pan=mono|c0=FL-FR",
        output_path,
    ])


def apply_vocal_extract_mdx(input_path: str, output_path: str,
                              chunk_seconds: int = 15,
                              device: str = "cuda") -> None:
    """
    Vocal extraction using MDX Kim_Vocal_2 model via audio-separator.
    Equivalent to --ff_vocal_extract mdx_kim2 in Faster-Whisper-XXL.
    """
    import tempfile
    from audio_separator.separator import Separator

    tmpdir = tempfile.mkdtemp(prefix="whisp_carier_mdx_")
    try:
        sep = Separator(
            output_dir=tmpdir,
            mdx_params={
                "hop_length": 1024,
                "segment_size": chunk_seconds * 44100 // 1024,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": False,
            },
        )
        sep.load_model(model_filename="Kim_Vocal_2.onnx")
        stems = sep.separate(input_path)

        vocal_file = next(
            (s for s in stems if "vocal" in os.path.basename(s).lower()),
            stems[0]
        )
        if not os.path.isabs(vocal_file):
            vocal_file = os.path.join(tmpdir, vocal_file)

        extract_audio(vocal_file, output_path)
    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def apply_vocal_extract_roformer(input_path: str, output_path: str,
                                  device: str = "cuda") -> None:
    """
    Vocal extraction using Mel-Band-Roformer model via audio-separator.
    Equivalent to --ff_vocal_extract mb-roformer in Faster-Whisper-XXL Pro.
    """
    import tempfile
    from audio_separator.separator import Separator

    # TemporaryDirectoryを使わず固定tmpフォルダを使う（ffmpegが後で参照できるように）
    tmpdir = tempfile.mkdtemp(prefix="whisp_carier_roformer_")
    try:
        sep = Separator(output_dir=tmpdir)
        sep.load_model(model_filename="vocals_mel_band_roformer.ckpt")
        stems = sep.separate(input_path)

        # separate()はフルパスを返す場合とファイル名のみの場合がある
        vocal_file = next(
            (s for s in stems if "vocal" in os.path.basename(s).lower()),
            stems[0]
        )
        # フルパスでなければtmpdirと結合
        if not os.path.isabs(vocal_file):
            vocal_file = os.path.join(tmpdir, vocal_file)

        extract_audio(vocal_file, output_path)
    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def preprocess(input_path: str, args) -> str:
    """
    Run the full preprocessing chain based on CLI args.
    Returns path to the processed temp WAV file.
    """
    tmp_files = []

    def next_tmp(suffix=".wav") -> str:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.close()
        tmp_files.append(f.name)
        return f.name

    current = next_tmp()
    extract_audio(input_path, current, track=getattr(args, "ff_track", 1))

    if getattr(args, "ff_fc", False):
        out = next_tmp()
        select_channel_fc(current, out)
        current = out

    if getattr(args, "ff_lc", False):
        out = next_tmp()
        select_channel_left(current, out)
        current = out

    if getattr(args, "ff_invert", False):
        out = next_tmp()
        invert_mix(current, out)
        current = out

    if getattr(args, "ff_rnndn_sh", False):
        out = next_tmp()
        apply_rnndn_sh(current, out)
        current = out

    if getattr(args, "ff_rnndn_xiph", False):
        out = next_tmp()
        apply_rnndn_xiph(current, out)
        current = out

    ff_fftdn = getattr(args, "ff_fftdn", 0)
    if ff_fftdn and ff_fftdn > 0:
        out = next_tmp()
        apply_fftdn(current, out, strength=ff_fftdn)
        current = out

    if getattr(args, "ff_gate", False):
        out = next_tmp()
        apply_gate(current, out)
        current = out

    if getattr(args, "ff_speechnorm", False):
        out = next_tmp()
        apply_speechnorm(current, out)
        current = out

    if getattr(args, "ff_loudnorm", False):
        out = next_tmp()
        apply_loudnorm(current, out)
        current = out

    ff_silence = getattr(args, "ff_silence_suppress", [0, 3.0])
    if ff_silence and ff_silence[0] != 0:
        out = next_tmp()
        apply_silence_suppress(current, out, noise_db=ff_silence[0], min_duration=ff_silence[1])
        current = out

    if getattr(args, "ff_lowhighpass", False):
        out = next_tmp()
        apply_lowhighpass(current, out)
        current = out

    ff_tempo = getattr(args, "ff_tempo", 1.0)
    if ff_tempo and ff_tempo != 1.0:
        out = next_tmp()
        apply_tempo(current, out, tempo=ff_tempo)
        current = out

    vocal_extract = getattr(args, "ff_vocal_extract", None)
    if vocal_extract == "mdx_kim2":
        out = next_tmp()
        apply_vocal_extract_mdx(
            current, out,
            chunk_seconds=getattr(args, "mdx_chunk", 15),
            device=getattr(args, "voc_device", "cuda"),
        )
        current = out
    elif vocal_extract == "mb-roformer":
        out = next_tmp()
        apply_vocal_extract_roformer(
            current, out,
            device=getattr(args, "voc_device", "cuda"),
        )
        current = out

    # Clean up intermediate files (keep the final one)
    for f in tmp_files[:-1]:
        try:
            os.unlink(f)
        except Exception:
            pass

    return current
