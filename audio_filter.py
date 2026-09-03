"""
audio_filter.py
Audio preprocessing utilities for whisp-carrier.
Wraps ffmpeg for noise reduction, vocal extraction, normalization, etc.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _frozen() -> bool:
    """True when running from a PyInstaller build."""
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")


def _load_separator():
    """Import audio-separator, or explain why it is unavailable.

    Not bundled into the exe. It was bundled once, under
    WHISP_CARRIER_FULL=1, and the result was worse than leaving it out: the
    package imported fine but died inside scipy ("The `scipy` install you are
    using seems to be broken ... please try reinstalling"), which sends the
    user off to reinstall an installation that is not broken. Reporting the
    real reason is more useful than shipping a path that cannot work.

    Worded like vad._missing_backend, which solves the same problem for the
    external VAD backends.
    """
    try:
        from audio_separator.separator import Separator

        return Separator
    except ImportError as e:
        if _frozen():
            raise ImportError(
                "--ff_vocal_extract is not available in this exe build "
                "(audio-separator is not bundled). Run the script version "
                "with audio-separator installed, or drop the option: "
                "measurements on this project's own test set were taken "
                "without any --ff_* filter, and vocal extraction has not been "
                "shown to help subtitle accuracy."
            ) from e
        raise ImportError(
            "audio-separator not installed. Run: pip install audio-separator"
        ) from e


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


def _run(cmd: list[str]) -> str:
    """Run ffmpeg, raise on a non-zero exit, and return stderr.

    stderr is returned rather than discarded because ffmpeg writes everything
    worth knowing there: stream layouts, volumedetect results, and warnings
    about filters that did nothing. A zero exit code on its own says very
    little, which is the whole reason the stage checks below exist.
    """
    result = subprocess.run(cmd, capture_output=True)
    stderr = result.stderr.decode(errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{stderr}")
    return stderr


class FilterStageError(RuntimeError):
    """A filter stage produced output that cannot be transcribed."""


# Digital silence measured -91.0 dB in this pipeline (see HANDOVER, 既知の問題
# 2-a), and normal speech sits around -20 dB mean / -1 dB peak. Anything whose
# peak is below this has nothing left to recognise.
SILENCE_PEAK_DB = -80.0

# Stages that are not supposed to change the length are allowed this much drift
# from resampling and frame alignment.
DURATION_TOLERANCE = 0.02

_RE_MAX = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB")
_RE_MEAN = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB")
_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")


def measure(path: str) -> dict:
    """Peak level, mean level and duration of an audio file.

    One volumedetect pass gives all three: the filter reports the levels and
    ffmpeg's own input banner carries the duration. Decoding the whole file
    costs a few seconds per stage, which only ever happens when --ff_* was
    requested.
    """
    stderr = _run([
        FFMPEG, "-hide_banner",
        "-i", path,
        "-af", "volumedetect",
        "-f", "null", "-",
    ])

    max_m = _RE_MAX.search(stderr)
    mean_m = _RE_MEAN.search(stderr)
    dur_m = _RE_DURATION.search(stderr)

    duration = None
    if dur_m:
        h, m, s = dur_m.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    return {
        # A file with no audio at all reports no max_volume; treat that as
        # silence rather than as "unknown", because it is not transcribable
        # either way.
        "peak_db": float(max_m.group(1)) if max_m else float("-inf"),
        "mean_db": float(mean_m.group(1)) if mean_m else float("-inf"),
        "duration": duration,
        "measured": bool(max_m),
    }


def _peak_str(m: dict) -> str:
    if m["peak_db"] == float("-inf"):
        return "silent"
    return f"{m['peak_db']:.1f} dB"


def _format_measurement(label: str, m: dict) -> str:
    mean = "-" if m["mean_db"] == float("-inf") else f"{m['mean_db']:.1f}dB"
    dur = "?" if m["duration"] is None else f"{m['duration']:.2f}s"
    return f"  [FF] {label}: peak={_peak_str(m)} mean={mean} dur={dur}"


def verify_stage(label: str, before: dict, after: dict, *,
                 expect_duration: bool = True) -> None:
    """Fail loudly when a stage silenced the audio or changed its length.

    Both failures used to pass unnoticed: ffmpeg exits 0 while writing digital
    silence (--ff_fc on stereo input did exactly that), and a truncated output
    looks like a success too. Raising here surfaces as a failed file and a
    non-zero exit code.
    """
    if after["peak_db"] <= SILENCE_PEAK_DB:
        raise FilterStageError(
            f"filter stage '{label}' silenced the audio "
            f"(peak {_peak_str(before)} -> {_peak_str(after)}). "
            f"Nothing would be transcribed. Drop that option, or check that "
            f"the requested channel exists in the source layout."
        )

    if (expect_duration and before["duration"] and after["duration"]):
        drift = abs(after["duration"] - before["duration"]) / before["duration"]
        if drift > DURATION_TOLERANCE:
            raise FilterStageError(
                f"filter stage '{label}' changed the duration from "
                f"{before['duration']:.2f}s to {after['duration']:.2f}s "
                f"({drift:.1%}). The output is truncated or padded, so the "
                f"timestamps would not line up with the source."
            )


# Channel selections are expressed by index where possible, so that they do not
# depend on ffmpeg's channel naming for the input layout. FC has no index
# equivalent, so it stays a name and fails the silence check on stereo input.
_CHANNEL_FILTERS = {
    "fc": "pan=mono|c0=FC",
    "left": "pan=mono|c0=c0",
    "invert": "pan=mono|c0=c0-c1",
}


def extract_audio(input_path: str, output_path: str, track: int = 1,
                  channel: str | None = None) -> None:
    """Extract one audio track to 16kHz mono WAV, optionally picking a channel.

    Channel selection belongs in this call rather than in a stage of its own.
    It used to run afterwards, against audio this function had already
    downmixed to mono, where selecting FL or FC produced complete digital
    silence (measured: -91.0 dB across every sample). ffmpeg applies an
    explicit -af before the automatic conversion it inserts for -ac, so
    selecting a channel and downmixing in one command happens in the right
    order.
    """
    cmd = [
        FFMPEG, "-y",
        "-i", input_path,
        "-map", f"0:a:{track - 1}",
    ]
    if channel:
        try:
            cmd += ["-af", _CHANNEL_FILTERS[channel]]
        except KeyError:
            raise ValueError(
                f"unknown channel selector {channel!r}; expected one of "
                f"{', '.join(sorted(_CHANNEL_FILTERS))}"
            ) from None
    cmd += [
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_path,
    ]
    _run(cmd)


def extract_track_native(input_path: str, output_path: str,
                         track: int = 1) -> None:
    """Extract one audio track to WAV without touching its rate or layout.

    Only the vocal separation stage needs this. The separation models are
    trained on 44.1kHz stereo music, so handing them the 16kHz mono audio
    extract_audio produces throws away every band above 8kHz and the stereo
    image the separation depends on, before the model has seen any of it. That
    is the same class of mistake as running channel selection after the
    downmix (see extract_audio), except that it degrades the result quietly
    instead of producing silence, so no stage check catches it.

    Everything else in the chain is an ffmpeg filter that works fine at 16kHz
    mono, and 16kHz mono is what the recogniser wants, so the conversion still
    happens -- just after the separation rather than before it.
    """
    _run([
        FFMPEG, "-y",
        "-i", input_path,
        "-map", f"0:a:{track - 1}",
        "-c:a", "pcm_s16le",
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


# Channel selection used to live here as three standalone stages
# (select_channel_fc / select_channel_left / invert_mix). They ran after
# extract_audio had already downmixed to mono, so `pan=mono|c0=FL` and
# `c0=FL-FR` addressed channels that no longer existed and wrote digital
# silence: all 113,640 samples of the test file landed in the -91 dB bin.
# ffmpeg exited 0 each time.
#
# They are gone rather than fixed in place, because the ordering cannot be
# repaired from a separate stage. extract_audio takes a `channel=` argument
# instead, which puts the selection in the same command as the downmix, and the
# selectors are index-based (`c0`, `c0-c1`) so they do not depend on the input
# layout's channel names. See _CHANNEL_FILTERS.


# audio-separator names its output "<input>_(<Stem>)_<model>.<ext>", so the
# stem label is the parenthesised part. Matching against the whole file name
# instead picks the wrong file whenever the model name itself contains the word
# (see _pick_vocal_stem).
_RE_STEM_LABEL = re.compile(r"\(([^)]+)\)")


def _pick_vocal_stem(stems: list, tmpdir: str, model: str) -> str:
    """Return the vocals stem, or refuse to guess.

    The test used to be `"vocal" in os.path.basename(path)`, applied to the
    whole file name. Both models write a name that contains the model's own
    name, and both model names contain "vocal":

        tmp_(other)_vocals_mel_band_roformer.wav     <- accompaniment
        tmp_(vocals)_vocals_mel_band_roformer.wav    <- what was wanted
        tmp_(Instrumental)_Kim_Vocal_2.wav           <- accompaniment
        tmp_(Vocals)_Kim_Vocal_2.wav                 <- what was wanted

    The accompaniment sorts first, so every run so far transcribed the audio
    with the voices taken out. On a 30 minute recording that left 27.1s of
    speech for the VAD to find out of 1800s, and language detection came back
    "en" on Japanese dialogue. This is where "vocal extraction removes even the
    anime voices" came from: the model was doing its job and the wrong output
    was being read.

    There is no fallback to stems[0] on purpose. Guessing is what produced a
    silently wrong result for as long as this option has existed; a failure that
    names the stems is more useful.

    audio-separator also logs model failures inside separate() and returns an
    empty list rather than raising, and it returns bare file names in some
    versions and absolute paths in others.
    """
    if not stems:
        raise FilterStageError(
            f"vocal separation with {model} produced no output. "
            f"audio-separator reports the reason on the lines above; nothing "
            f"would be left to transcribe, so the run stops here."
        )

    def stem_label(path: str) -> str:
        found = _RE_STEM_LABEL.search(os.path.basename(path))
        return found.group(1).lower() if found else ""

    vocal_file = next(
        (s for s in stems if "vocal" in stem_label(s)),
        None,
    )
    if vocal_file is None:
        names = ", ".join(os.path.basename(s) for s in stems)
        raise FilterStageError(
            f"vocal separation with {model} produced no stem labelled as "
            f"vocals (got: {names}). Transcribing the accompaniment instead "
            f"would look like a working run, so the run stops here."
        )
    if not os.path.isabs(vocal_file):
        vocal_file = os.path.join(tmpdir, vocal_file)
    return vocal_file


def apply_vocal_extract_mdx(input_path: str, output_path: str,
                              segment_size: int = 0,
                              device: str = "cuda") -> None:
    """
    Vocal extraction using MDX Kim_Vocal_2 model via audio-separator.
    Equivalent to --ff_vocal_extract mdx_kim2 in Faster-Whisper-XXL.

    segment_size counts STFT frames, not seconds. It used to be derived from a
    duration (--mdx_chunk seconds * 44100 // 1024, so 646 by default) and
    Kim_Vocal_2's graph does not accept that: every run died inside onnx2torch
    with "The size of tensor a (160) must match the size of tensor b (161)",
    which audio-separator caught and turned into an empty result. 0 leaves
    audio-separator's own default in place, which is what the model ships with.
    """
    import tempfile

    Separator = _load_separator()

    tmpdir = tempfile.mkdtemp(prefix="whisp_carrier_mdx_")
    try:
        params = {"output_dir": tmpdir}
        if segment_size:
            params["mdx_params"] = {
                "hop_length": 1024,
                "segment_size": segment_size,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": False,
            }
        sep = Separator(**params)
        sep.load_model(model_filename="Kim_Vocal_2.onnx")
        stems = sep.separate(input_path)
        vocal_file = _pick_vocal_stem(stems, tmpdir, "Kim_Vocal_2")

        # The stem keeps its rate and layout here. preprocess() converts to
        # 16kHz mono once, after this stage.
        extract_track_native(vocal_file, output_path)
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

    Separator = _load_separator()

    # TemporaryDirectoryを使わず固定tmpフォルダを使う（ffmpegが後で参照できるように）
    tmpdir = tempfile.mkdtemp(prefix="whisp_carrier_roformer_")
    try:
        sep = Separator(output_dir=tmpdir)
        sep.load_model(model_filename="vocals_mel_band_roformer.ckpt")
        stems = sep.separate(input_path)
        vocal_file = _pick_vocal_stem(stems, tmpdir, "mel_band_roformer")

        # The stem keeps its rate and layout here. preprocess() converts to
        # 16kHz mono once, after this stage.
        extract_track_native(vocal_file, output_path)
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

    Every stage is measured and checked. Before this existed, a stage could
    silence the audio or truncate it while ffmpeg exited 0, and the only
    symptom was a transcript with almost nothing in it. Two real cases:
    --ff_fc produced digital silence on stereo input, and --ff_lowhighpass was
    blamed for "losing the second half" without anyone having measured a single
    stage. The per-stage numbers are printed so that the next such report comes
    with evidence attached.
    """
    tmp_files = []

    def next_tmp(suffix=".wav") -> str:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.close()
        tmp_files.append(f.name)
        return f.name

    # Channel selection is folded into the extraction command rather than run
    # as its own stage, because doing it afterwards means selecting a channel
    # from audio that has already been downmixed to mono. See extract_audio.
    channel = None
    requested = [
        name for name, flag in (
            ("fc", getattr(args, "ff_fc", False)),
            ("left", getattr(args, "ff_lc", False)),
            ("invert", getattr(args, "ff_invert", False)),
        ) if flag
    ]
    if len(requested) > 1:
        raise FilterStageError(
            "--ff_fc, --ff_lc and --ff_invert select different channels and "
            f"cannot be combined (got: {', '.join(requested)})."
        )
    if requested:
        channel = requested[0]

    source = input_path
    source_track = getattr(args, "ff_track", 1)

    # Vocal separation runs first, on the source rate and layout. It used to run
    # last, which meant the separation models -- trained on 44.1kHz stereo --
    # only ever saw audio already reduced to 16kHz mono, with everything above
    # 8kHz and the whole stereo image discarded. Nothing downstream needs the
    # original format, so the 16kHz mono conversion moves to just after this.
    vocal_extract = getattr(args, "ff_vocal_extract", None)
    if vocal_extract:
        native = next_tmp()
        extract_track_native(input_path, native, track=source_track)
        native_state = measure(native)
        print(_format_measurement("source", native_state), flush=True)
        if native_state["peak_db"] <= SILENCE_PEAK_DB:
            raise FilterStageError(
                f"audio extraction produced silence "
                f"(peak {_peak_str(native_state)}); track "
                f"{source_track} of the source carries no audio."
            )

        separated = next_tmp()
        label = f"vocal_extract:{vocal_extract}"
        if vocal_extract == "mdx_kim2":
            apply_vocal_extract_mdx(
                native, separated,
                segment_size=getattr(args, "mdx_chunk", 0),
                device=getattr(args, "voc_device", "cuda"),
            )
        elif vocal_extract == "mb-roformer":
            apply_vocal_extract_roformer(
                native, separated,
                device=getattr(args, "voc_device", "cuda"),
            )
        else:
            raise FilterStageError(
                f"unknown --ff_vocal_extract model {vocal_extract!r}; "
                f"expected mdx_kim2 or mb-roformer"
            )
        separated_state = measure(separated)
        print(_format_measurement(label, separated_state), flush=True)
        verify_stage(label, native_state, separated_state)

        # What the separator wrote is a single-track WAV, so the source track
        # index no longer applies to it.
        source, source_track = separated, 1

    current = next_tmp()
    extract_audio(
        source, current,
        track=source_track,
        channel=channel,
    )

    label = "extract" if channel is None else f"extract+{channel}"
    state = measure(current)
    print(_format_measurement(label, state), flush=True)
    if state["peak_db"] <= SILENCE_PEAK_DB:
        detail = (
            f" The '{channel}' channel is probably absent from the source "
            f"layout; only 5.1-style sources carry FC."
            if channel else ""
        )
        raise FilterStageError(
            f"audio extraction produced silence (peak {_peak_str(state)})."
            f"{detail}"
        )

    def stage(name, fn, *a, expect_duration=True, **kw):
        """Run one filter stage, then measure and check what it produced."""
        nonlocal current, state
        out = next_tmp()
        fn(current, out, *a, **kw)
        after = measure(out)
        print(_format_measurement(name, after), flush=True)
        verify_stage(name, state, after, expect_duration=expect_duration)
        current, state = out, after

    if getattr(args, "ff_rnndn_sh", False):
        stage("rnndn_sh", apply_rnndn_sh)

    if getattr(args, "ff_rnndn_xiph", False):
        stage("rnndn_xiph", apply_rnndn_xiph)

    ff_fftdn = getattr(args, "ff_fftdn", 0)
    if ff_fftdn and ff_fftdn > 0:
        stage("fftdn", apply_fftdn, strength=ff_fftdn)

    if getattr(args, "ff_gate", False):
        stage("gate", apply_gate)

    if getattr(args, "ff_speechnorm", False):
        stage("speechnorm", apply_speechnorm)

    if getattr(args, "ff_loudnorm", False):
        stage("loudnorm", apply_loudnorm)

    ff_silence = getattr(args, "ff_silence_suppress", [0, 3.0])
    if ff_silence and ff_silence[0] != 0:
        # Removing silence is the point of this one, so the length must change.
        stage(
            "silence_suppress", apply_silence_suppress,
            noise_db=ff_silence[0], min_duration=ff_silence[1],
            expect_duration=False,
        )

    if getattr(args, "ff_lowhighpass", False):
        stage("lowhighpass", apply_lowhighpass)

    ff_tempo = getattr(args, "ff_tempo", 1.0)
    if ff_tempo and ff_tempo != 1.0:
        # Changing the length is what tempo does.
        stage("tempo", apply_tempo, tempo=ff_tempo, expect_duration=False)

    # Clean up intermediate files (keep the final one)
    for f in tmp_files[:-1]:
        try:
            os.unlink(f)
        except Exception:
            pass

    return current
