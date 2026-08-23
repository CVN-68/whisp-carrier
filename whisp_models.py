"""
whisp_models.py
Model aliases and on-demand CTranslate2 conversion for whisp-carrier.

Why this exists
---------------
faster-whisper can only load CTranslate2 models. Most interesting Whisper
fine-tunes are published in transformers format, so they need a one-time
conversion before this pipeline can touch them. Doing that by hand means
remembering the right ct2-transformers-converter incantation, and getting one
detail wrong fails quietly rather than loudly (see the tokenizer note below).
So the conversion happens here, keyed by a cache directory, and the model name
stays a single CLI argument.

    python whisp_carrier.py video.mp4 -m anime-whisper

The first run converts and caches; later runs load straight from the cache.

Alias table
-----------
An alias carries three things: where the weights come from, which format they
are in, and the option defaults that particular model wants. The last part
matters more than it sounds: anime-whisper collapses into hallucination when an
initial prompt is passed, which is a property of the model rather than of the
caller, so the knowledge belongs next to the model definition.

Any Hugging Face repo id also works without being listed here. The repo is
inspected once to decide whether it is already CTranslate2 or needs converting.

The tokenizer note
------------------
faster-whisper looks for tokenizer.json inside the model directory and, when it
is missing, silently downloads the tokenizer of openai/whisper-tiny instead.
For a large-v3 based model (vocab 51866 against whisper-tiny's 51865) that
produces plausible-looking but wrong text, with no error anywhere. Several
repos, litagin/anime-whisper among them, ship only vocab.json + merges.txt, so
the conversion step builds tokenizer.json itself and refuses to finish without
it.

The alignment heads note
-----------------------
Word timestamps are read off specific cross-attention heads, listed as
(decoder layer, head) pairs in the model's generation_config.json. Distilled
models inherit that list from their teacher, so a model with a 2 layer decoder
can advertise heads on layer 25. CTranslate2 copies the list as given, then
indexes straight into it at alignment time, and the process dies without a
Python traceback. litagin/anime-whisper is exactly this case: 2 decoder layers,
heads listed for layers 7 to 25 (whisper-large-v3's). Conversion therefore
validates the list against the real decoder shape and replaces it when needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Written into every converted directory so a stale cache can be detected.
MARKER_NAME = "whisp-carrier-model.json"

# Bumped when a conversion fix makes older cached copies unsafe to reuse.
# 2: alignment heads are validated against the decoder shape.
MARKER_FORMAT = 2

# Quantization values understood by the CTranslate2 converter.
CONVERTER_QUANTIZATIONS = {
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
}

# Model sizes faster-whisper resolves to a Systran CTranslate2 repo on its own.
FALLBACK_BUILTIN_MODELS = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large",
    "distil-large-v2", "distil-medium.en", "distil-small.en",
    "distil-large-v3", "distil-large-v3.5",
    "large-v3-turbo", "turbo",
}

# Files worth carrying into the converted directory when the source has them.
COPY_CANDIDATES = ("preprocessor_config.json", "tokenizer.json")

# Variants that keep large-v3's encoder but cut the decoder down: turbo runs 4
# decoder layers against large-v3's 32, and the distil family fewer still.
#
# Worth saying out loud rather than loading quietly, because the decoder is
# exactly where this project's measured advantage lives. Loop hallucinations,
# segments running past 30s and the timestamp tokens that place every cue are all
# decoder work, and condition_on_previous_text is off here by design, so there is
# no carried context to recover with. Faster, and not measured against the
# reference set.
#
# The concrete way this arrives uninvited: Amatsukaze's whisper-model set to
# 'auto' passes -m large-v3-turbo. Nobody typed it, and without this line the
# only clue is the model name in the [MODEL] row.
REDUCED_DECODER_HINTS = ("turbo", "distil")


def _reduced_decoder_notice(source: str) -> List[str]:
    """Lines warning that this model is not the one the figures come from."""
    lowered = source.lower()
    if not any(hint in lowered for hint in REDUCED_DECODER_HINTS):
        return []
    return [
        f"[MODEL] NOTE: '{source}' is a reduced-decoder Whisper variant "
        "(turbo runs 4 decoder layers against large-v3's 32).",
        "[MODEL]   The accuracy figures this project publishes are large-v3. "
        "This model is not measured against the reference set, and loop "
        "hallucinations and over-long segments are decoder-side failures.",
        "[MODEL]   Amatsukaze's whisper-model 'auto' resolves to "
        "large-v3-turbo; pick large-v3 explicitly to match the published "
        "numbers.",
    ]


class ModelError(Exception):
    """Raised for anything that stops a model from being made loadable."""


# ─────────────────────────────────────────────
# Alias table
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo: str
    summary: str
    license: str
    # Options this model wants, applied only where the caller stayed silent.
    defaults: Dict[str, Any] = field(default_factory=dict)
    # dest -> why it hurts. Warned about, never silently changed.
    avoid: Dict[str, str] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()


ALIASES: Dict[str, ModelSpec] = {
    "anime-whisper": ModelSpec(
        alias="anime-whisper",
        repo="litagin/anime-whisper",
        summary="Japanese anime / visual novel speech. kotoba-whisper-v2.0 fine-tuned "
                "on 5,300h of acted dialogue.",
        license="MIT",
        defaults={
            "language": "ja",
            # The published CER numbers were measured with this setting, and it
            # is the model author's own remedy for repetition loops.
            "no_repeat_ngram_size": 5,
        },
        avoid={
            "initial_prompt": "anime-whisper degrades badly with an initial prompt "
                              "(hallucination loops); the model card says to omit it",
        },
        notes=(
            "NOT RECOMMENDED for subtitles on recorded TV anime. Measured here over "
            "9 episodes (36k reference characters from ARIB captions): CER 41.8% "
            "against large-v3's 24.3%, losing on all 9. The model card's 13.0 vs "
            "16.5 did not reproduce. Kept as an alias because the non-verbal "
            "transcription is genuinely different, not because it scores better.",
            "Segments roughly 4x coarser than large-v3 (651 vs 2805 over the same "
            "9 episodes), so one cue can hold 30s and several speakers. Subtitle "
            "formatting such as --standard_asia is effectively required.",
            "Distilled decoder (32 encoder / 2 decoder layers): fast, and lighter on VRAM.",
            "Transcribes non-verbal speech (laughs, gasps, stutters) instead of dropping it.",
            "Writes half-width ! ? and digits, and usually omits the sentence-final .",
        ),
    ),
    "kotoba-v2": ModelSpec(
        alias="kotoba-v2",
        repo="kotoba-tech/kotoba-whisper-v2.0-faster",
        summary="Japanese general speech, distilled from whisper-large-v3. "
                "Base model of anime-whisper, already published as CTranslate2.",
        license="Apache-2.0",
        defaults={"language": "ja"},
    ),
}


def alias_names() -> List[str]:
    return sorted(ALIASES)


def describe_aliases() -> List[str]:
    """Lines for --list_models."""
    lines = ["Model aliases:"]
    for name in alias_names():
        spec = ALIASES[name]
        lines.append(f"  {name}")
        lines.append(f"      source   {spec.repo} ({spec.license})")
        lines.append(f"      about    {spec.summary}")
        if spec.defaults:
            applied = ", ".join(f"{k}={v!r}" for k, v in sorted(spec.defaults.items()))
            lines.append(f"      defaults {applied}")
        for note in spec.notes:
            lines.append(f"      note     {note}")
        for reason in spec.avoid.values():
            lines.append(f"      warning  {reason}")
    lines.append("")
    lines.append("Any Whisper model name, local directory or Hugging Face repo id also works.")
    lines.append("Repos published in transformers format are converted to CTranslate2 on")
    lines.append("first use and cached under the model directory.")
    return lines


# ─────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────

@dataclass
class ResolvedModel:
    requested: str
    path: str
    spec: Optional[ModelSpec] = None
    converted: bool = False
    lines: List[str] = field(default_factory=list)


def builtin_models() -> Set[str]:
    try:
        from faster_whisper.utils import available_models

        return set(available_models())
    except Exception:
        return set(FALLBACK_BUILTIN_MODELS)


def quantization_for(compute_type: Optional[str], device: str) -> str:
    """Pick the weight format to convert to, following --compute_type."""
    if compute_type in CONVERTER_QUANTIZATIONS:
        return str(compute_type)
    return "float16" if device == "cuda" else "int8"


def _slug(source: str) -> str:
    keep = []
    for ch in source:
        keep.append(ch if (ch.isalnum() or ch in "-._") else "-")
    return "".join(keep).strip("-") or "model"


def resolve(
    name: str,
    *,
    cache_root: Path,
    compute_type: Optional[str] = None,
    device: str = "cuda",
    force_convert: bool = False,
) -> ResolvedModel:
    """Turn --model into something WhisperModel can actually load.

    Built-in sizes and CTranslate2 directories or repos pass through untouched.
    A transformers-format source is converted into cache_root once.
    """
    requested = str(name).strip()
    spec = ALIASES.get(requested.lower())
    lines: List[str] = []

    source = spec.repo if spec else requested
    if spec:
        lines.append(f"[MODEL] alias '{spec.alias}' -> {spec.repo} ({spec.license})")

    local = Path(source).expanduser()
    is_local_dir = local.is_dir()

    if is_local_dir:
        if (local / "model.bin").is_file():
            lines.append(f"[MODEL] local CTranslate2 directory: {local}")
            return ResolvedModel(requested, str(local), spec, False, lines)
        if not (local / "config.json").is_file():
            raise ModelError(
                f"'{local}' is neither a CTranslate2 model (no model.bin) nor a "
                "transformers model (no config.json)"
            )
    elif source in builtin_models():
        lines.append(f"[MODEL] built-in Whisper model: {source}")
        lines.extend(_reduced_decoder_notice(source))
        return ResolvedModel(requested, source, spec, False, lines)

    quantization = quantization_for(compute_type, device)
    out_dir = cache_root / f"ct2-{_slug(source)}-{quantization}"

    # The cache is checked before the hub, so a converted model keeps working
    # with no network at all.
    if not force_convert and _cache_is_usable(out_dir, quantization, lines):
        return ResolvedModel(requested, str(out_dir), spec, False, lines)

    if not is_local_dir and _repo_form(source, lines) == "ctranslate2":
        return ResolvedModel(requested, source, spec, False, lines)

    convert(source, out_dir, quantization=quantization, lines=lines)
    return ResolvedModel(requested, str(out_dir), spec, True, lines)


def _repo_form(repo: str, lines: List[str]) -> str:
    """Decide whether a Hugging Face repo is CTranslate2 or transformers.

    An unreachable hub is not fatal: the name is handed to faster-whisper as
    before, which either finds it cached or reports its own error.
    """
    files = _repo_files(repo)
    if files is None:
        lines.append(
            f"[MODEL] cannot inspect '{repo}' (offline?), passing it to faster-whisper as is"
        )
        return "ctranslate2"

    if "model.bin" in files:
        lines.append(f"[MODEL] {repo} is already CTranslate2")
        return "ctranslate2"

    weights = any(
        f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt")
        for f in files
    )
    if "config.json" in files and weights:
        lines.append(f"[MODEL] {repo} is in transformers format, conversion required")
        return "transformers"

    raise ModelError(
        f"'{repo}' does not look like a Whisper model: no model.bin (CTranslate2) "
        "and no config.json plus weights (transformers)"
    )


def _repo_files(repo: str) -> Optional[Set[str]]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=False)
        return {sibling.rfilename for sibling in (info.siblings or [])}
    except Exception:
        return None


def _cache_is_usable(out_dir: Path, quantization: str, lines: List[str]) -> bool:
    if not (out_dir / "model.bin").is_file():
        return False

    # A cache without tokenizer.json would load with the wrong vocabulary.
    if not (out_dir / "tokenizer.json").is_file():
        lines.append(f"[MODEL] cached model at {out_dir} has no tokenizer.json, reconverting")
        return False

    marker = out_dir / MARKER_NAME
    if not marker.is_file():
        lines.append(
            f"[MODEL] cached model at {out_dir} was not produced by whisp-carrier, "
            "reconverting"
        )
        return False

    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        recorded = {}

    if recorded.get("quantization") not in (None, quantization):
        lines.append(
            f"[MODEL] cached model at {out_dir} was converted as "
            f"{recorded.get('quantization')}, reconverting as {quantization}"
        )
        return False

    if int(recorded.get("format", 0)) < MARKER_FORMAT:
        lines.append(
            f"[MODEL] cached model at {out_dir} predates the current conversion "
            "fixes, reconverting"
        )
        return False

    lines.append(f"[MODEL] using converted model: {out_dir}")
    return True


# ─────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────

def convert(
    source: str,
    out_dir: Path,
    *,
    quantization: str = "float16",
    lines: Optional[List[str]] = None,
) -> Path:
    """Convert a transformers Whisper model into a CTranslate2 directory.

    Writes to a sibling .tmp directory first so that an interrupted run cannot
    leave a half-converted model behind that would later look like a valid cache.
    """
    log = lines if lines is not None else []
    converter_cls = _load_converter()
    transformers = _load_transformers()

    out_dir = Path(out_dir)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    copy_files = _copyable(source)
    log.append(
        f"[MODEL] converting {source} -> {out_dir} (quantization={quantization}). "
        "This downloads the model once and takes a few minutes."
    )

    started = time.time()
    try:
        converter = converter_cls(
            source,
            copy_files=list(copy_files),
            load_as_float16=quantization in ("float16", "int8_float16"),
        )
        converter.convert(str(tmp_dir), quantization=quantization, force=True)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ModelError(f"conversion of {source} failed: {e}") from e

    try:
        _ensure_tokenizer(source, tmp_dir, transformers, log)
        _fix_alignment_heads(source, tmp_dir, transformers, log)
        _write_marker(tmp_dir, source, quantization, transformers)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp_dir, out_dir)

    log.append(f"[MODEL] converted in {time.time() - started:.1f}s: {out_dir}")
    return out_dir


def _frozen() -> bool:
    """True when running from a PyInstaller build."""
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")


def _no_conversion_error(detail: str) -> ModelError:
    """Explain a missing conversion toolchain in terms of the current run mode.

    The exe deliberately ships without transformers, so this is the expected
    outcome there rather than a broken install, and the way out is different:
    convert once with the script version and hand the exe the result. Worded
    like vad._missing_backend for the same reason.
    """
    if _frozen():
        return ModelError(
            "this exe build cannot convert transformers-format models "
            f"({detail}).\n"
            "Convert once with the script version, then pass the converted "
            "directory:\n"
            "  python whisp_carrier.py <input> -m anime-whisper\n"
            "  whisp-carrier.exe <input> -m "
            "<path>\\_models\\ct2-litagin-anime-whisper-float16\n"
            "Built-in sizes (large-v3 etc.) and CTranslate2 models work here "
            "as usual."
        )
    return ModelError(
        f"converting a transformers model needs the transformers package "
        f"({detail}):\n"
        "  pip install transformers\n"
        "Alternatively point --model at an already converted CTranslate2 "
        "directory or repo."
    )


def _load_converter():
    # This import succeeds even in a build without transformers, because
    # ctranslate2/converters/transformers.py wraps its transformers import in
    # try/except ImportError. The failure therefore surfaces from
    # _load_transformers below, which is where the useful message lives. This
    # path still reports the same thing so that a genuinely missing
    # ctranslate2.converters does not produce a bare traceback.
    try:
        from ctranslate2.converters import TransformersConverter

        return TransformersConverter
    except ImportError as e:
        raise _no_conversion_error(str(e)) from e


def _load_transformers():
    try:
        import transformers

        return transformers
    except ImportError as e:
        raise _no_conversion_error(str(e)) from e


def _copyable(source: str) -> Sequence[str]:
    """Which of the useful side files the source actually has.

    The converter raises if asked to copy a file that is not there, and
    preprocessor_config.json is not optional in practice: without it the feature
    extractor falls back to defaults that may not match the model's mel bins.
    """
    local = Path(source).expanduser()
    if local.is_dir():
        available = {p.name for p in local.iterdir() if p.is_file()}
    else:
        available = _repo_files(source) or set()
    return [name for name in COPY_CANDIDATES if name in available]


def _ensure_tokenizer(source: str, out_dir: Path, transformers, log: List[str]) -> None:
    """Guarantee a tokenizer.json next to the converted weights."""
    target = out_dir / "tokenizer.json"
    if target.is_file():
        return

    log.append("[MODEL] source has no tokenizer.json, building one from the fast tokenizer")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, use_fast=True)
    except Exception as e:
        raise ModelError(f"cannot load a tokenizer for {source}: {e}") from e

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ModelError(
            f"{source} only provides a slow tokenizer, so tokenizer.json cannot be "
            "generated. faster-whisper would fall back to the whisper-tiny "
            "tokenizer and produce wrong text, so the conversion is stopped here."
        )

    backend.save(str(target))
    if not target.is_file():
        raise ModelError(f"failed to write {target}")


def _fix_alignment_heads(source: str, out_dir: Path, transformers, log: List[str]) -> None:
    """Drop alignment heads that do not exist in this model's decoder.

    A distilled model usually keeps its teacher's head list, which points at
    decoder layers it does not have. CTranslate2 stores the list verbatim and
    reads it during alignment, so --word_timestamps then kills the process with
    no Python-level error at all. Anything out of range is removed here, and if
    that empties the list, CTranslate2's own default (every head of the upper
    half of the decoder) is written instead.
    """
    config_path = out_dir / "config.json"
    if not config_path.is_file():
        return

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ModelError(f"cannot read the converted config at {config_path}: {e}") from e

    heads = config.get("alignment_heads")
    if not heads:
        return

    try:
        model_config = transformers.AutoConfig.from_pretrained(source)
    except Exception as e:
        raise ModelError(f"cannot read the model configuration of {source}: {e}") from e

    layers = int(getattr(model_config, "decoder_layers", 0) or 0)
    per_layer = int(getattr(model_config, "decoder_attention_heads", 0) or 0)
    if layers <= 0 or per_layer <= 0:
        return

    def usable(pair) -> bool:
        try:
            layer, head = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            return False
        return 0 <= layer < layers and 0 <= head < per_layer

    kept = [list(pair) for pair in heads if usable(pair)]
    if len(kept) == len(heads):
        return

    dropped = len(heads) - len(kept)
    if kept:
        replacement = kept
        how = f"dropped {dropped} of {len(heads)} heads pointing outside the decoder"
    else:
        replacement = [
            [layer, head]
            for layer in range(layers // 2, layers)
            for head in range(per_layer)
        ]
        how = (
            f"all {len(heads)} heads pointed outside the decoder, falling back to "
            f"every head of decoder layer(s) {layers // 2}..{layers - 1}"
        )

    config["alignment_heads"] = replacement
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.append(
        f"[MODEL] alignment heads repaired for a {layers}-layer decoder: {how}"
    )


def _write_marker(out_dir: Path, source: str, quantization: str, transformers) -> None:
    import ctranslate2

    payload = {
        "format": MARKER_FORMAT,
        "source": source,
        "quantization": quantization,
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ctranslate2": getattr(ctranslate2, "__version__", "?"),
        "transformers": getattr(transformers, "__version__", "?"),
    }
    (out_dir / MARKER_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────
# Per-model option defaults
# ─────────────────────────────────────────────

def _prompt_is_set(value: Any) -> bool:
    return value not in (None, "", "None", "auto")


def apply_model_defaults(
    args: argparse.Namespace,
    spec: Optional[ModelSpec],
    explicit: Set[str],
) -> List[str]:
    """Fold a model's preferred options into args.

    Anything the caller set, on the command line or in the YAML file, is left
    alone and reported, so that a comparison run always shows which settings
    were actually in force.
    """
    if spec is None:
        return []

    lines: List[str] = []

    for dest in sorted(spec.defaults):
        value = spec.defaults[dest]
        if not hasattr(args, dest):
            continue
        if dest in explicit:
            current = getattr(args, dest)
            if current != value:
                lines.append(
                    f"[MODEL]   {dest} = {current!r} kept as given "
                    f"({spec.alias} would use {value!r})"
                )
            continue
        if getattr(args, dest) == value:
            continue
        setattr(args, dest, value)
        lines.append(f"[MODEL]   {dest} = {value!r}  (default for {spec.alias})")

    for dest, reason in sorted(spec.avoid.items()):
        value = getattr(args, dest, None)
        if dest == "initial_prompt":
            if not _prompt_is_set(value):
                continue
        elif not value:
            continue
        lines.append(f"[MODEL]   WARNING: {dest}={value!r}: {reason}")

    return lines
