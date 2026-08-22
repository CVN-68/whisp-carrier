#!/usr/bin/env python3
"""
eval/run.py
Run whisp-carrier over the evaluation set in one configuration and cache the
JSON output for eval/score.py.

One invocation is one configuration. Comparing two means running it twice with
different --config or --model, then pointing score.py at both output folders.

    python eval/run.py --config ext-collect
    python eval/run.py --config ext-clip
    python eval/score.py --hyp _eval/hyp/ext-collect --hyp _eval/hyp/ext-clip

--no_config is always passed. A stray whisp-carrier.yaml next to the script
would otherwise be picked up and silently change the settings under test, which
is the trap recorded in HANDOVER's 注意 section.

Existing output is skipped, so an interrupted sweep resumes. Each run's console
output is kept next to the JSON, because the [VAD] and [MODEL] lines are the
record of what the settings actually resolved to.

The child's output is streamed rather than collected, so a long file shows
progress while it runs. stdout and stderr are merged into the log in arrival
order; logs written before this change have stderr appended under a
"--- stderr ---" separator instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parent.parent

# VAD configurations. The model is a separate axis, chosen with --model.
CONFIGS: Dict[str, List[str]] = {
    # faster-whisper's bundled silero, the default path.
    "builtin": ["--vad_method", "silero_v5_fw"],
    # External silero routed through collect_chunks. Was the default until the
    # 5h22m measurement, where it lost on every exactly-computed metric.
    "ext-collect": ["--vad_method", "silero_v5", "--vad_segment_mode", "collect"],
    # External silero via clip_timestamps, now the default.
    "ext-clip": ["--vad_method", "silero_v5", "--vad_segment_mode", "clip"],
    # TEN VAD (Apache-2.0) in place of silero, same routing and same segment
    # aggregation, so the difference is the model. HANDOVER 測定結果 #17 showed the
    # speech missed on 死亡遊戯 sits below silero's probability floor, which no
    # silero-side parameter can reach; this is the test of whether another model
    # can (測定結果 #18).
    "ten-clip": ["--vad_method", "ten", "--vad_segment_mode", "clip"],
    # Regions produced by a segmenter that cannot be installed here, dumped by
    # eval/ext_vad_dump.py from its own virtualenv. Pass the JSON with
    # --extra=--vad_segments_json --extra=_eval/vad-<backend>.json.
    "ext-json": ["--vad_method", "precomputed", "--vad_segment_mode", "clip"],
    # No VAD at all, as a control for what the VAD is actually buying.
    "novad": ["--vad_filter", "false"],
}

# Child lines always echoed: the record of what the settings resolved to and of
# what the segment repair had to do.
ECHOED = ("[VAD]", "[MODEL]", "[FIX]", "[FMT]", "[STT] Done")

# Anything else is shown at most this often, as a sign of life. Whisper's own
# progress lines arrive every few seconds, so this only decides how coarse the
# heartbeat is.
HEARTBEAT_SECONDS = 30.0


def die(message: str) -> None:
    print(f"[run] error: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Run whisp-carrier over the evaluation set in one configuration.",
    )
    parser.add_argument("--config", default="ext-collect", choices=sorted(CONFIGS),
                        help="VAD configuration to run.")
    parser.add_argument("--model", "-m", default="large-v3",
                        help="Model passed to whisp-carrier.")
    parser.add_argument("--tag", default=None,
                        help="Output folder name under --hyp-root. Defaults to "
                             "the config name, or config@model when --model is "
                             "not the default.")
    parser.add_argument("--wav-dir", default="_eval/wav",
                        help="Folder of extracted WAVs from eval/prep.py.")
    parser.add_argument("--hyp-root", default="_eval/hyp",
                        help="Where per-configuration output folders are created.")
    parser.add_argument("--only", action="append", default=[], metavar="SUBSTR",
                        help="Only run files whose name contains this. Repeatable.")
    parser.add_argument("--exclude", action="append", default=[], metavar="SUBSTR",
                        help="Skip files whose name contains this. Applied after "
                             "--only. Repeatable.")
    parser.add_argument("--extra", action="append", default=[], metavar="ARG",
                        help="Extra argument passed straight to whisp-carrier. "
                             "Repeatable.")
    parser.add_argument("--redo", action="store_true",
                        help="Re-run even when the JSON already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands and exit.")
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir).expanduser()
    if not wav_dir.is_dir():
        die(f"not a directory: {wav_dir}. Run eval/prep.py first.")

    wavs = sorted(wav_dir.glob("*.wav"))
    if args.only:
        wavs = [w for w in wavs if any(token in w.name for token in args.only)]
    if args.exclude:
        wavs = [w for w in wavs if not any(token in w.name for token in args.exclude)]
    if not wavs:
        die(f"no WAVs to run under {wav_dir}")

    tag = args.tag
    if tag is None:
        tag = args.config if args.model == "large-v3" else (
            f"{args.config}@{Path(args.model).name}"
        )
    out_dir = Path(args.hyp_root).expanduser() / tag
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    carrier = REPO / "whisp_carrier.py"
    if not carrier.is_file():
        die(f"cannot find {carrier}")

    print(f"[run] config={args.config} model={args.model} -> {out_dir}", flush=True)
    print(f"[run] {len(wavs)} file(s)", flush=True)

    pending: List[Path] = []
    for wav in wavs:
        if (out_dir / f"{wav.stem}.json").is_file() and not args.redo:
            print(f"[run] skip (done): {wav.name}", flush=True)
            continue
        pending.append(wav)

    if not pending:
        print("[run] nothing to do", flush=True)
        return

    failures: List[str] = []
    for index, wav in enumerate(pending, 1):
        command = [
            sys.executable, str(carrier), str(wav),
            "-m", args.model,
            "-o", str(out_dir),
            "-f", "json",
            "--no_config",
            "--beep_off",
        ] + CONFIGS[args.config] + args.extra

        if args.dry_run:
            print("[run] " + " ".join(command), flush=True)
            continue

        print(f"\n[run] [{index}/{len(pending)}] {wav.name}", flush=True)
        started = time.time()
        # The child writes to a pipe, where Python would otherwise pick the
        # locale encoding. On a Japanese Windows install that is cp932, so the
        # media file names whisp-carrier echoes would land in the log as
        # mojibake. The log is the record of the run, so it has to be readable.
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        # Read the child line by line rather than collecting it at the end.
        # A 5h22m file spends ~20 minutes between the [VAD] line and [STT] Done,
        # and with capture_output there was no way to tell a slow run from a hung
        # one. stderr is merged so the log keeps the real ordering of warnings
        # against progress; the previous format appended it under a separator.
        process = subprocess.Popen(
            command, cwd=str(REPO), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        captured: List[str] = []
        last_beat = started
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            captured.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            if any(token in stripped for token in ECHOED):
                # What was actually in force, and what the segment repair had to
                # do. Always shown.
                print("  " + stripped, flush=True)
                last_beat = time.time()
            elif time.time() - last_beat >= HEARTBEAT_SECONDS:
                print("  ... " + stripped, flush=True)
                last_beat = time.time()
        returncode = process.wait()
        elapsed = time.time() - started

        log = out_dir / f"{wav.stem}.log"
        log.write_text("\n".join(captured) + "\n", encoding="utf-8")

        target = out_dir / f"{wav.stem}.json"
        if returncode != 0 or not target.is_file():
            failures.append(wav.name)
            print(f"  FAILED (exit {returncode}); see {log}", flush=True)
            for line in [l for l in captured if l.strip()][-5:]:
                print(f"    {line}", flush=True)
            continue

        segments = len(json.loads(target.read_text(encoding="utf-8")).get("segments") or [])
        print(f"  ok in {elapsed:.0f}s | {segments} segments -> {target.name}", flush=True)

    if failures:
        print(f"\n[run] {len(failures)} failure(s): {', '.join(failures)}", flush=True)
        raise SystemExit(1)
    print("\n[run] done", flush=True)


if __name__ == "__main__":
    main()
