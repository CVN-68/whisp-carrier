"""
config.py
YAML profile support for whisp-carrier.

Motivation: the calling application (Amatsukaze) stores its extra-options string
in its own settings UI, which is tedious to edit and easy to get wrong while
comparing transcription settings. A YAML file that sits next to the script (or
next to the exe) lets a profile be changed without touching the caller at all.

File layout
-----------
    override: true             # let this file win over command line options
    active_profile: anime      # which entry under profiles: to apply

    beam_size: 10              # options listed flat apply to every profile
    best_of: 10

    profiles:
      anime:
        language: ja
        standard_asia: true
      race:
        ff_loudnorm: true

Resolution order: flat top-level options first, then the active profile on top.
Precedence against the command line depends on override mode.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

CONFIG_NAMES = ("whisp-carrier.yaml", "whisp-carrier.yml")

# Top-level keys that control the config file itself rather than a CLI option.
RESERVED_KEYS = {"override", "active_profile", "profiles"}

# Options that must not come from the config file: they select the input, ask
# for immediate output, or control config loading itself.
BLOCKED_DESTS = {
    "audio", "version", "checkcuda", "list_models",
    "config", "no_config", "profile", "config_override",
    # A model conversion is a one-off action, not a setting: from a config file
    # it would silently reconvert on every single run.
    "reconvert",
}

_TRUTHY = {"true", "yes", "on", "1"}
_FALSY = {"false", "no", "off", "0"}


class ConfigError(Exception):
    """Raised for anything wrong with the config file, with a usable message."""


@dataclass
class ConfigResult:
    """What the config file actually did, for logging."""
    path: Optional[Path] = None
    profile: Optional[str] = None
    override: bool = False
    applied: Dict[str, Any] = field(default_factory=dict)
    overridden: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    ignored: Dict[str, Any] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return self.path is not None


# ─────────────────────────────────────────────
# Discovery and loading
# ─────────────────────────────────────────────

def base_dir() -> Path:
    """Directory to look in for a config file.

    For a PyInstaller build this is the folder holding the exe (user editable),
    not the temporary _MEIPASS extraction directory.
    """
    if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def discover(explicit: Optional[str]) -> Optional[Path]:
    """Return the config file to use, or None."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path.resolve()

    for name in CONFIG_NAMES:
        candidate = base_dir() / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def load(path: Path) -> Dict[str, Any]:
    """Parse a YAML config file into a mapping."""
    try:
        import yaml
    except ImportError as e:
        raise ConfigError(
            f"reading a config file requires PyYAML: {e}\n"
            "  pip install PyYAML"
        ) from e

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ConfigError(f"cannot parse config file ({path}): {e}") from e

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"top level of the config file must be a mapping ({path}): "
            f"found {type(raw).__name__}"
        )
    return raw


def resolve_options(
    raw: Dict[str, Any],
    profile_override: Optional[str],
) -> Tuple[Dict[str, Any], Optional[str], bool]:
    """Flatten the config into a single option mapping.

    Returns (options, active_profile_name, override_mode).
    """
    override = _as_bool(raw.get("override", False), "override")

    profiles = raw.get("profiles") or {}
    if profiles and not isinstance(profiles, dict):
        raise ConfigError("'profiles' must be a mapping keyed by profile name")

    active = profile_override or raw.get("active_profile")
    if active is not None:
        active = str(active)

    # Flat top-level keys form the base for every profile.
    options: Dict[str, Any] = {
        k: v for k, v in raw.items() if k not in RESERVED_KEYS
    }

    if active:
        if active not in profiles:
            known = ", ".join(sorted(profiles)) or "(none defined)"
            raise ConfigError(
                f"profile '{active}' is not in the config file. defined: {known}"
            )
        entry = profiles[active]
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            raise ConfigError(f"profile '{active}' must be a mapping")
        options.update(entry)

    return options, active, override


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ConfigError(f"'{key}' expects true or false: {value!r}")


# ─────────────────────────────────────────────
# Which options came from the command line
# ─────────────────────────────────────────────

def cli_specified(parser_factory, argv: List[str]) -> Set[str]:
    """Return the dest names that were given explicitly on the command line.

    Reparses with every default suppressed, so only what the user actually typed
    lands in the namespace. Abbreviated long options are handled correctly this
    way. Falls back to a direct token scan if that reparse fails for any reason.
    """
    try:
        probe = parser_factory()
        for action in probe._actions:
            action.default = argparse.SUPPRESS
        namespace, _ = probe.parse_known_args(argv)
        return set(vars(namespace))
    except SystemExit:
        raise
    except Exception:
        return _scan_argv(parser_factory(), argv)


def _scan_argv(parser: argparse.ArgumentParser, argv: List[str]) -> Set[str]:
    """Map option tokens in argv to their dest names."""
    option_actions = parser._option_string_actions
    found: Set[str] = set()
    for token in argv:
        if not isinstance(token, str) or token in ("-", "--") or not token.startswith("-"):
            continue
        name = token.split("=", 1)[0]
        action = option_actions.get(name)
        if action is not None:
            found.add(action.dest)
    return found


# ─────────────────────────────────────────────
# Type coercion against the parser definition
# ─────────────────────────────────────────────

def _action_map(parser: argparse.ArgumentParser) -> Dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS
    }


def _is_flag(action: argparse.Action) -> bool:
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))


def _coerce_scalar(action: argparse.Action, key: str, value: Any) -> Any:
    if _is_flag(action):
        return _as_bool(value, key)

    if action.type is not None and isinstance(value, str):
        try:
            return action.type(value)
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"cannot interpret the value of '{key}': {value!r} ({e})") from e

    if action.type is int:
        if isinstance(value, bool):
            raise ConfigError(f"'{key}' expects an integer: {value!r}")
        if isinstance(value, float):
            if not float(value).is_integer():
                raise ConfigError(f"'{key}' expects an integer: {value!r}")
            return int(value)

    if action.type is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)

    return value


def _coerce(action: argparse.Action, key: str, value: Any) -> Any:
    if value is None:
        return None

    # Flags carry nargs == 0, so they must be settled before the list check.
    if _is_flag(action):
        return _as_bool(value, key)

    wants_list = action.nargs in ("*", "+") or (
        isinstance(action.nargs, int) and action.nargs >= 1
    )
    if wants_list:
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        coerced = [_coerce_scalar(action, key, v) for v in items]
        if isinstance(action.nargs, int) and len(coerced) != action.nargs:
            raise ConfigError(
                f"'{key}' expects exactly {action.nargs} values: {value!r}"
            )
        return coerced

    if isinstance(value, (list, tuple)):
        raise ConfigError(f"'{key}' does not take a list: {value!r}")
    return _coerce_scalar(action, key, value)


def _validate_choices(action: argparse.Action, key: str, value: Any) -> None:
    if action.choices is None:
        return
    candidates = value if isinstance(value, list) else [value]
    for item in candidates:
        if item not in action.choices:
            allowed = ", ".join(str(c) for c in action.choices)
            raise ConfigError(f"invalid value for '{key}': {item!r} (choose from: {allowed})")


# ─────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────

def apply(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    parser_factory,
    argv: List[str],
) -> ConfigResult:
    """Load a config file if present and fold its values into args.

    Config values normally fill in only what the command line left alone. In
    override mode the config wins even against explicit command line options.
    """
    if getattr(args, "no_config", False):
        return ConfigResult()

    path = discover(getattr(args, "config", None))
    if path is None:
        return ConfigResult()

    raw = load(path)
    options, profile, file_override = resolve_options(raw, getattr(args, "profile", None))
    override = bool(getattr(args, "config_override", False)) or file_override

    result = ConfigResult(path=path, profile=profile, override=override)
    if not options:
        return result

    actions = _action_map(parser)
    from_cli = cli_specified(parser_factory, argv)

    for key in sorted(options):
        value = options[key]
        dest = str(key).replace("-", "_")

        if dest in BLOCKED_DESTS:
            raise ConfigError(f"'{key}' cannot be set from the config file")
        action = actions.get(dest)
        if action is None:
            raise ConfigError(
                f"unknown option '{key}'. use the option names from --help "
                "with the leading '--' removed"
            )

        coerced = _coerce(action, str(key), value)
        _validate_choices(action, str(key), coerced)

        if dest in from_cli and not override:
            result.ignored[dest] = coerced
            continue

        if dest in from_cli:
            result.overridden[dest] = (getattr(args, dest, None), coerced)

        setattr(args, dest, coerced)
        result.applied[dest] = coerced

    return result


def describe(result: ConfigResult) -> List[str]:
    """Human readable log lines for what the config file did."""
    if not result.loaded:
        return []

    header = f"[CONFIG] {result.path}"
    if result.profile:
        header += f" | profile={result.profile}"
    header += f" | override={'on' if result.override else 'off'}"
    lines = [header]

    for key in sorted(result.applied):
        value = result.applied[key]
        if key in result.overridden:
            cli_value, _ = result.overridden[key]
            lines.append(f"[CONFIG]   {key} = {value!r}  (overrides CLI {cli_value!r})")
        else:
            lines.append(f"[CONFIG]   {key} = {value!r}")

    for key in sorted(result.ignored):
        lines.append(
            f"[CONFIG]   {key}: kept the CLI value, ignoring config "
            f"{result.ignored[key]!r} (enable override to reverse this)"
        )

    if not result.applied and not result.ignored:
        lines.append("[CONFIG]   no options to apply")

    return lines
