"""Load settings.toml into typed, read-only objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.toml"


class ConfigError(Exception):
    """Settings are missing, malformed or invalid."""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Absolute paths under the data root."""

    root: Path
    db_file: Path
    job_dirs: Mapping[str, Path]

    def job_dir(self, job: str) -> Path:
        """Folder for a job. Raises ConfigError for an unknown job name."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str
    format: Literal["json"]
    max_field_chars: int


@dataclass(frozen=True, slots=True)
class Settings:
    timezone: str
    paths: PathsConfig
    logging: LoggingConfig


def load_settings(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Read the TOML file, apply env overrides, resolve paths and validate.

    path defaults to LAB_SETTINGS, then DEFAULT_SETTINGS_PATH. env defaults to os.environ.
    """
    raise NotImplementedError


def _read_toml(path: Path) -> dict[str, Any]:
    raise NotImplementedError


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Return a copy of raw with LAB_ROOT and LAB_LOG_LEVEL applied."""
    raise NotImplementedError


def default_data_root(env: Mapping[str, str]) -> Path:
    """$XDG_DATA_HOME/shuru, or ~/.local/share/shuru."""
    raise NotImplementedError


def _resolve_paths(raw_paths: Mapping[str, Any]) -> PathsConfig:
    """Make paths absolute under root. Reject any that escape it."""
    raise NotImplementedError


def _validate(raw: Mapping[str, Any]) -> None:
    """Raise ConfigError listing every problem found."""
    raise NotImplementedError


def get_settings() -> Settings:
    """Settings for this process, loaded on first call."""
    raise NotImplementedError
