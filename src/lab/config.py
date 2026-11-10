"""Load settings.toml into typed, read-only objects.

Each ``[section]`` of the file maps to one frozen dataclass below. The dataclass
defaults are the real defaults; ``config/settings.toml`` repeats them so they can be
read and changed in one place. Loading collects every problem it finds (unknown keys,
wrong types, values out of range) and reports them together in one ConfigError.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.toml"
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

DIR_NAMES = ("db", "models", "exports", "reports", "backups", "labelling")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_FORMATS = ("text", "json")

_SERVICE_NAME = re.compile(r"^[\w@.\-]+$")
_T = TypeVar("_T")


class ConfigError(Exception):
    """Settings are missing, malformed or invalid."""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Absolute paths under the data root."""

    root: Path
    db_file: Path
    dirs: Mapping[str, Path]

    def dir(self, name: str) -> Path:
        """Folder for one kind of output. Raises ConfigError for an unknown name."""
        try:
            return self.dirs[name]
        except KeyError:
            raise ConfigError(f"unknown data folder {name!r}") from None

    @property
    def lock_file(self) -> Path:
        """Lock file that stops two heavy jobs from running at once."""
        return self.dirs["db"] / "jobs.lock"


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "text"
    max_field_chars: int = 2000


@dataclass(frozen=True, slots=True)
class HttpConfig:
    user_agent: str = "shuru/0.1 (+https://github.com/camronibra/shuru)"
    timeout_s: float = 20.0
    max_bytes: int = 3_000_000
    default_delay_s: float = 2.0
    max_retries: int = 2
    max_redirects: int = 5
    max_host_failures: int = 5


@dataclass(frozen=True, slots=True)
class IngestConfig:
    max_workers: int = 8
    max_pages_per_source: int = 200
    max_seconds_per_source: int = 900
    batch_size: int = 50
    retry_after_hours: int = 24
    max_tries: int = 3


@dataclass(frozen=True, slots=True)
class CleanConfig:
    script: str = "arabic"
    min_script_ratio: float = 0.6
    min_chars: int = 200
    max_chars: int = 100_000


@dataclass(frozen=True, slots=True)
class DedupConfig:
    shingle_size: int = 3
    num_perm: int = 64
    bands: int = 16
    threshold: float = 0.8


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    target_chars: int = 0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hash_bits: int = 18
    ngram_min: int = 1
    ngram_max: int = 4
    epochs: int = 5
    learning_rate: float = 0.5
    l2: float = 1e-6
    seed: int = 13
    min_sentence_chars: int = 30
    max_sentence_chars: int = 400
    max_examples_per_source: int = 20_000
    max_examples: int = 200_000
    test_date_fraction: float = 0.2
    test_from: str = ""
    max_score_chars: int = 2000


@dataclass(frozen=True, slots=True)
class NewsConfig:
    max_items_per_feed: int = 100


@dataclass(frozen=True, slots=True)
class DriftConfig:
    window_days: int = 7
    min_items: int = 50
    psi_alert: float = 0.25


@dataclass(frozen=True, slots=True)
class LabellingConfig:
    queue_size: int = 100
    max_per_feed: int = 25


@dataclass(frozen=True, slots=True)
class HealthConfig:
    min_mem_available_mb: int = 150
    max_swap_used_mb: int = 512
    max_load_per_cpu: float = 2.0
    min_disk_free_mb: int = 2000
    max_wal_mb: int = 200
    max_cpu_temp_c: float = 80.0
    watch_services: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NotifyConfig:
    ntfy_server: str = "https://ntfy.sh"
    # Only ever set from the NTFY_TOPIC environment variable, never from the file.
    topic: str = ""


@dataclass(frozen=True, slots=True)
class BackupConfig:
    keep: int = 7


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything a job needs to know about its environment."""

    config_dir: Path
    paths: PathsConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    http: HttpConfig = field(default_factory=HttpConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    labelling: LabellingConfig = field(default_factory=LabellingConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)


# Section name in the TOML file -> dataclass that holds it.
_SECTIONS: dict[str, type[Any]] = {
    "logging": LoggingConfig,
    "http": HttpConfig,
    "ingest": IngestConfig,
    "clean": CleanConfig,
    "dedup": DedupConfig,
    "corpus": CorpusConfig,
    "model": ModelConfig,
    "news": NewsConfig,
    "drift": DriftConfig,
    "labelling": LabellingConfig,
    "health": HealthConfig,
    "notify": NotifyConfig,
    "backup": BackupConfig,
}


def load_settings(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Read the TOML file, apply env overrides, resolve paths and validate.

    path defaults to LAB_SETTINGS, then DEFAULT_SETTINGS_PATH. env defaults to os.environ.
    """
    env = os.environ if env is None else env
    if path is None:
        path = Path(env["LAB_SETTINGS"]) if env.get("LAB_SETTINGS") else DEFAULT_SETTINGS_PATH
    path = path.expanduser().resolve()
    raw = _apply_env_overrides(_read_toml(path), env)

    problems: list[str] = []
    for name in raw:
        if name != "paths" and name not in _SECTIONS:
            problems.append(f"unknown section [{name}]")

    sections = {
        name: _build(cls, raw.get(name, {}), name, problems) for name, cls in _SECTIONS.items()
    }
    paths = _resolve_paths(raw.get("paths", {}), env, problems)
    settings = Settings(config_dir=path.parent, paths=paths, **sections)
    _validate(settings, problems)
    if problems:
        raise ConfigError(f"{path}: " + "; ".join(problems))
    return settings


def read_toml(path: Path) -> dict[str, Any]:
    """Parse one TOML file. Raises ConfigError if it is missing or malformed."""
    return _read_toml(path)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Return a copy of raw with the LAB_* and NTFY_* environment variables applied."""
    merged = {name: dict(section) for name, section in raw.items() if isinstance(section, dict)}
    for name, section in raw.items():
        if not isinstance(section, dict):
            merged[name] = section  # reported later as an unknown section
    if env.get("LAB_ROOT"):
        merged.setdefault("paths", {})["root"] = env["LAB_ROOT"]
    if env.get("LAB_LOG_LEVEL"):
        merged.setdefault("logging", {})["level"] = env["LAB_LOG_LEVEL"].upper()
    if env.get("LAB_LOG_FORMAT"):
        merged.setdefault("logging", {})["format"] = env["LAB_LOG_FORMAT"].lower()
    if env.get("NTFY_SERVER"):
        merged.setdefault("notify", {})["ntfy_server"] = env["NTFY_SERVER"]
    notify = merged.setdefault("notify", {})
    notify.pop("topic", None)  # a topic in the file would be a committed secret
    notify["topic"] = env.get("NTFY_TOPIC", "")
    return merged


def default_data_root(env: Mapping[str, str]) -> Path:
    """$XDG_DATA_HOME/shuru, or ~/.local/share/shuru."""
    base = env.get("XDG_DATA_HOME")
    if base:
        return Path(base).expanduser() / "shuru"
    home = Path(env["HOME"]) if env.get("HOME") else Path.home()
    return home / ".local" / "share" / "shuru"


def _resolve_paths(
    raw_paths: Mapping[str, Any], env: Mapping[str, str], problems: list[str]
) -> PathsConfig:
    """Make paths absolute under root. Reject any that escape it."""
    for key in raw_paths:
        if key not in ("root", "db_file", "dirs"):
            problems.append(f"[paths] unknown key {key!r}")

    root_value = raw_paths.get("root", "")
    if not isinstance(root_value, str):
        problems.append("[paths] root must be a string")
        root_value = ""
    root = (Path(root_value).expanduser() if root_value else default_data_root(env)).resolve()

    def under_root(value: object, label: str) -> Path:
        if not isinstance(value, str) or not value:
            problems.append(f"[paths] {label} must be a non-empty string")
            return root
        resolved = (root / value).resolve()
        if resolved != root and root not in resolved.parents:
            problems.append(f"[paths] {label} = {value!r} escapes the data root")
            return root
        return resolved

    raw_dirs = raw_paths.get("dirs", {})
    if not isinstance(raw_dirs, dict):
        problems.append("[paths.dirs] must be a table")
        raw_dirs = {}
    for key in raw_dirs:
        if key not in DIR_NAMES:
            problems.append(f"[paths.dirs] unknown key {key!r}")
    dirs = {name: under_root(raw_dirs.get(name, name), f"dirs.{name}") for name in DIR_NAMES}
    db_file = under_root(raw_paths.get("db_file", "db/lab.db"), "db_file")
    return PathsConfig(root=root, db_file=db_file, dirs=dirs)


def _build(cls: type[_T], raw: object, section: str, problems: list[str]) -> _T:
    """Fill one section dataclass from its TOML table, checking key names and types.

    The expected type of each key is the type of its default value.
    """
    if not isinstance(raw, dict):
        problems.append(f"[{section}] must be a table")
        return cls()
    known = {f.name: f.default for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    values: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            problems.append(f"[{section}] unknown key {key!r}")
            continue
        default = known[key]
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, int):
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, float):
            ok = isinstance(value, int | float) and not isinstance(value, bool)
            value = float(value) if ok else value
        elif isinstance(default, tuple):
            ok = isinstance(value, list) and all(isinstance(item, str) for item in value)
            value = tuple(value) if ok else value
        else:
            ok = isinstance(value, str)
        if ok:
            values[key] = value
        else:
            wanted = "list of strings" if isinstance(default, tuple) else type(default).__name__
            problems.append(f"[{section}] {key} must be a {wanted}")
    return cls(**values)


def _validate(s: Settings, problems: list[str]) -> None:
    """Append a message for every value that is out of range."""

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    check(s.logging.level in LOG_LEVELS, f"[logging] level must be one of {LOG_LEVELS}")
    check(s.logging.format in LOG_FORMATS, f"[logging] format must be one of {LOG_FORMATS}")
    check(s.logging.max_field_chars >= 100, "[logging] max_field_chars must be >= 100")

    check(bool(s.http.user_agent.strip()), "[http] user_agent must not be empty")
    check(s.http.timeout_s > 0, "[http] timeout_s must be > 0")
    check(s.http.max_bytes >= 1024, "[http] max_bytes must be >= 1024")
    check(s.http.default_delay_s >= 0, "[http] default_delay_s must be >= 0")
    check(0 <= s.http.max_retries <= 5, "[http] max_retries must be between 0 and 5")
    check(0 <= s.http.max_redirects <= 10, "[http] max_redirects must be between 0 and 10")
    check(s.http.max_host_failures >= 1, "[http] max_host_failures must be >= 1")

    check(1 <= s.ingest.max_workers <= 32, "[ingest] max_workers must be between 1 and 32")
    check(s.ingest.max_pages_per_source >= 1, "[ingest] max_pages_per_source must be >= 1")
    check(s.ingest.max_seconds_per_source >= 1, "[ingest] max_seconds_per_source must be >= 1")
    check(s.ingest.batch_size >= 1, "[ingest] batch_size must be >= 1")
    check(s.ingest.retry_after_hours >= 0, "[ingest] retry_after_hours must be >= 0")
    check(s.ingest.max_tries >= 1, "[ingest] max_tries must be >= 1")

    check(0 <= s.clean.min_script_ratio <= 1, "[clean] min_script_ratio must be between 0 and 1")
    check(s.clean.min_chars >= 1, "[clean] min_chars must be >= 1")
    check(s.clean.max_chars >= s.clean.min_chars, "[clean] max_chars must be >= min_chars")

    check(s.dedup.shingle_size >= 1, "[dedup] shingle_size must be >= 1")
    check(s.dedup.bands >= 1, "[dedup] bands must be >= 1")
    check(
        s.dedup.num_perm >= s.dedup.bands and s.dedup.num_perm % s.dedup.bands == 0,
        "[dedup] num_perm must be a multiple of bands",
    )
    check(0 < s.dedup.threshold <= 1, "[dedup] threshold must be in (0, 1]")

    check(s.corpus.target_chars >= 0, "[corpus] target_chars must be >= 0")

    check(10 <= s.model.hash_bits <= 24, "[model] hash_bits must be between 10 and 24")
    check(
        1 <= s.model.ngram_min <= s.model.ngram_max <= 8,
        "[model] need 1 <= ngram_min <= ngram_max <= 8",
    )
    check(s.model.epochs >= 1, "[model] epochs must be >= 1")
    check(s.model.learning_rate > 0, "[model] learning_rate must be > 0")
    check(s.model.l2 >= 0, "[model] l2 must be >= 0")
    check(
        1 <= s.model.min_sentence_chars <= s.model.max_sentence_chars,
        "[model] need 1 <= min_sentence_chars <= max_sentence_chars",
    )
    check(s.model.max_examples_per_source >= 1, "[model] max_examples_per_source must be >= 1")
    check(s.model.max_examples >= 1, "[model] max_examples must be >= 1")
    check(0 < s.model.test_date_fraction < 1, "[model] test_date_fraction must be in (0, 1)")
    check(
        not s.model.test_from or bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.model.test_from)),
        "[model] test_from must be empty or YYYY-MM-DD",
    )
    check(s.model.max_score_chars >= 1, "[model] max_score_chars must be >= 1")

    check(s.news.max_items_per_feed >= 1, "[news] max_items_per_feed must be >= 1")
    check(s.drift.window_days >= 1, "[drift] window_days must be >= 1")
    check(s.drift.min_items >= 1, "[drift] min_items must be >= 1")
    check(s.drift.psi_alert > 0, "[drift] psi_alert must be > 0")
    check(s.labelling.queue_size >= 1, "[labelling] queue_size must be >= 1")
    check(s.labelling.max_per_feed >= 1, "[labelling] max_per_feed must be >= 1")

    check(s.health.max_load_per_cpu > 0, "[health] max_load_per_cpu must be > 0")
    for service in s.health.watch_services:
        check(
            bool(_SERVICE_NAME.match(service)),
            f"[health] watch_services: {service!r} is not a valid unit name",
        )
    check(
        s.notify.ntfy_server.startswith(("https://", "http://")),
        "[notify] ntfy_server must be an http(s) URL",
    )
    check(s.backup.keep >= 1, "[backup] keep must be >= 1")


def load_dotenv(path: Path, env: dict[str, str] | None = None) -> None:
    """Copy KEY=VALUE lines from a .env file into env, without replacing existing keys.

    A small reader for local use: no quoting rules beyond stripping one pair of quotes,
    no variable expansion. A missing file is not an error.
    """
    target = os.environ if env is None else env
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in target:
            target[key] = value


def ensure_dirs(paths: PathsConfig) -> None:
    """Create the data root and its folders if they are missing."""
    for folder in (paths.root, paths.db_file.parent, *paths.dirs.values()):
        folder.mkdir(parents=True, exist_ok=True)
