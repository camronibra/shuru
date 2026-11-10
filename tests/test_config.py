from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from conftest import SettingsFactory
from lab.config import (
    DEFAULT_SETTINGS_PATH,
    ConfigError,
    Settings,
    default_data_root,
    load_dotenv,
    load_settings,
)


def test_shipped_settings_file_matches_the_defaults_in_code(tmp_path: Path) -> None:
    """config/settings.toml documents the defaults, so it must not drift from them."""
    env = {"LAB_ROOT": str(tmp_path)}
    shipped = load_settings(DEFAULT_SETTINGS_PATH, env)
    empty = tmp_path / "settings.toml"
    empty.write_text("")
    defaults = load_settings(empty, env)
    for field in dataclasses.fields(Settings):
        if field.name not in ("config_dir", "paths"):
            assert getattr(shipped, field.name) == getattr(defaults, field.name), field.name
    assert shipped.paths == defaults.paths


def test_environment_overrides(make_settings: SettingsFactory, tmp_path: Path) -> None:
    settings = make_settings(
        '[logging]\nlevel = "INFO"\n', LAB_LOG_LEVEL="debug", NTFY_TOPIC="secret-topic"
    )
    assert settings.logging.level == "DEBUG"
    assert settings.notify.topic == "secret-topic"
    assert settings.paths.root == (tmp_path / "data").resolve()
    assert settings.paths.db_file == settings.paths.root / "db" / "lab.db"


def test_topic_in_the_file_is_ignored(make_settings: SettingsFactory) -> None:
    # The topic is a secret; only the environment may set it.
    with pytest.raises(ConfigError, match="unknown key 'topics'"):
        make_settings('[notify]\ntopics = "x"\n')
    assert make_settings('[notify]\ntopic = "from-file"\n').notify.topic == ""


def test_all_problems_are_reported_together(make_settings: SettingsFactory) -> None:
    with pytest.raises(ConfigError) as error:
        make_settings(
            '[http]\ntimeout_s = "slow"\nmax_retries = 99\n[dedup]\nnum_perm = 10\nbands = 3\n'
            "[nonsense]\nx = 1\n[ingest]\nworkers = 2\n"
        )
    message = str(error.value)
    assert "timeout_s must be a float" in message
    assert "max_retries must be between 0 and 5" in message
    assert "num_perm must be a multiple of bands" in message
    assert "unknown section [nonsense]" in message
    assert "[ingest] unknown key 'workers'" in message


def test_paths_may_not_escape_the_data_root(make_settings: SettingsFactory) -> None:
    with pytest.raises(ConfigError, match="escapes the data root"):
        make_settings('[paths]\ndb_file = "../outside.db"\n')
    with pytest.raises(ConfigError, match="escapes the data root"):
        make_settings('[paths.dirs]\nmodels = "/etc"\n')


def test_default_data_root_follows_xdg() -> None:
    assert default_data_root({"XDG_DATA_HOME": "/x"}) == Path("/x/shuru")
    assert default_data_root({"HOME": "/home/u"}) == Path("/home/u/.local/share/shuru")


def test_dotenv_does_not_replace_existing_values(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nLAB_ROOT=/from/file\nNTFY_TOPIC='quoted'\nbroken line\n")
    env = {"LAB_ROOT": "/already/set"}
    load_dotenv(dotenv, env)
    assert env == {"LAB_ROOT": "/already/set", "NTFY_TOPIC": "quoted"}
    load_dotenv(tmp_path / "missing.env", env)  # a missing file is fine
