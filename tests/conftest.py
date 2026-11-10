"""Shared fixtures: an isolated settings object and a migrated database per test."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from lab.config import Settings, load_settings
from lab.db import open_database
from lab.runs import Run

SettingsFactory = Callable[..., Settings]


@pytest.fixture
def make_settings(tmp_path: Path) -> SettingsFactory:
    """Build Settings from TOML text, with the data root inside the test's temp folder."""

    def factory(toml: str = "", **env: str) -> Settings:
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        path = config_dir / "settings.toml"
        path.write_text(toml, encoding="utf-8")
        return load_settings(path, {"LAB_ROOT": str(tmp_path / "data"), **env})

    return factory


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    # No politeness delay: tests talk to localhost or to a mock transport.
    return make_settings("[http]\ndefault_delay_s = 0.0\n")


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = open_database(settings.paths.db_file)
    yield connection
    connection.close()


@pytest.fixture
def run() -> Run:
    """A Run handle that is not tied to a row, for functions that only record metrics."""
    return Run(id=0, job="test")
