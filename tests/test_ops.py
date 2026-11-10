"""Health checks, alerts, backups and the command line."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from conftest import SettingsFactory
from lab.backup import backup_database
from lab.cli import main
from lab.config import HealthConfig, Settings
from lab.health.collect import HealthSample, check, collect, read_meminfo
from lab.notify import Notifier

HEALTHY = HealthSample(
    taken_at="x", mem_total_mb=1900, mem_available_mb=900, swap_used_mb=10, load1=0.5,
    load5=0.5, load15=0.5, cpus=4, disk_total_mb=100_000, disk_free_mb=50_000, db_mb=10.0,
    wal_mb=1.0, cpu_temp_c=55.0, throttled=0,
)  # fmt: skip


def test_healthy_sample_has_no_problems() -> None:
    assert check(HEALTHY, HealthConfig()) == {}


def test_each_threshold() -> None:
    from dataclasses import replace

    sick = replace(
        HEALTHY, mem_available_mb=100, swap_used_mb=900, load5=12.0, disk_free_mb=500,
        wal_mb=300.0, cpu_temp_c=85.0, throttled=0x50005, inactive_services=("pihole-FTL",),
    )  # fmt: skip
    problems = check(sick, HealthConfig())
    assert set(problems) == {
        "memory", "swap", "load", "disk", "wal", "temperature", "throttled", "service:pihole-FTL",
    }  # fmt: skip
    assert problems["throttled"] == "Raspberry Pi firmware reports: under-voltage, throttled"


def test_missing_metrics_are_skipped_not_failed() -> None:
    from dataclasses import replace

    bare = replace(HEALTHY, mem_available_mb=None, swap_used_mb=None, load5=None, cpu_temp_c=None)
    assert check(bare, HealthConfig()) == {}


def test_collect_reads_this_host(settings: Settings) -> None:
    sample = collect(settings)
    assert sample.cpus >= 1 and sample.disk_total_mb > 0
    if Path("/proc/meminfo").exists():
        assert sample.mem_total_mb and sample.mem_total_mb > 0
    assert read_meminfo(Path("/no/such/file")) == {}


def test_notifier_posts_json_and_never_raises() -> None:
    received: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    notifier = Notifier("https://ntfy.test/", "my-topic", client=client)
    assert notifier.send("title\x1b[31m", "body\x00text", priority=9)
    assert received == [
        {"topic": "my-topic", "title": "title [31m", "message": "body text", "priority": 5}
    ]
    failing = Notifier("https://ntfy.test", "t", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))))  # fmt: skip
    assert failing.send("a", "b") is False
    assert Notifier("https://ntfy.test", "").send("a", "b") is False  # no topic: log only


def test_backup_is_consistent_compressed_and_pruned(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    conn.execute("INSERT INTO runs (job, status, started_at) VALUES ('x', 'ok', 'now')")
    folder = tmp_path / "backups"
    for stamp in ("20200101T000000Z", "20200102T000000Z", "20200103T000000Z"):
        (folder / f"lab-{stamp}.db.gz").parent.mkdir(exist_ok=True)
        (folder / f"lab-{stamp}.db.gz").write_bytes(b"old")
    path = backup_database(conn, folder, keep=2)
    assert sorted(p.name for p in folder.iterdir()) == ["lab-20200103T000000Z.db.gz", path.name]
    restored = tmp_path / "restored.db"
    restored.write_bytes(gzip.decompress(path.read_bytes()))
    copy = sqlite3.connect(restored)
    assert copy.execute("SELECT job FROM runs").fetchall() == [("x",)]


def test_cli_end_to_end_without_network(
    make_settings: SettingsFactory,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_settings("[health]\nmin_disk_free_mb = 1\nmax_load_per_cpu = 1000.0\n")
    settings_file = str(tmp_path / "config" / "settings.toml")
    monkeypatch.setenv("LAB_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.chdir(tmp_path)

    def shuru(*args: str) -> tuple[int, str]:
        code = main(["--settings", settings_file, *args])
        return code, capsys.readouterr().out

    assert shuru("init")[0] == 0
    code, out = shuru("sources")
    assert code == 0 and "No sources configured" in out
    code, out = shuru("ingest")
    assert code == 0 and "0 new document(s)" in out
    assert shuru("health")[0] in (0, 1)  # 1 = a real problem on this machine, still a valid run
    assert shuru("backup")[0] == 0
    code, out = shuru("runs")
    assert code == 0 and "ingest" in out and "backup" in out
    code, _ = shuru("train", "--force")
    assert code == 1  # empty corpus: a clear error, not a traceback
    code, out = shuru("runs", "--id", "1")
    assert code == 0 and "run 1: ingest ok" in out
    assert shuru("schema")[1].startswith("-- Generated")


def test_cli_reports_bad_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "settings.toml"
    bad.write_text("[http]\ntimeout_s = -1\n")
    assert main(["--settings", str(bad), "init"]) == 2
    assert "timeout_s must be > 0" in capsys.readouterr().err
