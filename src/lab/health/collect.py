"""Host health checks: take a sample, compare it with thresholds, alert on change.

The numbers come straight from the kernel (``/proc``, ``/sys``) so no extra
dependency is needed. A metric that this host cannot provide is stored as NULL and
its check is skipped: temperature and throttle flags exist only on a Raspberry Pi.

An alert is sent when the set of failing checks *changes*, not on every sample. A
disk that stays full sends one alert, and one more when it recovers.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lab.config import HealthConfig, Settings
from lab.db import file_size_mb
from lab.logs import get_logger
from lab.notify import Notifier
from lab.runs import Run
from lab.timeutil import utc_now

log = get_logger(__name__)

_MEMINFO = Path("/proc/meminfo")
_THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
_MB = 1024 * 1024
# Bits of `vcgencmd get_throttled` that describe the present moment.
_THROTTLE_NOW = {
    0: "under-voltage",
    1: "CPU frequency capped",
    2: "throttled",
    3: "soft temp limit",
}


@dataclass(frozen=True, slots=True)
class HealthSample:
    taken_at: str
    mem_total_mb: int | None
    mem_available_mb: int | None
    swap_used_mb: int | None
    load1: float | None
    load5: float | None
    load15: float | None
    cpus: int
    disk_total_mb: int
    disk_free_mb: int
    db_mb: float
    wal_mb: float
    cpu_temp_c: float | None
    throttled: int | None
    inactive_services: tuple[str, ...] = ()


def read_meminfo(path: Path = _MEMINFO) -> dict[str, int]:
    """/proc/meminfo as {field: MiB}. Empty on systems without it (macOS)."""
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="ascii", errors="replace").splitlines():
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[name] = int(parts[0]) // 1024  # the file reports kB
    except OSError:
        pass
    return values


def _read_cpu_temp(path: Path = _THERMAL) -> float | None:
    try:
        return round(int(path.read_text().strip()) / 1000, 1)  # millidegrees Celsius
    except (OSError, ValueError):
        return None


def _run_quiet(command: list[str]) -> tuple[int, str] | None:
    """Run a system tool with a fixed argument list. None if it is missing or hangs."""
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - fixed program, arguments validated by config
            [executable, *command[1:]], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.returncode, done.stdout.strip()


def _read_throttled() -> int | None:
    """Raspberry Pi firmware flags, e.g. "throttled=0x50005". None anywhere else."""
    result = _run_quiet(["vcgencmd", "get_throttled"])
    if result is None or result[0] != 0 or "=" not in result[1]:
        return None
    try:
        return int(result[1].split("=", 1)[1], 16)
    except ValueError:
        return None


def _inactive_services(names: tuple[str, ...]) -> tuple[str, ...]:
    """The watched systemd units that are not active. Empty if systemd is not there."""
    inactive: list[str] = []
    for name in names:
        result = _run_quiet(["systemctl", "is-active", "--quiet", name])
        if result is not None and result[0] != 0:
            inactive.append(name)
    return tuple(inactive)


def collect(settings: Settings) -> HealthSample:
    """Take one sample of the host's state."""
    meminfo = read_meminfo()
    swap_total, swap_free = meminfo.get("SwapTotal"), meminfo.get("SwapFree")
    try:
        load: tuple[float, float, float] | None = os.getloadavg()
    except (OSError, AttributeError):
        load = None
    root = settings.paths.root
    disk = shutil.disk_usage(root if root.exists() else root.parent)
    db_file = settings.paths.db_file
    return HealthSample(
        taken_at=utc_now(),
        mem_total_mb=meminfo.get("MemTotal"),
        mem_available_mb=meminfo.get("MemAvailable"),
        swap_used_mb=(
            swap_total - swap_free if swap_total is not None and swap_free is not None else None
        ),
        load1=round(load[0], 2) if load else None,
        load5=round(load[1], 2) if load else None,
        load15=round(load[2], 2) if load else None,
        cpus=os.cpu_count() or 1,
        disk_total_mb=disk.total // _MB,
        disk_free_mb=disk.free // _MB,
        db_mb=file_size_mb(db_file),
        wal_mb=file_size_mb(db_file.with_name(db_file.name + "-wal")),
        cpu_temp_c=_read_cpu_temp(),
        throttled=_read_throttled(),
        inactive_services=_inactive_services(settings.health.watch_services),
    )


def check(sample: HealthSample, config: HealthConfig) -> dict[str, str]:
    """Every threshold the sample breaks, as {check name: short sentence}. Empty = healthy."""
    problems: dict[str, str] = {}
    if sample.mem_available_mb is not None and (
        sample.mem_available_mb < config.min_mem_available_mb
    ):
        problems["memory"] = (
            f"memory available {sample.mem_available_mb} MB < {config.min_mem_available_mb} MB"
        )
    if sample.swap_used_mb is not None and sample.swap_used_mb > config.max_swap_used_mb:
        problems["swap"] = f"swap used {sample.swap_used_mb} MB > {config.max_swap_used_mb} MB"
    # The 5-minute average, so one busy moment does not page anyone.
    if sample.load5 is not None and sample.load5 / sample.cpus > config.max_load_per_cpu:
        problems["load"] = (
            f"load {sample.load5} on {sample.cpus} CPUs > {config.max_load_per_cpu} per CPU"
        )
    if sample.disk_free_mb < config.min_disk_free_mb:
        problems["disk"] = f"disk free {sample.disk_free_mb} MB < {config.min_disk_free_mb} MB"
    if sample.wal_mb > config.max_wal_mb:
        problems["wal"] = f"WAL file {sample.wal_mb} MB > {config.max_wal_mb} MB"
    if sample.cpu_temp_c is not None and sample.cpu_temp_c > config.max_cpu_temp_c:
        problems["temperature"] = (
            f"CPU temperature {sample.cpu_temp_c} C > {config.max_cpu_temp_c} C"
        )
    if sample.throttled:
        active = [name for bit, name in _THROTTLE_NOW.items() if sample.throttled & (1 << bit)]
        if active:
            problems["throttled"] = "Raspberry Pi firmware reports: " + ", ".join(active)
    for service in sample.inactive_services:
        problems[f"service:{service}"] = f"service {service} is not active"
    return problems


def save(
    conn: sqlite3.Connection, run_id: int, sample: HealthSample, problems: dict[str, str]
) -> None:
    conn.execute(
        "INSERT INTO health_samples (run_id, taken_at, mem_total_mb, mem_available_mb, "
        "swap_used_mb, load1, load5, load15, cpus, disk_total_mb, disk_free_mb, db_mb, wal_mb, "
        "cpu_temp_c, throttled, problems) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            sample.taken_at,
            sample.mem_total_mb,
            sample.mem_available_mb,
            sample.swap_used_mb,
            sample.load1,
            sample.load5,
            sample.load15,
            sample.cpus,
            sample.disk_total_mb,
            sample.disk_free_mb,
            sample.db_mb,
            sample.wal_mb,
            sample.cpu_temp_c,
            sample.throttled,
            json.dumps(problems, ensure_ascii=False, sort_keys=True),
        ),
    )


def previous_problems(conn: sqlite3.Connection) -> set[str]:
    """Names of the checks that were failing at the last sample."""
    row = conn.execute("SELECT problems FROM health_samples ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return set()
    loaded = json.loads(row["problems"])
    return set(loaded) if isinstance(loaded, dict) else set()


def run_health(
    settings: Settings, conn: sqlite3.Connection, run: Run, notifier: Notifier
) -> tuple[HealthSample, dict[str, str]]:
    """Sample, check, store, and alert if the set of failing checks changed."""
    sample = collect(settings)
    problems = check(sample, settings.health)
    before = previous_problems(conn)
    save(conn, run.id, sample, problems)
    run.set("problems", len(problems))
    if set(problems) != before:
        if problems:
            notifier.send("shuru: host health", "\n".join(problems.values()), priority=4)
        else:
            notifier.send("shuru: host health", "All checks are passing again.", priority=2)
    for message in problems.values():
        log.warning("health: %s", message)
    return sample, problems
