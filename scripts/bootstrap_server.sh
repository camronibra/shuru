#!/usr/bin/env bash
# Optional server setup for a Debian-family Linux host with systemd (for example
# Raspberry Pi OS Bookworm). Not needed to run the project locally.
#
#   sudo ./scripts/bootstrap_server.sh            # install or update
#
# Every step is safe to re-run. Nothing is downloaded with `curl | sh`: Python
# packages come from PyPI through pip, pinned by the hashes in requirements.lock.
#
# Layout:
#   /opt/shuru        code and virtualenv, owned by root (the jobs cannot modify it)
#   /srv/lab          data root, owned by the `shuru` user
#   /etc/shuru/env    secrets (NTFY_TOPIC), mode 0640 root:shuru
set -euo pipefail

LAB_ROOT="${LAB_ROOT:-/srv/lab}"
REPO_DIR="${REPO_DIR:-/opt/shuru}"
REPO_URL="${REPO_URL:-https://github.com/camronibra/shuru.git}"
SERVICE_USER="shuru"
DATA_DIRS=(db models exports reports backups labelling)
JOBS=(health ingest news score stats backup)

log() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'bootstrap: %s\n' "$*" >&2; exit 1; }

require_root() {
    [[ ${EUID} -eq 0 ]] || die "run as root (sudo $0)"
}

check_platform() {
    [[ $(uname -s) == Linux ]] || die "Linux only"
    command -v systemctl >/dev/null || die "systemd is required"
    command -v apt-get >/dev/null || die "apt is required (Debian, Ubuntu, Raspberry Pi OS)"
}

install_packages() {
    log "installing system packages"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv sqlite3 git >/dev/null
    python3 - <<'PY' || die "Python 3.11 or newer is required"
import sqlite3, sys
assert sys.version_info >= (3, 11), sys.version
assert sqlite3.sqlite_version_info >= (3, 38, 0), sqlite3.sqlite_version
PY
}

create_user() {
    if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
        log "creating system user ${SERVICE_USER}"
        useradd --system --no-create-home --home-dir /nonexistent \
            --shell /usr/sbin/nologin "${SERVICE_USER}"
    fi
    # Raspberry Pi only: the health job reads throttle flags through /dev/vchiq.
    if getent group video >/dev/null; then
        usermod -a -G video "${SERVICE_USER}"
    fi
}

create_dirs() {
    log "creating ${LAB_ROOT}"
    install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${LAB_ROOT}"
    for dir in "${DATA_DIRS[@]}"; do
        install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${LAB_ROOT}/${dir}"
    done
    install -d -m 0750 -o root -g "${SERVICE_USER}" /etc/shuru
    if [[ ! -f /etc/shuru/env ]]; then
        install -m 0640 -o root -g "${SERVICE_USER}" /dev/null /etc/shuru/env
        printf '# NTFY_TOPIC=choose-a-long-random-topic\n' >/etc/shuru/env
    fi
}

deploy_code() {
    if [[ -d "${REPO_DIR}/.git" ]]; then
        log "updating ${REPO_DIR}"
        git -C "${REPO_DIR}" pull --ff-only
    else
        log "cloning into ${REPO_DIR}"
        git clone --depth 1 "${REPO_URL}" "${REPO_DIR}"
    fi
    chown -R root:root "${REPO_DIR}"
    chmod -R go-w "${REPO_DIR}"

    log "installing Python packages (hash-checked)"
    python3 -m venv "${REPO_DIR}/.venv"
    "${REPO_DIR}/.venv/bin/pip" install --quiet --upgrade pip
    "${REPO_DIR}/.venv/bin/pip" install --quiet --require-hashes --only-binary :all: \
        -r "${REPO_DIR}/requirements.lock"
    # The project itself is not built (that would download a build backend without a
    # hash check). The venv is pointed at the source folder, and a two-line launcher
    # stands in for the `shuru` command that `pip install .` would have made.
    local site_packages
    site_packages=$("${REPO_DIR}/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    printf '%s\n' "${REPO_DIR}/src" >"${site_packages}/shuru.pth"
    printf '#!%s\nfrom lab.cli import main\nraise SystemExit(main())\n' \
        "${REPO_DIR}/.venv/bin/python" >"${REPO_DIR}/.venv/bin/shuru"
    chmod 0755 "${REPO_DIR}/.venv/bin/shuru"

    log "creating or migrating the database"
    runuser -u "${SERVICE_USER}" -- env LAB_ROOT="${LAB_ROOT}" \
        LAB_SETTINGS="${REPO_DIR}/config/settings.toml" "${REPO_DIR}/.venv/bin/shuru" init
}

install_units() {
    log "installing systemd units"
    install -m 0644 "${REPO_DIR}"/deploy/systemd/lab-* /etc/systemd/system/
    systemctl daemon-reload
    for job in "${JOBS[@]}"; do
        systemctl enable --now "lab-${job}.timer" >/dev/null
    done
    systemctl list-timers 'lab-*' --no-pager
}

main() {
    require_root
    check_platform
    install_packages
    create_user
    create_dirs
    deploy_code
    install_units
    cat <<EOF

Done. Next:
  1. Put your sources in ${REPO_DIR}/config/sources.local.toml and feeds in
     news_feeds.local.toml (as root; the jobs can read but not change them).
  2. Set NTFY_TOPIC in /etc/shuru/env for phone alerts.
  3. Train on your PC, then copy the model here and run:
       sudo -u ${SERVICE_USER} env LAB_ROOT=${LAB_ROOT} ${REPO_DIR}/.venv/bin/shuru model install FILE
  4. Watch it work: journalctl -u 'lab-*' -f    and    sudo -u ${SERVICE_USER} ... shuru runs
EOF
}

main "$@"
