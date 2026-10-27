#!/usr/bin/env bash
# Optional server setup for a Debian-family Linux host with systemd.
# Not needed to run the project locally. Every step should be safe to re-run.
set -euo pipefail

LAB_ROOT="${LAB_ROOT:-/srv/lab}"
REPO_DIR="/opt/shuru"

not_implemented() {
    echo "bootstrap: $1 is not implemented yet" >&2
    exit 1
}

require_root() {
    # exit unless running as root
    not_implemented "${FUNCNAME[0]}"
}

check_platform() {
    # Linux with systemd and apt, Python >= 3.11
    not_implemented "${FUNCNAME[0]}"
}

install_packages() {
    # python3, sqlite3, git, jq
    not_implemented "${FUNCNAME[0]}"
}

install_uv() {
    # pinned version, checksum verified. No curl | sh as root.
    not_implemented "${FUNCNAME[0]}"
}

create_users() {
    # one system user per job, no login shell, no home
    # lab-db owns the database; user list depends on the writer design (undecided)
    not_implemented "${FUNCNAME[0]}"
}

create_dirs() {
    # LAB_ROOT plus one folder per job from settings.toml [paths.job_dirs], mode 0750
    not_implemented "${FUNCNAME[0]}"
}

deploy_code() {
    # clone to REPO_DIR, root-owned, then uv sync --locked
    not_implemented "${FUNCNAME[0]}"
}

main() {
    require_root
    check_platform
    install_packages
    install_uv
    create_users
    create_dirs
    deploy_code
}

main "$@"
