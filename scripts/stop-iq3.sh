#!/usr/bin/env bash
# Stop this project's server on PORT (default 18200), using the shared launcher.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/start-iq3.sh" --stop "$@"
