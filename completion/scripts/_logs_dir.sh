# Shared log directory: completion/logs (not /tmp).
# Usage: source "$(dirname "$0")/_logs_dir.sh"
_LOGS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
LOG_ROOT="${LOG_ROOT:-$(cd "$_LOGS_SCRIPT_DIR/.." && pwd)/logs}"
mkdir -p "$LOG_ROOT"
