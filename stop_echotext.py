"""Stop Script for EchoText (`stop_echotext.py`).

Reads `logs/echotext_pids.json`, terminates all running EchoText processes cleanly,
and sets `proxy_status.txt` to `AVAILABLE` (fail-safe silence on shutdown).
"""

from __future__ import annotations

import json
import os
import signal

from config import PID_FILE_PATH, set_proxy_status, setup_logging

logger = setup_logging("echotext.stopper")


def stop_all() -> dict[str, str]:
    """Set proxy status to AVAILABLE and terminate all tracked EchoText PIDs."""
    set_proxy_status("AVAILABLE")
    outcomes: dict[str, str] = {}

    if not PID_FILE_PATH.exists():
        logger.info("No PID file found at %s; status reset to AVAILABLE.", PID_FILE_PATH)
        return {"status_file": "AVAILABLE", "processes": "none_tracked"}

    try:
        pids: dict[str, int] = json.loads(PID_FILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pids = {}

    for name, pid in pids.items():
        try:
            os.kill(int(pid), signal.SIGTERM)
            outcomes[name] = f"terminated (PID {pid})"
            logger.info("Terminated %s (PID=%d)", name, pid)
        except ProcessLookupError:
            outcomes[name] = f"already stopped (PID {pid})"
        except Exception as exc:
            outcomes[name] = f"error stopping PID {pid}: {exc}"

    PID_FILE_PATH.unlink(missing_ok=True)
    print(f"🛑 EchoText stopped and status set to AVAILABLE: {outcomes}")
    return outcomes


if __name__ == "__main__":
    stop_all()
