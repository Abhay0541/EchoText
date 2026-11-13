"""Ordered Launcher for EchoText (`start_echotext.py`).

Starts EchoText services in dependency order:
1. Validates `.env` settings and warns if ``my_proxy.session`` is inside a cloud-synced folder.
2. Starts the FastAPI Brain server (`server.py`) first with rotating logs in `logs/echotext.log`.
3. Polls `http://127.0.0.1:8000/health` until the server and database are ready before
   starting `telegram_listener.py`, so early incoming messages are never dropped.
4. Starts the Streamlit Dashboard strictly bound to `--server.address 127.0.0.1`.
5. Saves child process PIDs to `logs/echotext_pids.json` so `stop_echotext.py` can cleanly
   stop all services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess  # nosec B404
import sys
import time
import urllib.error
import urllib.request

from config import (
    BASE_DIR,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    LOG_DIR,
    PID_FILE_PATH,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_URL,
    setup_logging,
    validate_env,
)

logger = setup_logging("echotext.launcher")


def wait_for_server_ready(health_url: str, timeout_seconds: int = 90) -> bool:
    """Poll ``/health`` until the Brain server responds with HTTP 200."""
    if not health_url.startswith(("http://127.0.0.1", "http://localhost")):
        logger.warning("Non-local health URL configured: %s", health_url)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:  # nosec B310  # noqa: S310
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        return True
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            logger.debug("Waiting for server readiness: %s", exc)
        time.sleep(1.0)
    return False


def _spawn_logged_process(cmd: list[str], log_path: Path) -> subprocess.Popen[bytes]:
    """Spawn a background process with stdout/stderr redirected to ``log_path``."""
    with log_path.open("a", encoding="utf-8") as log_handle:
        return subprocess.Popen(  # nosec B603  # noqa: S603
            cmd,
            cwd=str(BASE_DIR),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )


def launch_all(
    start_listener: bool = True,
    start_dashboard: bool = True,
    start_phone_bot: bool = False,
    wait_timeout: int = 90,
) -> dict[str, int]:
    """Start EchoText services in dependency order and write PIDs to ``PID_FILE_PATH``."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    errors = validate_env(
        require_telegram=start_listener,
        require_dashboard_auth=start_dashboard,
        require_phone_bot=start_phone_bot,
    )
    for e in errors:
        if e.startswith("WARNING:"):
            logger.warning(e)
        else:
            logger.error(e)

    fatal = [e for e in errors if not e.startswith("WARNING:")]
    if fatal:
        print("EchoText Startup Check Failed:")
        for f in fatal:
            print(f"  - {f}")
        raise SystemExit(1)

    pids: dict[str, int] = {}
    py_exe = sys.executable

    server_proc = _spawn_logged_process(
        [
            py_exe,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
        ],
        LOG_DIR / "server.out.log",
    )
    pids["server"] = server_proc.pid
    logger.info(
        "Started Brain server (PID=%d) on %s:%d",
        server_proc.pid,
        SERVER_HOST,
        SERVER_PORT,
    )

    health_url = f"{SERVER_URL.rstrip('/')}/health"
    logger.info("Waiting for Brain server readiness at %s ...", health_url)
    if not wait_for_server_ready(health_url, timeout_seconds=wait_timeout):
        logger.error("Brain server did not become ready within %ds. Stopping.", wait_timeout)
        server_proc.terminate()
        raise SystemExit(1)
    logger.info("Brain server is ready.")

    if start_listener:
        listener_proc = _spawn_logged_process(
            [py_exe, str(BASE_DIR / "telegram_listener.py")],
            LOG_DIR / "listener.out.log",
        )
        pids["telegram_listener"] = listener_proc.pid
        logger.info("Started Telegram listener (PID=%d)", listener_proc.pid)

    if start_dashboard:
        dash_proc = _spawn_logged_process(
            [
                py_exe,
                "-m",
                "streamlit",
                "run",
                str(BASE_DIR / "app.py"),
                "--server.address",
                DASHBOARD_HOST,
                "--server.port",
                str(DASHBOARD_PORT),
                "--server.headless",
                "true",
            ],
            LOG_DIR / "dashboard.out.log",
        )
        pids["dashboard"] = dash_proc.pid
        logger.info(
            "Started Streamlit dashboard (PID=%d) at http://%s:%d",
            dash_proc.pid,
            DASHBOARD_HOST,
            DASHBOARD_PORT,
        )

    if start_phone_bot:
        bot_proc = _spawn_logged_process(
            [py_exe, str(BASE_DIR / "phone_bot.py")],
            LOG_DIR / "phone_bot.out.log",
        )
        pids["phone_bot"] = bot_proc.pid
        logger.info("Started Private Phone Approval Bot (PID=%d)", bot_proc.pid)

    PID_FILE_PATH.write_text(json.dumps(pids, indent=2), encoding="utf-8")
    print(f"✅ EchoText launched! PIDs recorded in {PID_FILE_PATH}: {pids}")
    return pids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start EchoText services safely.")
    parser.add_argument(
        "--no-listener", action="store_true", help="Skip starting Telegram listener"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Skip starting Streamlit dashboard"
    )
    parser.add_argument(
        "--with-phone-bot", action="store_true", help="Also start phone approval bot"
    )
    args = parser.parse_args()
    launch_all(
        start_listener=not args.no_listener,
        start_dashboard=not args.no_dashboard,
        start_phone_bot=args.with_phone_bot,
    )
