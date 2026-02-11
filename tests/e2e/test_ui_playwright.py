import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright


UI_DIR = Path(__file__).parent.parent.parent / "echochat" / "ui"


def start_static_server(port=8000):
    # start a simple HTTP server serving the UI dir
    cmd = [sys.executable, "-m", "http.server", str(port), "-d", str(UI_DIR)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait briefly
    time.sleep(0.5)
    return proc


def stop_proc(proc):
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except Exception:
        proc.kill()


def test_ui_chat_flow():
    # Require backend running at default API_BASE
    api_base = "http://127.0.0.1:5000"
    # quick health check
    try:
        r = requests.get(f"{api_base}/")
    except Exception:
        # backend must be running before test
        raise RuntimeError("Backend not reachable at http://127.0.0.1:5000")

    server = start_static_server(8000)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://127.0.0.1:8000")

            # Inject API base so UI points to backend
            page.evaluate("(base) => { window.ECHOCHAT_API_BASE = base; }", api_base)
            # reload so scripts use API_BASE
            page.reload()

            # Apply an existing session directly to the UI so we can chat
            session_id = "f7226ab5fdfb4605"
            data = {
                "session_id": session_id,
                "echo_person": "Echo",
                "senders": [{"name": "Echo", "count": 100}],
                "message_count": 100,
            }
            page.evaluate("(d) => applySession(d)", data)

            # send a test message
            page.fill('#message', 'Hello from E2E test')
            page.click('#composer button')

            # wait for bot response to appear
            page.wait_for_selector('.msg.bot', timeout=15000)
            bot_text = page.locator('.msg.bot').all_text_contents()[-1]
            assert bot_text and len(bot_text) > 0

            browser.close()
    finally:
        stop_proc(server)
