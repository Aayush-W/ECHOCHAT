Running end-to-end UI tests

Prerequisites:
- Backend API running: `python -m echochat.backend.api` (default at http://127.0.0.1:5000)
- Virtualenv activated and dev requirements installed:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m playwright install
```

Run the E2E test (uses Playwright + pytest):

```powershell
pytest -q tests/e2e/test_ui_playwright.py
```

Notes:
- The test serves the UI static files on http://127.0.0.1:8000 and injects an existing session id into the page. Ensure migration has run so the session exists.
- Browsers are installed by `playwright install` step.
