#!/usr/bin/env bash
# Boots the FastAPI backend (internal) and the Streamlit dashboard (public)
# inside a single Render web service.
set -uo pipefail

PORT="${PORT:-8501}"

# Render exposes `python`; macOS often only has `python3`.
PY_BIN="$(command -v python || command -v python3)"

echo "[EAPS] starting FastAPI on 127.0.0.1:8000 ..."
"$PY_BIN" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# The API builds its SHAP explainer at import time, so wait for it to answer
# before putting the dashboard in front of users.
echo "[EAPS] waiting for API to become ready ..."
"$PY_BIN" - <<'PY'
import sys, time, urllib.request
for _ in range(90):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/", timeout=2)
        print("[EAPS] API is ready")
        sys.exit(0)
    except Exception:
        time.sleep(1)
print("[EAPS] API did not become ready in 90s", file=sys.stderr)
sys.exit(1)
PY

if [ $? -ne 0 ]; then
  kill -TERM "$API_PID" 2>/dev/null || true
  exit 1
fi

echo "[EAPS] starting Streamlit on 0.0.0.0:${PORT} ..."
"$PY_BIN" -m streamlit run dashboard/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false &
UI_PID=$!

# If either process dies, bring the container down so Render restarts it
# cleanly instead of serving a half-broken app.
# `wait -n` needs bash >= 4.3 (Render has 5.x; stock macOS bash is 3.2).
if ((BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3))); then
  wait -n
else
  while kill -0 "$API_PID" 2>/dev/null && kill -0 "$UI_PID" 2>/dev/null; do sleep 5; done
fi

echo "[EAPS] a process exited — shutting down" >&2
kill -TERM "$API_PID" "$UI_PID" 2>/dev/null || true
exit 1
