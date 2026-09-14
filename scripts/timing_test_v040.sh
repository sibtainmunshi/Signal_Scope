#!/usr/bin/env bash
# Fresh-clone timing test for v0.4.0. Run only after the v0.4.0 GitHub release
# (head.pt) is published; the tower asset is already public under v0.3.0.
set -euo pipefail
TARGET="/c/Users/Sibtainhaidar/OneDrive/Desktop/signal_scope_v040_timing"
rm -rf "$TARGET"

START=$(date +%s.%N)
git clone --branch v0.4.0 https://github.com/sibtainmunshi/Signal_Scope.git "$TARGET"
cd "$TARGET"
CLONE_DONE=$(date +%s.%N)

python scripts/setup.py
SETUP_DONE=$(date +%s.%N)

.venv/Scripts/python.exe model/predict.py --image "$OLDPWD/tmp/smoke_test_v040.jpg" --device cpu > /tmp/v040_timing_predict.json
PREDICT_DONE=$(date +%s.%N)

python - "$START" "$CLONE_DONE" "$SETUP_DONE" "$PREDICT_DONE" <<'PYEOF'
import json, sys
from datetime import UTC, datetime
start, clone_done, setup_done, predict_done = (float(x) for x in sys.argv[1:5])
result = {
    "source": "fresh public Git clone of the v0.4.0 tag",
    "release": "v0.4.0",
    "checked_utc": datetime.now(UTC).isoformat(),
    "clone_seconds": round(clone_done - start, 2),
    "setup_seconds": round(setup_done - clone_done, 2),
    "first_prediction_seconds": round(predict_done - setup_done, 2),
    "total_seconds": round(predict_done - start, 2),
    "note": "Includes the 608 MB tower + 398 KB head download inside setup; "
            "package_download_cache state depends on this machine's pip cache.",
}
with open("report/reproducibility/v0.4.0_windows_cpu.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
    f.write("\n")
print(json.dumps(result, indent=2))
PYEOF
