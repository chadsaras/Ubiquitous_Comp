import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aggregate.timeline import build_timeline
from src.query.engine import answer

# 1. Pick a model
model_path = Path("models/model_hgb.joblib")  # or "models/model_hgb.joblib"
rec_path = Path("tests/fixtures/00EABED2_590_160/recording.csv")

# 2. Build Timeline
timeline = build_timeline(rec_path, model_path)
print(f"Generated {len(timeline.intervals)} intervals.")

# 3. Answer a question
res = answer("How long was the user walking?", timeline)
print(res)
