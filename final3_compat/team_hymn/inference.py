from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from team_hymn_geometry import load_model, predict_batch
