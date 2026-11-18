from pathlib import Path
import sys

_src = str(Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from brainprint_pipeline import load_model, predict_batch
