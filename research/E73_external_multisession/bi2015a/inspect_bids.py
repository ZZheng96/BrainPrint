from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "cache" / "raw"
NEMAR_BASE = "https://data.nemar.org/nm000216/v1.0.0"
HEADERS = {"User-Agent": "BrainPrint-BI2015a-inspect/1.0"}

def api(path):
    r = requests.get(f"{NEMAR_BASE}/{path.strip('/')}/", headers=HEADERS, timeout=(20, 40))
    r.raise_for_status()
    return r.json()

def files(subject=1):
    result = []
    for session in range(3):
        rel = f"sub-{subject}/ses-{session}/eeg"
        for item in api(rel).get("children", []):
            name = item.get("name", "")
            if item.get("kind") == "file" and (name.endswith("_events.tsv") or name.endswith("_events.json") or name.endswith("_eeg.bdf")):
                result.append({"session": session, "name": name, "size": item.get("size"), "url": f"{NEMAR_BASE}/{rel}/{name}", "path": rel + "/" + name})
    return result

def download_small(item):
    dst = RAW / item["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (item["size"] is None or dst.stat().st_size == item["size"]):
        return dst
    r = requests.get(item["url"], headers=HEADERS, timeout=(20, 120))
    r.raise_for_status()
    dst.write_bytes(r.content)
    if item["size"] is not None and dst.stat().st_size != item["size"]:
        raise ValueError(f"{dst}: size {dst.stat().st_size} != {item['size']}")
    return dst

def parse_events(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for row in rows:
        row["onset"] = float(row["onset"])
        row["duration"] = float(row["duration"])
    return rows

def bdf_metadata(path):
    try:
        import mne
        raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
        return {"parser": "mne", "channels": len(raw.ch_names), "sfreq": float(raw.info["sfreq"]), "duration_seconds": float(raw.n_times / raw.info["sfreq"]), "complete": True}
    except ImportError:
        return {"parser": None, "complete": False, "error": "neither mne nor pyedflib is installed"}
    except Exception as exc:
        return {"parser": "mne", "complete": False, "error": f"{type(exc).__name__}: {exc}"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    args = p.parse_args()
    items = files(args.subject)
    report = {"subject": args.subject, "source": NEMAR_BASE, "sessions": [], "files": items}
    for session in range(3):
        selected = [x for x in items if x["session"] == session]
        event_files = [x for x in selected if x["name"].endswith("_events.tsv")]
        json_files = [x for x in selected if x["name"].endswith("_events.json")]
        record = {"session": session, "event_rows": None, "unique_trial_type": [], "bdf": None, "errors": []}
        try:
            if not event_files or not json_files:
                raise FileNotFoundError("events.tsv/json missing from NEMAR listing")
            tsv = download_small(event_files[0]); download_small(json_files[0])
            rows = parse_events(tsv)
            record["event_rows"] = len(rows)
            record["unique_trial_type"] = sorted({str(x.get("trial_type", "")) for x in rows})
        except Exception as exc:
            record["errors"].append(f"events: {type(exc).__name__}: {exc}")
        bdfs = [x for x in selected if x["name"].endswith("_eeg.bdf")]
        if bdfs:
            bdf = RAW / bdfs[0]["path"]
            record["bdf"] = bdf_metadata(bdf) if bdf.exists() and not bdf.with_suffix(bdf.suffix + ".part").exists() else {"complete": False, "error": "BDF absent or incomplete .part; not retried"}
        report["sessions"].append(record)
    (ROOT / "bids_inspection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
