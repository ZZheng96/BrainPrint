"""Download the public BCIC IV-2a external sessions without using MOABB.

Only BNCI2014_001 files are allowed here. BNCI2014_004 is BCIC IV-2b and is
intentionally not downloaded. No path outside this E73 directory is accessed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

E73_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = E73_DIR / "manifest.json"
DEFAULT_BASE_URL = "https://lampx.tugraz.at/~bci/database/001-2014"
DEFAULT_LABELS_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"
DATASET_URL = "https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html"
LICENSE_URL = "https://creativecommons.org/licenses/by-nd/4.0/"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "dataset": "BNCI2014_001",
        "common_name": "BCI Competition IV dataset 2a",
        "identifier_correction": (
            "BNCI2014_004 is BCI Competition IV dataset 2b (3 EEG channels, "
            "5 sessions), not dataset 2a; it is excluded from E73 downloads."
        ),
        "dataset_documentation": DATASET_URL,
        "license": "CC BY-ND 4.0 (as reported by MOABB/NEMAR dataset documentation)",
        "license_url": LICENSE_URL,
        "license_note": (
            "Raw files retain upstream provenance. CC BY-ND 4.0 restricts distribution "
            "of adapted material; local derived caches are not for redistribution."
        ),
        "intended_use": "external multi-session training/validation data preparation only",
        "access_policy": {
            "competition_test_accessed": False,
            "competition_train_accessed": False,
            "model_training_started": False,
            "allowed_root": str(E73_DIR),
            "original_T_files": "external session 1; original public training recording",
            "original_E_files": (
                "external session 2; original public evaluation recording plus public "
                "true labels, assigned only to E73 external training/validation"
            ),
        },
        "hash_provenance": "SHA-256 computed locally after download; not an upstream-signed checksum",
        "files": {},
        "runs": [],
    }


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def download_one(url: str, destination: Path, timeout: float) -> dict:
    record = {
        "url": url,
        "path": destination.relative_to(E73_DIR).as_posix(),
        "retrieved_at": utc_now(),
    }
    if destination.exists() and destination.stat().st_size > 0:
        record.update(
            status="existing",
            bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
        )
        return record

    if shutil.disk_usage(destination.parent).free < 2 * 1024**3:
        raise OSError("Less than 2 GiB free; download aborted")
    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists():
        part.unlink()
    digest = hashlib.sha256()
    size = 0
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, timeout),
            allow_redirects=True,
            headers={"User-Agent": "BrainPrint-E73-public-data-preparation/1.0"},
        ) as response:
            response.raise_for_status()
            record["final_url"] = response.url
            record["http_status"] = response.status_code
            content_type = response.headers.get("Content-Type", "")
            record["content_type"] = content_type
            if "text/html" in content_type.lower():
                raise ValueError(f"Expected MAT data, received Content-Type={content_type!r}")
            with part.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if not block:
                        continue
                    size += len(block)
                    if size > 256 * 1024**2:
                        raise ValueError("Single file exceeds the 256 MiB E73 safety limit")
                    digest.update(block)
                    handle.write(block)
        if size == 0:
            raise ValueError("Downloaded file is empty")
        part.replace(destination)
        record.update(status="downloaded", bytes=size, sha256=digest.hexdigest())
        return record
    except Exception:
        if part.exists():
            part.unlink()
        raise


def extract_label(archive: Path, subject: int, destination: Path, source_url: str) -> dict:
    member = f"A{subject:02d}E.mat"
    with zipfile.ZipFile(archive) as bundle:
        if member not in bundle.namelist():
            raise ValueError(f"{member} is absent from {archive.name}")
        payload = bundle.read(member)
    if not payload:
        raise ValueError(f"{member} in {archive.name} is empty")
    status = "extracted"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"Existing {destination.name} differs from the official label archive")
        status = "existing"
    else:
        destination.write_bytes(payload)
    return {
        "url": source_url,
        "archive_member": member,
        "path": destination.relative_to(E73_DIR).as_posix(),
        "retrieved_at": utc_now(),
        "status": status,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "kind": "public_true_labels",
        "subject": subject,
        "session": "E",
    }


def parse_subjects(value: str) -> list[int]:
    subjects = [int(item) for item in value.split(",") if item.strip()]
    if not subjects or any(subject < 1 or subject > 9 for subject in subjects):
        raise ValueError("Subjects must be comma-separated integers in 1..9")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Subjects must not contain duplicates")
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated subject IDs in 1..9; default: all 9 subjects",
    )
    parser.add_argument("--sessions", default="T,E", help="T, E, or T,E; default: T,E")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--labels-url", default=DEFAULT_LABELS_URL)
    parser.add_argument(
        "--skip-e-labels",
        action="store_true",
        help="Skip the separate public true_labels archive (verification of E will then fail)",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Read timeout; no automatic retry")
    parser.add_argument("--workers", type=int, default=4, help="Parallel first-attempt downloads (1..8)")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    sessions = [item.strip().upper() for item in args.sessions.split(",") if item.strip()]
    if not sessions or any(session not in {"T", "E"} for session in sessions):
        raise ValueError("Sessions must contain only T and/or E")
    if len(set(sessions)) != len(sessions):
        raise ValueError("Sessions must not contain duplicates")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")
    base_url = args.base_url.rstrip("/")
    manifest = load_manifest()
    for previous_run in manifest["runs"]:
        if previous_run.get("status") == "running":
            previous_run["status"] = "interrupted"
            previous_run["finished_at"] = utc_now()
    run = {
        "started_at": utc_now(),
        "base_url": base_url,
        "labels_url": args.labels_url,
        "subjects": subjects,
        "sessions": sessions,
        "status": "running",
    }
    manifest["runs"].append(run)
    save_manifest(manifest)

    try:
        raw_root = E73_DIR / "cache" / "bciciv2a" / "raw"
        label_archive = None
        if "E" in sessions and not args.skip_e_labels:
            raw_root.mkdir(parents=True, exist_ok=True)
            label_archive = raw_root / "true_labels.zip"
            try:
                record = download_one(args.labels_url, label_archive, args.timeout)
            except Exception as exc:
                record = {
                    "url": args.labels_url,
                    "path": label_archive.relative_to(E73_DIR).as_posix(),
                    "kind": "public_true_labels_archive",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "attempted_at": utc_now(),
                }
                manifest["files"][record["path"]] = record
                raise
            record["kind"] = "public_true_labels_archive"
            manifest["files"][record["path"]] = record
            save_manifest(manifest)
            print(json.dumps(record, ensure_ascii=False), flush=True)

        jobs = []
        for subject in subjects:
            subject_dir = raw_root / f"subject_{subject:02d}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            for session in sessions:
                stem = f"A{subject:02d}{session}"
                jobs.append(
                    (
                        subject,
                        session,
                        f"{base_url}/{stem}.mat",
                        subject_dir / f"{stem}.mat",
                    )
                )
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_jobs = {
                executor.submit(download_one, url, destination, args.timeout): (subject, session, url, destination)
                for subject, session, url, destination in jobs
            }
            for future in concurrent.futures.as_completed(future_jobs):
                subject, session, url, destination = future_jobs[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "url": url,
                        "path": destination.relative_to(E73_DIR).as_posix(),
                        "kind": "recording",
                        "status": "failed",
                        "subject": subject,
                        "session": session,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "attempted_at": utc_now(),
                    }
                    failures.append(record)
                else:
                    record["kind"] = "recording"
                    record["subject"] = subject
                    record["session"] = session
                manifest["files"][record["path"]] = record
                save_manifest(manifest)
                print(json.dumps(record, ensure_ascii=False), flush=True)
        if failures:
            raise RuntimeError(
                f"{len(failures)} recording downloads failed after one attempt each; no automatic retry"
            )

        if label_archive is not None:
            for subject in subjects:
                stem = f"A{subject:02d}E"
                label_path = raw_root / f"subject_{subject:02d}" / f"{stem}_labels.mat"
                try:
                    label_record = extract_label(label_archive, subject, label_path, args.labels_url)
                except Exception as exc:
                    label_record = {
                        "url": args.labels_url,
                        "archive_member": f"{stem}.mat",
                        "path": label_path.relative_to(E73_DIR).as_posix(),
                        "kind": "public_true_labels",
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "attempted_at": utc_now(),
                    }
                    manifest["files"][label_record["path"]] = label_record
                    raise
                manifest["files"][label_record["path"]] = label_record
                save_manifest(manifest)
                print(json.dumps(label_record, ensure_ascii=False), flush=True)
        run["status"] = "success"
    except Exception as exc:
        run["status"] = "failed"
        run["error_type"] = type(exc).__name__
        run["error"] = str(exc)
        raise
    finally:
        run["finished_at"] = utc_now()
        save_manifest(manifest)


if __name__ == "__main__":
    main()
