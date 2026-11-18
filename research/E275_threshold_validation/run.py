"""E275 Train-only independent validation. Run from project root with the project venv."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

H = Path(__file__).resolve().parent
R = H.parents[1]
C = R / "BrainPrint_cache"
PK = R / "final3_compat" / "team_hymn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 275
CANDIDATE_OFFSET = 0.06
FINAL3_OFFSET = 0.03
FIT_SCRIPT = R / "research" / "E270_new_calibration" / "run.py"
FIT_SCOPE = runpy.run_path(str(FIT_SCRIPT), run_name="e275_fit_rank_source")
FIT_RANK = FIT_SCOPE["fit"]


def load_package():
    spec = importlib.util.spec_from_file_location("e275_compat", PK / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    geometry_spec = importlib.util.spec_from_file_location("e275_compat_geometry", PK / "src" / "team_hymn_geometry.py")
    geometry = importlib.util.module_from_spec(geometry_spec)
    assert geometry_spec.loader is not None
    geometry_spec.loader.exec_module(geometry)
    module._task_indices = geometry._task_indices
    module._features = lambda artifact, features: features
    module.enroll = lambda artifact, features, meta, labels: enroll_cached(geometry, artifact, features, labels)
    module.predict_batch = lambda artifact, features, meta: predict_cached(geometry, artifact, features, meta)
    return module


def enroll_cached(geometry, artifact, features, labels):
    artifact = dict(artifact, prototypes=artifact["prototypes"].clone(), valid=artifact["valid"].clone())
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    x = torch.as_tensor(features, dtype=torch.float32, device=artifact["device"])
    for uid in np.unique(labels):
        if uid not in (21, 22, 23):
            continue
        rows = torch.as_tensor(labels == uid, device=artifact["device"])
        for task_index in range(4):
            projected = geometry._project(artifact, x[rows], task_index)
            count = len(projected)
            if count >= artifact["n_proto"]:
                for j in range(artifact["n_proto"]):
                    artifact["prototypes"][task_index, uid - 1, j] = projected[
                        j * count // artifact["n_proto"]:(j + 1) * count // artifact["n_proto"]].mean(0)
            else:
                artifact["prototypes"][task_index, uid - 1] = projected.mean(0)
            artifact["valid"][task_index, uid - 1] = True
    return artifact


@torch.no_grad()
def predict_cached(geometry, artifact, features, meta):
    feat = torch.as_tensor(features, dtype=torch.float32, device=artifact["device"])
    n = len(feat)
    tasks = geometry._task_indices(meta, n)
    sim = torch.full((n, 23), -float("inf"), device=artifact["device"])
    for task_index in range(4):
        if not bool(artifact["valid"][task_index].any()):
            continue
        take = torch.as_tensor((tasks == task_index) | (tasks < 0), device=artifact["device"])
        if not bool(take.any()):
            continue
        projected = geometry._project(artifact, feat[take], task_index)
        prototypes = artifact["prototypes"][task_index]
        query = geometry._unit(projected @ artifact["whiten"])
        proto = geometry._unit(prototypes.reshape(-1, prototypes.shape[-1]) @ artifact["whiten"]).reshape(23, artifact["n_proto"], -1)
        score = (query[:, None, None, :] * proto[None]).sum(-1).max(2).values
        score = score.masked_fill(~artifact["valid"][task_index][None], -float("inf"))
        sim[take] = torch.maximum(sim[take], score)
    top2 = sim.topk(2, dim=1).values
    best, second = top2[:, 0], top2[:, 1]
    finite = torch.isfinite(best)
    gate = best + artifact["beta"] * (best - second)
    threshold = torch.where(sim.argmax(1) >= 20, artifact["tau"] - 0.03, artifact["tau"])
    rejected = (~finite) | (gate < threshold)
    scores = torch.empty((n, 24), dtype=torch.float32, device=artifact["device"])
    scores[:, 1:] = sim
    scores[:, 0] = best - 1.0
    scores[rejected, 0] = best[rejected] + 1.0
    scores[~finite, 0] = 0.0
    return scores.cpu().numpy()


def f1_macro(truth, pred):
    labels = np.union1d(truth, pred)
    return float(np.mean([2 * ((truth == k) & (pred == k)).sum() / ((truth == k).sum() + (pred == k).sum()) for k in labels]))


def auc_binary(known, score):
    n = int(known.sum())
    if n == 0 or n == len(score):
        return float("nan")
    return float((rankdata(score)[known].sum() - n * (n + 1) / 2) / (n * (len(score) - n)))


def fit_artifact(model, X, y, task, early, old, new, fold):
    scope = FIT_RANK.__globals__
    (H / "fits").mkdir(exist_ok=True)
    enrolled = {}
    original_enroll = model.enroll

    def capture_enroll(artifact, features, meta, labels):
        enrolled["artifact"] = original_enroll(artifact, features, meta, labels)
        return enrolled["artifact"]

    model.enroll = capture_enroll
    scope.update(X=X, y=y, t=task, early=early, pkg=model, W=H / "fits", tag=fold)
    artifact, registration_rows = FIT_RANK(old, new, fold)
    model.enroll = original_enroll
    artifact = enrolled["artifact"]
    fit_rows = np.flatnonzero(early & np.isin(y, old))
    torch.save({key: value.cpu() if torch.is_tensor(value) else value for key, value in artifact.items()}, H / "fits" / f"{fold}_compat.pt")
    return artifact, fit_rows, registration_rows


def registration_counts(y, task, rows, new):
    return {str(p): {str(k): int((task[rows[y[rows] == p]] == k).sum()) for k in range(4)} for p in new}


def predict(model, artifact, features, meta_rows, rows):
    out = []
    for group in np.array_split(rows, max(1, (len(rows) + 127) // 128)):
        out.append(model.predict_batch(artifact, features[group], meta_rows[group]))
    return np.concatenate(out)


def evaluate(scores, y_query, new, unknown, task_query, offset):
    yq = y_query
    winner = scores[:, 1:].argmax(1) + 1
    top = np.sort(scores[:, 1:], axis=1)[:, -2:]
    gate = top[:, 1] + 0.5 * (top[:, 1] - top[:, 0])
    threshold = np.where(winner > 20, 0.40755866169929506 - offset, 0.40755866169929506)
    accepted = gate >= threshold
    pred = np.where(accepted, winner, 0)
    truth = yq.copy()
    truth[np.isin(truth, unknown)] = 0
    for j, p in enumerate(new):
        truth[yq == p] = 21 + j
    known = truth != 0
    old = known & (truth <= 20)
    new_mask = truth >= 21
    unknown_mask = truth == 0
    ranking_score = gate - threshold
    return {
        "n": int(len(truth)),
        "old_accuracy": float((pred[old] == truth[old]).mean()),
        "new_accuracy": float((pred[new_mask] == truth[new_mask]).mean()),
        "unknown_rejection": float((pred[unknown_mask] == 0).mean()),
        "AUC": auc_binary(known, ranking_score),
        "F1": f1_macro(truth, pred),
        "accepted_unknown": int((pred[unknown_mask] != 0).sum()),
        "truth_counts": {"old": int(old.sum()), "new": int(new_mask.sum()), "unknown": int(unknown_mask.sum())},
    }


@torch.inference_mode()
def main():
    torch.set_grad_enabled(False)
    model = load_package()
    model._features = lambda artifact, features: features
    model.load_model = lambda device=DEVICE: {"device": torch.device(device)}
    meta = pd.read_csv(C / "train_metadata.csv")
    if len(meta) != 11304 or not meta.path.str.startswith("Train/").all() or not meta.path.is_unique:
        raise AssertionError("仅允许使用 Train 元数据")
    y = meta.subject.to_numpy(dtype=np.int64)
    task = model._task_indices(meta[["task"]].to_dict("records"), len(meta))
    fraction = (meta.trial / meta.groupby(["subject", "task"]).trial.transform("max")).to_numpy()
    early, late = fraction <= 0.6, fraction > 0.8
    clean = torch.from_numpy(np.load(C / "e262_wave_pair" / "clean.npy")).to(DEVICE)
    stress = torch.from_numpy(np.load(C / "e262_wave_pair" / "stress.npy")).to(DEVICE)
    if clean.shape != stress.shape or clean.shape != (len(meta), 855):
        raise AssertionError("特征缓存形状不符")
    meta_rows = meta[["task"]].to_dict("records")
    ids = list(range(1, 21))
    folds = []
    for fold, seed in enumerate((BASE_SEED + 11, BASE_SEED + 37, BASE_SEED + 71)):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(ids).tolist()
        old, new, unknown = sorted(perm[:14]), sorted(perm[14:17]), sorted(perm[17:])
        fold_tag = f"seed{seed}"
        artifact, fit_rows, reg_rows = fit_artifact(model, clean, y, task, early, old, new, fold_tag)
        reg_counts = registration_counts(y, task, reg_rows, new)
        query_rows = np.flatnonzero(late & np.isin(y, old + new + unknown))
        scenarios = {}
        for name, features, missing_task in (
            ("clean_complete", clean, None),
            ("stress_complete", stress, None),
            ("clean_missing_EC", clean, 0),
            ("stress_missing_Imagery", stress, 2),
        ):
            selected = query_rows if missing_task is None else query_rows[task[query_rows] != missing_task]
            metas = np.asarray(meta_rows, dtype=object)[selected].copy()
            if missing_task is not None:
                for row in metas:
                    row.pop("task", None)
            test_artifact = artifact
            if missing_task is not None:
                test_artifact = dict(artifact, prototypes=artifact["prototypes"].clone(), valid=artifact["valid"].clone())
                test_artifact["valid"][missing_task].fill_(False)
            scores = predict(model, test_artifact, features[selected], metas, np.arange(len(selected), dtype=np.int64))
            prediction = scores[:, 1:].argmax(1) + 1
            if missing_task is None and len(selected) and prediction[prediction >= 21].size and not set(prediction[prediction >= 21]).issubset({21, 22, 23}):
                raise AssertionError("compat enrollment returned unexpected new identity ids")
            paired = {"final3": evaluate(scores, y[selected], new, unknown, task[selected], FINAL3_OFFSET),
                      "candidate": evaluate(scores, y[selected], new, unknown, task[selected], CANDIDATE_OFFSET)}
            scenarios[name] = paired
        folds.append({"fold": fold, "seed": seed, "old_ids": old, "new_ids": new, "unknown_ids": unknown,
                      "fit_count": int(len(fit_rows)), "registration_count": int(len(reg_rows)),
                      "registration_task_counts": reg_counts, "late_query_count": int(len(query_rows)),
                      "scenarios": scenarios})
        print(f"fold {fold} done", flush=True)
    summary = {}
    for method in ("final3", "candidate"):
        summary[method] = {}
        for scenario in ("clean_complete", "stress_complete", "clean_missing_EC", "stress_missing_Imagery"):
            metrics = [f["scenarios"][scenario][method] for f in folds]
            summary[method][scenario] = {k: float(np.mean([m[k] for m in metrics])) for k in ("old_accuracy", "new_accuracy", "unknown_rejection", "AUC", "F1")}
    result = {"protocol": {"device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                            "data": "Train only", "folds": 3, "roles": "14 old / 3 new / 3 unknown",
                            "selection": "seeded non-cyclic permutations", "fit_fraction": "<=0.6", "query_fraction": ">0.8",
                            "registration": "E272 fit_rank selects five early Train rows per new identity per task and calls final3 compatibility package enroll",
                            "final3_offset": FINAL3_OFFSET, "candidate_offset": CANDIDATE_OFFSET,
                            "tau_score": 0.40755866169929506, "beta": 0.5,
                            "scenario_weight": "four scenarios equal-weight proxy, not real cross-device evaluation"},
             "summary": summary, "folds": folds,
             "hashes": {"metadata": hashlib.sha256((C / "train_metadata.csv").read_bytes()).hexdigest(),
                        "clean": hashlib.sha256((C / "e262_wave_pair" / "clean.npy").read_bytes()).hexdigest(),
                        "stress": hashlib.sha256((C / "e262_wave_pair" / "stress.npy").read_bytes()).hexdigest()}}
    (H / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
