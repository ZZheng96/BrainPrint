# Checklist

- [ ] 根目录不存在 `赞美诗_第3次.csv`
- [ ] 根目录不存在 `__pycache__/`、`.pytest_cache/`、`.screenshots/`、`.trae-html-share-packages/`
- [ ] `.gitignore` 已包含 `.pytest_cache/`、`.screenshots/`
- [ ] `research/` 根级仅剩 `README.md`、`brainprint_research_and_experiments.md`、`对话交接_实验历程与下一步.md`、`问题梳理.md` 及子目录
- [ ] `research/archive/submissions/` 包含 18 个 `submission_*.csv`
- [ ] `research/archive/diagnostics/` 包含全部 `diagnostics_*`、`validation_*`、`baseline_*` json
- [ ] `research/archive/train_analysis/` 包含全部 `train_*` json 及 `data_audit.json` 等
- [ ] `calibrated_pseudo_ground_truth_v3/scenarios/` 中无 MD5 重复的 `pseudo_answer.csv`
- [ ] `calibrated_pseudo_ground_truth_v4/scenarios/main_evidence_consensus/` 已删除
- [ ] 全仓无残留 `research/submission_`、`research/diagnostics_`、`research/train_`、`research/baseline_`、`research/validation_` 旧路径引用
- [ ] `research/README.md` 存在且每个子目录均有方法、结论（含 AUC/F1）、报告链接
- [ ] 根目录 `README.md` 存在且说明项目结构、数据准备、复现入口
- [ ] `py -3 -m pytest test/ -q` 通过
- [ ] 变更已提交并推送到 `origin/main`
