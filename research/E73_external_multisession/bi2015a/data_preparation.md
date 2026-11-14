# BI2015a 数据准备

## 数据说明

BI2015a（Brain Invaders calibration-less P300-based BCI with modulation of flash duration）来自 Zenodo record 3266930；NEMAR `nm000216/v1.0.0` 提供同一数据的 BIDS 目录。NEMAR 资料确认：43 名受试者、每人 3 个 session、32 个 EEG 通道；当前直链主体文件为 BDF，不是 MAT，例如 `sub-1/ses-0/eeg/sub-1_ses-0_task-p300_run-0_eeg.bdf`，事件在对应的 `*_events.tsv`/`*_events.json` 中。每 session 可能只有一个 run，trial 数为 variable，不能直接套用固定 504 trial、6×84 run 协议，也不进行伪造 MAT 转换。

## 处理协议

`inspect_bids.py` 先通过 NEMAR API 获取三 session 的 events.tsv/json 与 BDF 清单，优先下载小文件；已有 `.part` 的 BDF 不重复重试。events.tsv 的 onset/duration/trial_type 仅用于按 onset 切 1 秒 trial 和校验，标签不作为身份输入。每 session 的实际事件序列通过 `np.array_split` 划为固定 6 段，记录各段 counts，不丢 trial，也不补齐；manifest 的 trial_counts/run_counts 动态生成。当前 BDF 不完整时明确阻塞。

## 当前状态

本次受控 NEMAR CLI 实验先检查 CLI：`nemar --help` 不可用（PowerShell 无法识别 `nemar`），`git annex version` 不可用（`git: 'annex' is not a git command`）。因此未安装依赖、未创建临时目录、未执行 `nemar dataset clone nm000216`，也未复制或读取 BDF；项目已有 `.part` 文件保持不变。

已实现 NEMAR BDF 的 4 MiB 分块 Range 下载：每块独立 timeout，使用 `.part` 续传，严格校验 206/Content-Range，服务器不支持 Range 或请求超时时不覆盖 `.part`，完成后核验大小与 SHA256。支持 `--only-bdf --chunk-size-mib --subjects`，按 ses-2、ses-1、ses-0 且小文件优先尝试；本次 TLS/重定向在首个 Range 响应前超时，三 session 均未完成，未伪造完成状态。未伪造 MAT、未开始训练。subject-1 可用文件如下：

- `sub-1/ses-0/eeg/sub-1_ses-0_task-p300_run-0_eeg.bdf`：59,279,104 bytes；`*_events.tsv`：91,416 bytes；`*_events.json`：1,012 bytes。
- `sub-1/ses-1/eeg/sub-1_ses-1_task-p300_run-0_eeg.bdf`：39,070,657 bytes；`*_events.tsv`：58,237 bytes；`*_events.json`：1,012 bytes。
- `sub-1/ses-2/eeg/sub-1_ses-2_task-p300_run-0_eeg.bdf`：17,785,504 bytes；`*_events.tsv`：24,983 bytes；`*_events.json`：1,012 bytes。

主体文件为 BDF，且实际每 session 一个 run、trial 数 variable，与原固定 504/6×84 协议冲突，当前状态为 blocked。后续仅按实际事件顺序切 trial；只有 trial 数可被 6 整除时才划分 6 个连续等分 run，否则明确阻塞。

脚本支持 `.part` 续传、重跑、SHA256 核验和状态持久化；`--source nemar|zenodo` 默认 NEMAR，`--subjects` 可限定受试者。模型指标：AUC=N/A；F1=N/A（本任务未训练模型）。

## 文件索引

- `download_data.py`：Zenodo API 元数据、43 个 subject zip 下载、续传、哈希和 MD5 核验。
- `derive_data.py`：MAT 解包、事件核验、固定 trial/run 派生和回读检查。
- `manifest.json`：顶层 `files` 列表；目标每项包括 path、sha256、subject、session、shape、dtype、run_counts。
- `preparation_status.json`：下载/派生状态和错误。
- `cache/raw/`：原始 zip 与解包 MAT（尚未产生）。
- `cache/derived/subject_XX/session_Y.npz`：目标派生布局（尚未产生）。

模型指标：AUC=N/A；F1=N/A（本任务未训练模型）。
