# E241 原始EEG鲁棒原型增强

- 实际完成 20/20 outer folds，设备 cuda，耗时 373.1s。
- 完整阅读并复用E215特征路径、任务标准化、PCA/k12原型、GPU距离、分折与指标。PCA改为CUDA float64 SVD，数学目标不变。
- 只在当前fit行上生成确定性原始EEG扰动，再调用E65 GPU提取器；样本级缓存不含跨样本统计。原始聚类分配固定，aug_mean/aug_trimmed以原始及五个增强视图更新每个簇中心；trimmed逐维去掉两端15%。
- inner覆盖所有非空时间块，另划分两个身份组，pseudo-unknown身份完全不参与相应内层标准化/PCA/原型。所有outer held身份完全不参与其阈值标定。
- 每个候选独立用inner unknown OOF的90%分位数设阈值（strict < 的边界采用nextafter）；同时记录对应known OOF分位数。90%是inner目标，不保证outer实测恰为90%；不得依据outer或Test回调阈值。
- 两个inner身份组的注册库比outer小，存在校准迁移偏差；不应将inner=0.90冒充完整LOSO=0.90。
- E215 acceptance=0.90不是unknown rejection=0.90。这里baseline也重新标定，不直接引用旧结果。
- AUC为每outer fold unknown-positive AUC均值（使用负距离置信度）；F1为unknown-positive二分类F1逐折均值。Test无真实身份标签，官方AUC/F1均N/A。S1/S2只作会话描述，非官方准确率。
- 三个候选与扰动强度预先固定，未依据Test选型；历史反复查看Test的研究偏差仍存在。

|候选|LOSO AUC|unknown-positive F1|LOSO拒识micro|known接受micro|Test S1拒识|Test S2接受|
|---|---:|---:|---:|---:|---:|---:|
|baseline|0.910173|0.343120|0.814429|0.848327|0.9485657764589516|0.3456655219065459|
|aug_mean|0.907881|0.337469|0.814863|0.844049|0.9452687108473459|0.34483296909147676|
|aug_trimmed|0.908431|0.338991|0.815298|0.844919|0.948236069897791|0.3453012800499532|

## 结论
仅当完整20折实测拒识约0.90且S2高于同协议baseline时，才可宣称满足约束的提升。未满足者为负结果；部分折不代表完整LOSO。

## 复现与产物
命令：`.venv-py311\Scripts\python.exe research\E241_robust_prototype_aug\run_e241.py`；可行性阶段使用`--folds 1 --no-test`。
config.json固定扰动与协议；cache保存原始波形重提特征；models保存每内外折fit/query行ID、标准化/PCA状态、三种原型与分数，完整运行保存最终模型/阈值/OOF分数。文件名以脚本+config哈希隔离，既有实验只读。
