import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(r"d:\WORK\BCIC2026\BrainPrint")
OUT = ROOT / "research/E11_session_blind_candidate/pseudo_ground_truth_v1"
P1 = ROOT / "commit/answer/赞美诗_第1次.csv"
P2 = ROOT / "commit/answer/赞美诗_第2次.csv"
LOG = ROOT / "research/E11_session_blind_candidate/nested_pseudounknown_corrspd_v1/candidate_logcov.csv"
CORR = ROOT / "research/E11_session_blind_candidate/nested_pseudounknown_corrspd_v1/candidate_corrspd.csv"
META = ROOT / "BrainPrint_cache/test_metadata.csv"
PAIR = ROOT / "BrainPrint_cache/test_pair_candidates.csv"
WAVE = ROOT / "BrainPrint_cache/waveform_pairs.csv"

p1, p2, log, corr = [pd.read_csv(p) for p in (P1, P2, LOG, CORR)]
meta = pd.read_csv(META)
pair = pd.read_csv(PAIR)
wave = pd.read_csv(WAVE)
for d in (p2, log, corr):
    if not d.Sample.equals(p1.Sample): raise ValueError("Sample顺序不一致")
if len(p1) != 22251 or len(p1) != len(p2): raise ValueError("正式CSV行数不符")

x = p1.merge(p2, on='Sample', suffixes=('_p1','_p2')).merge(log.rename(columns={'Label':'Label_logcov'}), on='Sample').merge(corr.rename(columns={'Label':'Label_corrspd'}), on='Sample').merge(meta[['sample','session','task']], left_on='Sample', right_on='sample', how='left').drop(columns='sample')
x['same_official'] = x.Label_p1.eq(x.Label_p2)
x['candidate_agree'] = x.Label_logcov.eq(x.Label_corrspd)
x['candidate_any_agree'] = x.Label_logcov.isin([x.Label_p1, x.Label_p2]) | x.Label_corrspd.isin([x.Label_p1, x.Label_p2])
x['both_official_zero'] = x.Label_p1.eq(0) & x.Label_p2.eq(0)
x['candidate_reject'] = x.Label_logcov.eq(0) & x.Label_corrspd.eq(0)
x['task'] = x.task.fillna('unknown')
x['source'] = np.where(x.same_official, 'official_common', 'discordant_rule')
x['confidence'] = np.where(x.same_official, 'high', 'low')
# Common labels are retained exactly. For discordant rows, choose the label with stronger candidate support;
# tie-break by agreement with either official submission, then candidate agreement, then conservative rejection.
chosen=[]; reason=[]
for r in x.itertuples():
    if r.same_official:
        chosen.append(int(r.Label_p1)); reason.append('两次正式标签共同值')
        continue
    scores={int(r.Label_p1):0.0,int(r.Label_p2):0.0}
    # candidate agreement is primary evidence; reject/reject is a meaningful safety state.
    for lab in (int(r.Label_logcov), int(r.Label_corrspd)):
        if lab in scores: scores[lab]+=2.0
    if int(r.Label_logcov)==int(r.Label_corrspd):
        if int(r.Label_logcov) in scores: scores[int(r.Label_logcov)]+=1.0
    if r.Label_p1==0 or r.Label_p2==0:
        if 0 in scores: scores[0]+=0.5
    best=max(scores.values()); winners=[k for k,v in scores.items() if v==best]
    if len(winners)>1:
        # deterministic tie-break: official first submission, then second, then 0.
        pick=next((int(v) for v in (r.Label_p1,r.Label_p2,0) if int(v) in winners), winners[0])
        why='候选支持并列，按正式第1次/第2次/拒识顺序裁决'
    else:
        pick=winners[0]; why='候选一致性/拒识状态支持'
    chosen.append(pick); reason.append(why)
x['Label']=chosen; x['rule_reason']=reason
pseudo=x[['Sample','Label']]
pseudo.to_csv(OUT/'pseudo_answer.csv', index=False)

labels=list(range(21))
def eval_candidate(name, pred):
    y=x.Label.to_numpy(); pred=np.asarray(pred)
    accepted=y!=0
    pred_accept=pred!=0
    accepted_acc=float((pred[accepted]==y[accepted]).mean())
    overall=float((pred==y).mean())
    macro=float(f1_score(y,pred,labels=labels,average='macro',zero_division=0))
    cm=confusion_matrix(y,pred,labels=labels).tolist()
    out={
      'candidate':name,'n':len(y),'accepted_n':int(accepted.sum()),'accepted_accuracy':accepted_acc,
      'overall_accuracy':overall,'macro_f1':macro,'true_rejection_n':int((y==0).sum()),
      'predicted_rejection_n':int((pred==0).sum()),'true_rejection_pred0_rate':float(((y==0)&(pred==0)).sum()/max((y==0).sum(),1)),
      'accepted_predicted_rejection_rate':float(((y!=0)&(pred==0)).sum()/max((y!=0).sum(),1)),
      'false_accept_n':int(((y==0)&(pred!=0)).sum()),'confusion_labels':labels,'confusion_matrix':cm
    }
    return out
metrics={'inputs':{'n':len(x),'official_common_n':int(x.same_official.sum()),'official_discordant_n':int((~x.same_official).sum()),'candidate_logcov_n':len(log),'candidate_corrspd_n':len(corr)},'candidates':[eval_candidate('logcov',x.Label_logcov),eval_candidate('corrspd',x.Label_corrspd)],'pseudo_rule':{'common_labels':'保留两次正式CSV共同标签','discordant_primary':'候选logcov/corrspd对两次正式标签的支持；候选一致性加分；拒识状态半分；并列按正式第1次、第2次、0裁决','discordant_confidence':'low','not_true_recovery':True}}
(OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['# pseudo ground truth v1','',f'- 生成时间：Python运行；样本数 **{len(x)}**。','- 输入：两次正式提交、`candidate_logcov.csv`、`candidate_corrspd.csv`；正式提交文件未修改。','- 两次正式标签共同的 **{int(x.same_official.sum())}** 行直接保留；不同的 **{int((~x.same_official).sum())}** 行按候选一致性、拒识状态与确定性并列规则构造。','- 这不是真实标签恢复：共同标签仅表示两次提交相同，discordant 行为低置信伪标签；不得用于声称官方真值。','', '## 候选相对伪答案']
for z in metrics['candidates']:
 lines += [f"### {z['candidate']}",f"- accepted accuracy（伪答案非0子集，n={z['accepted_n']}）：**{z['accepted_accuracy']:.6f}**",f"- overall accuracy：**{z['overall_accuracy']:.6f}**",f"- macro-F1：**{z['macro_f1']:.6f}**",f"- 伪真值0且预测0：{z['true_rejection_pred0_rate']:.6f}；伪真值0误接受数：{z['false_accept_n']}；非0被拒识率：{z['accepted_predicted_rejection_rate']:.6f}",f"- confusion matrix：见 `metrics.json`，标签顺序 0–20。",'']
lines += ['## 官方分数边界','赛题官方分数需要真实样本分组与真实标签：Clean Accuracy、RA、ASafeR、U。当前只有伪答案，且没有可核验的真实 Clean/Adversarial/Unknown 分组以及真实 RA/ASafeR/U，因此不计算、不声称官方分。','若强行把伪答案的 0 当未知、非0 当已知并把候选0视作拒识，只能得到“clean 伪答案条件分/上界”，但这依赖未经验证的分组假设，不能替代官方分；本次仅报告通用 accepted/overall/macro-F1 与拒识统计。','', '## 限制','- 任务元数据仅用于核验分布和审计；规则没有把 task/session 当作真实标签。','- 伪答案会继承两次正式提交的系统性错误；discordant 行尤其不确定。','- 当前候选无概率分数，无法计算可靠 AUC。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# Audit detail is included in candidate_eval.csv without changing requested pseudo_answer schema.
out=x[['Sample','Label','Label_p1','Label_p2','Label_logcov','Label_corrspd','same_official','candidate_agree','candidate_any_agree','candidate_reject','task','session','source','confidence','rule_reason']]
out.to_csv(OUT/'candidate_eval.csv',index=False)
print(json.dumps({'out':str(OUT),'n':len(x),'common':int(x.same_official.sum()),'discordant':int((~x.same_official).sum()),'metrics':metrics['candidates']},ensure_ascii=False))
