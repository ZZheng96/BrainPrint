from __future__ import annotations
import csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

ROOT=Path(__file__).resolve().parent; RAW=ROOT/'cache'/'raw'; DERIVED=ROOT/'cache'/'derived'; STATUS=ROOT/'preparation_status.json'; MANIFEST=ROOT/'manifest.json'
CFG={'input_sfreq_hz':512,'output_sfreq_hz':300,'bandpass_hz':[1,40],'trial_seconds':1,'channels':32,'runs':6,'scaling':'per-trial median/IQR with near-zero IQR=1','labels_used':False}
def now(): return datetime.now(timezone.utc).isoformat()
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def load_status(): return json.loads(STATUS.read_text(encoding='utf-8')) if STATUS.exists() else {'dataset':'BI2015a','errors':[]}
def save_status(s): STATUS.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def event_rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f,delimiter='\t'))
 return [(float(r['onset']),float(r['duration']),r.get('trial_type','')) for r in rows]
def load_bdf(path):
 try:
  import mne
 except ImportError as exc: raise RuntimeError('mne is required to parse BDF') from exc
 raw=mne.io.read_raw_bdf(path,preload=True,verbose='ERROR')
 eeg=[i for i,n in enumerate(raw.ch_names) if n.upper().startswith('EEG')]
 if len(eeg)!=32 or raw.info['sfreq']!=512: raise ValueError(f'{path}: expected 32 EEG channels/512Hz, got {len(eeg)}/{raw.info["sfreq"]}')
 return raw.get_data(picks=eeg), float(raw.info['sfreq']), raw

def process(subject,session,bdf,events_path):
 rows=event_rows(events_path)
 if len(rows) < CFG['runs']: raise ValueError(f'{events_path}: {len(rows)} events cannot form six non-empty runs')
 data,sfreq,raw=load_bdf(bdf); starts=[]; n_samples=int(round(sfreq))
 for onset,_,_ in rows:
  start=int(round(onset*sfreq))
  if start<0 or start+n_samples>data.shape[1]: raise ValueError(f'{bdf}: event onset {onset} exceeds recording')
  starts.append(start)
 x=np.stack([data[:,s:s+n_samples] for s in starts])
 y=resample_poly(x,CFG['output_sfreq_hz'],CFG['input_sfreq_hz'],axis=-1)
 sos=butter(4,CFG['bandpass_hz'],btype='bandpass',fs=CFG['output_sfreq_hz'],output='sos'); y=sosfiltfilt(sos,y,axis=-1)
 med=np.median(y,axis=-1,keepdims=True); iqr=np.percentile(y,75,axis=-1,keepdims=True)-np.percentile(y,25,axis=-1,keepdims=True); y=((y-med)/np.where(iqr>1e-12,iqr,1)).astype(np.float32)
 parts=np.array_split(np.arange(len(rows)),CFG['runs']); run=np.empty(len(rows),dtype=np.int64)
 for rid,idx in enumerate(parts): run[idx]=rid
 out=DERIVED/f'subject_{subject:02d}'; out.mkdir(parents=True,exist_ok=True); dst=out/f'session_{session+1}.npz'; np.savez_compressed(dst,raw=y,run=run,trial_index=np.arange(len(rows),dtype=np.int64))
 return dst, {'subject':subject,'session':session+1,'trial_count':len(rows),'run_counts':{str(i):int(len(p)) for i,p in enumerate(parts)},'unique_trial_type':sorted({r[2] for r in rows}),'shape':list(y.shape),'sfreq':sfreq,'channels':32,'labels_used':False,'sha256':sha256(dst)}
def main():
 s=load_status(); s.update(status='deriving',updated_at=now()); save_status(s)
 manifest={'schema_version':3,'dataset':'BI2015a','record':'3266930','source':'nemar','preprocessing':CFG,'files':[],'sessions':{}}
 try:
  for subject in [1]:
   for session in range(3):
    base=RAW/f'sub-{subject}/ses-{session}/eeg'; bdfs=list(base.glob('*_eeg.bdf')); evs=list(base.glob('*_events.tsv'))
    if not bdfs or not evs: raise FileNotFoundError(f'sub-{subject}/ses-{session}: BDF/events missing')
    part=bdfs[0].with_suffix(bdfs[0].suffix+'.part')
    if part.exists(): raise RuntimeError(f'{bdfs[0]} incomplete .part exists; resume download before deriving')
    dst,meta=process(subject,session,bdfs[0],evs[0]); manifest['files'].append({'path':str(dst.relative_to(ROOT)).replace('\\','/'),**meta}); manifest['sessions'][f'{subject}:{session+1}']=meta
  manifest['validation']={'status':'complete','derived_files':len(manifest['files'])}; MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); s.update(status='complete',derived=len(manifest['files']),updated_at=now())
 except Exception as exc:
  manifest['validation']={'status':'blocked','error':f'{type(exc).__name__}: {exc}'}; MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); s.setdefault('errors',[]).append({'stage':'derive','type':type(exc).__name__,'error':str(exc)}); s.update(status='blocked',updated_at=now()); save_status(s); raise
 save_status(s)
if __name__=='__main__': main()
