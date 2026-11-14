from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'cache'/'raw'; STATUS=ROOT/'preparation_status.json'; MANIFEST=ROOT/'manifest.json'
ZENODO_API='https://zenodo.org/api/records/3266930'; ZENODO_BASE='https://zenodo.org/records/3266930/files'
NEMAR_BASE='https://data.nemar.org/nm000216/v1.0.0'
HEADERS={'User-Agent':'BrainPrint-BI2015a-preparation/1.0'}

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def save_status(status): STATUS.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load_status():
 if STATUS.exists(): return json.loads(STATUS.read_text(encoding='utf-8'))
 return {'dataset':'BI2015a','record':'3266930','status':'not_started','downloaded':0,'download_total':43,'derived':0,'derived_total':129,'errors':[],'updated_at':now()}
def download(url,dst,expected=None,chunk_size=4*1024*1024):
    """Download each byte range independently; preserve .part on any failure."""
    part=dst.with_suffix(dst.suffix+'.part')
    if expected is None:
        raise ValueError(f'Range download requires expected size: {url}')
    if dst.exists():
        if dst.stat().st_size != expected: raise ValueError(f'existing size mismatch: {dst.stat().st_size} != {expected}')
        return sha256(dst)
    start=part.stat().st_size if part.exists() else 0
    if start > expected: raise ValueError(f'.part larger than expected: {start} > {expected}')
    while start < expected:
        end=min(start+chunk_size,expected)-1
        headers={**HEADERS,'Range':f'bytes={start}-{end}'}
        with requests.get(url,headers=headers,stream=True,timeout=(20,120),allow_redirects=True) as r:
            if r.status_code != 206:
                raise RuntimeError(f' server does not support Range (status {r.status_code})')
            content_range=r.headers.get('Content-Range','')
            expected_range=f'bytes {start}-{end}/{expected}'
            if content_range != expected_range:
                raise RuntimeError(f' invalid Content-Range {content_range!r}; expected {expected_range!r}')
            received=0
            with part.open('ab') as f:
                for block in r.iter_content(1024*1024):
                    if block: f.write(block); received += len(block)
            if received != end-start+1: raise IOError(f'chunk size mismatch: {received} != {end-start+1}')
        start=part.stat().st_size
    if start != expected: raise ValueError(f'size mismatch: {start} != {expected}')
    part.replace(dst)
    return sha256(dst)
def nemar_json(path):
 r=requests.get(f'{NEMAR_BASE}/{path.strip("/")}/',headers=HEADERS,timeout=(20,40)); r.raise_for_status(); return r.json()
def nemar_files(subject):
 found=[]
 for session in range(3):
  path=f'sub-{subject}/ses-{session}/eeg'
  for item in nemar_json(path).get('children',[]):
   name=item.get('name','')
   if item.get('kind')=='file' and (name.lower().endswith('.bdf') or name.lower().endswith('_events.tsv') or name.lower().endswith('_events.json')):
    found.append({'key':f'{path}/{name}','name':name,'size':item.get('size'),'url':f'{NEMAR_BASE}/{path}/{name}' ,'subject':subject,'session':session})
 return found
def parse_subjects(value):
 subjects=[int(x) for x in value.split(',')] if value else [1]
 if any(x not in range(1,44) for x in subjects): raise ValueError('--subjects must contain values in 1..43')
 return subjects
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--source',choices=('nemar','zenodo'),default='nemar')
 parser.add_argument('--subjects',default='1',help='comma-separated subject numbers, e.g. 1,2,3')
 parser.add_argument('--only-bdf',action='store_true')
 parser.add_argument('--chunk-size-mib',type=int,default=4)
 args=parser.parse_args(); subjects=parse_subjects(args.subjects)
 if args.chunk_size_mib <= 0: raise ValueError('--chunk-size-mib must be positive')
 status=load_status(); status.update(status='downloading',source=args.source,updated_at=now(),errors=[]); save_status(status)
 if args.source=='zenodo':
  files=[]
  try:
   r=requests.get(ZENODO_API,headers=HEADERS,timeout=(20,40)); r.raise_for_status()
   files=[f for f in r.json().get('files',[]) if f.get('key','').endswith('.zip') and 'subject_' in f.get('key','')]
   if len(files)!=43: raise ValueError(f'Zenodo API returned {len(files)} subject zip files, expected 43')
   files=[f for i,f in enumerate(sorted(files,key=lambda x:x['key']),1) if i in subjects]
  except Exception as exc:
   status.update(status='blocked',updated_at=now(),errors=[{'stage':'metadata','type':type(exc).__name__,'error':str(exc),'url':ZENODO_API}]); save_status(status); raise
  normalized=[{'key':f['key'],'url':f.get('links',{}).get('self') or f'{ZENODO_BASE}/{f["key"]}?download=1','size':f.get('size'),'subject':subjects[i]} for i,f in enumerate(files)]
  if args.only_bdf: normalized=[]
 else:
  normalized=[]
  try:
   for subject in subjects: normalized.extend(nemar_files(subject))
   if args.only_bdf: normalized=[x for x in normalized if x['name'].lower().endswith('.bdf')]
   normalized.sort(key=lambda x: (x.get('session', 0) != 2, x.get('session', 0) != 1, x.get('session', 0), x.get('size') or 0))
  except Exception as exc:
   status.update(status='blocked',updated_at=now(),errors=[{'stage':'metadata','type':type(exc).__name__,'error':str(exc),'url':NEMAR_BASE}]); save_status(status); raise
 RAW.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.exists() else {'schema_version':2,'dataset':'BI2015a','files':[]}
 manifest.update(source_api=ZENODO_API if args.source=='zenodo' else NEMAR_BASE,source=args.source)
 for item in normalized:
  key=item['key']; dst=RAW/key
  try:
   dst.parent.mkdir(parents=True,exist_ok=True)
   digest=sha256(dst) if dst.exists() and (item.get('size') is None or item['size']==dst.stat().st_size) else download(item['url'],dst,item.get('size'),args.chunk_size_mib*1024*1024)
   entry={'path':str(dst.relative_to(ROOT)).replace('\\','/'),'url':item['url'],'size':dst.stat().st_size,'sha256':digest,'subject':item['subject'],'status':'downloaded'}
   if 'session' in item: entry['session']=item['session']; entry['format']=dst.suffix.lower().lstrip('.')
   manifest['files']=[x for x in manifest.get('files',[]) if x.get('path')!=entry['path']]; manifest['files'].append(entry)
   status['downloaded']=status.get('downloaded',0)+1; status['updated_at']=now(); save_status(status); MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
  except Exception as exc:
   status['errors'].append({'stage':'download','subject':item['subject'],'session':item.get('session'),'file':key,'type':type(exc).__name__,'error':str(exc),'updated_at':now()}); status.update(status='blocked',updated_at=now()); save_status(status)
   continue
 if args.source=='nemar':
  status.update(status='blocked',blocked_reason='NEMAR v1.0.0 provides BIDS BDF/events rather than MAT; fixed 504-trial/6x84 protocol is not validated. Batch download stopped.',updated_at=now())
 else: status.update(status='download_complete',updated_at=now())
 save_status(status)
if __name__=='__main__': main()
