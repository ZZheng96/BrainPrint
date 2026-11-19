"""Build an enrolled two-prototype consensus package without rewriting weights."""
from pathlib import Path
import hashlib, zipfile
root=Path(__file__).resolve().parents[2]/'final3_compat/submission_enrolled'
src=root/'relative_margin/team_hymn.zip'; dest=root/'prototype_consensus/team_hymn.zip'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='995860f8608a91537a4d95926d3043b26bb39c6c2db795d054a630f101cc2251'
dest.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(src) as z, zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as out:
 for name in z.namelist():
  data=z.read(name)
  if name.endswith('team_hymn_geometry.py'):
   old=b'(q[:, None, None, :] * pn[None]).sum(-1).max(2).values'; assert data.count(old)==1
   data=data.replace(old,b'(q[:, None, None, :] * pn[None]).sum(-1).topk(2, dim=2).values.mean(2)')
  if name.endswith('README.md'):
   old='每个身份取四个原型中最大的余弦分数。'.encode(); assert old in data
   data=data.replace(old,'每个身份取四个原型中最高的两个余弦分数的平均值。'.encode())
  out.writestr(name,data)
print(dest,hashlib.sha256(dest.read_bytes()).hexdigest())
