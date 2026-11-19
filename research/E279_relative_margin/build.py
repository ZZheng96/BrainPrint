"""Build the private candidate from the verified enrolled archive."""
import hashlib, zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'final3_compat/submission_enrolled'; src=root/'single_whitening/team_hymn.zip'; dest=root/'relative_margin/team_hymn.zip'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='14410e8ed2ddaefd4ccdaa6b044b938ad882eccff29e60d2e25cd674a5f2dcd6'
dest.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(src) as z, zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as out:
 for name in z.namelist():
  data=z.read(name)
  if name.endswith('team_hymn_geometry.py'):
   s=data.decode(); old='    reject = (~finite) | (gate < threshold)'; assert s.count(old)==1
   s=s.replace(old,'''    mask = torch.isfinite(sim)
    count = mask.sum(dim=1).clamp_min(1)
    mean = sim.masked_fill(~mask, 0).sum(dim=1) / count
    sd = ((sim - mean[:, None]).masked_fill(~mask, 0).square().sum(dim=1) / count).sqrt()
    rescue = (count >= 2) & (best - second >= sd)
    reject = (~finite) | ((gate < threshold) & ~rescue)''').replace('Column 0 wins exactly when','The absolute gate first checks whether').replace('i.e. when the best match','before the relative margin check, i.e. when the best match'); data=s.encode()
  if name.endswith('README.md'):
   s=data.decode().replace('低于阈值则拒识。','若低于阈值，仅当第一、第二名分差至少达到该查询所有有效身份分数的总体标准差且有至少两个有效身份时放行，否则拒识。'); data=s.encode()
  out.writestr(name,data)
print(dest,hashlib.sha256(dest.read_bytes()).hexdigest())
