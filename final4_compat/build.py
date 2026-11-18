"""Build the submission from the reviewed source tree and pinned frozen branches."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import torch

out = Path(__file__).resolve().parent
team = out / 'team_hymn'
names = ('README.md', 'enroll.py', 'inference.py', 'model.py', 'model/weights.pt',
         'requirements.txt', 'src/euclidean_gate.py', 'src/cosine_rank.py',
         'src/brainprint_pipeline.py', 'team.json', 'weights.pt')


def main():
    weights = {}
    for source, branch, digest in (
        ('final', 'euclidean_gate', '01716fbe179f7287368d95692f72fd575a3a87aa7ac5efa12d56241dff19777a'),
        ('final2', 'cosine_rank', 'fcf1146f4891715dbec89a3288ae76ed5fc435ec470343a608a747b52cce430c'),
    ):
        with ZipFile(out.parent / source / 'team_hymn.zip') as z:
            raw = z.read('team_hymn/weights.pt')
        assert sha256(raw).hexdigest() == digest
        w = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
        if branch == 'euclidean_gate':
            w.update(format='euclidean_prototypes', source='subwindow_log_covariance_pca')
        else:
            w['protocol'] = 'nested leave-one-subject-out whitened cosine prototypes'
        weights[branch] = w
    buf = BytesIO()
    torch.save(weights, buf)
    for name in ('weights.pt', 'model/weights.pt'):
        (team / name).write_bytes(buf.getvalue())
    with ZipFile(out / 'team_hymn.zip', 'w') as z:
        for name in names:
            z.writestr(ZipInfo('team_hymn/' + name), (team / name).read_bytes(), compress_type=ZIP_DEFLATED)


if __name__ == '__main__':
    main()
