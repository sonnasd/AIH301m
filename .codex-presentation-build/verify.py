import csv
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / '.codex-presentation-build'
POLICY = json.loads((ROOT / 'artifacts/ai_policy.json').read_text())
METRICS = json.loads((ROOT / 'artifacts/evaluation_metrics.json').read_text())


def records(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


train = records(ROOT / 'data/trajectories_train.csv')
test = records(ROOT / 'data/trajectories_test.csv')
val = records(ROOT / 'data/trajectories_val.csv')
patient_sets = [{r['subject_id'] for r in rows} for rows in [train, val, test]]
assert list(map(len, patient_sets)) == [420, 90, 90]
assert all(not patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3))
matrices = np.load(ROOT / 'artifacts/mdp_matrices.npz')
assert matrices['transitions'].shape == (77, 25, 77)
assert matrices['valid_mask'][:75].sum() == 327
np.testing.assert_allclose(matrices['transitions'].sum(axis=2)[matrices['valid_mask']], 1)
assert np.all(matrices['V'][75:] == 0)
counts = matrices['counts'].sum(axis=2)[:75]
assert np.array_equal(np.flatnonzero(counts.max(axis=1) < 5), [27, 28, 52, 58, 71])
assert (counts.sum(axis=1) > 0).all()
assert [x['actions_changed'] for x in POLICY['history']] == [53, 8, 0]
np.testing.assert_allclose(matrices['V'][:75].mean(), POLICY['mean_v'])
np.testing.assert_allclose(matrices['td_V'][:75].mean(), POLICY['td_clinician_mean_v'])
behavior = (counts + 0.05) / (counts.sum(axis=1, keepdims=True) + 25 * 0.05)
episodes = defaultdict(list)
for r in test:
    episodes[tuple(r[c] for c in ['subject_id', 'hadm_id', 'icustay_id'])].append(r)
returns, logs = [], []
for episode in episodes.values():
    episode.sort(key=lambda r: int(r['time_step']))
    assert [int(r['time_step']) for r in episode] == list(range(len(episode)))
    returns.append(sum(0.99 ** t * float(r['reward']) for t, r in enumerate(episode)))
    log_weight = 0
    for r in episode:
        s, a = int(r['state']), int(r['action'])
        target = 0.99 if a == POLICY['policy'][s] else 0.01 / 24
        log_weight += math.log(target) - math.log(behavior[s, a])
    logs.append(min(log_weight, math.log(500)))
weights = np.exp(np.array(logs) - max(logs))
weights /= weights.sum()
np.testing.assert_allclose(weights @ returns, METRICS['point_wis'], atol=1e-10)
np.testing.assert_allclose(np.mean(returns), METRICS['point_clinician'], atol=1e-10)
np.testing.assert_allclose(1 / (weights @ weights), METRICS['ess'], atol=1e-10)
assert math.isclose(0.5 * 0.99 * 60 + 0.5 * 100, 79.7)
assert math.isclose(0.8 * 100 - 0.2 * 100, 60)
assert math.isclose(50 / (1 - 0.495), 99.00990099009901)
assert len(re.findall(r'^    def test_', (ROOT / 'test_regressions.py').read_text(), re.M)) == 19

deck = Path(sys.argv[1]) if len(sys.argv) > 1 else BUILD / 'candidate.pptx'
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main', 'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
with zipfile.ZipFile(deck) as package:
    slide_files = [f for f in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', f)]
    note_files = [f for f in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', f)]
    assert len(slide_files) == len(note_files) == 32
    durations = []
    for i in range(1, 33):
        slide = ET.fromstring(package.read(f'ppt/slides/slide{i}.xml'))
        note = ET.fromstring(package.read(f'ppt/notesSlides/notesSlide{i}.xml'))
        content = '\n'.join(t.text or '' for t in slide.findall('.//a:t', ns))
        notes = '\n'.join(t.text or '' for t in note.findall('.//a:t', ns))
        assert 'Nguồn:' in notes and 'Ý chính để tự nói:' in notes
        assert '•' not in content
        durations.append(int(re.search(r'Thời lượng dự kiến: (\d+) giây', notes).group(1)))
        who = 'Bùi Đức Giang' if i <= 9 else 'Nguyễn Huy Sơn' if i <= 20 else 'Cường' if i <= 28 else 'Nhóm'
        assert f'Người trình bày: {who}' in notes
        for props in slide.findall('.//a:rPr', ns):
            font = props.find('a:latin', ns)
            if font is not None:
                assert font.attrib.get('typeface') == 'Arial', (i, font.attrib)
        if i in [12, 19, 24, 25, 26]:
            assert 'Mô phỏng' in content or 'mô phỏng' in content
        if i in [24, 25]:
            assert 'ESS' in content and '1,90' in content
        assert not slide.findall('.//p:pic', ns)
    assert [sum(durations[:9]), sum(durations[9:20]), sum(durations[20:28]), sum(durations[28:])] == [540, 720, 540, 0]
    assert len([f for f in package.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/[^/]+\.xml', f)]) == 1
print('PASS: saved data, WIS, ESS, Bellman example, 32 slides/notes, 1,800 seconds, speakers, editable objects.')
