import json
import pytest
from radreport.data import freeze_splits


def make_rows(n, with_patient=False):
    return [{'uid':str(i),'patient_id':f'patient_{i//2}' if with_patient else '',
             'impression':'test', 'image_path':'not-used'} for i in range(n)]


def test_default_refuses_unverified_patient_ids(tmp_path):
    with pytest.raises(ValueError, match='patient-level split cannot be claimed'):
        freeze_splits(make_rows(20), tmp_path)


def test_study_split_labelled_as_study(tmp_path):
    meta=freeze_splits(make_rows(20), tmp_path, require_patient_ids=False)
    assert meta['split_level']=='study'
    assert meta['train_count']+meta['val_count']+meta['test_count']==20
    assert (tmp_path/'manifest.json').exists()


def test_patient_groups_never_cross_splits(tmp_path):
    data=make_rows(40, with_patient=True)
    meta=freeze_splits(data,tmp_path)
    assert meta['split_level']=='patient'
    groups=[]
    for split in ('train','val','test'):
        rows=[json.loads(x) for x in (tmp_path/f'{split}.jsonl').read_text().splitlines()]
        groups.append({x['patient_id'] for x in rows})
    assert not(groups[0]&groups[1] or groups[1]&groups[2] or groups[0]&groups[2])
