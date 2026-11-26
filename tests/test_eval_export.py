import csv
from scripts.export_eval_for_chexpert import export


def test_exports_chexpert_reference_and_generation_in_same_order(tmp_path):
    testfile = tmp_path/'predictions.csv'
    with testfile.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,['uid','reference','generated']); wr.writeheader()
        wr.writerow({'uid':'a','reference':'No pneumonia, no effusion.','generated':'There is pneumonia, left lower lobe.'})
        wr.writerow({'uid':'b','reference':'Heart normal.','generated':'Heart normal.'})
    n, filenames=export(testfile,tmp_path/'export')
    assert n==2
    def rows(path):
        with open(path,encoding='utf-8',newline='') as f:return list(csv.reader(f))
    assert rows(filenames['reference'])==[['No pneumonia, no effusion.'], ['Heart normal.']]
    assert rows(filenames['generated'])==[['There is pneumonia, left lower lobe.'],['Heart normal.']]
