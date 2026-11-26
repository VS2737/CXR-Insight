from radreport.v8_retrieval import build_index
import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='data/processed')
    p.add_argument('--features',default='data/v8_rad_features')
    p.add_argument('--output',default='models_v8/retrieval')
    a=p.parse_args()
    n=build_index(a.data_dir,a.features,a.output)
    print(f'Saved {n} training-only reference analogues. They are NOT diagnoses.')
if __name__=='__main__':main()
