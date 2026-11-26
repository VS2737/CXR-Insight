"""Small-scale Optuna learning rate / accumulation ablation with MLflow option."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import optuna

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=3)
    p.add_argument('--epochs',type=int,default=2);a=p.parse_args()
    def objective(trial):
        lr=trial.suggest_float('lr',5e-6,8e-5,log=True)
        accum=trial.suggest_categorical('accumulation',[2,4,8])
        folder=f'models/optuna_{trial.number}'
        cmd=[sys.executable,'-m','scripts.train_generator','--epochs',str(a.epochs),
             '--lr',str(lr),'--accumulation',str(accum),'--output',folder]
        subprocess.run(cmd,check=True)
        meta=json.loads((Path(folder)/'generator_metadata.json').read_text())
        return float(meta['validation_loss'])
    study=optuna.create_study(direction='minimize');study.optimize(objective,n_trials=a.trials)
    print('Best parameters:',study.best_params,'best val loss:',study.best_value)
