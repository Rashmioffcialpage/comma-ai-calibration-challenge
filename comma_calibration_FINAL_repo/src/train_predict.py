from pathlib import Path
import argparse, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from model import make_features, fit, predict

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,default=Path('submission'));a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    Xs=[];ys=[]
    for i in range(5):
        X=make_features(np.load(a.data/f'{i}.hevc.feat.npy')); y=np.loadtxt(a.data/f'{i}.txt'); n=min(len(X),len(y));X,y=X[:n],y[:n];g=np.isfinite(y).all(1);Xs.append(X[g]);ys.append(y[g])
    models=fit(np.vstack(Xs),np.vstack(ys))
    for i in range(5,10):
        X=make_features(np.load(a.data/f'{i}.hevc.feat.npy')); p=predict(models,X);np.savetxt(a.out/f'{i}.txt',p,fmt='%.10e');print('wrote',a.out/f'{i}.txt')
if __name__=='__main__':main()
