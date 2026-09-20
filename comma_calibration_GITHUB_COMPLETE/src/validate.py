from pathlib import Path
import argparse, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from model import make_features, fit, predict

def score(gt, pred):
    pred = np.nan_to_num(pred)
    return np.mean(np.nanmean((gt - pred) ** 2, axis=0))

def load_one(data, i):
    raw = np.load(data / f"{i}.hevc.feat.npy")
    X = make_features(raw)
    y = np.loadtxt(data / f"{i}.txt")
    n = min(len(X), len(y)); X, y = X[:n], y[:n]
    good = np.isfinite(y).all(axis=1)
    return X[good], y[good]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); a=ap.parse_args()
    videos=[load_one(a.data,i) for i in range(5)]
    mses=[]; zeros=[]
    for h in range(5):
        X=np.vstack([videos[j][0] for j in range(5) if j!=h])
        y=np.vstack([videos[j][1] for j in range(5) if j!=h])
        Xt,yt=videos[h]; p=predict(fit(X,y),Xt)
        m,z=score(yt,p),score(yt,np.zeros_like(yt)); mses.append(m);zeros.append(z)
        print(f"video {h}: {100*m/z:.2f}%")
    print(f"aggregate: {100*np.mean(mses)/np.mean(zeros):.2f}%")
if __name__=='__main__': main()
