import cv2, numpy as np, sys, os, time
from scipy.signal import savgol_filter

def feats(path, scale=.25):
 cap=cv2.VideoCapture(path); ok,fr=cap.read(); rows=[]
 if not ok: raise RuntimeError(path)
 fr=cv2.resize(fr,None,fx=scale,fy=scale); prev=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY); h,w=prev.shape; f=910*scale
 rows.append([np.nan,np.nan,0,0])
 while True:
  ok,fr=cap.read();
  if not ok: break
  g=cv2.cvtColor(cv2.resize(fr,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY)
  p0=cv2.goodFeaturesToTrack(prev,300,.02,6)
  if p0 is None: rows.append([np.nan,np.nan,0,0]); prev=g; continue
  p1,st,err=cv2.calcOpticalFlowPyrLK(prev,g,p0,None,winSize=(15,15),maxLevel=2)
  a=p0[st.ravel()==1].reshape(-1,2); b=p1[st.ravel()==1].reshape(-1,2); d=b-a; mag=np.linalg.norm(d,axis=1)
  keep=(mag>.08)&(mag<np.percentile(mag,95) if len(mag)>5 else True); a=a[keep]; d=d[keep]; mag=mag[keep]
  if len(a)<8: rows.append([np.nan,np.nan,np.median(mag) if len(mag) else 0,len(a)]); prev=g; continue
  # optical flow translation model: u = tx - x*tz/f ; v = ty-y*tz/f => u = A + B*x, v=C+B*y. joint LS
  x=a[:,0]-w/2; y=a[:,1]-h/2; u=d[:,0]; v=d[:,1]
  # robust iterative least squares for A,C,B
  M=np.zeros((2*len(a),3)); z=np.r_[u,v]
  M[:len(a),0]=1; M[:len(a),2]=x; M[len(a):,1]=1; M[len(a):,2]=y
  mask=np.ones(len(z),bool)
  beta=np.linalg.lstsq(M,z,rcond=None)[0]
  for _ in range(2):
   r=z-M@beta; s=1.4826*np.median(np.abs(r-np.median(r)))+1e-6; mask=np.abs(r)<2.5*s
   if mask.sum()>10: beta=np.linalg.lstsq(M[mask],z[mask],rcond=None)[0]
  A,C,B=beta
  if abs(B)>1e-5:
   foe_x=-A/B; foe_y=-C/B
   yaw=np.arctan2(foe_x,f); pitch=np.arctan2(foe_y,f)
  else: yaw=pitch=np.nan
  rows.append([pitch,yaw,float(np.median(mag)),float(mask.mean())]); prev=g
 cap.release(); return np.array(rows)

if __name__=='__main__':
 for p in sys.argv[1:]:
  t=time.time(); a=feats(p); out=p+'.feat.npy'; np.save(out,a); print(p,a.shape,'valid',np.isfinite(a[:,0]).mean(),'sec',time.time()-t)
