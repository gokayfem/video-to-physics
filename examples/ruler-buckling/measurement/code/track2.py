import cv2, numpy as np, json, sys
ROOT=__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','..'))  # case root
S=sys.argv[1]
fr=np.load(S+'/frames.npy',mmap_mode='r'); N=len(fr)
bg=cv2.GaussianBlur(np.load(S+'/bg.npy'),(5,5),0).astype(np.float32)
out=[]
for i in range(N):
    f=np.ascontiguousarray(fr[i]); fb=cv2.GaussianBlur(f,(5,5),0)
    d=np.linalg.norm(fb.astype(np.float32)-bg,axis=2)
    sat=cv2.cvtColor(fb,cv2.COLOR_BGR2HSV)[...,1].astype(np.float32)
    thr=22.0
    xc=872.0; w_prev=50; pts=[]; miss=0; tiprow=None
    for y in range(640,5,-1):
        lo=int(max(xc-140,0)); hi=int(min(xc+140,1919)); seg=d[y,lo:hi]
        idx=np.where(seg>thr)[0]
        ok=False
        if len(idx):
            runs=np.split(idx,np.where(np.diff(idx)>2)[0]+1)
            c=min(runs,key=lambda r_:abs(lo+0.5*(r_[0]+r_[-1])-xc))
            a,b=c[0],c[-1]
            L=a-1+(thr-seg[a-1])/(seg[a]-seg[a-1]+1e-9) if a>0 else a
            Rr=b+(seg[b]-thr)/(seg[b]-seg[b+1]+1e-9) if b+1<len(seg) else b
            cx=lo+(L+Rr)/2; w=Rr-L
            ms=np.median(sat[y,lo+a:lo+b+1])
            if ms>95 and len(pts)>50:   # brass reached
                tiprow=y; break
            if abs(cx-xc)<max(10,0.5*w_prev) and 5<w<200:
                ok=True; pts.append((cx+0.5,y+0.5,w)); xc=cx; w_prev=w; miss=0
        if not ok:
            miss+=1
            if miss>45: break
    out.append(np.array(pts))
np.save(S+'/cl2.npy',np.array(out,dtype=object),allow_pickle=True)
print([ (i,(out[i][-1] if len(out[i]) else np.zeros(3)).round(1).tolist(),len(out[i])) for i in range(0,N,10)])
