import cv2, numpy as np, sys
S=sys.argv[1]
fr=np.load(S+'/frames.npy',mmap_mode='r'); N=len(fr)
bg=cv2.GaussianBlur(np.load(S+'/bg.npy'),(5,5),0).astype(np.float32)
out=[]; stack=[]
for i in range(N):
    f=np.ascontiguousarray(fr[i]); fb=cv2.GaussianBlur(f,(5,5),0)
    d=np.linalg.norm(fb.astype(np.float32)-bg,axis=2)
    hsv=cv2.cvtColor(fb,cv2.COLOR_BGR2HSV)
    brass=((hsv[...,1]>95)&(d>22)).astype(np.uint8)
    m=((d>18)&(hsv[...,1]<=95)).astype(np.uint8)
    m[655:]=0
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,41)))
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    seed=lab[600:650,855:895]; vals=np.bincount(seed[seed>0].ravel()) if (seed>0).any() else None
    comp=(lab==np.argmax(vals)) if vals is not None else np.zeros_like(m,bool)
    pts=[]
    ys=np.where(comp.any(1))[0]
    for y in range(ys.max(),ys.min()-1,-1):
        xs=np.where(comp[y])[0]
        if len(xs)==0: continue
        a,b=xs[0],xs[-1]; row=d[y]; thr=18.
        L=a-1+(thr-row[a-1])/(row[a]-row[a-1]+1e-9); Rr=b+(row[b]-thr)/(row[b]-row[b+1]+1e-9)
        L=np.clip(L,a-1,a); Rr=np.clip(Rr,b,b+1)
        pts.append((0.5*(L+Rr)+0.5,y+0.5,Rr-L))
    pts=np.array(pts); out.append(pts)
    # brass component near tip
    nb,lb,sb,cb=cv2.connectedComponentsWithStats(cv2.morphologyEx(brass,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)))
    tip=pts[-1,:2]; best=None
    for k in range(1,nb):
        if sb[k,4]<150: continue
        yy,xx=np.where(lb==k); dist=np.min(np.hypot(xx-tip[0],yy-tip[1]))
        best=(k,dist,sb[k,4],xx,yy) if (best is None or dist<best[1]) else best
    if best and best[1]<12:
        xx,yy=best[3],best[4]
        # tangent at tip from last 30 px of centreline
        q=pts[-30:,:2]; tv=q[-1]-q[0]; tv/=np.linalg.norm(tv)
        s=(xx-tip[0])*tv[0]+(yy-tip[1])*tv[1]
        stack.append((i,best[2],float(np.percentile(s,99.5)),float(np.percentile(s,0.5))))
    else: stack.append((i,0,0.,0.))
    # all brass blobs in upper image (floating weights)
np.save(S+'/cl3.npy',np.array(out,dtype=object),allow_pickle=True)
np.save(S+'/stack.npy',np.array(stack))
for i in range(0,N,10): print(i,out[i][0].round(1),out[i][-1].round(1),len(out[i]),np.round(stack[i],1))
