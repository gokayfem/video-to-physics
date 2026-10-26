import cv2, numpy as np, json, sys
ROOT=__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','..'))  # case root
S=sys.argv[1]
fr=np.load(S+'/frames.npy'); N=len(fr)
cam=json.load(open(ROOT+'/config/camera_ff.json'))
K=np.array(cam['intrinsics']['K']); R=np.array(cam['extrinsics_world_to_camera_opencv']['R']); t=np.array(cam['extrinsics_world_to_camera_opencv']['t_m'])
C=-R.T@t
def proj(X): x=R@X+t; return np.array([K[0,0]*x[0]/x[2]+K[0,2],K[1,1]*x[1]/x[2]+K[1,2]])
# ---- 1. camera stability
g0=cv2.cvtColor(fr[0],cv2.COLOR_BGR2GRAY)
H,W=g0.shape
mask=np.full((H,W),255,np.uint8)
cv2.rectangle(mask,(450,0),(1400,720),0,-1)       # strip/tip swept region
cv2.rectangle(mask,(1000,780),(1180,960),0,-1)      # bench weight stack (check separately)
orb=cv2.ORB_create(4000)
k0,d0=orb.detectAndCompute(g0,mask)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
probe=np.float32([[200,200],[1700,200],[200,950],[1700,950],[872,653]]).reshape(-1,1,2)
cam_rows=[]
for i in range(N):
    g=cv2.cvtColor(fr[i],cv2.COLOR_BGR2GRAY)
    k,d=orb.detectAndCompute(g,mask)
    m=bf.match(d0,d)
    p0=np.float32([k0[a.queryIdx].pt for a in m]); p1=np.float32([k[a.trainIdx].pt for a in m])
    Hm,inl=cv2.findHomography(p1,p0,cv2.RANSAC,2.0)
    inl=inl.ravel().astype(bool)
    r=np.linalg.norm(cv2.perspectiveTransform(p1[inl].reshape(-1,1,2),Hm).reshape(-1,2)-p0[inl],axis=1)
    disp=np.linalg.norm(cv2.perspectiveTransform(probe,Hm).reshape(-1,2)-probe.reshape(-1,2),axis=1)
    cam_rows.append((int(inl.sum()),float(np.median(r)),float(disp.max()),float(disp[4])))
cam_rows=np.array(cam_rows)
np.save(S+'/cam.npy',cam_rows)
print('camera: inliers min',cam_rows[:,0].min(),'max probe disp px',cam_rows[:,2].max(),'clamp-pt disp max',cam_rows[:,3].max())
# ---- 2. strip centreline (row scan, bottom-up, from jaw top)
def score(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV).astype(np.float32)
    return (hsv[...,2]/255.)*(1-hsv[...,1]/255.)   # bright & unsaturated
def brass(im):
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    return ((hsv[...,0]>=8)&(hsv[...,0]<=28)&(hsv[...,1]>110)&(hsv[...,2]>90))
y_start=640  # just above jaw top (653 px) to avoid jaw
res=[]
for i in range(N):
    s=cv2.GaussianBlur(score(fr[i]),(3,3),0)
    xc=872.0; pts=[]; widths=[]
    miss=0
    for y in range(y_start,15,-1):
        row=s[y]; lo=int(max(xc-60,0)); hi=int(min(xc+60,W-1))
        seg=row[lo:hi]; thr=0.5*(seg.max()+np.median(seg)) if seg.max()>0.55 else None
        if thr is None: miss+=1
        else:
            on=seg>thr
            # run containing argmax nearest to xc
            idx=np.where(on)[0]
            if len(idx)==0: miss+=1
            else:
                # split runs
                runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1)
                c=min(runs,key=lambda r_:abs(lo+r_.mean()-xc))
                a,b=c[0],c[-1]
                # subpixel edges by linear interp of threshold crossing
                def xing(j0,j1):
                    v0,v1=seg[j0],seg[j1]; return j0+(thr-v0)/(v1-v0+1e-9)*(j1-j0)
                L=xing(a-1,a) if a>0 else a; Rr=xing(b,b+1) if b+1<len(seg) else b
                w=Rr-L
                if abs(lo+(L+Rr)/2-xc)>12 or w<6 or w>110: miss+=1
                else:
                    miss=0; xc=lo+(L+Rr)/2; pts.append((xc,y+0.5)); widths.append(w)
        if miss>4: break
    pts=np.array(pts); widths=np.array(widths)
    res.append((pts,widths))
np.save(S+'/cl.npy',np.array([r[0] for r in res],dtype=object),allow_pickle=True)
np.save(S+'/wd.npy',np.array([r[1] for r in res],dtype=object),allow_pickle=True)
print('tips', [ (i,res[i][0][-1].round(1).tolist()) for i in range(0,N,20)])
