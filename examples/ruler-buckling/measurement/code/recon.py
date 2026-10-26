import numpy as np, json, sys
ROOT=__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','..'))  # case root
S=sys.argv[1]
cam=json.load(open(ROOT+'/config/camera_ff.json'))
K=np.array(cam['intrinsics']['K']); R=np.array(cam['extrinsics_world_to_camera_opencv']['R']); t=np.array(cam['extrinsics_world_to_camera_opencv']['t_m'])
C=-R.T@t; Ki=np.linalg.inv(K)
def ray(u,v): d=R.T@(Ki@np.array([u,v,1.0])); return d/np.linalg.norm(d)
def hit(u,v,n,p0=np.zeros(3)):
    d=ray(u,v); s=np.dot(n,p0-C)/np.dot(n,d); return C+s*d
cl=np.load(S+'/cl3.npy',allow_pickle=True)
rows=[]; curves={}
for i,p in enumerate(cl):
    p=p[p[:,1]<640]
    # plane hypotheses through clamp axis: normal n(phi) = (cos phi, sin phi, 0); phi=0 -> plane x=0 (physical weak-axis plane)
    res={}
    for phi in np.radians(np.arange(-80,81,5)):
        n=np.array([np.cos(phi),np.sin(phi),0.])
        X=np.array([hit(u,v,n) for u,v,_ in p[::3]])
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(X,axis=0),axis=1))]+ (X[0,2]-0.095)
        res[round(float(np.degrees(phi)))]=(X,s)
    X,s=res[0]
    # strip top (platform bottom) at s=0.280 along curve from clamp
    k=np.searchsorted(s,0.280); k=min(k,len(s)-1)
    tipS=X[k]
    tipImg=p[::3][k,:2]
    chain_len={ph:float(v[1][-1]) for ph,v in res.items()}
    rows.append(dict(frame=i,t=i/24,tip_u=float(tipImg[0]),tip_v=float(tipImg[1]),tip_x=float(tipS[0]),tip_y=float(tipS[1]),tip_z=float(tipS[2]),
                     delta_mm=float(tipS[1]*1000),chain_top_u=float(p[-1,0]),chain_top_v=float(p[-1,1]),chain_len_x0plane_mm=chain_len[0]*1000,
                     chain_len_y0plane_mm=chain_len[90] *1000 if 90 in chain_len else np.nan))
    curves[i]=X
np.save(S+'/curves.npy',np.array([curves[i] for i in range(len(cl))],dtype=object),allow_pickle=True)
# also the y=0 hypothesis (strong-axis plane) for out-of-plane evidence
for r,p in zip(rows,cl):
    p=p[p[:,1]<640]
    X=np.array([hit(u,v,np.array([0,1.,0])) for u,v,_ in p[::3]])
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(X,axis=0),axis=1))]+(X[0,2]-0.095)
    r['chain_len_y0plane_mm']=float(s[-1]*1000)
    k=min(np.searchsorted(s,0.280),len(s)-1); r['tipx_if_y0plane_mm']=float(X[k,0]*1000)
json.dump(rows,open(S+'/rows.json','w'))
for r in rows[::10]: print(r['frame'],round(r['delta_mm'],1),round(r['tip_z']*1000,1),round(r['chain_len_x0plane_mm']),round(r['chain_len_y0plane_mm']),round(r['tipx_if_y0plane_mm'],1))
