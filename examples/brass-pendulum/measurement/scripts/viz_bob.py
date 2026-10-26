import cv2, numpy as np, os, sys, pickle
sys.path.insert(0, os.path.dirname(__file__))
from common import *
F = load_frames(); d = np.load(os.path.join(CACHE, "bob_raw.npz")); E = pickle.load(open(os.path.join(CACHE, "bob_edges.pkl"), "rb"))
def overlay(n, z=4, half=80):
    c = d["cA"][n]; R = d["radS"][n]
    u0, v0 = int(c[0]) - half, int(c[1]) - half
    img = cv2.resize(np.ascontiguousarray(F[n][v0:v0 + 2 * half, u0:u0 + 2 * half]), None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    T = lambda u, v: (int(round((u - u0) * z)), int(round((v - v0) * z)))
    P, wt = E[n]
    for (pu, pv), w in zip(P, wt):
        cv2.circle(img, T(pu, pv), 2, (0, 255, 0) if w > 0 else (0, 0, 255), -1)
    cv2.circle(img, T(*c), int(R * z), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.drawMarker(img, T(*c), (255, 255, 0), cv2.MARKER_CROSS, 20, 1)
    b = d["cB"][n, :2] - np.median(d["cB"][:, :2] - d["cA"], axis=0)
    cv2.drawMarker(img, T(*b), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 1)
    cv2.putText(img, "f%d t=%.3f A=(%.2f,%.2f) R=%.2f" % (n, n / 24, c[0], c[1], R), (6, 20), 0, 0.6, (255, 255, 255), 2)
    return img
if __name__ == "__main__":
    out = sys.argv[1]; frames = [int(x) for x in sys.argv[2:]]
    ims = [overlay(n) for n in frames]
    rows = [np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)]
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(out, np.vstack(rows))
