"""Helper: write a zoomed crop of a reference frame with a pixel grid (for landmark picking)."""
import sys, cv2
fn, x0, y0, x1, y1, s, out = sys.argv[1], *map(int, sys.argv[2:7]), sys.argv[7]
im = cv2.imread(fn)[y0:y1, x0:x1]
im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST if s >= 4 else cv2.INTER_CUBIC)
step = int(sys.argv[8]) if len(sys.argv) > 8 else 20
for x in range((x0 // step + 1) * step, x1, step):
    X = (x - x0) * s
    cv2.line(im, (X, 0), (X, im.shape[0]), (0, 255, 255) if x % (5*step) == 0 else (0, 160, 0), 1)
    if x % (5*step) == 0: cv2.putText(im, str(x), (X + 2, 12), 0, 0.4, (0, 255, 255), 1)
for y in range((y0 // step + 1) * step, y1, step):
    Y = (y - y0) * s
    cv2.line(im, (0, Y), (im.shape[1], Y), (0, 255, 255) if y % (5*step) == 0 else (0, 160, 0), 1)
    if y % (5*step) == 0: cv2.putText(im, str(y), (2, Y - 2), 0, 0.4, (0, 255, 255), 1)
cv2.imwrite(out, im)
