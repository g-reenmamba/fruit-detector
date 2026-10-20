"""
make_background.py
------------------
Draws the pastel pixel-art landscape behind the app (original artwork,
generated procedurally so it can be tweaked and re-rendered).

    python3 scripts/make_background.py      ->  assets/background.png

The PNG is tiny (320x180 pixels); the app scales it up with
`image-rendering: pixelated`, so every pixel stays a crisp square.
"""

import math
import os
import random

from PIL import Image

W, H = 320, 180
rng = random.Random(7)

# palette (sampled to sit with the app's colour tokens)
SKY = [(176, 186, 236), (186, 196, 240), (197, 207, 242), (208, 220, 240), (214, 232, 236)]
CLOUD_PINK = [(236, 166, 190), (244, 190, 206), (250, 214, 222)]
CLOUD_BUTTER = [(244, 214, 150), (250, 232, 178), (255, 244, 206)]
CLOUD_MINT = [(170, 214, 200), (196, 228, 214)]
MTN_FAR = [(168, 166, 214), (182, 180, 224)]
MTN_NEAR = [(140, 138, 196), (154, 152, 208), (126, 124, 182)]
PINE = [(38, 92, 78), (58, 128, 90), (90, 164, 104), (140, 200, 120)]
TRUNK = [(92, 66, 96), (120, 86, 112)]
GRASS = [(110, 176, 104), (138, 198, 112), (170, 216, 126)]
PATH = [(206, 178, 150), (222, 198, 168)]
ROCK = [(150, 150, 178), (178, 178, 204), (206, 206, 226), (120, 120, 150)]
SPARK = [(255, 250, 220), (255, 236, 160)]

img = Image.new("RGB", (W, H))
px = img.load()

BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]


def put(x, y, c):
    if 0 <= x < W and 0 <= y < H:
        px[x, y] = c


# --- sky: banded gradient with ordered dithering between bands --------------
for y in range(H):
    t = y / 120 * (len(SKY) - 1)
    i = min(int(t), len(SKY) - 2)
    f = t - i
    for x in range(W):
        px[x, y] = SKY[i + 1] if f * 16 > BAYER[y % 4][x % 4] else SKY[i]


# --- clouds: clumps of quantised circles, shaded from the top ---------------
def cloud(cx, cy, w, palette):
    blobs = [(cx + rng.randint(-w // 2, w // 2), cy + rng.randint(-3, 3), rng.randint(5, 11))
             for _ in range(max(3, w // 6))]
    for x in range(cx - w, cx + w):
        for y in range(cy - 16, cy + 12):
            inside = [b for b in blobs if (x - b[0]) ** 2 + ((y - b[1]) * 1.4) ** 2 < b[2] ** 2]
            if not inside:
                continue
            if y > cy + 6:
                continue  # flat-ish bottom
            top = min(b[1] - b[2] / 1.4 for b in inside)
            depth = y - top
            shade = 2 if depth < 3 else (1 if depth < 8 else 0)
            put(x, y, palette[min(shade, len(palette) - 1)])


for cx, cy, w, pal in [(-4, 92, 40, CLOUD_PINK), (100, 84, 30, CLOUD_BUTTER), (226, 70, 36, CLOUD_PINK),
                       (330, 86, 40, CLOUD_BUTTER), (18, 52, 34, CLOUD_PINK), (62, 70, 26, CLOUD_PINK), (40, 30, 16, CLOUD_MINT),
                       (250, 46, 30, CLOUD_MINT), (300, 74, 34, CLOUD_BUTTER), (212, 84, 22, CLOUD_PINK),
                       (150, 24, 18, CLOUD_MINT), (110, 96, 18, CLOUD_BUTTER), (282, 104, 26, CLOUD_PINK)]:
    cloud(cx, cy, w, pal)


# --- mountains: overlapping triangular peaks, lit from the left -----------
def ridge(peaks, floor):
    out = []
    for x in range(W):
        h = floor
        for px_, top, half in peaks:
            d = abs(x - px_)
            if d < half:
                h = min(h, int(top + (floor - top) * d / half))
        out.append(h)
    return out


def jitter(line, amount, seed):
    r = random.Random(seed)
    return [v + (r.randint(0, amount) if (i // 3) % 2 else 0) for i, v in enumerate(line)]


far = jitter(ridge([(20, 92, 46), (96, 80, 60), (176, 88, 50), (238, 74, 64), (310, 90, 50)], 132), 1, 1)
near = jitter(ridge([(-10, 108, 50), (58, 100, 46), (140, 112, 40), (196, 96, 52), (268, 104, 44),
                     (330, 98, 40)], 150), 0, 2)
for x in range(W):
    for y in range(far[x], H):
        put(x, y, MTN_FAR[1])
    if far[x] < 140:
        lit = x > 0 and far[x] <= far[x - 1]
        put(x, far[x], MTN_FAR[0] if lit else MTN_FAR[1])
    for y in range(near[x], H):
        rising = x > 0 and near[x] < near[x - 1]   # left face of a peak -> lit
        flat = x > 0 and near[x] == near[x - 1]
        c = MTN_NEAR[1] if (rising or flat) else MTN_NEAR[2]
        if y < near[x] + 2 and (rising or flat):
            c = MTN_NEAR[0]
        put(x, y, c)
# --- ground ------------------------------------------------------------------
GROUND = 150
for x in range(W):
    edge = GROUND + int(2 * math.sin(x / 9)) + (1 if rng.random() < 0.15 else 0)
    for y in range(edge, H):
        c = GRASS[2] if y < edge + 2 else (GRASS[1] if (x * 5 + y * 3) % 23 else GRASS[0])
        put(x, y, c)
# dirt path winding toward the front
for y in range(GROUND + 6, H):
    t = (y - GROUND) / (H - GROUND)
    cx = 168 + int(18 * math.sin(t * 3.0)) - int(30 * t)
    half = int(6 + 26 * t)
    for x in range(cx - half, cx + half):
        put(x, y, PATH[1] if (x + y) % 5 else PATH[0])


# --- pine trees: stacked sawtooth tiers, lit from the upper left -----------
def pine(cx, base, height, width):
    trunk_h = max(3, height // 7)
    for x in range(cx - 1, cx + 2):
        for y in range(base - trunk_h - 2, base + 1):
            put(x, y, TRUNK[0] if x == cx + 1 else TRUNK[1])
    crown_bottom = base - trunk_h
    top = base - height
    tiers = max(3, height // 11)
    tier_h = (crown_bottom - top) / tiers * 1.55
    for t in range(tiers):
        ty = int(top + t * (crown_bottom - top - tier_h) / max(tiers - 1, 1))
        tw = width * (0.45 + 0.55 * (t + 1) / tiers)
        for y in range(ty, int(ty + tier_h)):
            frac = (y - ty) / tier_h
            half = int(tw / 2 * frac) + (1 if t == 0 and y < ty + 2 else 0)
            for x in range(cx - half, cx + half + 1):
                rel = (x - cx) / max(half, 1)
                bottom_row = y >= int(ty + tier_h) - 2
                if bottom_row and (x + t) % 4 in (0, 1):
                    continue                       # sawtooth fringe under each tier
                if bottom_row:
                    c = PINE[0]
                elif rel < -0.45 and frac > 0.25:
                    c = PINE[3] if (x + 2 * y) % 5 == 0 else PINE[2]
                elif rel > 0.35:
                    c = PINE[0]
                else:
                    c = PINE[1]
                put(x, y, c)


for cx, h, w in [(14, 46, 22), (44, 34, 18), (288, 52, 26), (312, 38, 20), (250, 30, 16),
                 (78, 26, 14), (120, 92, 46), (204, 40, 20)]:
    pine(cx, GROUND + 4 + rng.randint(0, 6), h, w)


# --- rocks ---------------------------------------------------------------
def rock(cx, cy, rw, rh):
    for x in range(cx - rw, cx + rw + 1):
        for y in range(cy - rh, cy + 1):
            if ((x - cx) / rw) ** 2 + ((y - cy) / rh) ** 2 > 1:
                continue
            fx, fy = (x - cx) / rw, (y - cy) / rh
            c = ROCK[2] if fx + fy < -0.9 else (ROCK[1] if fx < 0.35 else ROCK[0])
            if y == cy:
                c = ROCK[3]
            put(x, y, c)


for args in [(96, 172, 10, 6), (238, 168, 14, 8), (150, 160, 5, 3), (190, 175, 6, 4),
             (30, 176, 12, 6), (276, 160, 6, 4), (60, 160, 4, 3)]:
    rock(*args)


# --- sparkles (4-point stars) and floating flecks -------------------------
def star(x, y, r):
    put(x, y, SPARK[0])
    for d in range(1, r + 1):
        c = SPARK[0] if d < r else SPARK[1]
        for dx, dy in [(d, 0), (-d, 0), (0, d), (0, -d)]:
            put(x + dx, y + dy, c)


for x, y, r in [(122, 12, 3), (96, 36, 1), (178, 30, 2), (232, 22, 1), (270, 18, 2), (60, 14, 1),
                (196, 60, 1), (30, 92, 1), (304, 30, 1), (140, 58, 1)]:
    star(x, y, r)
for _ in range(60):
    x, y = rng.randrange(W), rng.randrange(20, GROUND)
    if px[x, y] in SKY or px[x, y] in MTN_FAR or px[x, y] in PINE:
        put(x, y, (240, 244, 255))

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "background.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
img.save(out, optimize=True)
print("saved", out, os.path.getsize(out), "bytes")
