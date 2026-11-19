"""Test 1, step 1: verify how blocks, tiles, and UAV positions map onto the city RS image.
Run from the better-uav-cvgl repo root:  python ood/check_geometry.py
"""
import json, os, random
import numpy as np, pandas as pd
from PIL import Image

ROOT = 'data/Bearing_UAV_90K'
CSV = f'{ROOT}/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv'
RSI = {
    '34bc': '35.67091338738739_139.69289911300856_1791.95_1024_1024_4326_city',
    '36bc': '25.030947387387386_121.51462868800057_1791.95_1024_1024_4326_city',
    '37bc': '1.2897673873873876_103.84197619336068_1791.95_1024_1024_4326_city',
    '38bc': '37.75538738738739_-122.4533351740761_1791.95_1024_1024_4326_city',
}
TILE, UNIT = 256, 128
N_ROWS = 20
random.seed(0)

def path(p):
    return os.path.join('data', p[2:])

def city_of(p):
    return p.split('/')[3].split('_')[2]

def block_tile_ids(p):
    name = os.path.basename(p)[:-4].split('_')
    return int(name[1]), int(name[2]), int(name[4][0]), int(name[4][1])

df = pd.read_csv(CSV)
print(f'rows: {len(df)}   x_norm range [{df.x_norm.min():.3f}, {df.x_norm.max():.3f}]'
      f'   y_norm range [{df.y_norm.min():.3f}, {df.y_norm.max():.3f}]'
      f'   block_x max {df.block_x.max()}   block_y max {df.block_y.max()}')

rows = df.iloc[random.sample(range(len(df)), N_ROWS)]
rsi_img, rsi_json = {}, {}
HYP = {
    'block(a,b)=(x,y) tile(i,j)=(x,y)': lambda a, b, i, j: ((a + i) * TILE, (b + j) * TILE),
    'block(a,b)=(x,y) tile(i,j)=(y,x)': lambda a, b, i, j: ((a + j) * TILE, (b + i) * TILE),
    'block(a,b)=(y,x) tile(i,j)=(x,y)': lambda a, b, i, j: ((b + i) * TILE, (a + j) * TILE),
    'block(a,b)=(y,x) tile(i,j)=(y,x)': lambda a, b, i, j: ((b + j) * TILE, (a + i) * TILE),
}
tile_err = {h: [] for h in HYP}
csv_block_matches_name = True
pos_err = {'x=block_x, y=block_y': [], 'x=block_y, y=block_x': []}

for _, r in rows.iterrows():
    city = city_of(r.p1_path)
    if city not in rsi_img:
        rsi_img[city] = np.asarray(Image.open(f'{ROOT}/city_rsi/{RSI[city]}.jpg').convert('RGB'), dtype=np.float32)
        rsi_json[city] = json.load(open(f'{ROOT}/city_rsi/{RSI[city]}.json'))
    img, J = rsi_img[city], rsi_json[city]
    for col in ['p1_path', 'p2_path', 'p3_path', 'p4_path']:
        a, b, i, j = block_tile_ids(r[col])
        csv_block_matches_name &= (a == r.block_x and b == r.block_y)
        tile = np.asarray(Image.open(path(r[col])).convert('RGB'), dtype=np.float32)
        for h, f in HYP.items():
            x0, y0 = f(a, b, i, j)
            crop = img[y0:y0 + TILE, x0:x0 + TILE]
            tile_err[h].append(np.abs(crop - tile).mean() if crop.shape == tile.shape else np.inf)
    u = json.load(open(path(r.target_path).replace('.jpg', '.json')))
    px = (u['lng'] - J['left_mid'][1]) / J['lng_per_pixel']
    py = (J['top_mid'][0] - u['lat']) / J['lat_per_pixel']
    for h, (bx, by) in {'x=block_x, y=block_y': (r.block_x, r.block_y),
                        'x=block_y, y=block_x': (r.block_y, r.block_x)}.items():
        cx, cy = bx * TILE + TILE, by * TILE + TILE
        pos_err[h].append(np.hypot(cx + r.x_norm * UNIT - px, cy + r.y_norm * UNIT - py))

print(f'\nCSV block_x/block_y match the tile file names: {csv_block_matches_name}')
print('\n(1) Tile vs city-image crop, mean |pixel difference| (0-255 scale; correct one should be small):')
for h, e in tile_err.items():
    print(f'   {h}:  {np.mean(e):7.2f}')
print('\n(2) UAV position from CSV vs from GPS, mean error in pixels (correct one should be < 1):')
for h, e in pos_err.items():
    print(f'   {h}:  {np.mean(e):7.2f}')
