"""Test 1: Bearing-UAV under controlled block shifts (Cases 1-4 at adjustable severity).

For every test sample (the authors' seed-42 85/5/10 split) and every shift distance d,
the 2x2 reference block is re-cropped from the city image, displaced by d meters in a
random direction (one direction per sample, shared by all d). The model is run on the
shifted block and everything is logged per (sample, d).

Geometry (verified with ood/check_geometry.py):
  city image 4096 px, 0.25 m/px; tile 256 px = 64 m; block 512 px = 128 m;
  1 normalized unit = 128 px = 32 m; block edges at +-2 units; training positions in [-1, 1].
  block_x/block_y are x/y tile offsets; y points down. p1..p4 = top-left, top-right,
  bottom-left, bottom-right.

Cases (from the true position (x', y') relative to the shifted block, in units):
  1  |x'|,|y'| <= 1            position in the training region
  2  inside block, not case 1  position inside block (<= 2) but outside the training region
  3  outside block, overlap > 0
  4  footprint does not overlap the block

Grid mode (--grid): shifts by whole tiles (k = 0..3 tiles, one of 8 directions per sample),
so every block stays aligned to the 256-px tile grid the model was trained on and that
navigation uses. This separates the out-of-block effect from crop misalignment.

Run from better-uav-cvgl/:
  python ood/eval_shift.py                     # full run (GPU)
  python ood/eval_shift.py --grid              # grid-aligned run
  python ood/eval_shift.py --limit 20          # quick check
  python ood/eval_shift.py --dry-run --limit 200   # labels only, no model
"""
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd
from PIL import Image

TILE, BLOCK, UNIT, M_PER_PX = 256, 512, 128, 0.25
M_PER_UNIT = UNIT * M_PER_PX                      # 32 m
CITY_PX = 4096
RSI = {
    '34bc': '35.67091338738739_139.69289911300856_1791.95_1024_1024_4326_city',
    '36bc': '25.030947387387386_121.51462868800057_1791.95_1024_1024_4326_city',
    '37bc': '1.2897673873873876_103.84197619336068_1791.95_1024_1024_4326_city',
    '38bc': '37.75538738738739_-122.4533351740761_1791.95_1024_1024_4326_city',
}
DEFAULT_D = [0, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/Bearing_UAV_90K')
    ap.add_argument('--dset', default='c4m_254k_96bc_b15_s100_v3d')
    ap.add_argument('--bearing', default='external/bearinguav', help='Bearing-UAV code dir')
    ap.add_argument('--weights', default='external/bearinguav/Bearing_UAV/cross_view/best_model.pth')
    ap.add_argument('--d', type=float, nargs='+', default=DEFAULT_D, help='shift distances (m)')
    ap.add_argument('--seed', type=int, default=0, help='seed for shift directions')
    ap.add_argument('--limit', type=int, default=0, help='use only the first N test samples')
    ap.add_argument('--batch', type=int, default=4, help='test samples per batch (x len(d) blocks)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='outputs/test1')
    ap.add_argument('--dry-run', action='store_true', help='compute labels only, no model')
    ap.add_argument('--grid', action='store_true', help='grid-aligned shifts by whole tiles')
    ap.add_argument('--tiles', type=int, nargs='+', default=[0, 1, 2, 3], help='grid mode: shifts in tiles')
    return ap.parse_args()


# ---------------------------------------------------------------- data split
def test_indices(n):
    """Same split as Bearing-UAV's par_dataloader (85/5/10, torch seed 42)."""
    import torch
    from torch.utils.data import random_split
    tr, va = int(0.85 * n), int(0.05 * n)
    _, _, te = random_split(range(n), [tr, va, n - tr - va], generator=torch.Generator().manual_seed(42))
    return list(te.indices)


# ---------------------------------------------------------------- geometry
def city_of(p):          # './Bearing_UAV_90K/citya/sat_254k_34bc_b15_s100/...' -> '34bc'
    return p.split('/')[3].split('_')[2]


def shifted_origin(bx, by, d_m, phi):
    """Top-left pixel of the shifted block. An axis that would leave the image is reflected."""
    s = d_m / M_PER_PX
    ox, oy = bx * TILE, by * TILE
    sx, sy = s * math.cos(phi), s * math.sin(phi)
    nx, ny = ox + sx, oy + sy
    lim = CITY_PX - BLOCK
    if not 0 <= nx <= lim:
        nx = ox - sx
    if not 0 <= ny <= lim:
        ny = oy - sy
    return int(round(nx)), int(round(ny))


_U, _V = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41))   # footprint sample grid (units)


def overlap_ratio(x, y, theta_deg):
    """Fraction of the 2x2-unit footprint (64 m square, rotated by theta) inside the block [-2,2]^2.
    Approximation: square footprint, theta counter-clockwise in a y-up frame."""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    px = x + c * _U - s * _V
    py_up = -y + s * _U + c * _V          # work in y-up, then flip back
    inside = (np.abs(px) <= 2) & (np.abs(py_up) <= 2)
    return float(inside.mean())


def label(x, y, theta):
    ov = overlap_ratio(x, y, theta)
    inside = abs(x) <= 2 and abs(y) <= 2
    if abs(x) <= 1 and abs(y) <= 1:
        case = 1
    elif inside:
        case = 2
    elif ov > 0:
        case = 3
    else:
        case = 4
    dist_out = math.hypot(max(abs(x) - 2, 0), max(abs(y) - 2, 0)) * M_PER_UNIT
    return case, inside, ov, dist_out


DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
N_BLOCKS = CITY_PX // TILE - 1                       # 15 block positions per axis (0..14)


def grid_origin(bx, by, k, dx, dy):
    """Top-left pixel of the block k tiles away in direction (dx, dy); an axis leaving the grid is reflected."""
    nbx, nby = bx + dx * k, by + dy * k
    if not 0 <= nbx < N_BLOCKS:
        nbx = bx - dx * k
    if not 0 <= nby < N_BLOCKS:
        nby = by - dy * k
    return nbx * TILE, nby * TILE


def plan(df, idx, ds, seed, grid=False, tiles=(0, 1, 2, 3)):
    """One row per (test sample, shift): shifted block origin, new target, labels."""
    rng = np.random.default_rng(seed)
    phis = rng.uniform(0, 2 * math.pi, len(idx))
    dirs = rng.integers(0, 8, len(idx))
    out = []
    for k, (i, phi, di) in enumerate(zip(idx, phis, dirs)):
        r = df.iloc[i]
        bx, by = int(r.block_x), int(r.block_y)
        px = (bx + 1) * TILE + r.x_norm * UNIT          # UAV center, city-image pixels
        py = (by + 1) * TILE + r.y_norm * UNIT
        for t, d in enumerate(tiles if grid else ds):
            if grid:
                nx, ny = grid_origin(bx, by, d, *DIRS8[di])
            else:
                nx, ny = shifted_origin(bx, by, d, phi)
            x, y = (px - (nx + TILE)) / UNIT, (py - (ny + TILE)) / UNIT
            case, inside, ov, dist_out = label(x, y, r.theta)
            dxm, dym = (nx - bx * TILE) * M_PER_PX, (ny - by * TILE) * M_PER_PX
            out.append(dict(sample=k, row=i, city=city_of(r.p1_path), block_x=bx, block_y=by,
                            theta=r.theta, d_m=round(math.hypot(dxm, dym), 1) if grid else d,
                            shift_tiles=d if grid else np.nan, shift_x_m=dxm, shift_y_m=dym,
                            phi_deg=math.degrees(math.atan2(dym, dxm)) % 360 if (dxm or dym) else 0.0,
                            x_orig=r.x_norm, y_orig=r.y_norm, x_new=x, y_new=y,
                            case=case, inside=inside, overlap=ov, dist_out_m=dist_out,
                            origin_x=nx, origin_y=ny, target_path=r.target_path,
                            cos_t=r.x_cosa, sin_t=r.y_sina))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- model run
def run_model(P, args, n_d):
    import torch
    from torch.utils.data import Dataset, DataLoader
    sys.path.insert(0, os.path.abspath(args.bearing))
    from cvphr.models.posaglreg.models import MODEL_CLASS_DICT, model_kwargs_par_ca_sgm_v5a
    from cvphr.utils.utils_transform import transform_pipeline3

    tf = transform_pipeline3()                                       # ToTensor + ImageNet Normalize
    city_img = {c: np.asarray(Image.open(f'{args.data}/city_rsi/{RSI[c]}.jpg').convert('RGB'))
                for c in P.city.unique()}

    class ShiftSet(Dataset):                 # one item = one test sample with all its shifted blocks
        def __len__(self):
            return len(P) // n_d

        def __getitem__(self, k):
            rows = P.iloc[k * n_d:(k + 1) * n_d]
            uav = tf(Image.open(os.path.join(os.path.dirname(args.data), rows.target_path.iloc[0][2:])).convert('RGB'))
            img = city_img[rows.city.iloc[0]]
            blocks = []
            for nx, ny in zip(rows.origin_x, rows.origin_y):
                b = img[ny:ny + BLOCK, nx:nx + BLOCK]
                tiles = [b[:TILE, :TILE], b[:TILE, TILE:], b[TILE:, :TILE], b[TILE:, TILE:]]   # p1..p4
                blocks.append(torch.stack([tf(Image.fromarray(t)) for t in tiles] + [uav]))
            return torch.stack(blocks)                                 # [n_d, 5, 3, 256, 256]

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODEL_CLASS_DICT['PARCASGM_v5a'](**model_kwargs_par_ca_sgm_v5a).to(dev)
    ckpt = torch.load(args.weights, map_location=dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    sims = []                                                          # pre-softmax tile similarities
    model.sim_pos_prior.cos.register_forward_hook(lambda m, i, o: sims.append(o.detach().float().cpu()))

    loader = DataLoader(ShiftSet(), batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    pos, dirs = [], []
    t0 = time.time()
    with torch.no_grad():
        for j, x in enumerate(loader):
            x = x.flatten(0, 1).to(dev, non_blocking=True)            # [B*n_d, 5, 3, H, W]
            p, d = model(x)
            pos.append(p.float().cpu()); dirs.append(d.float().cpu())
            if j % 50 == 0:
                print(f'  batch {j + 1}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
    pos = torch.cat(pos).numpy(); dirs = torch.cat(dirs).numpy(); sim = torch.cat(sims).numpy()

    P = P.copy()
    P['pred_x'], P['pred_y'] = pos[:, 0], pos[:, 1]
    P['pos_err_m'] = np.hypot(pos[:, 0] - P.x_new, pos[:, 1] - P.y_new) * M_PER_UNIT
    gt = np.stack([P.cos_t, P.sin_t], 1)
    cosang = (dirs * gt).sum(1) / (np.linalg.norm(dirs, axis=1) * np.linalg.norm(gt, axis=1) + 1e-8)
    P['head_err_deg'] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    for t in range(4):
        P[f'sim{t + 1}'] = sim[:, t]
    e = np.exp(sim - sim.max(1, keepdims=True))
    P['max_prob'] = (e / e.sum(1, keepdims=True)).max(1)
    print(f'model run: {len(P)} blocks in {time.time() - t0:.0f}s on {dev}')
    return P


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    df = pd.read_csv(f'{args.data}/{args.dset}/metadata/metadata.csv')
    idx = test_indices(len(df))
    if args.limit:
        idx = idx[:args.limit]
    n_shift = len(args.tiles) if args.grid else len(args.d)
    print(f'test samples: {len(idx)}   ' + (f'grid shifts (tiles): {args.tiles}' if args.grid else f'shifts (m): {args.d}'))

    P = plan(df, idx, args.d, args.seed, grid=args.grid, tiles=args.tiles)
    if not args.dry_run:
        P = run_model(P, args, n_shift)

    os.makedirs(args.out, exist_ok=True)
    tag = time.strftime('%Y%m%d_%H%M%S') + ('_grid' if args.grid else '') + ('_dry' if args.dry_run else '')
    P.drop(columns=['target_path']).to_csv(f'{args.out}/shift_{tag}.csv', index=False)
    json.dump(vars(args), open(f'{args.out}/shift_{tag}_args.json', 'w'), indent=2)

    pd.set_option('display.width', 200)
    cols = {'n': ('case', 'size'), 'overlap': ('overlap', 'mean')}
    for c in range(1, 5):
        cols[f'case{c}'] = ('case', lambda s, c=c: (s == c).mean())
    if not args.dry_run:
        cols.update(pos_err_m=('pos_err_m', 'mean'), head_err_deg=('head_err_deg', 'mean'),
                    max_prob=('max_prob', 'mean'))
    key = 'shift_tiles' if args.grid else 'd_m'
    print(f'\nBy shift ({key}):')
    print(P.groupby(key).agg(**cols).round(3).to_string())
    if not args.dry_run:
        print('\nBy case:')
        print(P.groupby('case').agg(n=('case', 'size'), pos_err_m=('pos_err_m', 'mean'),
                                    head_err_deg=('head_err_deg', 'mean'),
                                    max_prob=('max_prob', 'mean')).round(3).to_string())
    print(f'\nsaved: {args.out}/shift_{tag}.csv')


if __name__ == '__main__':
    main()
