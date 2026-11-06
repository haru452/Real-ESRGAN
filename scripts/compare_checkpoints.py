"""学習途中の重み（net_g_*.pth）ごとに、val の画像を超解像して並べて比較するスクリプト。

val の画像から指定した枚数（既定 2 枚）を選び、重みごとに超解像して
  - 比較画像（compare_<画像名>.png）: LQ 入力・各重みの出力・GT（あれば）を並べたもの
  - 各重みの出力画像（<画像名>/<重み名>.png）
  - psnr.csv: 画像 × 重みの PSNR 表（GT があるとき）
を出力フォルダに保存する。

使い方:
    python scripts/compare_checkpoints.py                          # x2 の実験フォルダ・既定の val を使う
    python scripts/compare_checkpoints.py --iters 10000 50000 100000 150000
    python scripts/compare_checkpoints.py --images 0001 0002 --crop 256
    python scripts/compare_checkpoints.py --official weights/RealESRGAN_x2plus.pth   # 公式の重みも並べる
"""
import argparse
import csv
import glob
import os
import random
import re

import cv2
import matplotlib
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def find_checkpoints(exp_dir, iters):
    """experiments/<name>/models/net_g_<iter>.pth を iter 順に返す。[(ラベル, パス), ...]"""
    found = []
    for p in glob.glob(os.path.join(exp_dir, 'models', 'net_g_*.pth')):
        m = re.search(r'net_g_(\d+)\.pth$', p)
        if m:  # net_g_latest.pth は除く
            found.append((int(m.group(1)), p))
    found.sort()
    if iters:
        found = [(i, p) for i, p in found if i in iters]
        missing = set(iters) - {i for i, _ in found}
        if missing:
            print(f'見つからない iter: {sorted(missing)}')
    return [(f'{i // 1000}k' if i % 1000 == 0 else str(i), p) for i, p in found]


def list_images(d):
    return sorted(p for p in glob.glob(os.path.join(d, '*')) if p.lower().endswith(IMG_EXTS))


def pick_images(lq_dir, names, num, seed):
    paths = list_images(lq_dir)
    if not paths:
        raise SystemExit(f'画像が見つかりません: {lq_dir}')
    if names:
        by_stem = {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
        missing = [n for n in names if n not in by_stem]
        if missing:
            raise SystemExit(f'指定した画像が lq_dir にありません: {missing}')
        return [by_stem[n] for n in names]
    return random.Random(seed).sample(paths, min(num, len(paths)))  # seed 固定なので毎回同じ画像になる


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0**2 / mse)


def find_gt(gt_dir, stem):
    if not gt_dir:
        return None
    for ext in IMG_EXTS:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def build_upsampler(path, scale, args):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    return RealESRGANer(
        scale=scale, model_path=path, model=model, tile=args.tile, half=not args.fp32, gpu_id=args.gpu_id)


def center_crop(img, size):
    if not size:
        return img
    h, w = img.shape[:2]
    size = min(size, h, w)
    y, x = (h - size) // 2, (w - size) // 2
    return img[y:y + size, x:x + size]


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_grid(path, title, panels, cols, crop):
    """panels: [(ラベル, BGR 画像), ...] を cols 列で並べて保存する。"""
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (label, img) in zip(axes.flat, panels):
        ax.imshow(to_rgb(center_crop(img, crop)))
        ax.set_title(label, fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', default=os.path.join(root, 'experiments', 'finetune_RealESRGANx2plus'),
                   help='実験フォルダ（中に models/net_g_*.pth がある）')
    p.add_argument('--lq_dir', default=os.path.join(root, 'dataset', 'val', 'lq'), help='val の LQ 画像フォルダ')
    p.add_argument('--gt_dir', default=os.path.join(root, 'dataset', 'val', 'gt'), help='val の GT 画像フォルダ（無ければ PSNR は出さない）')
    p.add_argument('--images', nargs='*', default=None, help='使う画像名（拡張子なし）。省略時は val からランダムに選ぶ')
    p.add_argument('--num', type=int, default=2, help='ランダムに選ぶ枚数')
    p.add_argument('--seed', type=int, default=0, help='ランダム選択の seed')
    p.add_argument('--iters', nargs='*', type=int, default=None, help='比較する iter（例: 10000 50000）。省略時は全部')
    p.add_argument('--official', default=None, help='公式の重み（例: weights/RealESRGAN_x2plus.pth）。指定すると並べる')
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--cols', type=int, default=3, help='比較画像の列数')
    p.add_argument('--crop', type=int, default=0, help='比較画像で中央を切り出す大きさ（px, 0 で全体）。差を見やすくする用')
    p.add_argument('--tile', type=int, default=0, help='タイルサイズ（GPU メモリ不足のとき）')
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--gpu-id', type=int, default=None)
    p.add_argument('--output', default=None, help='出力先（既定: <exp>/compare）')
    args = p.parse_args()

    out_dir = args.output or os.path.join(args.exp, 'compare')
    os.makedirs(out_dir, exist_ok=True)

    weights = find_checkpoints(args.exp, set(args.iters) if args.iters else None)
    if args.official:
        weights.insert(0, ('official', args.official))
    if not weights:
        raise SystemExit(f'重みが見つかりません: {os.path.join(args.exp, "models")}')

    images = pick_images(args.lq_dir, args.images, args.num, args.seed)
    print('画像:', [os.path.basename(i) for i in images])
    print('重み:', [w[0] for w in weights])

    # 画像の読み込み
    items = []
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt_path = find_gt(args.gt_dir, stem)
        items.append({
            'stem': stem,
            'lq': cv2.imread(path, cv2.IMREAD_COLOR),
            'gt': cv2.imread(gt_path, cv2.IMREAD_COLOR) if gt_path else None,
            'outs': [],
        })
        os.makedirs(os.path.join(out_dir, stem), exist_ok=True)

    # 重みを 1 つずつ読み込んで、全画像を処理する（重みの読み込み回数を最小にする）
    for label, wpath in weights:
        print(f'超解像: {label}')
        up = build_upsampler(wpath, args.scale, args)
        for it in items:
            out, _ = up.enhance(it['lq'], outscale=args.scale)
            it['outs'].append((label, out))
            cv2.imwrite(os.path.join(out_dir, it['stem'], f'{label}.png'), out)
        del up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 比較画像と PSNR 表
    psnr_rows = []
    for it in items:
        lq_up = cv2.resize(it['lq'], None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        panels = [('LQ (nearest x%d)' % args.scale, lq_up)]
        gt = it['gt']
        row = {'image': it['stem']}
        for label, out in it['outs']:
            text = label
            if gt is not None and gt.shape == out.shape:
                v = psnr(out, gt)
                row[label] = round(v, 4)
                text += f'  PSNR {v:.2f}'
            panels.append((text, out))
        if gt is not None:
            panels.append(('GT', gt))
        psnr_rows.append(row)
        save_grid(os.path.join(out_dir, f'compare_{it["stem"]}.png'), it['stem'], panels, args.cols, args.crop)

    if any(len(r) > 1 for r in psnr_rows):
        cols = ['image'] + [w[0] for w in weights]
        with open(os.path.join(out_dir, 'psnr.csv'), 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=cols, restval='')
            wr.writeheader()
            wr.writerows(psnr_rows)
        for r in psnr_rows:
            print(r)
    print('保存先:', out_dir)


if __name__ == '__main__':
    main()
