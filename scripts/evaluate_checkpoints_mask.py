"""重みごとの超解像画像に障害物検出（先行研究）をかけ、GT のマスクとの違いを線グラフにするスクリプト。

compare_checkpoints.py の出力（<exp>/compare/<画像名>/<重み名>.png）を読む。先にそれを実行しておくこと。
重みごとに、SR 画像と GT 画像から作ったマスクを比べる。
  - SR のみ / GT のみ にある塊（クレーター）の数と面積
  - マスク同士の IoU（高いほど GT のマスクに近い）

出力（<exp>/compare/ の中）:
    mask_diff_curves.png   iter ごとの線グラフ（画像ごとに 1 本。公式の重みは横線）
    mask_metrics.csv       画像 × 重みの数値
    <画像名>/diff_<重み名>.png   差分を重ねた画像（赤: SR のみ, 緑: GT のみ）

使い方:
    python scripts/evaluate_checkpoints_mask.py
    python scripts/evaluate_checkpoints_mask.py --ratio 0.5 --no-save-images
"""
import argparse
import csv
import glob
import os
import re

import cv2
import matplotlib
import numpy as np

from crater_mask import OriginalCNN, compare_masks, overlay_mask_on_image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def label_to_iter(label):
    """'10k' -> 10000, '12345' -> 12345。数字でなければ None（official など）。"""
    m = re.fullmatch(r'(\d+)(k?)', label)
    if not m:
        return None
    return int(m.group(1)) * (1000 if m.group(2) else 1)


def find_gt(gt_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def plot_curves(path, results, ratio):
    """results: {画像名: [(label, iter or None, metrics), ...]}"""
    metrics = [
        ('Differing craters (SR-only + GT-only)', lambda m: m['sr_only_count'] + m['gt_only_count']),
        ('Differing area [px] (SR-only + GT-only)', lambda m: m['sr_only_px'] + m['gt_only_px']),
        ('Mask IoU with GT mask', lambda m: m['mask_iou']),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ci, (stem, rows) in enumerate(results.items()):
        color = colors[ci % len(colors)]
        curve = sorted((r for r in rows if r[1] is not None), key=lambda r: r[1])
        for ax, (title, fn) in zip(axes, metrics):
            ax.plot([r[1] for r in curve], [fn(r[2]) for r in curve], marker='o', color=color, label=stem)
            for r in rows:
                if r[1] is None:  # 公式の重みなどは横線で示す
                    ax.axhline(fn(r[2]), color=color, linestyle='--', alpha=0.6, label=f'{stem} ({r[0]})')
    for ax, (title, _) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_xlabel('iter')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'Crater-mask difference between SR and GT (overlap ratio threshold {ratio})')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--compare_dir', default=os.path.join(root, 'experiments', 'finetune_RealESRGANx2plus', 'compare'),
                   help='compare_checkpoints.py の出力フォルダ')
    p.add_argument('--gt_dir', default=os.path.join(root, 'dataset', 'val', 'gt'), help='GT 画像のフォルダ')
    p.add_argument('--ratio', type=float, default=0.3, help='重なり割合がこれ以下の塊を「違う塊」とする（元のコードは 0.3）')
    p.add_argument('--no-save-images', action='store_true', help='差分を重ねた画像を保存しない')
    args = p.parse_args()

    stems = sorted(d for d in os.listdir(args.compare_dir) if os.path.isdir(os.path.join(args.compare_dir, d)))
    if not stems:
        raise SystemExit(f'画像フォルダがありません。先に compare_checkpoints.py を実行してください: {args.compare_dir}')

    detector = OriginalCNN()
    results, csv_rows = {}, []
    for stem in stems:
        gt_path = find_gt(args.gt_dir, stem)
        if gt_path is None:
            print(f'GT が見つからないので飛ばします: {stem}')
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = detector.forward(gt)
        rows = []
        for sr_path in sorted(glob.glob(os.path.join(args.compare_dir, stem, '*.png'))):
            label = os.path.splitext(os.path.basename(sr_path))[0]
            if label.startswith('diff_'):
                continue
            sr = cv2.imread(sr_path, cv2.IMREAD_GRAYSCALE)
            if sr.shape != gt.shape:  # 元のコードと同じく、GT を SR の大きさに合わせる
                gt_r = cv2.resize(gt, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_LINEAR)
                gt_mask_r = cv2.resize(gt_mask, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_r, gt_mask_r = gt, gt_mask
            m = compare_masks(detector.forward(sr), gt_mask_r, args.ratio)
            rows.append((label, label_to_iter(label), m))
            print(f'{stem} {label}: SR-only {m["sr_only_count"]}, GT-only {m["gt_only_count"]}, IoU {m["mask_iou"]:.3f}')
            if not args.no_save_images:
                out = overlay_mask_on_image(gt_r, m['sr_only_mask'], color=(0, 0, 255), alpha=0.15)
                out = overlay_mask_on_image(out, m['gt_only_mask'], color=(0, 255, 0), alpha=0.15)
                cv2.imwrite(os.path.join(args.compare_dir, stem, f'diff_{label}.png'), out)
            csv_rows.append({'image': stem, 'weight': label, 'iter': label_to_iter(label) or '',
                             **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in m.items() if not k.endswith('_mask')}})
        results[stem] = rows

    if not csv_rows:
        raise SystemExit('評価する画像がありませんでした')
    with open(os.path.join(args.compare_dir, 'mask_metrics.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader()
        wr.writerows(csv_rows)
    plot_curves(os.path.join(args.compare_dir, 'mask_diff_curves.png'), results, args.ratio)
    print('保存先:', args.compare_dir)


if __name__ == '__main__':
    main()
