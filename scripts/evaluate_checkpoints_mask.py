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
    python scripts/evaluate_checkpoints_mask.py --detector filter   # 太陽の向きを使うフィルター検出（lroc_filters.py）

--detector filter のときは、出力の名前に _filter が付く（mask_metrics_filter.csv など）。元の結果は上書きしない。
"""
import argparse
import csv
import glob
import os
import re

import cv2
import matplotlib
import numpy as np

from crater_mask import add_detector_args, build_detector, compare_masks, overlay_mask_on_image
from progress import Progress

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


METRICS = [
    ('Differing craters (SR-only + GT-only)', lambda m: m['sr_only_count'] + m['gt_only_count']),
    ('Differing area [px] (SR-only + GT-only)', lambda m: m['sr_only_px'] + m['gt_only_px']),
    ('Mask IoU with GT mask', lambda m: m['mask_iou']),
]


def value_table(results, fn, baseline=None):
    """{画像名: {iter: 値}} と、iter の無い重み（official など）の {画像名: {ラベル: 値}} を返す。

    baseline にラベル（例: 'official'）を渡すと、各画像で「その重みの値 − baseline の値」に直す。
    画像ごとの差（クレーターの数など）を消して、重みの効果だけを見るため。
    """
    table, fixed = {}, {}
    for stem, rows in results.items():
        base = next((fn(m) for label, _, m in rows if label == baseline), None) if baseline else None
        if baseline and base is None:
            continue  # この画像には baseline が無いので使わない
        table[stem] = {it: fn(m) - (base or 0) for _, it, m in rows if it is not None}
        fixed[stem] = {label: fn(m) - (base or 0) for label, it, m in rows if it is None}
    return table, fixed


def mean_curve(table):
    """全画像にそろっている iter だけで、平均と標準偏差を出す。(iters, mean, std)

    画像によって iter の集合が違うと、iter ごとに平均する画像の数が変わってしまうため、共通の iter だけを使う。
    """
    common = set.intersection(*(set(v) for v in table.values())) if table else set()
    iters = sorted(common)
    vals = np.array([[table[s][i] for i in iters] for s in table])
    return iters, vals.mean(axis=0), vals.std(axis=0)


def plot_curves(path, results, suptitle, baseline=None, metrics=METRICS, ref_line=None):
    """上の段: 値そのもの。下の段（baseline があるとき）: baseline との差。どちらも画像ごとの薄い線と全画像の平均（太線＋帯）。"""
    has_base = baseline is not None and any(label == baseline for rows in results.values() for label, _, _ in rows)
    metrics = list(metrics)
    modes = [(None, 'raw')] + ([(baseline, f'minus {baseline}')] if has_base else [])
    fig, axes = plt.subplots(len(modes), len(metrics), figsize=(6 * len(metrics), 4.5 * len(modes)), squeeze=False)
    few = len(results) <= 5  # 枚数が少ないときだけ、画像ごとの凡例を出す
    for r, (base, tag) in enumerate(modes):
        for ax, (title, fn) in zip(axes[r], metrics):
            table, fixed = value_table(results, fn, base)
            for stem, vals in table.items():
                its = sorted(vals)
                ax.plot(its, [vals[i] for i in its], color=None if few else 'gray',
                        alpha=0.8 if few else 0.25, linewidth=1, label=stem if few else None)
            iters, mean, std = mean_curve(table)
            ax.fill_between(iters, mean - std, mean + std, color='tab:red', alpha=0.12)
            ax.plot(iters, mean, color='tab:red', marker='o', linewidth=2.5,
                    label=f'mean of {len(table)} images (±1 std)')
            if base is None:
                if ref_line is not None:
                    ax.axhline(ref_line, color='gray', linestyle=':', alpha=0.8)
                for label in {k for v in fixed.values() for k in v}:
                    ax.axhline(np.mean([v[label] for v in fixed.values() if label in v]), color='black',
                               linestyle='--', alpha=0.7, label=f'{label} (mean)')
            else:
                ax.axhline(0, color='black', linestyle='--', alpha=0.7)
            ax.set_title(title if base is None else f'{title} [{tag}]')
            ax.set_xlabel('iter')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--compare_dir', default=os.path.join(root, 'experiments', 'finetune_RealESRGANx2plus1106', 'compare'),
                   help='compare_checkpoints.py の出力フォルダ')
    p.add_argument('--gt_dir', default=os.path.join(root, 'dataset', 'val', 'gt'), help='GT 画像のフォルダ')
    p.add_argument('--lq_dir', default=os.path.join(root, 'dataset', 'val', 'lq'),
                   help='LQ 画像のフォルダ（--detector filter のとき、ここの *.xlsx をメタデータとして使う）')
    p.add_argument('--ratio', type=float, default=0.3, help='重なり割合がこれ以下の塊を「違う塊」とする（元のコードは 0.3）')
    p.add_argument('--baseline', default='official',
                   help='この重みとの差のグラフも作る（下の段）。compare_checkpoints.py に --official を付けて作った official が既定。無ければ作らない')
    p.add_argument('--no-save-images', action='store_true', help='差分を重ねた画像を保存しない')
    add_detector_args(p)
    args = p.parse_args()
    tag = '' if args.detector == 'original' else f'_{args.detector}'

    stems = sorted(d for d in os.listdir(args.compare_dir) if os.path.isdir(os.path.join(args.compare_dir, d)))
    if not stems:
        raise SystemExit(f'画像フォルダがありません。先に compare_checkpoints.py を実行してください: {args.compare_dir}')

    detector = build_detector(args)
    total = sum(len([f for f in glob.glob(os.path.join(args.compare_dir, st, '*.png'))
                     if not os.path.basename(f).startswith('diff_')]) for st in stems)
    progress = Progress(total, '評価')
    results, csv_rows = {}, []
    for stem in stems:
        gt_path = find_gt(args.gt_dir, stem)
        if gt_path is None:
            print(f'GT が見つからないので飛ばします: {stem}')
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        try:
            gt_mask = detector.mask(gt, stem)
        except KeyError as e:  # filter でメタデータが見つからないとき
            print(f'メタデータが無いので飛ばします: {e}')
            continue
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
            m = compare_masks(detector.mask(sr, stem, gt.shape), gt_mask_r, args.ratio)
            rows.append((label, label_to_iter(label), m))
            progress.step(f'{stem} {label}: SR-only {m["sr_only_count"]}, GT-only {m["gt_only_count"]}, IoU {m["mask_iou"]:.3f}')
            if not args.no_save_images:
                out = overlay_mask_on_image(gt_r, m['sr_only_mask'], color=(0, 0, 255), alpha=0.15)
                out = overlay_mask_on_image(out, m['gt_only_mask'], color=(0, 255, 0), alpha=0.15)
                cv2.imwrite(os.path.join(args.compare_dir, stem, f'diff{tag}_{label}.png'), out)
            csv_rows.append({'image': stem, 'weight': label, 'iter': label_to_iter(label) or '',
                             **{k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in m.items() if not k.endswith('_mask')}})
        results[stem] = rows

    if not csv_rows:
        raise SystemExit('評価する画像がありませんでした')
    keys = [k for k in csv_rows[0] if k not in ('image', 'weight', 'iter')]
    for label in dict.fromkeys(r['weight'] for r in csv_rows):  # 重みごとの全画像の平均
        same = [r for r in csv_rows if r['weight'] == label]
        csv_rows.append({'image': 'MEAN', 'weight': label, 'iter': same[0]['iter'],
                         **{k: round(float(np.mean([r[k] for r in same])), 4) for k in keys}})
    with open(os.path.join(args.compare_dir, f'mask_metrics{tag}.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader()
        wr.writerows(csv_rows)
    iter_sets = {st: frozenset(it for _, it, _ in rows if it is not None) for st, rows in results.items()}
    if len(set(iter_sets.values())) > 1:
        common = frozenset.intersection(*iter_sets.values())
        print(f'注意: 画像によって iter の集合が違います（古い結果が混ざっていませんか）。平均は共通の {len(common)} 個の iter だけで取ります')
    if not any(label == args.baseline for rows in results.values() for label, _, _ in rows):
        print(f'注意: baseline "{args.baseline}" が無いので、差のグラフは作りません（compare_checkpoints.py に --official を付けて作り直すと作れます）')
    plot_curves(os.path.join(args.compare_dir, f'mask_diff_curves{tag}.png'), results,
                f'Crater-mask difference between SR and GT [{args.detector}] (overlap ratio threshold {args.ratio}, {len(results)} images)',
                args.baseline)
    print('保存先:', args.compare_dir)


if __name__ == '__main__':
    main()
