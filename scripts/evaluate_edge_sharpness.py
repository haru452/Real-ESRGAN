"""超解像画像が「GT よりくっきり（fresh に見える）」になっていないかを、クレーターの縁のシャープさで調べるスクリプト。

GAN の超解像は高周波を足すので、なだらかな縁（degraded なクレーター）が、くっきりした縁（fresh）に見えることがある。
マスクの重なりでは、それが測れないので、縁の勾配の強さを GT と比べる。

やり方:
  1. GT 画像から先行研究の障害物検出でマスクを作り、縁のあたり（マスクを少し太らせた領域）を塊ごとに分ける
  2. 塊ごとに、勾配の強さ（Sobel）の平均を、GT と SR で測る
  3. 比 = SR / GT。1 より大きければ SR のほうがくっきり（fresh 寄り）、小さければぼやけている
  4. 画像ごとに、比の中央値と、くっきりした塊の割合・ぼやけた塊の割合を出す

compare_checkpoints.py の出力（<exp>/compare/<画像名>/<重み名>.png）を読む。先にそれを実行しておくこと。
LQ 画像が見つかれば、bicubic で拡大したもの（'bicubic'）も基準として並べる（点線）。

出力（<exp>/compare/ の中）:
    edge_sharpness_curves.png   iter ごとの線グラフ（上: 値そのもの、下: --baseline との差）
    edge_sharpness.csv          画像 × 重みの数値（image が MEAN の行は全画像の平均）

使い方:
    python scripts/evaluate_edge_sharpness.py
    python scripts/evaluate_edge_sharpness.py --tol 0.2 --min_area 50
"""
import argparse
import csv
import glob
import os

import cv2
import matplotlib
import numpy as np

from crater_mask import OriginalCNN
from evaluate_checkpoints_mask import IMG_EXTS, find_gt, label_to_iter, plot_curves
from progress import Progress

matplotlib.use('Agg')

METRICS = [
    ('Edge sharpness SR/GT (median; >1 = sharper)', lambda m: m['median_ratio']),
    ('Craters sharper than GT [fraction]', lambda m: m['frac_sharper']),
    ('Craters blurrier than GT [fraction]', lambda m: m['frac_blurrier']),
]


def gradient_magnitude(gray):
    g = gray.astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def crater_regions(gt_mask, dilate, min_area):
    """GT のマスクを少し太らせて塊に分ける。(ラベル画像, ラベル数, 各ラベルの面積) を返す。

    太らせるのは、「GT で勾配が強かった画素だけ」を測ると、GT が必ず高く出てしまう（比が低く偏る）のを和らげるため。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    n, labels = cv2.connectedComponents(cv2.dilate(gt_mask, kernel))
    area = np.bincount(labels.ravel(), minlength=n)
    area[0] = 0  # 背景
    return labels, n, area


def sharpness_per_crater(grad, labels, n, area):
    """塊ごとの勾配の強さの平均（面積 0 の塊は 0）。"""
    total = np.bincount(labels.ravel(), weights=grad.ravel(), minlength=n)
    return np.divide(total, area, out=np.zeros(n), where=area > 0)


def sharpness_metrics(sr_grad, gt_sharp, labels, n, area, min_area, tol):
    keep = (area >= min_area) & (gt_sharp > 0)
    keep[0] = False
    if not keep.any():
        return {'median_ratio': float('nan'), 'frac_sharper': float('nan'), 'frac_blurrier': float('nan'), 'n_craters': 0}
    ratio = sharpness_per_crater(sr_grad, labels, n, area)[keep] / gt_sharp[keep]
    return {
        'median_ratio': float(np.median(ratio)),
        'frac_sharper': float(np.mean(ratio > 1 + tol)),
        'frac_blurrier': float(np.mean(ratio < 1 - tol)),
        'n_craters': int(keep.sum()),
    }


def find_lq(lq_dir, stem):
    for ext in IMG_EXTS:
        p = os.path.join(lq_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--compare_dir', default=os.path.join(root, 'experiments', 'finetune_RealESRGANx2plus1106', 'compare'),
                   help='compare_checkpoints.py の出力フォルダ')
    p.add_argument('--gt_dir', default=os.path.join(root, 'dataset', 'val', 'gt'), help='GT 画像のフォルダ')
    p.add_argument('--lq_dir', default=os.path.join(root, 'dataset', 'val', 'lq'),
                   help='LQ 画像のフォルダ。あれば bicubic 拡大を基準として並べる')
    p.add_argument('--baseline', default='official', help='この重みとの差のグラフも作る（無ければ作らない）')
    p.add_argument('--tol', type=float, default=0.1, help='比が 1±tol を超えたら「くっきり／ぼやけた」と数える')
    p.add_argument('--min_area', type=int, default=30, help='これより小さい塊は、ノイズが大きいので除く（px）')
    p.add_argument('--dilate', type=int, default=2, help='マスクを太らせる半径（px）')
    args = p.parse_args()

    stems = sorted(d for d in os.listdir(args.compare_dir) if os.path.isdir(os.path.join(args.compare_dir, d)))
    if not stems:
        raise SystemExit(f'画像フォルダがありません。先に compare_checkpoints.py を実行してください: {args.compare_dir}')

    def sr_files(stem):
        return [f for f in sorted(glob.glob(os.path.join(args.compare_dir, stem, '*.png')))
                if not os.path.basename(f).startswith('diff_')]

    detector = OriginalCNN()
    progress = Progress(sum(len(sr_files(st)) + 1 for st in stems), '縁の評価')
    results, csv_rows = {}, []
    for stem in stems:
        gt_path = find_gt(args.gt_dir, stem)
        if gt_path is None:
            print(f'GT が見つからないので飛ばします: {stem}')
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        labels, n, area = crater_regions(detector.forward(gt), args.dilate, args.min_area)
        gt_sharp = sharpness_per_crater(gradient_magnitude(gt), labels, n, area)

        images = [(os.path.splitext(os.path.basename(f))[0], cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in sr_files(stem)]
        lq_path = find_lq(args.lq_dir, stem)
        if lq_path:  # bicubic 拡大を基準として加える（iter が無いので点線で描かれる）
            lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE)
            images.append(('bicubic', cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)))

        rows = []
        for label, sr in images:
            if sr.shape != gt.shape:
                print(f'大きさが GT と違うので飛ばします: {stem} {label} {sr.shape} vs {gt.shape}')
                continue
            m = sharpness_metrics(gradient_magnitude(sr), gt_sharp, labels, n, area, args.min_area, args.tol)
            rows.append((label, label_to_iter(label), m))
            csv_rows.append({'image': stem, 'weight': label, 'iter': label_to_iter(label) or '',
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
            progress.step(f'{stem} {label}: 比の中央値 {m["median_ratio"]:.3f} (塊 {m["n_craters"]} 個)')
        results[stem] = rows

    if not csv_rows:
        raise SystemExit('評価する画像がありませんでした')
    keys = [k for k in csv_rows[0] if k not in ('image', 'weight', 'iter')]
    for label in dict.fromkeys(r['weight'] for r in csv_rows):  # 重みごとの全画像の平均
        same = [r for r in csv_rows if r['weight'] == label and r['image'] != 'MEAN']
        csv_rows.append({'image': 'MEAN', 'weight': label, 'iter': same[0]['iter'],
                         **{k: round(float(np.nanmean([r[k] for r in same])), 4) for k in keys}})
    with open(os.path.join(args.compare_dir, 'edge_sharpness.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        wr.writeheader()
        wr.writerows(csv_rows)

    if not any(label == args.baseline for rows in results.values() for label, _, _ in rows):
        print(f'注意: baseline "{args.baseline}" が無いので、差のグラフは作りません')
    plot_curves(os.path.join(args.compare_dir, 'edge_sharpness_curves.png'), results,
                f'Edge sharpness of craters, SR vs GT (tol {args.tol}, {len(results)} images)',
                args.baseline, metrics=METRICS, ref_line=None)
    print('保存先:', args.compare_dir)


if __name__ == '__main__':
    main()
