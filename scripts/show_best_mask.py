"""学習ログの val PSNR が最高の重みだけについて、SR と GT の検出マスクの違いを色で重ねて見るスクリプト。

赤: SR にだけある塊（ハルシネーションの候補）
緑: GT にだけある塊（見落としの候補）

compare_checkpoints.py の出力（<compare_dir>/<画像名>/<重み名>.png）を読む。先にそれを実行しておくこと。
best の iter は、実験フォルダの学習ログ（train_*.log）から選ぶ（plot_training_log.py の星印と同じ）。

出力（<compare_dir>/best_mask[_filter]/ の中）:
    <画像名>_<重み名>.png   左: GT に重ねたもの、右: SR に重ねたもの

使い方:
    python scripts/show_best_mask.py --detector filter \
        --compare_dir experiments/finetune_RealESRGANx2plus1106/compare_test \
        --gt_dir dataset/test/gt --lq_dir dataset/test/lq
    python scripts/show_best_mask.py --weight 85k        # best ではなく、重みを自分で選ぶ
"""
import argparse
import os

import cv2
import numpy as np

from crater_mask import add_detector_args, build_detector, compare_masks, overlay_mask_on_image
from evaluate_checkpoints_mask import find_gt
from plot_training_log import find_logs, parse_logs

RED, GREEN = (0, 0, 255), (0, 255, 0)


def best_label(exp_dir, metric):
    """学習ログから、metric（例 psnr）が最高の iter を探し、compare_checkpoints.py の重み名（'85k' など）で返す。"""
    _, _, metrics = parse_logs(find_logs(exp_dir))
    data = next((d for (_, m), d in sorted(metrics.items()) if m == metric), None)
    if not data:
        raise SystemExit(f'学習ログに {metric} の検証結果がありません: {exp_dir}')
    lower_better = metric.lower() in ('niqe', 'lpips')
    it = (min if lower_better else max)(data, key=data.get)
    print(f'best: {metric} {data[it]:.4f} @ {it:,} iter')
    return f'{it // 1000}k' if it % 1000 == 0 else str(it)


def overlay(img, sr_only, gt_only, alpha):
    out = overlay_mask_on_image(img, sr_only, RED, alpha)
    out = overlay_mask_on_image(out, gt_only, GREEN, alpha)
    # 塗りだけだと小さい塊が見えにくいので、輪郭も描く
    for mask, color in ((sr_only, RED), (gt_only, GREEN)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
    return out


def put_title(img, text):
    out = cv2.copyMakeBorder(img, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp', default=os.path.join(root, 'experiments', 'finetune_RealESRGANx2plus1106'),
                   help='学習ログ（train_*.log）のある実験フォルダ')
    p.add_argument('--compare_dir', default=None, help='compare_checkpoints.py の出力フォルダ（既定: <exp>/compare）')
    p.add_argument('--gt_dir', default=os.path.join(root, 'dataset', 'val', 'gt'), help='GT 画像のフォルダ')
    p.add_argument('--lq_dir', default=os.path.join(root, 'dataset', 'val', 'lq'),
                   help='LQ 画像のフォルダ（--detector filter のとき、ここの *.xlsx をメタデータとして使う）')
    p.add_argument('--metric', default='psnr', help='best を選ぶ指標（学習ログの名前）')
    p.add_argument('--weight', default=None, help='best ではなく、この重み名（例 85k, official）を使う')
    p.add_argument('--ratio', type=float, default=0.3, help='重なり割合がこれ以下の塊を「違う塊」とする')
    p.add_argument('--alpha', type=float, default=0.35, help='色を重ねる濃さ（0〜1）')
    add_detector_args(p)
    args = p.parse_args()
    compare_dir = args.compare_dir or os.path.join(args.exp, 'compare')

    label = args.weight or best_label(args.exp, args.metric)
    detector = build_detector(args)
    out_dir = os.path.join(compare_dir, 'best_mask' + ('' if args.detector == 'original' else f'_{args.detector}'))
    os.makedirs(out_dir, exist_ok=True)

    stems = sorted(d for d in os.listdir(compare_dir) if os.path.isdir(os.path.join(compare_dir, d))
                   and not d.startswith('best_mask'))
    done = 0
    for stem in stems:
        sr_path = os.path.join(compare_dir, stem, f'{label}.png')
        gt_path = find_gt(args.gt_dir, stem)
        if not os.path.isfile(sr_path) or gt_path is None:
            print(f'飛ばします（{"SR" if not os.path.isfile(sr_path) else "GT"} がありません）: {stem}')
            continue
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        sr = cv2.imread(sr_path, cv2.IMREAD_GRAYSCALE)
        if sr.shape != gt.shape:
            print(f'大きさが GT と違うので飛ばします: {stem} {sr.shape} vs {gt.shape}')
            continue
        try:
            m = compare_masks(detector.mask(sr, stem, gt.shape), detector.mask(gt, stem), args.ratio)
        except KeyError as e:  # filter でメタデータが見つからないとき
            print(f'メタデータが無いので飛ばします: {e}')
            continue
        left = put_title(overlay(gt, m['sr_only_mask'], m['gt_only_mask'], args.alpha), 'GT')
        right = put_title(overlay(sr, m['sr_only_mask'], m['gt_only_mask'], args.alpha),
                          f'SR {label}  red: SR only {m["sr_only_count"]} / green: GT only {m["gt_only_count"]}')
        gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
        out_path = os.path.join(out_dir, f'{stem}_{label}.png')
        cv2.imwrite(out_path, np.hstack([left, gap, right]))
        print(f'{stem}: 赤（SR のみ） {m["sr_only_count"]} 個, 緑（GT のみ） {m["gt_only_count"]} 個, IoU {m["mask_iou"]:.3f}')
        done += 1

    if not done:
        raise SystemExit(f'重み {label} の SR 画像が 1 枚もありません。compare_checkpoints.py でその重みも出力してください')
    print('保存先:', out_dir)


if __name__ == '__main__':
    main()
