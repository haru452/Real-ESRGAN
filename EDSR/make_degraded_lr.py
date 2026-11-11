# -*- coding: utf-8 -*-
"""
HR フォルダーの画像を Real-ESRGAN と同じ方法で劣化させ、LR を別フォルダーに保存する。

LR は HR と同じファイル名で保存するので、そのまま edsr_fine_tuning2.py の --lr_dir に使えます。

    python EDSR/make_degraded_lr.py \
        --hr_dir ./dataset/high6 \
        --out_dir ./dataset/high6_lr_realesrgan \
        --scale 2 --seed 0

    python EDSR/edsr_fine_tuning2.py ... \
        --hr_dir ./dataset/high6 --lr_dir ./dataset/high6_lr_realesrgan

- 1 枚につき劣化は 1 通りだけです (学習中に毎回変わる edsr_fine_tuning_realesrgan.py と違います)。
- --seed が同じなら、同じ LR ができます。
- 劣化の設定は options/train_realesrgan_x2plus.yml と同じ値です。--deg_yml で別の yml から読めます。
- JPEG はこのリポジトリの Real-ESRGAN に合わせて既定で使いません (--use_jpeg で有効)。
- .jpg の HR は LR も .jpg (品質 100) で保存します。劣化を正確に残したいなら HR を png/tiff にしてください。
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realesrgan_degradation import degrade, load_opt, seed_everything  # noqa: E402

IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')


def get_args():
    p = argparse.ArgumentParser(description='Real-ESRGAN 方式で LR 画像を作って保存する')
    p.add_argument('--hr_dir', required=True, help='高解像度 (正解) 画像のフォルダー')
    p.add_argument('--out_dir', required=True, help='LR の保存先フォルダー')
    p.add_argument('--scale', type=int, choices=[2, 3, 4], default=2)
    p.add_argument('--seed', type=int, default=0, help='乱数の種 (同じなら同じ LR になる)')
    p.add_argument('--deg_yml', default=None,
                   help='劣化の設定を読む Real-ESRGAN の yml (省略時は x2plus と同じ値)')
    p.add_argument('--use_jpeg', action='store_true', help='JPEG 圧縮の劣化も入れる')
    p.add_argument('--overwrite', action='store_true', help='保存先に同名ファイルがあっても作り直す')
    return p.parse_args()


def main():
    args = get_args()
    opt = load_opt(args.deg_yml)
    opt['use_jpeg'] = args.use_jpeg
    seed_everything(args.seed)

    paths = sorted({p for ext in IMG_EXTS for p in glob.glob(os.path.join(args.hr_dir, ext))})
    if not paths:
        raise FileNotFoundError(f"{args.hr_dir} に画像が見つかりません")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(paths)} 枚を劣化させます -> {args.out_dir} "
          f"(scale={args.scale}, JPEG={'あり' if args.use_jpeg else 'なし'}, seed={args.seed})")

    start = time.time()
    skipped = 0
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  読めないのでスキップ: {name}")
            continue
        hr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 縦横を scale の倍数に切りそろえる (EDSR の学習と同じく左上基準)
        h, w = hr.shape[:2]
        hr = hr[:h - h % args.scale, :w - w % args.scale]

        lr = degrade(hr, args.scale, opt)
        lr = cv2.cvtColor(lr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        params = []
        if name.lower().endswith(('.jpg', '.jpeg')):
            params = [cv2.IMWRITE_JPEG_QUALITY, 100]
        cv2.imwrite(out_path, lr, params)

        if i % 50 == 0 or i == len(paths):
            elapsed = time.time() - start
            print(f"  {i}/{len(paths)} 枚 ({elapsed:.0f} 秒)")

    if skipped:
        print(f"  既にある {skipped} 枚はスキップしました (作り直すなら --overwrite)")
    print("完了")


if __name__ == '__main__':
    main()
