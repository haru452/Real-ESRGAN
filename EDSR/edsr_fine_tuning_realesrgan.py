# -*- coding: utf-8 -*-
"""
EDSR を Real-ESRGAN と同じ劣化で学習する (学習しながら毎回 LR を作る版)。

edsr_fine_tuning2.py の学習ループをそのまま使い、学習用 LR の作り方だけを
bicubic 縮小から Real-ESRGAN 方式の 2 段階劣化に差し替えます。
パッチごとに毎回ちがう劣化になるので、make_degraded_lr.py で作った固定の LR より種類が多くなります。
val は今まで通り bicubic です (--val_lr_dir を省略した場合)。

    python EDSR/edsr_fine_tuning_realesrgan.py \
        --weight_path ./edsr_x2-0edfb8a3.pt \
        --hr_dir ./dataset/high6 \
        --val_hr_dir ./dataset/val/gt \
        --scale 2 --patch_size 48 --batch_size 16 --epochs 30 --lr 1e-4 \
        --log_dir ./runs/edsr_realesrgan_deg

edsr_fine_tuning2.py の引数はすべて使えます (--lr_dir だけは使えません)。追加の引数:
  --deg_yml PATH   劣化の設定を Real-ESRGAN の yml から読む (省略時は train_realesrgan_x2plus.yml と同じ値)
  --use_jpeg       JPEG 圧縮の劣化も入れる (このリポジトリの Real-ESRGAN は切っているので既定はなし)
  --gt_usm         Real-ESRGAN と同じく、正解にアンシャープマスクをかけてから劣化させ、学習の正解にも使う
                   (val の正解は元画像のままなので、PSNR は下がりやすい)
  --deg_margin N   パッチの周りに何画素 (LR 換算) 余分に切り出して劣化させるか (ぼかしの端の影響を避ける)
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edsr_fine_tuning2 as base  # noqa: E402
from realesrgan_degradation import degrade, load_opt, usm_sharp  # noqa: E402


class RealESRGANPatchDataset(base.SRPatchDataset):
    """HR から少し大きめに切り出し、Real-ESRGAN 方式で劣化させて (LR, HR) パッチを作る。"""

    config = {}  # main() で設定する。__init__ でインスタンスに写すので、DataLoader の子プロセスにも届く

    def __init__(self, hr_dir, lr_dir, scale, patch_size, length_per_epoch=1000):
        if lr_dir is not None:
            raise ValueError('--lr_dir とは一緒に使えません (LR はこのスクリプトが作ります)')
        super().__init__(hr_dir, None, scale, patch_size, length_per_epoch)
        self.deg_opt = self.config['opt']
        self.gt_usm = self.config['gt_usm']
        self.margin = self.config['margin']

    def _read_hr(self, idx):
        path = self.hr_paths[idx % len(self.hr_paths)]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        s, lp = self.scale, self.patch_size
        h, w = hr.shape[:2]
        if h // s < lp or w // s < lp:
            # 画像が小さすぎる場合は拡大して最低サイズを確保
            r = max(lp * s / h, lp * s / w) + 0.01
            hr = cv2.resize(hr, (int(w * r) + s, int(h * r) + s), interpolation=cv2.INTER_CUBIC)
            h, w = hr.shape[:2]
        return hr[:h - h % s, :w - w % s]

    def __getitem__(self, idx):
        hr_full = self._read_hr(idx)
        s, lp, m = self.scale, self.patch_size, self.margin
        lh, lw = hr_full.shape[0] // s, hr_full.shape[1] // s

        # パッチの位置は画像全体から一様に選び、その周りに余白をつけた範囲を劣化させる
        ly = random.randint(0, lh - lp)
        lx = random.randint(0, lw - lp)
        rh, rw = min(lp + 2 * m, lh), min(lp + 2 * m, lw)
        ry = min(max(ly - m, 0), lh - rh)
        rx = min(max(lx - m, 0), lw - rw)
        hr_region = hr_full[ry * s:(ry + rh) * s, rx * s:(rx + rw) * s]
        if self.gt_usm:
            hr_region = usm_sharp(hr_region)
        lr_region = degrade(hr_region, s, self.deg_opt)

        oy, ox = ly - ry, lx - rx
        lr_patch = lr_region[oy:oy + lp, ox:ox + lp]
        hr_patch = hr_region[oy * s:(oy + lp) * s, ox * s:(ox + lp) * s]

        # 左右反転 / 90度回転 (edsr_fine_tuning2.py と同じデータ拡張)
        if random.random() < 0.5:
            lr_patch = lr_patch[:, ::-1, :]
            hr_patch = hr_patch[:, ::-1, :]
        if random.random() < 0.5:
            lr_patch = np.rot90(lr_patch)
            hr_patch = np.rot90(hr_patch)

        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_patch.transpose(2, 0, 1)))
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_patch.transpose(2, 0, 1)))
        return lr_tensor, hr_tensor


def parse_degradation_args():
    """このスクリプト用の引数だけ取り出し、残りを edsr_fine_tuning2.py に渡す。"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--deg_yml', default=None)
    p.add_argument('--use_jpeg', action='store_true')
    p.add_argument('--gt_usm', action='store_true')
    p.add_argument('--deg_margin', type=int, default=16)
    deg_args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return deg_args


def main():
    deg = parse_degradation_args()
    opt = load_opt(deg.deg_yml)
    opt['use_jpeg'] = deg.use_jpeg
    RealESRGANPatchDataset.config = {'opt': opt, 'gt_usm': deg.gt_usm, 'margin': deg.deg_margin}

    print("学習用 LR: Real-ESRGAN 方式の劣化 "
          f"(設定: {deg.deg_yml or 'train_realesrgan_x2plus.yml と同じ値'}, "
          f"JPEG: {'あり' if deg.use_jpeg else 'なし'}, 正解の USM: {'あり' if deg.gt_usm else 'なし'})")
    base.SRPatchDataset = RealESRGANPatchDataset
    base.main()


if __name__ == '__main__':
    main()
