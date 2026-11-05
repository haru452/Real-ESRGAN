# -*- coding: utf-8 -*-
"""
EDSR ファインチューニング用の検証（val）処理。

edsr_fine_tuning2.py から呼び出して、エポックごとに
  - val の L1 損失（学習損失と同じ尺度）
  - val の PSNR（Real-ESRGAN 側と同じ計算: RGB・0-255・周囲 scale 画素を除外）
を計算し、CSV とグラフに残す。
学習損失は下がり続けるのに val 損失が上がり始めたら、過学習のサインです。
"""

import csv
import glob
import os

import cv2
import numpy as np
import torch

IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # 学習時と同じ読み方 (16bit も 8bit になる)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


def list_images(dir_path):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(dir_path, ext)))
    return sorted(paths)


class ValSet:
    """
    val の (LR, HR) の組を読む。
    - lr_dir を指定: 同じファイル名の画像を組にする (例: dataset/val/lq と dataset/val/gt)
    - lr_dir を省略: HR を学習時と同じ bicubic で縮小して LR を作る
    """

    def __init__(self, hr_dir, lr_dir, scale, max_images=None):
        self.scale = scale
        self.lr_dir = lr_dir
        self.hr_paths = list_images(hr_dir)
        if max_images:
            self.hr_paths = self.hr_paths[:max_images]
        if not self.hr_paths:
            raise FileNotFoundError(f"{hr_dir} に val 画像が見つかりません")
        if lr_dir is not None:
            self.lr_paths = {os.path.basename(p): p for p in list_images(lr_dir)}
            missing = [os.path.basename(p) for p in self.hr_paths
                       if os.path.basename(p) not in self.lr_paths]
            if missing:
                raise FileNotFoundError(
                    f"{lr_dir} に対応する LR 画像がありません: {missing[:5]} など {len(missing)} 枚")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, i):
        s = self.scale
        hr = _read_rgb(self.hr_paths[i])
        if self.lr_dir is not None:
            lr = _read_rgb(self.lr_paths[os.path.basename(self.hr_paths[i])])
        else:
            h, w = hr.shape[:2]
            h, w = h - h % s, w - w % s
            hr = hr[:h, :w]
            lr = cv2.resize(hr, (w // s, h // s), interpolation=cv2.INTER_CUBIC)
        # 出力サイズ (LR × scale) に HR をそろえる
        hr = hr[:lr.shape[0] * s, :lr.shape[1] * s]
        return lr, hr


def calc_psnr(sr, hr, crop_border):
    """basicsr の calculate_psnr と同じ: 0-255 に丸めた RGB、周囲 crop_border 画素を除外"""
    sr = np.clip(np.round(sr), 0, 255).astype(np.float64)
    hr = np.clip(np.round(hr), 0, 255).astype(np.float64)
    if crop_border > 0:
        sr = sr[crop_border:-crop_border, crop_border:-crop_border]
        hr = hr[crop_border:-crop_border, crop_border:-crop_border]
    mse = np.mean((sr - hr) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


@torch.no_grad()
def evaluate(model, val_set, device):
    """val 全体の平均 L1 損失と平均 PSNR を返す (1 枚ずつ画像全体を推論)"""
    model.eval()
    l1_sum, psnr_sum = 0.0, 0.0
    for i in range(len(val_set)):
        lr, hr = val_set[i]
        x = torch.from_numpy(lr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        sr = model(x)[0].float().cpu().numpy().transpose(1, 2, 0)
        l1_sum += float(np.mean(np.abs(sr - hr)))
        psnr_sum += calc_psnr(sr, hr, crop_border=val_set.scale)
    model.train()
    n = len(val_set)
    return l1_sum / n, psnr_sum / n


class ValLogger:
    """エポックごとの train_loss / val_loss / val_psnr を CSV とグラフに保存する"""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'val_metrics.csv')
        self.png_path = os.path.join(out_dir, 'training_curves.png')
        self.rows = []

    def add(self, epoch, train_loss, val_loss, val_psnr):
        self.rows.append((epoch, train_loss, val_loss, val_psnr))
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'val_loss', 'val_psnr'])
            w.writerows(self.rows)
        self._plot()

    def _plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return  # matplotlib が無ければ CSV だけ残す
        ep = [r[0] for r in self.rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(ep, [r[1] for r in self.rows], 'o-', label='train L1')
        ax1.plot(ep, [r[2] for r in self.rows], 'o-', label='val L1')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('L1 loss (0-255)')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        psnr = [r[3] for r in self.rows]
        best = int(np.argmax(psnr))
        ax2.plot(ep, psnr, 'o-', color='tab:green')
        ax2.plot(ep[best], psnr[best], '*', color='red', markersize=14,
                 label=f'best: epoch {ep[best]} ({psnr[best]:.2f} dB)')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Validation PSNR')
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=120)
        plt.close(fig)
