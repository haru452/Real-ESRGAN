# -*- coding: utf-8 -*-
"""
EDSR ファインチューニング用の検証（val）処理。

edsr_fine_tuning2.py から呼び出して、エポックごとに
  - val の L1 損失（学習損失と同じ尺度）
  - val の PSNR / SSIM（Real-ESRGAN 側と同じ計算: RGB・0-255・周囲 scale 画素を除外）
  - bias（SR の平均明るさ − 正解の平均明るさ。0 から離れていれば明るさがずれている）
を計算し、CSV とグラフに残す。
比べる基準として、bicubic で拡大しただけの PSNR / SSIM も最初に 1 回だけ計算する。
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


def pick_evenly(paths, n):
    """名前順のリストから n 枚を等間隔に選ぶ (先頭だけだと同じ撮影帯に偏りやすいため)"""
    if not n or n >= len(paths):
        return paths
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in idx]


class ValSet:
    """
    val の (LR, HR) の組を読む。
    - lr_dir を指定: 同じファイル名の画像を組にする (例: dataset/val/lq と dataset/val/gt)
    - lr_dir を省略: HR を学習時と同じ bicubic で縮小して LR を作る
    max_images を指定すると、名前順に並べた中から等間隔に選ぶ。
    """

    def __init__(self, hr_dir, lr_dir, scale, max_images=None):
        self.scale = scale
        self.lr_dir = lr_dir
        self.hr_paths = pick_evenly(list_images(hr_dir), max_images)
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

    def name(self, i):
        return os.path.basename(self.hr_paths[i])

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


def bicubic_up(lr, scale):
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _prep(img, crop_border):
    img = np.clip(np.round(img), 0, 255).astype(np.float64)
    if crop_border > 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    return img


def calc_psnr(sr, hr, crop_border):
    """basicsr の calculate_psnr と同じ: 0-255 に丸めた RGB、周囲 crop_border 画素を除外"""
    sr, hr = _prep(sr, crop_border), _prep(hr, crop_border)
    mse = np.mean((sr - hr) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def _ssim_channel(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    win = np.outer(k, k.transpose())
    mu1 = cv2.filter2D(a, -1, win)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, win)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = cv2.filter2D(a ** 2, -1, win)[5:-5, 5:-5] - mu1_sq
    s2 = cv2.filter2D(b ** 2, -1, win)[5:-5, 5:-5] - mu2_sq
    s12 = cv2.filter2D(a * b, -1, win)[5:-5, 5:-5] - mu12
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return m.mean()


def calc_ssim(sr, hr, crop_border):
    """basicsr の calculate_ssim と同じ: 各チャンネルの SSIM の平均"""
    sr, hr = _prep(sr, crop_border), _prep(hr, crop_border)
    return float(np.mean([_ssim_channel(sr[..., c], hr[..., c]) for c in range(sr.shape[2])]))


@torch.no_grad()
def super_resolve(model, lr, device):
    """LR (HWC, 0-255) を 1 枚まるごと推論して SR (HWC, float) を返す"""
    x = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))).unsqueeze(0).to(device)
    return model(x)[0].float().cpu().numpy().transpose(1, 2, 0)


@torch.no_grad()
def evaluate(model, val_set, device):
    """val 全体の平均 l1 / psnr / ssim / bias を dict で返す"""
    model.eval()
    sums = {'l1': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'bias': 0.0}
    s = val_set.scale
    for i in range(len(val_set)):
        lr, hr = val_set[i]
        sr = super_resolve(model, lr, device)
        sums['l1'] += float(np.mean(np.abs(sr - hr)))
        sums['psnr'] += calc_psnr(sr, hr, crop_border=s)
        sums['ssim'] += calc_ssim(sr, hr, crop_border=s)
        sums['bias'] += float(np.mean(sr) - np.mean(hr))
    model.train()
    return {k: v / len(val_set) for k, v in sums.items()}


def evaluate_bicubic(val_set):
    """bicubic で拡大しただけの場合の psnr / ssim (超解像の効果をはかる基準)"""
    s = val_set.scale
    psnr, ssim = 0.0, 0.0
    for i in range(len(val_set)):
        lr, hr = val_set[i]
        up = bicubic_up(lr, s)
        psnr += calc_psnr(up, hr, crop_border=s)
        ssim += calc_ssim(up, hr, crop_border=s)
    return {'psnr': psnr / len(val_set), 'ssim': ssim / len(val_set)}


class ValLogger:
    """エポックごとの値を CSV とグラフに保存する"""

    COLUMNS = ['epoch', 'lr', 'train_loss', 'val_loss', 'val_psnr', 'val_ssim', 'val_bias']

    def __init__(self, out_dir, bicubic=None):
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'val_metrics.csv')
        self.png_path = os.path.join(out_dir, 'training_curves.png')
        self.bicubic = bicubic  # {'psnr', 'ssim'} または None
        self.rows = []

    def add(self, epoch, lr, train_loss, m):
        self.rows.append((epoch, lr, train_loss, m['l1'], m['psnr'], m['ssim'], m['bias']))
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            if self.bicubic:
                w.writerow([f"# bicubic: psnr={self.bicubic['psnr']:.4f} ssim={self.bicubic['ssim']:.4f}"])
            w.writerow(self.COLUMNS)
            w.writerows(self.rows)
        self._plot()

    def _plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return  # matplotlib が無ければ CSV だけ残す
        col = {c: [r[i] for r in self.rows] for i, c in enumerate(self.COLUMNS)}
        ep = col['epoch']
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        ax = axes[0]
        ax.plot(ep, col['train_loss'], 'o-', label='train L1 (epoch average)')
        ax.plot(ep, col['val_loss'], 'o-', label='val L1')
        ax.set_ylabel('L1 loss (0-255)')
        ax.set_title('Loss')

        for ax, key, label in ((axes[1], 'val_psnr', 'PSNR (dB)'), (axes[2], 'val_ssim', 'SSIM')):
            vals = col[key]
            best = int(np.argmax(vals))
            ax.plot(ep, vals, 'o-', color='tab:green', label='EDSR')
            ax.plot(ep[best], vals[best], '*', color='red', markersize=14,
                    label=f'best: epoch {ep[best]} ({vals[best]:.4g})')
            if self.bicubic:
                b = self.bicubic[key.split('_')[1]]
                ax.axhline(b, color='gray', linestyle='--', label=f'bicubic ({b:.4g})')
            ax.set_ylabel(label)
            ax.set_title(f'Validation {label.split()[0]}')

        for ax in axes:
            ax.set_xlabel('epoch')
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.png_path, dpi=120)
        plt.close(fig)
