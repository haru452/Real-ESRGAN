# -*- coding: utf-8 -*-
"""
Real-ESRGAN と同じ 2 段階の劣化で、HR 画像から LR 画像を作る (1 枚ずつ)。

処理の順番は realesrgan/models/realesrgan_model.py の feed_data と同じです。
  1 段目: ぼかし -> ランダムリサイズ -> ノイズ (ガウス or ポアソン) -> (JPEG)
  2 段目: ぼかし (確率) -> ランダムリサイズ -> ノイズ
          -> [1/scale に縮小 + sinc フィルタ] と [JPEG] を 50% で順番入れ替え
  最後に 8bit に丸める

- 既定値は options/train_realesrgan_x2plus.yml と同じです。
- このリポジトリの Real-ESRGAN 学習では JPEG をコメントアウトしているので、
  JPEG も既定では使いません (opt['use_jpeg'] = True で有効。cv2 の JPEG を使用)。
- basicsr なしで動くように、basicsr/data/degradations.py の処理を写しています。

入出力: RGB / float32 / 0-255 の HWC (EDSR のスクリプトと同じ)。
"""

import copy
import math
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.special import j1 as _bessel_j1
except ImportError:  # scipy がなければ torch で代用
    def _bessel_j1(x):
        return torch.special.bessel_j1(torch.from_numpy(np.asarray(x, dtype=np.float64))).numpy()

_KERNELS = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
_KERNEL_PROB = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]

# options/train_realesrgan_x2plus.yml と同じ値
DEFAULT_OPT = {
    # 1 段目
    'kernel_list': _KERNELS, 'kernel_prob': _KERNEL_PROB, 'sinc_prob': 0.1,
    'blur_sigma': [0.2, 3], 'betag_range': [0.5, 4], 'betap_range': [1, 2],
    'resize_prob': [0.2, 0.7, 0.1], 'resize_range': [0.15, 1.5],
    'gaussian_noise_prob': 0.5, 'noise_range': [1, 30], 'poisson_scale_range': [0.05, 3],
    'gray_noise_prob': 0.4, 'jpeg_range': [30, 95],
    # 2 段目
    'second_blur_prob': 0.8,
    'kernel_list2': _KERNELS, 'kernel_prob2': _KERNEL_PROB, 'sinc_prob2': 0.1,
    'blur_sigma2': [0.2, 1.5], 'betag_range2': [0.5, 4], 'betap_range2': [1, 2],
    'resize_prob2': [0.3, 0.4, 0.3], 'resize_range2': [0.3, 1.2],
    'gaussian_noise_prob2': 0.5, 'noise_range2': [1, 25], 'poisson_scale_range2': [0.05, 2.5],
    'gray_noise_prob2': 0.4, 'jpeg_range2': [30, 95],
    # 最後の sinc フィルタ
    'final_sinc_prob': 0.8,
    # JPEG を入れるか (このリポジトリの Real-ESRGAN では切っている)
    'use_jpeg': False,
}

KERNEL_RANGE = [2 * v + 1 for v in range(3, 11)]  # 7, 9, ..., 21
KERNEL_PAD_TO = 21


def load_opt(yml_path=None):
    """劣化の設定を返す。yml を渡すと Real-ESRGAN の学習設定から値を読む。"""
    opt = copy.deepcopy(DEFAULT_OPT)
    if yml_path is None:
        return opt
    import yaml
    with open(yml_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    # 劣化の値は yml の最上位と datasets.train に分かれている
    sources = [cfg, cfg.get('datasets', {}).get('train', {})]
    for src in sources:
        for key in opt:
            if key in src:
                opt[key] = src[key]
    return opt


# ======================
# ぼかしカーネル (basicsr/data/degradations.py と同じ式)
# ======================

def _quad_form(k, sig_x, sig_y, theta):
    """各画素で x^T Σ^-1 x を計算する (k×k)。"""
    d = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    u = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    inv = np.linalg.inv(u @ d @ u.T)
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    grid = np.stack([xx, yy], axis=-1)
    return np.sum(np.dot(grid, inv) * grid, 2)


def _random_beta(beta_range):
    if np.random.uniform() < 0.5:
        return np.random.uniform(beta_range[0], 1)
    return np.random.uniform(1, beta_range[1])


def _random_blur_kernel(kernel_list, kernel_prob, k, sigma_range, betag_range, betap_range):
    ktype = random.choices(kernel_list, kernel_prob)[0]
    if 'aniso' in ktype:
        sig_x = np.random.uniform(*sigma_range)
        sig_y = np.random.uniform(*sigma_range)
        theta = np.random.uniform(-math.pi, math.pi)
    else:
        sig_x = sig_y = np.random.uniform(*sigma_range)
        theta = 0
    q = _quad_form(k, sig_x, sig_y, theta)
    if ktype.startswith('generalized'):
        kernel = np.exp(-0.5 * np.power(q, _random_beta(betag_range)))
    elif ktype.startswith('plateau'):
        kernel = np.reciprocal(np.power(q, _random_beta(betap_range)) + 1)
    else:
        kernel = np.exp(-0.5 * q)
    return kernel / np.sum(kernel)


def circular_lowpass_kernel(cutoff, k, pad_to=0):
    """sinc フィルタ (リンギングや高周波の欠けを作る)。"""
    c = (k - 1) / 2
    y, x = np.mgrid[0:k, 0:k]
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = cutoff * _bessel_j1(cutoff * r) / (2 * np.pi * r)
    kernel[int(c), int(c)] = cutoff ** 2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > k:
        p = (pad_to - k) // 2
        kernel = np.pad(kernel, ((p, p), (p, p)))
    return kernel


def _pad21(kernel):
    p = (KERNEL_PAD_TO - kernel.shape[0]) // 2
    return np.pad(kernel, ((p, p), (p, p)))


def _stage_kernel(opt, suffix):
    """1 段目 (suffix='') / 2 段目 (suffix='2') のぼかしカーネル。"""
    k = random.choice(KERNEL_RANGE)
    if np.random.uniform() < opt['sinc_prob' + suffix]:
        omega_c = np.random.uniform(np.pi / 3, np.pi) if k < 13 else np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, k)
    else:
        kernel = _random_blur_kernel(opt['kernel_list' + suffix], opt['kernel_prob' + suffix], k,
                                     opt['blur_sigma' + suffix], opt['betag_range' + suffix],
                                     opt['betap_range' + suffix])
    return _pad21(kernel)


def make_kernels(opt):
    """1 段目・2 段目・最後の sinc のカーネルを作る (最後の sinc は使わないとき None)。"""
    kernel1 = _stage_kernel(opt, '')
    kernel2 = _stage_kernel(opt, '2')
    sinc = None
    if np.random.uniform() < opt['final_sinc_prob']:
        k = random.choice(KERNEL_RANGE)
        sinc = circular_lowpass_kernel(np.random.uniform(np.pi / 3, np.pi), k, pad_to=KERNEL_PAD_TO)
    return kernel1, kernel2, sinc


# ======================
# 劣化の部品 (x は 1×C×H×W, 0-1 の torch テンソル)
# ======================

def _filter2d(x, kernel):
    k = kernel.shape[-1]
    c, h, w = x.shape[1:]
    pad = k // 2
    mode = 'reflect' if min(h, w) > pad else 'replicate'
    x = F.pad(x, (pad, pad, pad, pad), mode=mode)
    weight = torch.from_numpy(np.ascontiguousarray(kernel)).float().view(1, 1, k, k).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, groups=c)


def _random_scale(prob, scale_range):
    updown = random.choices(['up', 'down', 'keep'], prob)[0]
    if updown == 'up':
        return np.random.uniform(1, scale_range[1])
    if updown == 'down':
        return np.random.uniform(scale_range[0], 1)
    return 1


def _resize(x, scale=None, size=None):
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    if size is not None:
        size = (max(1, size[0]), max(1, size[1]))
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, scale_factor=scale, mode=mode)


def _add_noise(x, gauss_prob, sigma_range, poisson_range, gray_prob):
    gray = np.random.uniform() < gray_prob and x.shape[1] == 3
    if np.random.uniform() < gauss_prob:
        sigma = np.random.uniform(*sigma_range) / 255.
        shape = (1, 1) + tuple(x.shape[2:]) if gray else x.shape
        noise = torch.randn(shape) * sigma
    else:
        scale = np.random.uniform(*poisson_range)
        src = x
        if gray:
            src = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        src = torch.clamp((src * 255.).round(), 0, 255) / 255.
        vals = len(torch.unique(src))
        vals = 2 ** math.ceil(math.log2(max(vals, 2)))
        noise = (torch.poisson(src * vals) / vals - src) * scale
    return torch.clamp(x + noise, 0, 1)  # noise は gray のとき全チャンネル共通


def _jpeg(x, quality_range):
    quality = int(np.random.uniform(*quality_range))
    img = (x[0].permute(1, 2, 0).numpy() * 255.).round().clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return torch.from_numpy(img.copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.


# ======================
# まとめ
# ======================

def degrade(hr, scale, opt=None):
    """
    HR (RGB, float32, 0-255, HWC) から Real-ESRGAN 方式で LR を作る。
    HR の縦横は scale の倍数にしておくこと。返り値の LR は (H/scale, W/scale) で 0-255 の整数値。
    """
    opt = opt or DEFAULT_OPT
    h, w = hr.shape[:2]
    out_size = (h // scale, w // scale)
    kernel1, kernel2, sinc = make_kernels(opt)

    x = torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1))).float().unsqueeze(0) / 255.
    with torch.no_grad():
        # ---- 1 段目 ----
        x = _filter2d(x, kernel1)
        x = _resize(x, scale=_random_scale(opt['resize_prob'], opt['resize_range']))
        x = _add_noise(x, opt['gaussian_noise_prob'], opt['noise_range'],
                       opt['poisson_scale_range'], opt['gray_noise_prob'])
        x = torch.clamp(x, 0, 1)
        if opt['use_jpeg']:
            x = _jpeg(x, opt['jpeg_range'])

        # ---- 2 段目 ----
        if np.random.uniform() < opt['second_blur_prob']:
            x = _filter2d(x, kernel2)
        s = _random_scale(opt['resize_prob2'], opt['resize_range2'])
        x = _resize(x, size=(int(h / scale * s), int(w / scale * s)))
        x = _add_noise(x, opt['gaussian_noise_prob2'], opt['noise_range2'],
                       opt['poisson_scale_range2'], opt['gray_noise_prob2'])

        # ---- 最後: [縮小 + sinc] と [JPEG] の順番を 50% で入れ替え ----
        if np.random.uniform() < 0.5:
            x = _resize(x, size=out_size)
            if sinc is not None:
                x = _filter2d(x, sinc)
            x = torch.clamp(x, 0, 1)
            if opt['use_jpeg']:
                x = _jpeg(x, opt['jpeg_range2'])
        else:
            x = torch.clamp(x, 0, 1)
            if opt['use_jpeg']:
                x = _jpeg(x, opt['jpeg_range2'])
            x = _resize(x, size=out_size)
            if sinc is not None:
                x = _filter2d(x, sinc)

        x = torch.clamp((x * 255.).round(), 0, 255)
    return x[0].permute(1, 2, 0).numpy().astype(np.float32)


def usm_sharp(img, weight=0.5, threshold=10, radius=50):
    """Real-ESRGAN が正解画像にかけるアンシャープマスク (0-255 の HWC を受け取り、同じ範囲で返す)。"""
    if radius % 2 == 0:
        radius += 1
    img = img / 255.
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = (np.abs(residual) * 255 > threshold).astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    sharp = np.clip(img + weight * residual, 0, 1)
    out = soft_mask * sharp + (1 - soft_mask) * img
    return (out * 255.).astype(np.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
