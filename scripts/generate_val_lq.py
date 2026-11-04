"""検証（val）用の低解像度画像を、高解像度画像から作るスクリプト。

学習時（realesrgan/models/realesrgan_model.py の feed_data）と同じ劣化を、
学習設定ファイル（yml）のパラメータでかける。JPEG 圧縮は学習側と同じく使わない。
乱数の種（--seed）を固定するので、何度実行しても同じ画像になる。

使い方:
    python scripts/generate_val_lq.py --opt options/finetune_realesrgan_x2plus.yml
    python scripts/generate_val_lq.py --scale 2 --mode bicubic   # 単純な縮小だけにする場合

出力ファイル名は高解像度画像と同じにする（PairedImageDataset が同名で組にするため）。
"""
import argparse
import cv2
import glob
import math
import numpy as np
import os
import random
import torch
import yaml
from basicsr.data.degradations import (circular_lowpass_kernel, random_add_gaussian_noise_pt,
                                       random_add_poisson_noise_pt, random_mixed_kernels)
from basicsr.utils import USMSharp
from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F

IMG_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
KERNEL_RANGE = [2 * v + 1 for v in range(3, 11)]  # ぼかしカーネルの大きさ 7〜21


def read_image(path):
    """画像を読み、(RGB の float テンソル [1,3,H,W], 画素の最大値) を返す。8bit と 16bit に対応。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'読み込めません: {path}')
    max_val = 65535.0 if img.dtype == np.uint16 else 255.0
    if img.ndim == 2:  # グレースケールは 3 チャンネルにそろえる（学習時と同じ）
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / max_val
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, max_val


def write_image(path, tensor, max_val):
    img = tensor.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dtype = np.uint16 if max_val > 255 else np.uint8
    cv2.imwrite(path, (img * max_val).round().astype(dtype))


def make_blur_kernel(d, suffix):
    """学習データセット（RealESRGANDataset）と同じ方法でぼかしカーネルを作る。suffix は '' か '2'。"""
    kernel_size = random.choice(KERNEL_RANGE)
    if np.random.uniform() < d[f'sinc_prob{suffix}']:
        low = np.pi / 3 if kernel_size < 13 else np.pi / 5
        kernel = circular_lowpass_kernel(np.random.uniform(low, np.pi), kernel_size, pad_to=False)
    else:
        sigma = d[f'blur_sigma{suffix}']
        kernel = random_mixed_kernels(
            d[f'kernel_list{suffix}'], d[f'kernel_prob{suffix}'], kernel_size, sigma, sigma, [-math.pi, math.pi],
            d[f'betag_range{suffix}'], d[f'betap_range{suffix}'], noise_range=None)
    pad = (21 - kernel_size) // 2
    return torch.FloatTensor(np.pad(kernel, ((pad, pad), (pad, pad))))


def make_sinc_kernel(d):
    if np.random.uniform() < d['final_sinc_prob']:
        kernel = circular_lowpass_kernel(np.random.uniform(np.pi / 3, np.pi), random.choice(KERNEL_RANGE), pad_to=21)
        return torch.FloatTensor(kernel)
    pulse = torch.zeros(21, 21)  # ぼかさないカーネル
    pulse[10, 10] = 1
    return pulse


def random_resize(out, opt, suffix, size=None):
    """拡大・縮小・そのままをランダムに選んでリサイズする。"""
    updown = random.choices(['up', 'down', 'keep'], opt[f'resize_prob{suffix}'])[0]
    lo, hi = opt[f'resize_range{suffix}']
    scale = np.random.uniform(1, hi) if updown == 'up' else np.random.uniform(lo, 1) if updown == 'down' else 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    if size is None:
        return F.interpolate(out, scale_factor=scale, mode=mode)
    return F.interpolate(out, size=(int(size[0] * scale), int(size[1] * scale)), mode=mode)


def add_noise(out, opt, suffix):
    gray_prob = opt[f'gray_noise_prob{suffix}']
    if np.random.uniform() < opt[f'gaussian_noise_prob{suffix}']:
        return random_add_gaussian_noise_pt(
            out, sigma_range=opt[f'noise_range{suffix}'], clip=True, rounds=False, gray_prob=gray_prob)
    return random_add_poisson_noise_pt(
        out, scale_range=opt[f'poisson_scale_range{suffix}'], gray_prob=gray_prob, clip=True, rounds=False)


def final_resize(out, size):
    return F.interpolate(out, size=size, mode=random.choice(['area', 'bilinear', 'bicubic']))


@torch.no_grad()
def degrade(gt, opt, usm):
    """realesrgan_model.py の feed_data と同じ 2 段階の劣化（JPEG なし）。"""
    d = opt['datasets']['train']
    scale = opt['scale']
    h, w = gt.shape[2:4]
    lq_size = (h // scale, w // scale)
    kernel1 = make_blur_kernel(d, '').unsqueeze(0)
    kernel2 = make_blur_kernel(d, '2').unsqueeze(0)
    sinc_kernel = make_sinc_kernel(d).unsqueeze(0)

    # 1 回目: ぼかし → リサイズ → ノイズ
    out = filter2D(usm(gt), kernel1)
    out = random_resize(out, opt, '')
    out = add_noise(out, opt, '')
    out = torch.clamp(out, 0, 1)

    # 2 回目: ぼかし → リサイズ → ノイズ
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
    out = random_resize(out, opt, '2', size=lq_size)
    out = add_noise(out, opt, '2')

    # 最終: 目標サイズへのリサイズ + sinc フィルタ（学習側と同じく 50% で順番を入れ替える）
    if np.random.uniform() < 0.5:
        out = filter2D(final_resize(out, lq_size), sinc_kernel)
        out = torch.clamp(out, 0, 1)
    else:
        out = torch.clamp(out, 0, 1)
        out = filter2D(final_resize(out, lq_size), sinc_kernel)
    return torch.clamp((out * 255.0).round(), 0, 255) / 255.


def bicubic(gt, scale):
    h, w = gt.shape[2:4]
    return F.interpolate(gt, size=(h // scale, w // scale), mode='bicubic', align_corners=False).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description='val 用の低解像度画像を作る')
    parser.add_argument('--gt', default='dataset/val/gt', help='高解像度画像のフォルダ')
    parser.add_argument('--lq', default='dataset/val/lq', help='低解像度画像の出力先フォルダ')
    parser.add_argument('--opt', default='options/finetune_realesrgan_x2plus.yml', help='劣化パラメータを読む学習設定')
    parser.add_argument('--scale', type=int, default=None, help='縮小倍率（省略時は設定ファイルの scale）')
    parser.add_argument('--mode', choices=['degrade', 'bicubic'], default='degrade',
                        help='degrade: 学習と同じ劣化 / bicubic: 単純な縮小だけ')
    parser.add_argument('--seed', type=int, default=0, help='乱数の種（同じ値なら毎回同じ結果）')
    args = parser.parse_args()

    opt = None
    if args.mode == 'degrade' or args.scale is None:
        with open(args.opt, encoding='utf-8') as f:
            opt = yaml.safe_load(f)
    scale = args.scale or opt['scale']
    if opt is not None:
        opt['scale'] = scale

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    paths = sorted(p for p in glob.glob(os.path.join(args.gt, '*')) if p.lower().endswith(IMG_EXTS))
    if not paths:
        raise SystemExit(f'画像が見つかりません: {args.gt}')
    os.makedirs(args.lq, exist_ok=True)
    usm = USMSharp()

    for i, path in enumerate(paths, 1):
        gt, max_val = read_image(path)
        h, w = gt.shape[2:4]
        if h % scale or w % scale:
            # 高解像度の大きさが倍率で割り切れないと、検証時に出力と正解の大きさがずれる
            print(f'[スキップ] {os.path.basename(path)}: {w}x{h} が {scale} で割り切れません')
            continue
        lq = degrade(gt, opt, usm) if args.mode == 'degrade' else bicubic(gt, scale)
        out_path = os.path.join(args.lq, os.path.basename(path))
        write_image(out_path, lq, max_val)
        print(f'[{i}/{len(paths)}] {os.path.basename(path)}: {w}x{h} -> {lq.shape[3]}x{lq.shape[2]}')

    print(f'完了: {args.lq}')


if __name__ == '__main__':
    main()
