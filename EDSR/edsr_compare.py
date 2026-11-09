# -*- coding: utf-8 -*-
"""
val 画像の一部を切り出して、
    LR (拡大表示) | bicubic | 学習前 EDSR | 学習後 EDSR | 正解 (HR)
を横に並べた PNG を保存する。PSNR だけでは分からない
「クレーターの縁や岩などの細かい形が本当に出ているか」を目で確かめるため。

切り出す場所は、正解画像の中で一番細かい模様が多い (ラプラシアンの分散が大きい) 所を選ぶ。
"""

import os

import cv2
import numpy as np

from edsr_val import bicubic_up, calc_psnr, pick_evenly, super_resolve


def find_detailed_crop(hr, size):
    """HR の中で細部が一番多い size×size の左上座標を返す"""
    h, w = hr.shape[:2]
    if h <= size or w <= size:
        return 0, 0
    gray = cv2.cvtColor(np.clip(hr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F) ** 2
    energy = cv2.boxFilter(lap, -1, (size, size), normalize=True)
    # boxFilter の値は窓の中心。端からはみ出さない範囲だけで最大を探す
    half = size // 2
    valid = energy[half:h - size + half + 1, half:w - size + half + 1]
    y, x = np.unravel_index(int(np.argmax(valid)), valid.shape)
    return int(y), int(x)


class Comparer:
    """学習前に「学習前 EDSR の出力」を取っておき、学習後に並べて保存する"""

    def __init__(self, val_set, n_images, crop_size, out_dir):
        self.val_set = val_set
        self.crop = crop_size
        self.out_dir = os.path.join(out_dir, 'compare')
        self.indices = pick_evenly(list(range(len(val_set))), n_images)
        self.items = []  # [(name, (y, x), lr, hr, bicubic, pretrained_sr)]

    def capture_before(self, model, device):
        s = self.val_set.scale
        for i in self.indices:
            lr, hr = self.val_set[i]
            c = min(self.crop, hr.shape[0], hr.shape[1])
            c -= c % s
            y, x = find_detailed_crop(hr, c)
            y, x = y - y % s, x - x % s
            self.items.append({
                'name': os.path.splitext(self.val_set.name(i))[0],
                'box': (y, x, c),
                'lr': lr, 'hr': hr,
                'bicubic': bicubic_up(lr, s),
                'before': super_resolve(model, lr, device),
            })

    def save_after(self, model, device, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        s = self.val_set.scale
        paths = []
        for it in self.items:
            after = super_resolve(model, it['lr'], device)
            y, x, c = it['box']
            cut = lambda img: img[y:y + c, x:x + c]
            lr_cut = it['lr'][y // s:(y + c) // s, x // s:(x + c) // s]
            lr_view = cv2.resize(lr_cut, (c, c), interpolation=cv2.INTER_NEAREST)
            hr = cut(it['hr'])
            psnr = lambda img: f"{calc_psnr(img, hr, crop_border=s):.2f}dB"
            panels = [
                ('LR (nearest)', lr_view),
                (f"bicubic {psnr(cut(it['bicubic']))}", cut(it['bicubic'])),
                (f"before {psnr(cut(it['before']))}", cut(it['before'])),
                (f"after {psnr(cut(after))}", cut(after)),
                ('HR (truth)', hr),
            ]
            tiles = [_tile(img, text) for text, img in panels]
            sheet = np.concatenate(tiles, axis=1)
            path = os.path.join(self.out_dir, f"{it['name']}_{tag}.png")
            cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            paths.append(path)
        return paths


def _tile(img, text, view=256):
    """見やすいように view 四方へ nearest で拡大し、上に文字を入れる"""
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    img = cv2.resize(img, (view, view), interpolation=cv2.INTER_NEAREST)
    bar = np.full((24, view, 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    pad = np.full((24 + view, 4, 3), 255, np.uint8)
    return np.concatenate([np.concatenate([bar, img], axis=0), pad], axis=1)
