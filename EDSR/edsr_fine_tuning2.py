# -*- coding: utf-8 -*-
"""
EDSR ファインチューニングスクリプト（.pt / .onnx 両対応版）

--weight_path に次のどちらでも指定できます。
  (A) EDSR-PyTorch 公式の重み (.pt)  例: edsr_x2-0edfb8a3.pt / edsr_baseline_x2-1bc95232.pt
      -> このスクリプト内の EDSR 定義に state_dict を読み込む
         (n_resblocks / n_feats / scale は重みから自動判定)
  (B) ailia 等の ONNX 重み (.onnx)    例: edsr_scale2.onnx
      -> onnx2torch で PyTorch モデルに変換

どちらの場合も、学習後は ONNX にエクスポートし直すので、
既存の ailia 推論スクリプト (recognize_from_image.py) にそのまま差し替えられます。

使い方 (.pt から x2 をファインチューニング):
    pip install torch onnx opencv-python        # .onnx 入力も使うなら onnx2torch も
    wget https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt

    python edsr_fine_tuning.py \
        --weight_path ./edsr_x2-0edfb8a3.pt \
        --hr_dir ./edsr/train_hr_images \
        --scale 2 \
        --patch_size 48 \
        --batch_size 16 \
        --epochs 100 \
        --lr 1e-5 \
        --save_every 10 \
        --io_names_from ./edsr/edsr_scale2.onnx \
        --output_path ./edsr/edsr_scale2_finetuned.onnx

    ※ --io_names_from は任意。元の ailia 用 ONNX の入出力名を引き継ぎたいときに指定します。

過学習の確認 (val):
    --val_hr_dir を付けると、--val_every エポックごとに val の L1 損失と PSNR を計算し、
    --log_dir に val_metrics.csv と training_curves.png を保存します。
    PSNR が一番良かったエポックの重みは checkpoint_best.pt に保存し、
    ONNX にもその重みを書き出します (--export_weights last で最後の重みに変更可)。

        python EDSR/edsr_fine_tuning2.py ... --val_hr_dir ./dataset/val/gt --val_max_images 200

    --val_lr_dir を省略すると、学習と同じ bicubic 縮小で val の LR を作ります。
    --val_max_images は時間短縮用 (名前順で先頭から指定枚数。省略で全部)。

入力の前提: RGB / float32 / 0-255 (正規化なし)。EDSR-PyTorch・ailia とも同じ。
"""

import argparse
import glob
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from edsr_compare import Comparer
from edsr_val import ValLogger, ValSet, evaluate, evaluate_bicubic


def format_seconds(sec):
    """秒を「1時間23分」「4分05秒」のような読みやすい形にする。"""
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}時間{m:02d}分"
    return f"{m}分{sec:02d}秒"


def format_metrics(m):
    return (f"val_loss={m['l1']:.4f}  val_psnr={m['psnr']:.3f} dB  "
            f"val_ssim={m['ssim']:.4f}  bias={m['bias']:+.2f}")


# ======================
# EDSR モデル定義 (EDSR-PyTorch の src/model/edsr.py, common.py 相当)
#   state_dict のキー名を公式と一致させているので .pt をそのまま読み込める
# ======================

def default_conv(in_ch, out_ch, k, bias=True):
    return nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, k, res_scale=1.0):
        super().__init__()
        self.body = nn.Sequential(conv(n_feats, n_feats, k), nn.ReLU(True),
                                  conv(n_feats, n_feats, k))
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x).mul(self.res_scale) + x


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats):
        m = []
        if (scale & (scale - 1)) == 0:  # 2, 4, 8...
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super().__init__(*m)


class EDSR(nn.Module):
    def __init__(self, n_resblocks, n_feats, scale, res_scale, rgb_range=255, n_colors=3):
        super().__init__()
        conv = default_conv
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(conv(n_colors, n_feats, 3))
        body = [ResBlock(conv, n_feats, 3, res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(conv(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(Upsampler(conv, scale, n_feats),
                                  conv(n_feats, n_colors, 3))

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return self.add_mean(x)


def build_edsr_from_pt(pt_path, scale_arg, device):
    """公式 .pt から構成を自動判定して EDSR を作り、重みを読み込む"""
    sd = torch.load(pt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # DataParallel 保存や 'model.' 接頭辞を除去
    sd = {k.replace('module.', '', 1).replace('model.', '', 1): v for k, v in sd.items()}

    n_feats = sd['head.0.weight'].shape[0]
    n_resblocks = len({k.split('.')[1] for k in sd if k.startswith('body.') and '.body.' in k})
    up_convs = [k for k in sd if k.startswith('tail.0.') and k.endswith('.weight')]
    if sd[up_convs[0]].shape[0] == 9 * n_feats:
        scale = 3
    else:
        scale = 2 ** len(up_convs)
    res_scale = 0.1 if n_feats == 256 else 1.0  # 公式: EDSR=0.1, baseline=1

    if scale != scale_arg:
        raise ValueError(f"重みは x{scale} 用ですが --scale {scale_arg} が指定されています")

    print(f"  検出した構成: n_resblocks={n_resblocks}, n_feats={n_feats}, "
          f"scale=x{scale}, res_scale={res_scale}")
    model = EDSR(n_resblocks, n_feats, scale, res_scale)
    # MeanShift は公式重みにも含まれるが、含まれない場合に備えて strict=False で補う
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if 'mean' not in k]
    if missing or unexpected:
        raise RuntimeError(f"重みの読み込みに失敗: missing={missing}, unexpected={unexpected}")
    return model.to(device)


def build_model_from_onnx(onnx_path, device):
    try:
        from onnx2torch import convert
    except ImportError:
        raise ImportError("onnx2torch が見つかりません。`pip install onnx2torch onnx` でインストールしてください。")
    return convert(onnx_path).to(device)


def get_onnx_io_names(onnx_path):
    import onnx
    m = onnx.load(onnx_path)
    return m.graph.input[0].name, m.graph.output[0].name


# ======================
# 引数
# ======================

def get_args():
    parser = argparse.ArgumentParser(description="EDSR fine-tuning")
    parser.add_argument('--weight_path', required=True,
                        help='元の重み (.pt: EDSR-PyTorch 公式 / .onnx: ailia 等)')
    parser.add_argument('--hr_dir', required=True, help='高解像度(正解)画像が入ったディレクトリ')
    parser.add_argument('--lr_dir', default=None,
                        help='低解像度画像のディレクトリ (省略時は hr_dir から自動生成)')
    parser.add_argument('--scale', type=int, choices=[2, 3, 4], default=2)
    parser.add_argument('--patch_size', type=int, default=48,
                        help='LR側のパッチサイズ (HR側は patch_size * scale)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--output_path', default=None,
                        help='ファインチューニング後の ONNX 出力先 (省略時は _finetuned.onnx を付加)')
    parser.add_argument('--io_names_from', default=None,
                        help='入出力名を引き継ぐ ONNX (省略時: .onnx 入力ならそれ、.pt なら input/output)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset バージョン')
    parser.add_argument('--save_every', type=int, default=10, help='何エポックごとに保存するか')
    parser.add_argument('--iters_per_epoch', type=int, default=None,
                        help='1エポックあたりに切り出すパッチ数 (省略時は画像枚数と同数)')
    parser.add_argument('--val_hr_dir', default=None,
                        help='val の高解像度(正解)画像フォルダ。指定すると過学習チェックを行う')
    parser.add_argument('--val_lr_dir', default=None,
                        help='val の低解像度画像フォルダ (同名で組にする。省略時は bicubic で自動生成)')
    parser.add_argument('--val_every', type=int, default=1, help='何エポックごとに val を行うか')
    parser.add_argument('--val_max_images', type=int, default=None,
                        help='val に使う最大枚数 (名前順で先頭から。省略時は全部)')
    parser.add_argument('--log_dir', default='.',
                        help='val_metrics.csv / training_curves.png / checkpoint_best.pt の保存先')
    parser.add_argument('--export_weights', choices=['best', 'last'], default='best',
                        help='ONNX に書き出す重み (best: val PSNR 最良 / last: 最後のエポック)')
    parser.add_argument('--lr_schedule', choices=['cosine', 'step', 'none'], default='cosine',
                        help='学習率の下げ方 (cosine: --epochs で --lr_min まで滑らかに / '
                             'step: --lr_step エポックごとに --lr_gamma 倍 / none: 固定)')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='cosine のときの最小学習率')
    parser.add_argument('--lr_step', type=int, default=10, help='step のときの間隔 (エポック)')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='step のときの倍率')
    parser.add_argument('--early_stop', type=int, default=10,
                        help='val PSNR が --min_delta 以上よくならない状態がこのエポック数続いたら止める '
                             '(0 で無効。--val_hr_dir が必要)')
    parser.add_argument('--min_delta', type=float, default=0.01,
                        help='「よくなった」とみなす val PSNR の最小の上がり幅 (dB)')
    parser.add_argument('--compare_images', type=int, default=4,
                        help='比較画像 (bicubic / 学習前 / 学習後 / 正解) を作る val 画像の枚数 (0 で作らない)')
    parser.add_argument('--compare_crop', type=int, default=128,
                        help='比較画像で切り出す HR 側の大きさ (画素)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


# ======================
# データセット
# ======================

class SRPatchDataset(Dataset):
    """
    HR画像から HRパッチ / LRパッチ のペアを切り出す。
    - lr_dir が指定されていれば、同名ファイルのペアとして扱う
    - 指定がなければ、HR を bicubic で 1/scale にダウンサンプルして LR を生成する
    値は 0-255 の float (RGB) のまま扱う。
    """

    IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')

    def __init__(self, hr_dir, lr_dir, scale, patch_size, length_per_epoch=1000):
        self.scale = scale
        self.patch_size = patch_size
        self.length_per_epoch = length_per_epoch

        self.hr_paths = []
        for ext in self.IMG_EXTS:
            self.hr_paths.extend(glob.glob(os.path.join(hr_dir, ext)))
        self.hr_paths.sort()
        if len(self.hr_paths) == 0:
            raise FileNotFoundError(f"{hr_dir} に画像が見つかりません")

        self.lr_dir = lr_dir
        if lr_dir is not None:
            self.lr_paths = {
                os.path.basename(p): p
                for ext in self.IMG_EXTS
                for p in glob.glob(os.path.join(lr_dir, ext))
            }

    def __len__(self):
        return self.length_per_epoch

    def _load_pair(self, idx):
        hr_path = self.hr_paths[idx % len(self.hr_paths)]
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)  # BGR
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.lr_dir is not None:
            fname = os.path.basename(hr_path)
            if fname not in self.lr_paths:
                raise FileNotFoundError(f"{fname} に対応する LR 画像が見つかりません")
            lr_img = cv2.imread(self.lr_paths[fname], cv2.IMREAD_COLOR)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            h, w = hr_img.shape[:2]
            # HR を scale の倍数に切りそろえてから縮小 (位置ずれ防止)
            h, w = h - h % self.scale, w - w % self.scale
            hr_img = hr_img[:h, :w]
            lr_img = cv2.resize(hr_img, (w // self.scale, h // self.scale),
                                interpolation=cv2.INTER_CUBIC)
        return hr_img, lr_img

    def __getitem__(self, idx):
        hr_full, lr_full = self._load_pair(idx)
        lp = self.patch_size
        hp = lp * self.scale

        lh, lw = lr_full.shape[:2]
        if lh < lp or lw < lp:
            # 画像が小さすぎる場合は拡大して最低サイズを確保
            s = max(lp / lh, lp / lw) + 0.01
            lr_full = cv2.resize(lr_full, (int(lw * s) + 1, int(lh * s) + 1))
            hr_full = cv2.resize(hr_full, (lr_full.shape[1] * self.scale,
                                           lr_full.shape[0] * self.scale))
            lh, lw = lr_full.shape[:2]

        ly = random.randint(0, lh - lp)
        lx = random.randint(0, lw - lp)
        hy, hx = ly * self.scale, lx * self.scale

        lr_patch = lr_full[ly:ly + lp, lx:lx + lp, :]
        hr_patch = hr_full[hy:hy + hp, hx:hx + hp, :]

        # 左右反転 / 90度回転 (データ拡張)
        if random.random() < 0.5:
            lr_patch = lr_patch[:, ::-1, :]
            hr_patch = hr_patch[:, ::-1, :]
        if random.random() < 0.5:
            lr_patch = np.rot90(lr_patch)
            hr_patch = np.rot90(hr_patch)

        # HWC -> CHW
        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_patch.transpose(2, 0, 1)))
        hr_tensor = torch.from_numpy(np.ascontiguousarray(hr_patch.transpose(2, 0, 1)))
        return lr_tensor, hr_tensor


# ======================
# メイン処理
# ======================

def main():
    args = get_args()
    is_onnx = args.weight_path.lower().endswith('.onnx')

    if args.output_path is None:
        base, _ = os.path.splitext(args.weight_path)
        args.output_path = f"{base}_finetuned.onnx"

    # 入出力名
    if args.io_names_from:
        input_name, output_name = get_onnx_io_names(args.io_names_from)
    elif is_onnx:
        input_name, output_name = get_onnx_io_names(args.weight_path)
    else:
        input_name, output_name = 'input', 'output'

    print(f"[1/4] モデル読み込み中: {args.weight_path}")
    if is_onnx:
        model = build_model_from_onnx(args.weight_path, args.device)
    else:
        model = build_edsr_from_pt(args.weight_path, args.scale, args.device)
    model.train()

    print("[2/4] データセット準備中")
    hr_count = sum(len(glob.glob(os.path.join(args.hr_dir, ext)))
                   for ext in SRPatchDataset.IMG_EXTS)
    length_per_epoch = args.iters_per_epoch or hr_count
    # バッチ数が 0 にならないよう最低 batch_size 個は確保
    length_per_epoch = max(length_per_epoch, args.batch_size)
    print(f"  検出した画像枚数: {hr_count} 枚 -> 1エポックあたり {length_per_epoch} パッチ")

    dataset = SRPatchDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        scale=args.scale,
        patch_size=args.patch_size,
        length_per_epoch=length_per_epoch,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    params = [p for p in model.parameters() if p.requires_grad]  # MeanShift は固定
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.L1Loss()  # EDSR は L1 loss が一般的
    if args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)
    elif args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    else:
        scheduler = None
    print(f"  学習率: {args.lr:g} (スケジュール: {args.lr_schedule})")

    val_set, val_logger, comparer = None, None, None
    best_psnr, best_epoch = -float('inf'), 0
    best_path = os.path.join(args.log_dir, 'checkpoint_best.pt')
    if args.val_hr_dir:
        val_set = ValSet(args.val_hr_dir, args.val_lr_dir, args.scale, args.val_max_images)
        lr_src = args.val_lr_dir or 'bicubic で自動生成'
        print(f"  val: {len(val_set)} 枚 (LR: {lr_src}) を {args.val_every} エポックごとに評価")
        # 超解像しない場合 (bicubic で拡大しただけ) の値を基準として記録
        bic = evaluate_bicubic(val_set)
        print(f"  bicubic (基準)       psnr={bic['psnr']:.3f} dB  ssim={bic['ssim']:.4f}")
        val_logger = ValLogger(args.log_dir, bicubic=bic)
        # 学習前の値 (epoch 0)
        m = evaluate(model, val_set, args.device)
        print(f"  epoch    0 (学習前)  {format_metrics(m)}")
        val_logger.add(0, args.lr, float('nan'), m)
        best_psnr = m['psnr']
        torch.save(model.state_dict(), best_path)
        if args.compare_images > 0:
            comparer = Comparer(val_set, args.compare_images, args.compare_crop, args.log_dir)
            comparer.capture_before(model, args.device)
    elif args.early_stop:
        print("  --val_hr_dir が無いので早期終了は使いません")

    print(f"[3/4] 学習開始 (device={args.device})")
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for lr_batch, hr_batch in loader:
            lr_batch = lr_batch.to(args.device)
            hr_batch = hr_batch.to(args.device)

            optimizer.zero_grad()
            pred = model(lr_batch)
            loss = criterion(pred, hr_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        cur_lr = optimizer.param_groups[0]['lr']
        msg = f"  epoch {epoch:4d}/{args.epochs}  lr={cur_lr:.2e}  loss={avg_loss:.4f}"
        if scheduler is not None:
            scheduler.step()

        stop = False
        if val_set is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            m = evaluate(model, val_set, args.device)
            msg += f"  {format_metrics(m)}"
            val_logger.add(epoch, cur_lr, avg_loss, m)
            if m['psnr'] > best_psnr:
                if m['psnr'] >= best_psnr + args.min_delta:
                    best_epoch = epoch  # 早期終了の判定は min_delta 以上の改善だけ数える
                best_psnr = m['psnr']
                torch.save(model.state_dict(), best_path)
                msg += "  (best 更新)"
            if args.early_stop and epoch - best_epoch >= args.early_stop:
                stop = True

        # 残り時間の目安: ここまでの 1 エポックあたりの平均（val の時間も含む）× 残りエポック数
        elapsed = time.time() - train_start
        remain = elapsed / epoch * (args.epochs - epoch)
        finish = time.strftime('%m/%d %H:%M', time.localtime(time.time() + remain))
        msg += f"  | 経過 {format_seconds(elapsed)}  残り約 {format_seconds(remain)} (終了予定 {finish})"
        print(msg)

        if epoch % args.save_every == 0 or epoch == args.epochs or stop:
            ckpt_path = f"checkpoint_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)  # .pt 入力時は EDSR-PyTorch 互換
            print(f"    -> チェックポイント保存: {ckpt_path}")

        if stop:
            print(f"  早期終了: {args.early_stop} エポックの間、val PSNR が "
                  f"{args.min_delta} dB 以上よくならなかったので止めます (epoch {epoch})")
            break

    if val_set is not None:
        print(f"  val PSNR 最良: {best_psnr:.3f} dB (bicubic より "
              f"{best_psnr - val_logger.bicubic['psnr']:+.3f} dB) -> {best_path}")
        print(f"  グラフ: {val_logger.png_path} / 表: {val_logger.csv_path}")
        if args.export_weights == 'best':
            model.load_state_dict(torch.load(best_path, map_location=args.device))
            print("  ONNX には val PSNR が最良の重みを書き出します")
        if comparer is not None:
            paths = comparer.save_after(model, args.device, tag=args.export_weights)
            print(f"  比較画像 ({len(paths)} 枚): {comparer.out_dir}")

    print(f"[4/4] ONNX へエクスポート中: {args.output_path}")
    model.eval()
    dummy_input = torch.rand(1, 3, args.patch_size, args.patch_size, device=args.device) * 255
    export_kwargs = dict(
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: 'batch', 2: 'height', 3: 'width'},
            output_name: {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=args.opset,
    )
    with torch.no_grad():
        try:
            # 新しい PyTorch では従来方式のエクスポータを明示 (ailia との互換性のため)
            torch.onnx.export(model, dummy_input, args.output_path, dynamo=False, **export_kwargs)
        except TypeError:
            # 古い PyTorch (dynamo 引数なし)
            torch.onnx.export(model, dummy_input, args.output_path, **export_kwargs)
    print("完了")


if __name__ == '__main__':
    main()
