# -*- coding: utf-8 -*-
"""
学習で保存された .pt (checkpoint_epoch10.pt など) を、学習はせずに ONNX に書き出す。

書き出し方は edsr_fine_tuning2.py の最後の処理と同じなので、
できた ONNX はそのまま infer_edsr.py で使えます。

    python EDSR/export_edsr_onnx.py \
        --weight_path ./checkpoint_epoch10.pt \
        --output_path ./EDSR/edsr_scale2_realesrgan_deg_epoch10.onnx \
        --scale 2

--output_path を省略すると、<重みの名前>.onnx を重みと同じ場所に作ります。
.pt から学習を始めたときのチェックポイントが対象です (.onnx から始めた学習の .pt は読めません)。
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edsr_fine_tuning2 import build_edsr_from_pt, get_onnx_io_names  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(description='EDSR の .pt を ONNX に書き出す')
    p.add_argument('--weight_path', required=True, help='書き出したい .pt (チェックポイント)')
    p.add_argument('--output_path', default=None, help='保存する ONNX のパス (省略時は <重みの名前>.onnx)')
    p.add_argument('--scale', type=int, choices=[2, 3, 4], default=2)
    p.add_argument('--io_names_from', default=None,
                   help='入出力名を引き継ぐ元の ONNX (省略時は input / output)')
    p.add_argument('--opset', type=int, default=11, help='ONNX opset バージョン')
    p.add_argument('--patch_size', type=int, default=48, help='書き出しに使うダミー入力の大きさ')
    return p.parse_args()


def main():
    args = get_args()
    if args.output_path is None:
        args.output_path = os.path.splitext(args.weight_path)[0] + '.onnx'
    if os.path.abspath(args.output_path) == os.path.abspath(args.weight_path):
        raise ValueError('--output_path が --weight_path と同じです')

    if args.io_names_from:
        input_name, output_name = get_onnx_io_names(args.io_names_from)
    else:
        input_name, output_name = 'input', 'output'

    print(f"重みを読み込み中: {args.weight_path}")
    model = build_edsr_from_pt(args.weight_path, args.scale, 'cpu')
    model.eval()

    print(f"ONNX へ書き出し中: {args.output_path}")
    dummy_input = torch.rand(1, 3, args.patch_size, args.patch_size) * 255
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
