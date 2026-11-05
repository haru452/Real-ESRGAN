"""
ファインチューニング後の EDSR (ONNX) を使って超解像推論を行うスクリプト。

ailia を使わず onnxruntime だけで動くので、finetune_edsr.py で書き出した
.onnx ファイルの動作確認にそのまま使えます。

前処理・後処理は元の recognize_from_image.py (ailia版) と揃えてあります:
  - 画像は RGB, 0-255 の float のまま (正規化なし)
  - 出力は 0-255 にクリップして BGR で保存

必要なパッケージ:
    pip install onnxruntime opencv-python numpy

使い方:
    # 画像1枚を指定
    python infer_edsr.py --weight_path ./edsr/edsr_scale4_finetuned.onnx \
        --input input.png --savepath output.png

        python edsrx2/infer_edsr.py --weight_path ./edsrx2/edsr_scale4_finetuned.onnx         --input ./inputs/input_M111674228R_clip (2)_clip.tiff --savepath ./results/edsx2/a.tiff

    # フォルダ内の画像をまとめて処理
    python infer_edsr.py --weight_path ./edsr/edsr_scale4_finetuned.onnx \
        --input ./test_images --savepath ./results
"""

import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort


IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp','*.tif','*.tiff')


def get_args():
    parser = argparse.ArgumentParser(description='EDSR inference (onnxruntime)')
    parser.add_argument('--weight_path', required=True, help='推論に使う ONNX ファイル')
    parser.add_argument('--input', required=True, help='入力画像ファイル、または画像が入ったフォルダ')
    parser.add_argument('--savepath', required=True,
                         help='保存先。--input がフォルダの場合はフォルダを指定してください')
    parser.add_argument('--providers', nargs='+', default=None,
                         help="onnxruntime の実行プロバイダ (例: CUDAExecutionProvider)。"
                              "省略時は自動選択")
    return parser.parse_args()


def list_input_images(input_path):
    if os.path.isdir(input_path):
        paths = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        paths.sort()
        if len(paths) == 0:
            raise FileNotFoundError(f"{input_path} に画像が見つかりません")
        return paths
    else:
        return [input_path]


def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # HWC -> CHW -> NCHW  (正規化なし。学習時と同じ 0-255 の値域)
    input_data = img_rgb.transpose(2, 0, 1)[np.newaxis, ...]
    return input_data


def postprocess(output_data):
    # NCHW -> HWC
    output_img = output_data[0].transpose(1, 2, 0)
    output_img = np.clip(output_img, 0, 255)
    output_img = output_img.astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    return output_img


def get_output_path(savepath, input_path, is_dir_output):
    if is_dir_output:
        os.makedirs(savepath, exist_ok=True)
        name, _ = os.path.splitext(os.path.basename(input_path))
        return os.path.join(savepath, f"{name}_sr.tif")
    return savepath


def main():
    args = get_args()

    providers = args.providers or ort.get_available_providers()
    print(f"使用する ExecutionProvider: {providers}")

    session = ort.InferenceSession(args.weight_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_paths = list_input_images(args.input)
    is_dir_output = os.path.isdir(args.input) or len(input_paths) > 1

    for image_path in input_paths:
        print(f"処理中: {image_path}")
        input_data = preprocess(image_path)

        outputs = session.run([output_name], {input_name: input_data})
        output_img = postprocess(outputs[0])

        out_path = get_output_path(args.savepath, image_path, is_dir_output)
        cv2.imwrite(out_path, output_img)
        print(f"  -> 保存: {out_path}")

    print("完了しました。")


if __name__ == '__main__':
    main()
