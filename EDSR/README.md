## this is only for me

# EDSR ファインチューニング

公式の EDSR の重みから始めて、自分の画像で追加学習（ファインチューニング）し、
ONNX に書き出して推論するためのフォルダです。

## ファイル

| ファイル | 役割 |
| --- | --- |
| `edsr_fine_tuning2.py` | 学習スクリプト。EDSR のモデル定義、学習データの作成、学習、ONNX への書き出し |
| `edsr_val.py` | 検証（val）の処理。val の損失・PSNR の計算、CSV とグラフの保存（学習スクリプトから呼ばれる） |
| `infer_edsr.py` | 書き出した ONNX で超解像する推論スクリプト（onnxruntime を使用） |

## 準備

```
pip install torch opencv-python numpy matplotlib onnx onnxruntime
# .onnx の重みから学習を始める場合だけ
pip install onnx2torch
```

公式の重み（EDSR-PyTorch）をダウンロードします。x2 の例：

```
wget https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt            # EDSR 本体（32 ブロック・256 チャンネル）
wget https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt   # 軽量版（16 ブロック・64 チャンネル）
```

ブロック数・チャンネル数・倍率は、重みファイルから自動で読み取ります。
`--scale` と重みの倍率が違うとエラーで止まります。

## 学習データの作り方

用意するのは **高解像度（HR）画像のフォルダだけ** です。低解像度（LR）画像は学習中に自動で作ります。

1 枚の学習データは、次の手順で作ります（`SRPatchDataset`）。

1. HR 画像を読み込む
   - RGB、値は 0〜255 のまま（正規化しない）。公式 EDSR と同じ
   - 16bit の TIFF は 8bit に、白黒画像は 3 チャンネルに変換される（OpenCV の `IMREAD_COLOR`）
2. HR 画像の縦横を倍率の倍数に切りそろえる（LR と位置がずれないように）
3. HR を **bicubic で 1/倍率に縮小** して LR を作る（OpenCV の `cv2.resize(..., INTER_CUBIC)`）
   - ぼかしやノイズ、JPEG 圧縮は加えない。Real-ESRGAN の学習データ（劣化をランダムに加える）とはここが違う
4. LR から `--patch_size`（初期値 48）四方を、ランダムな位置で切り出す。HR からは同じ位置を倍率倍の大きさで切り出す
   - 例: x2・patch_size 48 なら、LR 48×48 と HR 96×96 の組
   - LR が patch_size より小さい画像は、拡大してから切り出す
5. データ拡張: 50% の確率で左右反転、50% の確率で 90 度回転

**1 エポックの長さ:** 1 エポックで切り出すパッチの数は、初期値では HR 画像の枚数と同じです
（1 枚から 1 か所ずつ）。`--iters_per_epoch` で変えられます。
例えば HR 17000 枚・`--batch_size 16` なら、1 エポックは約 1062 回の更新です。

**LR を自分で用意する場合:** `--lr_dir` に LR 画像のフォルダを指定すると、自動縮小の代わりに
**HR と同じファイル名** の画像を組にして使います。LR の大きさは HR のちょうど 1/倍率にしてください
（ずれていると切り出し位置がずれます）。

**学習の設定:** 損失は L1、最適化は Adam です（公式 EDSR と同じ）。
`.pt` から始める場合、入力の平均を引く層（MeanShift）は固定され、学習されません。

## 学習

リポジトリのいちばん上のフォルダで実行します。

```
python EDSR/edsr_fine_tuning2.py \
    --weight_path ./edsr_x2-0edfb8a3.pt \
    --hr_dir <学習用 HR 画像のフォルダ> \
    --scale 2 \
    --patch_size 48 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-5 \
    --save_every 10 \
    --val_hr_dir ./dataset/val/gt \
    --val_max_images 200 \
    --log_dir ./edsr_logs \
    --output_path ./edsr_logs/edsr_scale2_finetuned.onnx
```

主な引数

| 引数 | 初期値 | 説明 |
| --- | --- | --- |
| `--weight_path` | （必須） | 元の重み。`.pt`（公式）か `.onnx` |
| `--hr_dir` | （必須） | 学習用 HR 画像のフォルダ |
| `--lr_dir` | なし | 学習用 LR 画像のフォルダ。省略すると bicubic で自動作成 |
| `--scale` | 2 | 倍率（2 / 3 / 4） |
| `--patch_size` | 48 | LR 側の切り出しサイズ（HR 側はこの倍率倍） |
| `--batch_size` | 8 | 1 回の更新で使うパッチ数 |
| `--epochs` | 50 | 学習するエポック数 |
| `--lr` | 1e-4 | 学習率。ファインチューニングなら 1e-5 くらいがおすすめ |
| `--iters_per_epoch` | HR 枚数 | 1 エポックで切り出すパッチ数 |
| `--save_every` | 10 | 何エポックごとに `checkpoint_epoch{N}.pt` を保存するか |
| `--output_path` | 元の重み名 + `_finetuned.onnx` | 書き出す ONNX の場所 |
| `--io_names_from` | なし | 入出力名を引き継ぎたい ONNX（ailia 用の元 ONNX など） |

`checkpoint_epoch{N}.pt` は **コマンドを実行したフォルダ** に保存されます。

## val（過学習の確認）

`--val_hr_dir` を付けると、エポックごとに val の **L1 損失** と **PSNR** を計算します。
PSNR の計算方法は Real-ESRGAN 側と同じ（RGB・0〜255・周囲を倍率分の画素だけ除く）です。

| 引数 | 初期値 | 説明 |
| --- | --- | --- |
| `--val_hr_dir` | なし | val の HR 画像のフォルダ。付けると val を行う |
| `--val_lr_dir` | なし | val の LR 画像のフォルダ（同じファイル名で組にする）。省略すると学習と同じ bicubic で作る |
| `--val_every` | 1 | 何エポックごとに val を行うか |
| `--val_max_images` | 全部 | val に使う枚数（名前順で先頭から）。時間短縮用 |
| `--log_dir` | `.` | 結果の保存先 |
| `--export_weights` | best | ONNX に書き出す重み。`best`: val の PSNR が最良 / `last`: 最後のエポック |

`--log_dir` に保存されるもの

- `val_metrics.csv`: エポックごとの学習損失・val 損失・val PSNR の表（epoch 0 は学習前の値）
- `training_curves.png`: 左が損失（学習と val）、右が val PSNR のグラフ。最良の時点に ★
- `checkpoint_best.pt`: val PSNR が最良だったときの重み

**見方:** 学習損失は下がっているのに、val 損失が上がる（PSNR が下がる）ようになったら過学習です。

**val の LR をどう作るか**

- 過学習を見るだけなら、`--val_lr_dir` を付けない（学習と同じ bicubic 縮小になるので条件がそろう）
- Real-ESRGAN と同じ条件で比べたいなら `--val_lr_dir ./dataset/val/lq` を付ける。
  この LR には Real-ESRGAN 用のぼかし・ノイズが入っていて、EDSR はそれを学習していないので、PSNR は低めに出る

EDSR は val 画像を 1 枚ずつ丸ごと処理するため、val の枚数が多いと時間がかかります
（1900 枚だと 1 回数分）。`--val_max_images` や `--val_every` で調整してください。

## 推論

```
# 1 枚
python EDSR/infer_edsr.py --weight_path ./edsr_logs/edsr_scale2_finetuned.onnx \
    --input input.png --savepath output.png

# フォルダ内をまとめて（<元の名前>_sr.tif で保存）
python EDSR/infer_edsr.py --weight_path ./edsr_logs/edsr_scale2_finetuned.onnx \
    --input ./test_images --savepath ./results

# GPU を使う場合
python EDSR/infer_edsr.py ... --providers CUDAExecutionProvider
```

入力は学習と同じく RGB・0〜255 で、出力は 0〜255 に切りそろえて保存します。

**白黒画像の場合:** モデルは 3 チャンネル（RGB）なので、白黒画像は同じ値を 3 チャンネルにコピーして処理します
（学習・val も同じ）。出力の 3 チャンネルはわずかにずれて色が付くことがあるため、
入力が白黒（1 チャンネル、または R=G=B）なら、出力を白黒の 1 チャンネルに戻して保存します
（Real-ESRGAN の推論と同じ）。
画像全体を一度に処理するので、大きな画像ではメモリが足りなくなることがあります。

## 注意

- 学習の LR は OpenCV の bicubic 縮小で作っています。公式 EDSR の学習データ（MATLAB の縮小）とは少し違います。
  評価用に LR を作るときも、同じ OpenCV の方法で縮小してください。
- `.onnx` の重みから学習を始めた場合、保存される `.pt` は公式 EDSR の形式にはなりません
  （このスクリプトで続きを学習するときは ONNX を使ってください）。
