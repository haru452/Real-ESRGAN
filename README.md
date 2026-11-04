# Real-ESRGAN（月面 NAC 画像向け改造版）

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) をフォークし、月面画像サイト **QuickMap** で入手できる **LROC NAC 画像**を超解像するために改造したものです。

- 本家の README は [README_original.md](README_original.md) に移動しました（インストール方法・推論方法・モデル一覧などはそちらを参照）。
- 本家からの分岐点はコミット `a4abfb2`（2024-04-03）です。差分は `git diff a4abfb2 HEAD` で確認できます。

---

## 本家からの変更点

### 1. データセット作成ノートブックを追加（`createDataset.ipynb`）

Google Colab 上で、QuickMap から保存した NAC 画像を学習用パッチに変換します。

1. Google Drive をマウントし、`/content/drive/My Drive/Research/data_input/*.tif` を更新日時の古い順に読み込む
2. `process_screenshot()` で余分な部分を切り落とす
   - 上 80px・下 120px を固定で削除（スマホのステータスバー・ナビゲーションバー分）
   - 周囲の黒い縁を自動でトリミング（明るさ 10 以下を「暗い」とみなす）
3. 1024×1024 のパッチに分割し、`/content/drive/My Drive/Research/B` に `.tif` で保存
   - 1024×1024 に満たない端の部分は保存しない
   - メモリ節約のため、1 枚ずつ処理して終わったら解放する
4. ノートブック末尾の「いらなくなったコードが下にある」以降は旧コード（未使用）

### 2. 劣化モデルから JPEG 圧縮を除去（`realesrgan/models/realesrgan_model.py`）

Real-ESRGAN は高解像度画像に「ぼかし → リサイズ → ノイズ → JPEG 圧縮」を 2 回かけて低解像度画像を合成します。
このうち **JPEG 圧縮（`DiffJPEG`）を 1 回目・2 回目とも無効化**しました（該当行をコメントアウト）。

- ぼかし・リサイズ・ノイズ・sinc フィルタはそのまま残っています
- ファイル全体に日本語の解説コメントを追加しています

### 3. マルチスケール画像の保存形式を TIFF に変更（`scripts/generate_multiscale_DF2K.py`）

縮小画像の出力を `.png` から `.tif` に変更しました。

### 4. `setup.py` のバージョン取得を固定値に変更

`get_version()` が `'0.3.0'` を直接返すようにしました（`version.py` の読み込みで失敗するのを回避するため）。

### 5. 学習設定（`options/`）

| ファイル | 状態 | 内容 |
|---|---|---|
| `finetune_realesrnet_x4plus.yml` | 新規 | x4 ESRNet（1 段階目）のファインチューニング。70k iter、50k で学習率半減、batch 4 |
| `finetune_realesrgan_x4plus.yml` | 変更 | x4 GAN のファインチューニング（公式の `RealESRNet_x4plus.pth` から開始）。データを `dataset/` に、seed 1、batch 4、150k iter（130k で学習率半減）、1k iter ごとに保存 |
| `finetune_realesrnet_x2plus.yml` | 新規 | x2 ESRNet。公式の `RealESRGAN_x2plus.pth` から開始（ESRNet x2 の重みは非公開のため）。100k iter、検証データあり |
| `finetune_realesrgan_x2plus.yml` | 新規 | x2 GAN。上の x2 ESRNet の結果（`experiments/finetune_RealESRNetx2plus/models/net_g_latest.pth`）から開始。150k iter、検証データあり |
| `train_realesrgan_x4plus.yml` | 変更 | データを `dataset/` に変更 |
| `train_realesrgan_x2plus.yml` | 変更 | Colab の Drive 上データを使用、gt_size 128、batch 4、worker 2、133.2k iter |

### 6. 検証用の低解像度画像を作るスクリプトを追加（`scripts/generate_val_lq.py`）

`dataset/val/gt` の高解像度画像から、`dataset/val/lq` に低解像度画像を作ります。

- 劣化のかけ方は学習時と同じです（USM シャープ → ぼかし・リサイズ・ノイズを 2 回 → sinc フィルタ。JPEG 圧縮なし）。パラメータは `--opt` で指定した学習設定から読みます
- 乱数の種を固定しているので、何度実行しても同じ画像になります
- 出力のファイル名は高解像度画像と同じです（検証時に同じ名前どうしで組になるため）
- 8bit と 16bit の画像に対応しています。グレースケールは 3 チャンネルにそろえます
- 縦横が倍率で割り切れない画像はスキップします（検証時に大きさがずれるため）

```bash
# 学習と同じ劣化（倍率は設定ファイルの scale）
python scripts/generate_val_lq.py --opt options/finetune_realesrgan_x2plus.yml

# 単純な bicubic 縮小だけにしたい場合
python scripts/generate_val_lq.py --mode bicubic --scale 2
```

| オプション | 既定値 | 意味 |
|---|---|---|
| `--gt` | `dataset/val/gt` | 高解像度画像のフォルダ |
| `--lq` | `dataset/val/lq` | 出力先フォルダ |
| `--opt` | `options/finetune_realesrgan_x2plus.yml` | 劣化パラメータを読む学習設定 |
| `--scale` | 設定ファイルの `scale` | 縮小倍率 |
| `--mode` | `degrade` | `degrade`（学習と同じ劣化）か `bicubic`（縮小のみ） |
| `--seed` | `0` | 乱数の種 |

共通のデータ配置（Colab 版の `train_realesrgan_x2plus.yml` 以外）:

```
dataset/
├── (高解像度パッチ画像)
├── meta_infometa_info_imagesmultiscale.txt   # 学習画像の一覧
└── val/                                       # x2 設定のみ使用
    ├── gt/   # 検証用の高解像度画像
    └── lq/   # 検証用の低解像度画像
```

---

## 使い方（学習の流れ）

1. `createDataset.ipynb` を Colab で実行し、1024×1024 のパッチを作る
2. パッチを `dataset/` に置き、画像一覧（meta_info）を作る

   ```bash
   python scripts/generate_meta_info.py \
     --input dataset/high6 \
     --root dataset \
     --meta_info dataset/meta_infometa_info_imagesmultiscale.txt \
     --check
   ```

3. 本家の事前学習済み重みを `experiments/pretrained_models/` に置く（`RealESRNet_x4plus.pth`、`RealESRGAN_x4plus_netD.pth` など）
4. 公式の重みから GAN をファインチューニングする

   ```bash
   python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml
   ```

学習した重みは `experiments/<設定ファイルの name>/models/` に保存されます。

- 途中経過: `net_g_<iter数>.pth`（例: `net_g_150000.pth`）
- 学習が最後まで終わったとき: `net_g_latest.pth`

---

## 推論（超解像の実行）

`inference_realesrgan.py` を使います。

- 自分で学習した重みは `--model_path` で指定します
- `-n` には、学習したモデルと同じ構造の公式モデル名を指定します。ネットワークの形を決めるためで、x4 なら `RealESRGAN_x4plus`、x2 なら `RealESRGAN_x2plus` です

```bash
# x4 モデル（finetune_realesrgan_x4plus.yml で学習したもの）
python inference_realesrgan.py \
  -n RealESRGAN_x4plus \
  --model_path experiments/finetune_RealESRGANx4plus_400k/models/net_g_latest.pth \
  -i inputs \
  -o results \
  -s 4

# x2 モデル（finetune_realesrgan_x2plus.yml で学習したもの）
python inference_realesrgan.py \
  -n RealESRGAN_x2plus \
  --model_path experiments/finetune_RealESRGANx2plus/models/net_g_latest.pth \
  -i inputs \
  -o results \
  -s 2
```

| オプション | 既定値 | 意味 |
|---|---|---|
| `-i` | `inputs` | 入力画像、または画像の入ったフォルダ |
| `-o` | `results` | 出力先フォルダ |
| `-n` | `RealESRGAN_x4plus` | モデルの構造（上の説明を参照） |
| `--model_path` | なし | 学習した重みファイル。省略すると公式の重みを使う（ダウンロードされる） |
| `-s` | `4` | 最終的な拡大倍率。**x2 モデルでは `-s 2` を付ける**（付けないと 4 倍に引き伸ばされる） |
| `-t` | `0`（分割なし） | 画像を分割して処理する大きさ。GPU メモリ不足（CUDA out of memory）になったら `-t 512` などを付ける |
| `--fp32` | 付けない（fp16） | GPU がない環境（CPU）で動かすときは付ける |
| `--suffix` | `out` | 出力ファイル名の末尾（`a.tif` → `a_out.tif`）。`--suffix ""` で元と同じ名前 |
| `--ext` | `auto` | 出力形式。`auto` は入力と同じ（tif なら tif） |

補足:

- グレースケール画像や 16bit の tif もそのまま入力できます
- NAC の元画像のように大きな画像は、`-t 512` などを付けて分割処理すると安全です
- 学習途中の重み（`net_g_<iter数>.pth`）を指定して、途中のモデルの結果を見ることもできます
- 公式の重みでの結果と比べたいときは、`--model_path` を外して実行します

---

## ライセンス

本家と同じ BSD-3-Clause です（[LICENSE](LICENSE)）。
