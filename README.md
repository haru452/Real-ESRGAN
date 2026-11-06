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
| `finetune_realesrgan_x2plus.yml` | 新規 | x2 GAN のファインチューニング（公式の `RealESRGAN_x2plus.pth` と `RealESRGAN_x2plus_netD.pth` から開始）。150k iter（125k で学習率半減）、batch 4、検証データあり |
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

### 7. 学習ログをグラフにするスクリプトを追加（`scripts/plot_training_log.py`）

学習ログから PSNR・損失・学習率のグラフを画像で保存します。使い方は下の「学習の様子をグラフで見る」を参照。

### 8. 途中保存の重みを見比べるスクリプトを追加（`scripts/compare_checkpoints.py`）

途中保存の重みごとに val の画像を超解像して、並べた比較画像を作ります。使い方は下の「途中保存の重みを見比べる」を参照。

### 9. 重みごとの障害物検出の違いを調べるスクリプトを追加（`scripts/evaluate_checkpoints_mask.py`）

先行研究の障害物検出を SR 画像と GT 画像にかけ、マスクの違いを重みごとに線グラフにします。使い方は下の「重みごとの障害物検出の違いを線グラフにする」を参照。

---

## 使い方（学習の流れ）

フォルダ構成（Colab 版の `train_realesrgan_x2plus.yml` 以外）:

```
dataset/
├── (高解像度パッチ画像)
├── meta_infometa_info_imagesmultiscale.txt   # 学習画像の一覧
└── val/                                       # 検証用（x2 の設定ファイルだけが使う）
    ├── gt/   # 検証用の高解像度画像（自分で置く）
    └── lq/   # 検証用の低解像度画像（手順 3 で作る）
```

1. `createDataset.ipynb` を Colab で実行し、1024×1024 のパッチを作る
2. パッチを `dataset/` に置き、画像一覧（meta_info）を作る

   ```bash
   python scripts/generate_meta_info.py \
     --input dataset/high6 \
     --root dataset \
     --meta_info dataset/meta_infometa_info_imagesmultiscale.txt \
     --check
   ```

3. （検証する場合）検証用の画像を作る
   1. 学習に使っていないパッチを何枚か `dataset/val/gt/` に置く
   2. 低解像度画像を作る（`dataset/val/lq/` に同じ名前で保存される）

      ```bash
      python scripts/generate_val_lq.py --opt options/finetune_realesrgan_x2plus.yml
      ```

   - `--opt` には、これから使う学習設定を指定する（倍率と劣化の強さがそこから決まる）
   - 細かいオプションは上の「6. 検証用の低解像度画像を作るスクリプトを追加」を参照
   - x4 の設定ファイル（`finetune_realesrgan_x4plus.yml`）は検証の設定がコメントアウトされているので、そのままならこの手順は不要。検証したいときは、そのファイルの `val:` の 2 か所（`datasets` の中と末尾）のコメントを外し、`dataroot_gt` を `dataset/val/gt`、`dataroot_lq` を `dataset/val/lq` にする

4. 本家の事前学習済み重みを `experiments/pretrained_models/` に置く
   - x4: `RealESRNet_x4plus.pth`、`RealESRGAN_x4plus_netD.pth`
   - x2: `RealESRGAN_x2plus.pth`、`RealESRGAN_x2plus_netD.pth`
5. 公式の重みから GAN をファインチューニングする

   ```bash
   # x4
   python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml

   # x2
   python realesrgan/train.py -opt options/finetune_realesrgan_x2plus.yml
   ```

6. できた重みで推論する（下の「推論（超解像の実行）」を参照）

学習した重みは `experiments/<設定ファイルの name>/models/` に保存されます。

- 途中経過: `net_g_<iter数>.pth`（例: `net_g_150000.pth`）
- 学習が最後まで終わったとき: `net_g_latest.pth`

---

## 学習の様子をグラフで見る

### 方法 1: 学習ログからグラフ画像を作る（`scripts/plot_training_log.py`）

学習ログ `experiments/<name>/train_*.log` を読み、PSNR などの評価値・損失・学習率のグラフを画像で保存します。
画面のないサーバーでも使えます。学習中に実行すれば、その時点までのグラフになります。

```bash
# x2（既定）
python scripts/plot_training_log.py

# x4
python scripts/plot_training_log.py experiments/finetune_RealESRGANx4plus_400k
```

実験フォルダの中に次の 2 つができます。

- `training_curves.png`: グラフ
  - PSNR は点と線で描き、一番良い点に ★ を付けます
  - 損失は揺れが大きいので、そのままの値（薄い線）と移動平均（濃い線）を重ねます
- `val_metrics.csv`: 検証ごとの評価値（iter と PSNR）

| オプション | 既定値 | 意味 |
|---|---|---|
| （最初の引数） | `experiments/finetune_RealESRGANx2plus` | 実験フォルダ、またはログファイル |
| `-o` | 実験フォルダ | 出力先フォルダ |
| `--smooth` | `20` | 損失の移動平均の幅（ログの行数。1 行 = 100 iter）。`1` で平均しない |

途中から再開してログが複数ある場合も、まとめて 1 本の線にします。

### 方法 2: TensorBoard

設定ファイルで `use_tb_logger: true` になっているので、`tb_logger/<name>/` に記録されています。

```bash
# サーバーで
tensorboard --logdir tb_logger --port 6006

# 手元のパソコンで（つないだまま、ブラウザで http://localhost:6006 を開く）
ssh -L 6006:localhost:6006 <ユーザー名>@<サーバー>
```

### グラフを見るときの注意

- PSNR は 1 万 iter ごとにしか記録されません（`val_freq`）
- GAN の学習では、見た目が良くなっても PSNR が下がることがあります。最終的には、途中保存の重みで実際に超解像した画像を見比べて選んでください
- `l_g_gan` と `l_d_*` は上下に揺れ続けるのが普通です。どれかが 0 付近に張り付いたら、学習がうまくいっていないサインです

---

## 途中保存の重みを見比べる（`scripts/compare_checkpoints.py`）

GAN の学習では PSNR が下がっても見た目が良くなることがあるので、重みごとに同じ画像を超解像して目で比べます。

`experiments/<name>/models/net_g_*.pth` を 1 つずつ読み込み、val の LQ 画像（既定は 2 枚）を超解像します。

```bash
# x2（既定）: 全部の重みで、ランダムな 2 枚
python scripts/compare_checkpoints.py

# 重みを絞る（おすすめ）
python scripts/compare_checkpoints.py --iters 10000 50000 100000 150000

# 画像を指定し、中央 256px を拡大して見る
python scripts/compare_checkpoints.py --images 0001 0002 --crop 256

# 公式の重みも並べる
python scripts/compare_checkpoints.py --official weights/RealESRGAN_x2plus.pth
```

出力は `experiments/<name>/compare/` に保存されます。

- `compare_<画像名>.png`: LQ・各重みの出力・GT を並べた画像（タイトルに PSNR）
- `<画像名>/<重み名>.png`: 各重みの出力画像
- `psnr.csv`: 画像ごと・重みごとの PSNR（GT があるとき）

| オプション | 既定値 | 意味 |
|---|---|---|
| `--exp` | `experiments/finetune_RealESRGANx2plus` | 実験フォルダ |
| `--lq_dir` / `--gt_dir` | `dataset/val/lq` / `dataset/val/gt` | val の画像フォルダ（GT が無ければ PSNR は出さない） |
| `--images` | （ランダム） | 使う画像名（拡張子なし） |
| `--num` / `--seed` | `2` / `0` | ランダムに選ぶ枚数と乱数の種（種が同じなら同じ画像） |
| `--iters` | （全部） | 比較する iter |
| `--official` | なし | 公式の重みのパス |
| `--scale` | `2` | 倍率（x4 の実験なら `4`、`--exp` も変える） |
| `--cols` | `3` | 比較画像の列数 |
| `--crop` | `0` | 中央を切り出す大きさ（px）。`0` で全体 |
| `--tile` | `0` | タイルサイズ。GPU メモリ不足のとき `256` など |
| `--output` | `<exp>/compare` | 出力先 |

- `--iters` を省略すると 5k 刻みで全部（30 個）を処理するので、比較画像が大きくなります
- 重みは学習時の val と同じ `params_ema` を読み込みます

## 重みごとの障害物検出の違いを線グラフにする（`scripts/evaluate_checkpoints_mask.py`）

「途中保存の重みを見比べる」の出力を使い、先行研究の障害物検出（`evaluate.py`）を SR 画像と GT 画像にかけて、マスクの違いを重みごとに調べます。
検出の処理は `scripts/crater_mask.py` にまとめてあります。先に `compare_checkpoints.py` を実行しておいてください（流れは次のとおり）。

```bash
# 1. 重みごとに超解像する（compare/ に画像ができる）
python scripts/compare_checkpoints.py --iters 10000 50000 100000 150000

# 2. 障害物検出をかけて、GT のマスクとの違いを線グラフにする
python scripts/evaluate_checkpoints_mask.py
```

オプションを付ける例:

```bash
python scripts/evaluate_checkpoints_mask.py --ratio 0.5 --no-save-images
```

`experiments/<name>/compare/` に次が保存されます。

- `mask_diff_curves.png`: iter ごとの線グラフ（画像ごとに 1 本。公式の重みは横の点線）
  - 違う塊（クレーター）の数（SR のみ + GT のみ）
  - 違う塊の面積（px）
  - GT のマスクとの IoU（高いほど GT に近い）
- `mask_metrics.csv`: 画像 × 重みの数値
- `<画像名>/diff_<重み名>.png`: 差分を重ねた画像（赤: SR のみ、緑: GT のみ）

| オプション | 既定値 | 意味 |
|---|---|---|
| `--compare_dir` | `experiments/finetune_RealESRGANx2plus/compare` | `compare_checkpoints.py` の出力 |
| `--gt_dir` | `dataset/val/gt` | GT 画像のフォルダ |
| `--ratio` | `0.3` | 他方のマスクとの重なり割合がこれ以下の塊を「違う塊」とする（元のコードと同じ値） |
| `--no-save-images` | なし | 差分を重ねた画像を保存しない |

- 検出は先行研究の方法（`evaluate.py`）をそのまま使っています。「違う塊」は、他方のマスクとの重なり割合が `--ratio` 以下の塊です
- 画像が少ないと線が揺れるので、`compare_checkpoints.py` の `--num` で枚数を増やしてください

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
