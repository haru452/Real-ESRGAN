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

推論は本家と同じく `inference_realesrgan.py` を使います（[README_original.md](README_original.md) 参照）。

---

## ライセンス

本家と同じ BSD-3-Clause です（[LICENSE](LICENSE)）。
