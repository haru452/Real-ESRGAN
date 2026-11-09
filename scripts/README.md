# 評価スクリプトの使い方

超解像（SR）の結果を、クレーター・ボルダーの検出で GT と比べるためのスクリプトです。

## 流れ

1. `compare_checkpoints.py` で、重みごとに val 画像を超解像します。
2. `evaluate_checkpoints_mask.py` で、SR と GT の検出マスクの違いを数えます。
3. `evaluate_edge_sharpness.py` で、クレーターの縁のシャープさを GT と比べます。

```bash
python scripts/compare_checkpoints.py --official weights/RealESRGAN_x2plus.pth
python scripts/evaluate_checkpoints_mask.py --detector filter
python scripts/evaluate_edge_sharpness.py --detector filter
```

2 と 3 は、1 の出力（`<exp>/compare/<画像名>/<重み名>.png`）を読みます。

## フォルダの指定

何も付けなければ、既定のフォルダ（val）を使います。

| スクリプト | 引数 | 既定 | 中身 |
|---|---|---|---|
| `compare_checkpoints.py` | `--exp` | `experiments/finetune_RealESRGANx2plus1106` | 重み（`models/net_g_*.pth`）のある実験フォルダ |
| | `--lq_dir` | `dataset/val/lq` | 超解像する低解像度画像 |
| | `--gt_dir` | `dataset/val/gt` | GT（元の高解像度画像） |
| | `--output` | `<exp>/compare` | SR 画像の出力先 |
| 評価の 2 つ | `--compare_dir` | `<exp>/compare` | `compare_checkpoints.py` の `--output` と同じフォルダ。結果もここに保存されます |
| | `--gt_dir` | `dataset/val/gt` | GT |
| | `--lq_dir` | `dataset/val/lq` | 低解像度画像。`filter` のときは、ここと `--gt_dir` の xlsx を読みます |

評価の 2 つのスクリプトには、同じフォルダを指定してください。

## val・train に入っていない画像で評価する

フォルダの指定を変えるだけです。評価には **GT（元の高解像度画像）が必要**です。
以下は `dataset/test` に置く場合の例です。

1. **画像を置く**
   - 高解像度画像を `dataset/test/gt/` に入れます。名前には Product ID を入れます（例 `M1101052726LE_crop03.tif`）。
   - QuickMap の xlsx も `dataset/test/gt/`（または `dataset/test/lq/`）に入れます。

2. **低解像度画像を作る**：学習と同じ劣化をかけます。
   ```bash
   python scripts/generate_val_lq.py --gt dataset/test/gt --lq dataset/test/lq --crop
   ```
   - GT の大きさが倍率（x2 なら 2）で割り切れないと、その画像は飛ばされます。
     `--crop` を付けると、GT の右と下の端を削って割り切れる大きさにしてから使います（例 2000×1567 → 2000×1566）。
   - **`--crop` は GT のファイルを上書きします。** 元の画像を残したいときは、先にコピーしておいてください。
     TIFF の地理情報などのタグは、上書きすると消えます。

   低解像度画像がすでにあれば、GT と同じ名前で `dataset/test/lq/` に置けば、この手順は要りません。

3. **超解像する**
   ```bash
   python scripts/compare_checkpoints.py --lq_dir dataset/test/lq --gt_dir dataset/test/gt \
       --output experiments/finetune_RealESRGANx2plus1106/compare_test --num 1000 \
       --official weights/RealESRGAN_x2plus.pth
   ```
   - `--num` の既定は 5 で、5 枚だけをランダムに選びます。全部使うなら、枚数より大きい値を渡します。
   - `--output` を分けると、val の結果と混ざりません。

4. **評価する**
   ```bash
   python scripts/evaluate_checkpoints_mask.py --detector filter \
       --compare_dir experiments/finetune_RealESRGANx2plus1106/compare_test \
       --gt_dir dataset/test/gt --lq_dir dataset/test/lq
   python scripts/evaluate_edge_sharpness.py --detector filter \
       --compare_dir experiments/finetune_RealESRGANx2plus1106/compare_test \
       --gt_dir dataset/test/gt --lq_dir dataset/test/lq
   ```

## 検出器の選び方（`--detector`）

| 値 | 中身 | ファイル |
|---|---|---|
| `original`（既定） | 先行研究の検出（adaptive threshold ＋ Sobel のエッジ） | `crater_mask.py` の `OriginalCNN` |
| `filter` | 太陽の向きに合わせたフィルターでの検出（学習なし） | `lroc_filters.py`（`crater_mask.py` の `FilterDetector` から使う） |

`filter` は、光の進む向きに沿った明暗の順番で凹凸を見分けます。

- クレーター（へこみ）：暗い → 明るい
- ボルダー（出っぱり）：明るい → 影

検出結果はマスクに直してから比べます。クレーターは塗りつぶした円、ボルダーは正方形です。
`evaluate_edge_sharpness.py` では、クレーターの円周（縁）だけを使います。

### `filter` に必要なもの

- **メタデータ（QuickMap からエクスポートした xlsx）**
  - `--lq_dir` と `--gt_dir` の中の `*.xlsx` を、すべて自動で読みます。別の場所なら `--meta a.xlsx b.xlsx` で指定します。
  - 使う列は `Product ID`、`Resolution`、`Sub solar azimuth`、`Incidence Angle` です。`Sub solar azimuth` は 67 列版のエクスポートにしかありません。
- **画像名に Product ID を入れること**（例 `M1101052726LE_crop03.png`）
  - `M` ＋ 9〜10 桁 ＋ `L`/`R` の部分で xlsx と照合します。末尾の `E`/`C` は無視します。
  - 名前に ID がない画像や、xlsx に載っていない画像は、「メタデータが無いので飛ばします」と出して飛ばします。

### `filter` の引数

| 引数 | 既定 | 意味 |
|---|---|---|
| `--meta` | 自動 | 使う xlsx のパス（複数可） |
| `--geometry` | `raw` | `raw`：投影していない切り出し。`north_up`：北が上に投影した画像 |
| `--gt_px_scale` | 1.0 | GT の 1 画素が、xlsx の `Resolution` の何画素分か。GT を元の NAC から 2 倍縮小して作ったなら 2 |
| `--k_sigma` | 6.0 | 検出の閾値。応答のばらつき（MAD）の何倍以上を検出とするか。下げると検出が増える |
| `--rim_excl` | 1.6 | クレーター中心から「半径 × この値」以内のボルダー候補を消す。クレーター内の本物のボルダーも消える |

`--k_sigma` と `--rim_excl` は、合成画像で決めた暫定値です。

### 最初に確かめること

`diff_filter_<重み名>.png` を開き、強いクレーターの**暗い内壁が太陽側**にあるかを見ます。
逆なら、光の向きが 180° ずれています。`--geometry` を変えてみてください。

## 出力（`<exp>/compare/` の中）

`--detector filter` のときは、名前に `_filter` が付きます。`original` の結果は上書きしません。

| ファイル | 中身 |
|---|---|
| `mask_metrics[_filter].csv` | 画像 × 重みの数値（SR にだけある塊・GT にだけある塊の数と面積、IoU） |
| `mask_diff_curves[_filter].png` | iter ごとの線グラフ（下の段は `--baseline`（既定 official）との差） |
| `<画像名>/diff[_filter]_<重み名>.png` | 差分を重ねた画像（赤：SR にだけある、緑：GT にだけある） |
| `edge_sharpness[_filter].csv` | 縁のシャープさの比（SR / GT）の中央値など |
| `edge_sharpness_curves[_filter].png` | その線グラフ |

## 例

```bash
# 北が上に投影した画像で、閾値を下げて試す
python scripts/evaluate_checkpoints_mask.py --detector filter --geometry north_up --k_sigma 5

# xlsx を別の場所から読む
python scripts/evaluate_checkpoints_mask.py --detector filter --meta D:/lroc/meta/nac_a.xlsx D:/lroc/meta/nac_b.xlsx
```

## 既知の限界（`filter`）

- 入射角が小さい画像（例 18°）では影が短く、ボルダーの検出が弱くなります。50〜70° が望ましいです。
- 急な斜面では、斜面全体の明暗が凹凸の信号と混ざります。
- クレーターの半径は、3〜40 m のうち一番強く反応したスケールの値です。正確な直径ではありません。
- 本物の LROC 画像では、まだ検証していません。
