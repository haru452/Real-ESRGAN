## this is only for me

# EDSR ファインチューニング

公式の EDSR の重みから始めて、自分の画像で追加学習（ファインチューニング）し、
ONNX に書き出して推論するためのフォルダです。

## ファイル

| ファイル | 役割 |
| --- | --- |
| `edsr_fine_tuning2.py` | 学習スクリプト。EDSR のモデル定義、学習データの作成、学習、ONNX への書き出し |
| `edsr_val.py` | 検証（val）の処理。val の損失・PSNR の計算、CSV とグラフの保存（学習スクリプトから呼ばれる） |
| `edsr_compare.py` | 比較画像（LR / bicubic / 学習前 / 学習後 / 正解）を作る処理（学習スクリプトから呼ばれる） |
| `infer_edsr.py` | 書き出した ONNX で超解像する推論スクリプト（onnxruntime を使用） |

## 実行の進め方

上から順に進めます。各ステップの詳しい説明は、後ろの節にあります。

### 1. 準備（最初の 1 回だけ）

- ライブラリを入れて、公式の重みをダウンロードする（→「準備」）
- 学習用の HR 画像フォルダと、val 用の HR 画像フォルダを用意する。**同じ画像を両方に入れない**
  （入れると val が甘くなり、過学習に気づけない）

### 2. 学習する

リポジトリのいちばん上のフォルダで実行します。`--log_dir` は実行ごとに別の名前にすると、結果を比べやすくなります。

```
python EDSR/edsr_fine_tuning2.py --weight_path ./EDSR/EDSR_x2.pt --hr_dir ./dataset/high6 --scale 2 --batch_size 16 --epochs 30 --lr 1e-5 --val_hr_dir ./dataset/val/gt --val_max_images 400 --log_dir ./EDSR/edsr_logs2 --output_path ./EDSR/edsr_scale2_finetuned.onnx
```

### 3. 学習の最初のログを見る（数分で出る）

```
  bicubic (基準)       psnr=xx.xxx dB  ssim=0.xxxx
  epoch    0 (学習前)  val_loss=...  val_psnr=...  val_ssim=...  bias=...
```

- **bicubic の値を控えておく。** これが「超解像しない場合」の点数です
- **epoch 0 の bias を見る。** 大きく 0 から離れていて、epoch 1 で 0 近くに戻るなら、
  最初の PSNR の急上昇は「明るさのずれが直った」だけです

### 4. 学習が終わったら、結果を見る

`--log_dir` の中を順に見ます。

1. **最後のログ** `val PSNR 最良: xx dB (bicubic より +x.xxx dB)`
   - この差が超解像の本当の効果。+0.1 dB 程度なら、ほぼ bicubic と同じです
2. **`training_curves.png`**
   - 緑の線が灰色の破線（bicubic）よりどれだけ上か
   - PSNR が途中から下がっていれば過学習
3. **`compare/*.png`**（いちばん大事）
   - 「after」が「bicubic」より、クレーターの縁や小さな岩がくっきりしているか
   - 「after」に、正解に無い模様（偽の細部）やにじみが出ていないか
   - PSNR が上がっていても、見た目がよくなっていなければ意味がありません
4. **`val_metrics.csv`** は、数字を表やグラフにまとめたいときに使う

### 5. 結果に応じて次を決める

| 見えたこと | 次にやること |
| --- | --- |
| 早期終了で数エポックで止まった・bicubic との差が小さい | 学習データを増やす / 種類を広げる。学習率を上げても（1e-4）同じか試す |
| 最後まで PSNR が上がり続けた | `--epochs` を増やす（cosine は `--epochs` に合わせて下がるので、増やしてやり直す） |
| PSNR がギザギザで落ち着かない | `--lr` を下げる、`--val_max_images` を増やす（val が少ないと揺れやすい） |
| PSNR は上がったが比較画像の見た目が変わらない | 明るさの補正だけの可能性。bias と比較画像を合わせて確認する |
| 比較画像に偽の模様が出る | 学習率を下げる / 早めのエポック（`checkpoint_epoch*.pt`）を使う |

### 6. 本番の画像で推論する

書き出された ONNX で、実際の NAC 画像を超解像します（→「推論」）。
**注意:** 学習は「HR を bicubic で縮小した LR」で行っているので、本物の低解像度画像では
val の数字ほどよくならないことがあります。本番画像の結果も必ず目で確かめてください。

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
    --weight_path ./EDSR/EDSR_x2.pt \
    --hr_dir ./dataset/high6 \
    --scale 2 \
    --patch_size 48 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-5 \
    --save_every 10 \
    --val_hr_dir ./dataset/val/gt \
    --val_max_images 200 \
    --log_dir ./EDSR/edsr_logs \
    --output_path ./EDSR/edsr_scale2_finetuned.onnx
```

コマンドを書くときの注意（エラー `the following arguments are required: --weight_path` の原因になりやすい）

- 行末の `\` の後ろに **空白を入れない**。`\` は「次の行に続く」という意味で、後ろに空白があると効かなくなる。
- 行頭に `\` を置かない。
- 引数名の前は **`--`**（例: `--val_hr_dir`）。`--` が無いと引数として読まれない。
- 綴りに注意（例: `--output_path`。`--opuput_path` ではない）。
- 迷ったら、全部を 1 行で書く。

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

**残り時間の目安:** 各エポックの表示の最後に、経過時間・残り時間・終了予定の時刻が出ます。

```
  epoch    3/100  loss=0.0123  val_loss=... val_psnr=... dB  | 経過 12分30秒  残り約 5時間 (終了予定 11/06 23:40)
```

残り時間は「ここまでの 1 エポックあたりの平均（val の時間も含む）× 残りのエポック数」です。最初の数エポックは、準備の時間が混ざるのでずれやすく、進むほど正確になります。

## val（過学習の確認）

`--val_hr_dir` を付けると、エポックごとに val の **L1 損失**・**PSNR**・**SSIM**・**bias** を計算します。
PSNR / SSIM の計算方法は Real-ESRGAN 側と同じ（RGB・0〜255・周囲を倍率分の画素だけ除く）です。

- **bicubic（基準）:** 学習の前に 1 回だけ、「bicubic で拡大しただけ」の PSNR / SSIM を計算します。
  EDSR がこれをどれだけ上回るかが、超解像の本当の効果です（グラフの灰色の破線）。
- **bias:** SR 画像の平均の明るさ − 正解の平均の明るさ（0〜255）。0 から離れていれば明るさがずれています。
  学習前（epoch 0）に bias が大きく、1 エポックで 0 近くに戻るなら、最初の PSNR の急上昇は
  「細部を覚えた」のではなく「明るさのずれが直った」ことによるものです。

| 引数 | 初期値 | 説明 |
| --- | --- | --- |
| `--val_hr_dir` | なし | val の HR 画像のフォルダ。付けると val を行う |
| `--val_lr_dir` | なし | val の LR 画像のフォルダ（同じファイル名で組にする）。省略すると学習と同じ bicubic で作る |
| `--val_every` | 1 | 何エポックごとに val を行うか |
| `--val_max_images` | 全部 | val に使う枚数。名前順に並べた中から **等間隔に** 選ぶ（先頭だけだと同じ撮影帯に偏るため）。時間短縮用 |
| `--log_dir` | `.` | 結果の保存先 |
| `--export_weights` | best | ONNX に書き出す重み。`best`: val の PSNR が最良 / `last`: 最後のエポック |
| `--early_stop` | 10 | val PSNR が `--min_delta` 以上よくならない状態がこのエポック数続いたら学習を止める。0 で無効 |
| `--min_delta` | 0.01 | 「よくなった」とみなす val PSNR の最小の上がり幅（dB） |
| `--compare_images` | 4 | 比較画像を作る val 画像の枚数。0 で作らない |
| `--compare_crop` | 128 | 比較画像で切り出す大きさ（HR 側の画素） |

`--log_dir` に保存されるもの

- `val_metrics.csv`: エポックごとの学習率・学習損失・val 損失・PSNR・SSIM・bias の表（epoch 0 は学習前の値。1 行目に bicubic の値）
- `training_curves.png`: 左から損失（学習と val）、val PSNR、val SSIM のグラフ。最良の時点に ★、bicubic は灰色の破線
- `checkpoint_best.pt`: val PSNR が最良だったときの重み
- `compare/<画像名>_best.png`: 比較画像。左から LR（そのまま拡大）・bicubic・学習前 EDSR・学習後 EDSR・正解。
  切り出す場所は、正解の中で一番細かい模様が多い所（クレーターの縁など）を自動で選び、各パネルに PSNR を書きます

**学習 L1 が val L1 より高いのは普通です:** 学習 L1 はエポック中の平均（途中の、まだ学習が進んでいない時点も含む）で、
学習パッチにはランダムな切り出し・反転・回転が入っているためです。

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

## メモ　－エポック数をどうするかについて　ー
見方の目安
- 最後まで val の PSNR が上がり続けている → 学習が足りないので、150〜250 エポックに増やします。Real-ESRGAN と同じ画素量なら約 250 エポックです。
- 途中で val の PSNR が下がり始める → 過学習なので、その手前までで十分です。
- 前回の結果（x2・lr 1e-5・100 エポック）では、1 エポック目で PSNR がほぼ上限に達し、残りはほとんど伸びませんでした。
  初期設定では早期終了（`--early_stop 10`）が入っているので、こうなった場合は自動で止まります。

## 学習率の下げ方

| 引数 | 初期値 | 説明 |
| --- | --- | --- |
| `--lr_schedule` | cosine | `cosine`: `--epochs` かけて `--lr_min` まで滑らかに下げる / `step`: `--lr_step` エポックごとに `--lr_gamma` 倍 / `none`: 固定（以前の動き） |
| `--lr_min` | 1e-7 | cosine の最小値 |
| `--lr_step` | 10 | step の間隔（エポック） |
| `--lr_gamma` | 0.5 | step の倍率 |

cosine は `--epochs` を基準に下げるので、早期終了を使うなら `--epochs` は 20〜30 くらいに短めにするのがおすすめです。

## 注意

- 学習の LR は OpenCV の bicubic 縮小で作っています。公式 EDSR の学習データ（MATLAB の縮小）とは少し違います。
  評価用に LR を作るときも、同じ OpenCV の方法で縮小してください。
- `.onnx` の重みから学習を始めた場合、保存される `.pt` は公式 EDSR の形式にはなりません
  （このスクリプトで続きを学習するときは ONNX を使ってください）。
