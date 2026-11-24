# このリポジトリにおける Real-ESRGAN 方式の劣化画像生成・学習パイプライン（まとめメモ）

## 生成・学習パイプラインの構造

- メイン実装: `EDSR/realesrgan_degradation.py`
    - basicsrなしで動作可能な、自作の2段階 degradation 実装
    - 外部からyamlでパラメータ制御 (`EDSR/degradation_lunar_mild.yml`など)
    - degrade関数 (degrade(hr, scale, opt)) で1枚ごとにHR→LR変換

### パイプライン全体 (実装コメントも参照)
1. 1段目（高解像度画像上での前処理）
    - ぼかし（iso/aniso 強調、σ=1.0~2.2, sinc型は使わない）
    - 拡大縮小は行わない（resizeはkeep固定）
    - ノイズ付加なし
    - JPEGはOFF（有効フラグのみ）

2. 2段目（scaleダウン＋ノイズ）
    - 追加ブラーはかけない（second_blur_prob: 0）
    - 1/scaleちょうどに縮小（方式はarea/bilinear/bicubicがランダム）
    - ノイズを加える（50%ガウシアン/50%ポアソン, 範囲1.5~7.0, grayのみ）
    - JPEGはOFF

3. 最終sinc/リンギングフィルタは使わない

### すべての値・適用有無・範囲はyml（例: degradation_lunar_mild.yml）で制御
- 実運用ではこれを`--deg_yml`で指定してデグレ画像を作る

## 標準Real-ESRGANとの差
- 多様・強めな劣化（ノイズ多, ダウンサンプルランダム, JPEG多用, リンギング強調）をやめ、「月面実画像に近い」弱めの劣化にしている
- 設定値は、実際の月面画像の測定値（ノイズσ=2.3~6.7, ブラーσ=1.3~1.8）に合わせて絞られている
- 色もノイズも必ずグレースケール（R=G=B）指定

### 劣化の強さの比較表（2026-11-25）

`options/finetune_realesrgan_x2plus.yml`（Real-ESRGAN の既定。ファインチューニングで使用）と `EDSR/degradation_lunar_mild.yml`（EDSR の 4 回目の学習で使用）の設定値を比べた。

| 項目 | Real-ESRGAN（既定） | EDSR（弱めの劣化） | 差の目安 |
|---|---|---|---|
| 1 回目のぼかし σ | 0.2〜3（平均 1.6） | 1.0〜2.2（平均 1.6） | 平均は同じで、最大は約 1.4 倍 |
| 2 回目のぼかし | 80% の確率で σ 0.2〜1.5 | なし | Real-ESRGAN だけ |
| ノイズの標準偏差 | 1〜30 を 2 回（平均 15.5） | 1.5〜7 を 1 回（平均 4.25） | 平均で約 3.6 倍、最大で約 4 倍、回数も 2 倍 |
| ポアソンノイズの強さ | 0.05〜3、2 回目は 0.05〜2.5 | 0.15〜0.6 | 最大で約 5 倍 |
| ランダムな拡大縮小 | 1 回目 0.15〜1.5 倍、2 回目 0.3〜1.2 倍 | なし（ちょうど 1/2 に縮小） | Real-ESRGAN だけ。0.15 倍まで縮小すると、ぼけ方がとても大きくなる |
| 最後の sinc（リンギング） | 80% の確率 | なし | Real-ESRGAN だけ |
| JPEG | なし（`realesrgan/models/realesrgan_model.py` でコメントアウト） | なし | 同じ |
| 正解画像のシャープ化（USM） | あり | なし | Real-ESRGAN は、正解をくっきりさせる方向にも学習している |

- まとめ: ノイズは最大で約 4 倍あり、それが 2 回かかる。ぼけは 1 回分の平均なら同じだが、2 回目のぼかし、大きな縮小、sinc フィルタが加わるので、全体ではかなり強い。
- 元の画像で測った値（ノイズ 2.3〜6.7、必要なぼかし σ 1.3〜1.8）を、Real-ESRGAN の既定は大きく超えている。
- 上の値は設定ファイルどおりのもの。ぼかしと縮小が重なったときの実際のぼけ方は、まだ計算していない。
- EDSR の 1〜2 回目の学習は bicubic 縮小だけで劣化を入れていないので、それと比べると差はもっと大きい。
- `options/finetune_realesrgan_x2plus_mild.yml`（2026-11-24 に作成）は、EDSR と同じ弱い劣化にそろえてある。この表はそれより前の既定の設定との比較。

## 学習スクリプト・生成スクリプト
- `EDSR/edsr_fine_tuning_realesrgan.py`, `EDSR/make_degraded_lr.py` などで、上記関数＋設定を使ってLR画像を生成＆学習

---

# 実際の推奨パラメータ例
(EDSR/degradation_lunar_mild.ymlより)
- 1段目: blur_sigma=[1.0,2.2], resize_prob=[0,0,1], gaussian_noise_prob=1, noise_range=[0,0]
- 2段目: gaussian_noise_prob2=0.5, noise_range2=[1.5,7], second_blur_prob=0, resize_prob2=[0,0,1]

---

# 備考
- 暗黙的な「絶対やらないこと」：ランダムアップ/ダウンscaling、カラーノイズ生成、JPEG artifacts混入、強いリンギングはすべてoff
- 学習再現性・論文再現のため、YAML設定を必ず添付・管理推奨

---
