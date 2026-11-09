# -*- coding: utf-8 -*-
"""
LROC NAC画像のクレーター/ボルダー検出(フィルターのみ・学習なし)
画像ファイル名に含まれる Product ID (例: M1101052726LE) から、
QuickMapでエクスポートしたxlsxのメタデータを自動で引いてくる。

使い方:
    meta_table = load_meta_table(glob.glob('./meta/*.xlsx'))
    meta = meta_for('./input_ground_truth/M1101052726LE_crop03.tif', meta_table)
    dets = detect(img, meta)

このリポジトリでは crater_mask.FilterDetector から使う(検出結果をマスクに直す)。
"""
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. メタデータ
# ---------------------------------------------------------------------------
# M + 9〜10桁 + L/R (+ E/C)。LE と LC は同じ観測なので L/R までで照合する
ID_RE = re.compile(r'(M\d{9,10}[LR])[EC]?', re.IGNORECASE)


def load_meta_table(xlsx_paths):
    """複数のxlsx(全シート)を読み込んで1つの表にする。キーは 'M1101052726L' の形。"""
    if isinstance(xlsx_paths, str):
        xlsx_paths = [xlsx_paths]
    frames = []
    for p in xlsx_paths:
        for _, df in pd.read_excel(p, sheet_name=None).items():
            if 'Product ID' in df.columns:
                frames.append(df)
    if not frames:
        raise ValueError('Product ID 列を持つシートが見つかりません')
    table = pd.concat(frames, ignore_index=True)
    table['_key'] = table['Product ID'].astype(str).str.upper().str.extract(ID_RE, expand=False)
    table = table.drop_duplicates('_key', keep='last').set_index('_key')
    return table


def product_key(path):
    m = ID_RE.search(os.path.basename(path).upper())
    return m.group(1) if m else None


def meta_for(path, table, fallback_path=None):
    """画像パスからメタデータ(dict)を返す。
    SR画像など名前にIDが無いときは fallback_path(対応するGT画像のパス)で引く。"""
    key = product_key(path) or (product_key(fallback_path) if fallback_path else None)
    if key is None:
        raise KeyError(f'ファイル名に Product ID がありません: {path}')
    if key not in table.index:
        raise KeyError(f'{key} がxlsxにありません ({path})。QuickMapから追加してください')
    return table.loc[key].to_dict()


# ---------------------------------------------------------------------------
# 2. 太陽の向き
# ---------------------------------------------------------------------------
def light_vector(meta, geometry='raw'):
    """画像上で『光が進む向き』の単位ベクトル (x右, y下)。

    geometry='raw'      : EDR/CDRをそのまま切り出した画像(地図投影していない)。
                          Sub solar azimuth は画像の3時方向から時計回りの角度(PDS/ISISの定義)。
    geometry='north_up' : 地図投影して北が上の画像。太陽直下点の方位から計算する。
    """
    if geometry == 'raw':
        az_sun = float(meta['Sub solar azimuth'])            # 画像上で太陽がある方向
    elif geometry == 'north_up':
        lat1, lon1 = np.deg2rad([meta['Center Latitude'], meta['Center Longitude']])
        lat2, lon2 = np.deg2rad([meta['Sub solar latitude'], meta['Sub solar longitude']])
        dlon = lon2 - lon1
        bearing = np.rad2deg(np.arctan2(np.sin(dlon) * np.cos(lat2),
                                        np.cos(lat1) * np.sin(lat2)
                                        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)))
        az_sun = bearing - 90.0                                # 北(上)基準 → 3時基準
    else:
        raise ValueError(geometry)
    a = np.deg2rad(az_sun + 180.0)                             # 光は太陽と反対向きに進む
    return np.array([np.cos(a), np.sin(a)], np.float32)


# ---------------------------------------------------------------------------
# 3. フィルター
# ---------------------------------------------------------------------------
def to_ratio(img, px_m, bg_m=60.0):
    """背景(半径~bg_m)で割った相対明るさ。露光・アルベド・cos(i) の違いを打ち消す。"""
    f = img.astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=bg_m / px_m[0], sigmaY=bg_m / px_m[1])
    return f / bg - 1.0


def lobe_kernel(R_m, u, px_m):
    """光の下流側に+、上流側に−のガウスローブ。メートル単位で作り、画素の縦横比も反映。"""
    sx, sy = px_m
    s = max(R_m / 2.0, 0.6 * min(sx, sy))
    hx = int(np.ceil((3 * s + R_m / 2) / sx)) + 1
    hy = int(np.ceil((3 * s + R_m / 2) / sy)) + 1
    y, x = np.mgrid[-hy:hy + 1, -hx:hx + 1].astype(np.float32)
    x *= sx
    y *= sy
    d = R_m / 2.0

    def g(cx, cy):
        return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))

    k = g(d * u[0], d * u[1]) - g(-d * u[0], -d * u[1])
    return k / np.abs(k).sum()


def _peaks(resp, thr, win_px):
    win = max(int(win_px) | 1, 3)
    m = cv2.dilate(resp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (win, win)))
    ys, xs = np.where((resp >= m) & (resp > thr))
    return xs, ys, resp[ys, xs]


def _nms(dets, px):
    """複数スケールで同じ物体が出るのを、強い順に残して周囲を消す。"""
    dets = sorted(dets, key=lambda d: -d['score'])
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            r_px = max(d['R_m'], k['R_m']) / px
            if (d['x'] - k['x']) ** 2 + (d['y'] - k['y']) ** 2 < r_px ** 2:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept


def detect(img, meta, scale=1.0, geometry='raw',
           crater_R_m=(3, 5, 8, 12, 18, 27, 40),
           boulder_R_m=(1.0, 1.5, 2.5, 4.0),
           k_sigma=6.0, square_pixels=True, rim_excl=1.6):
    """
    img   : グレースケール uint8
    meta  : meta_for() の結果
    scale : この画像の1画素がxlsxの何画素分か。GTそのまま=1、
            4倍縮小した入力=4、それを4倍SRした画像=1。
            (ground truth と縦横サイズを比べて gt_w / img_w を渡せばよい)
    k_sigma: 閾値。応答のばらつき(MAD)の何倍以上を検出とするか。画像ごとに自動で決まる。
    戻り値: [{'type','x','y','R_m','score'}, ...]
    """
    res = float(meta['Resolution']) * scale
    if square_pixels:
        px_m = (res, res)
    else:  # 生のEDR/CDRは縦横で画素サイズが違う(例 1.06 x 0.78 m)
        px_m = (float(meta['Scaled Pixel Width']) * scale,
                float(meta['Scaled Pixel Height']) * scale)
    px = float(np.mean(px_m))
    u = light_vector(meta, geometry)
    ratio = to_ratio(img, px_m)

    def run(radii, sign, kind):
        out = []
        for R in radii:
            if R / px < 1.0:          # 1画素未満の大きさは見えないので飛ばす
                continue
            r = sign * cv2.filter2D(ratio, cv2.CV_32F, lobe_kernel(R, u, px_m),
                                    borderType=cv2.BORDER_REFLECT)
            mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
            xs, ys, sc = _peaks(r, k_sigma * mad, 2 * R / px)
            out += [dict(type=kind, x=int(x), y=int(y), R_m=float(R), score=float(s / mad))
                    for x, y, s in zip(xs, ys, sc)]
        return _nms(out, px)

    # クレーター: 光の進む向きに 暗い内壁 → 明るい内壁
    # ボルダー  : 光の進む向きに 明るい面 → 影   (符号が逆)
    craters = run(crater_R_m, +1, 'crater')
    boulders = run(boulder_R_m, -1, 'boulder')

    # (a) ボルダーの明暗の両脇には小さな「クレーターもどき」が出るので、
    #     近くにもっと強いボルダーがあるクレーター候補は消す
    def near(a, b, f):
        return (a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2 < (f * (a['R_m'] + b['R_m']) / px + 2) ** 2
    craters = [c for c in craters
               if not any(near(c, b, 1.2) and b['score'] > 0.8 * c['score'] for b in boulders)]
    # (b) クレーターの内壁・縁の明暗をボルダーと誤認したものを消す
    #     (クレーター内の本物のボルダーも消えるので、岩塊の多いクレーターを調べるなら rim_excl を小さく)
    boulders = [b for b in boulders
                if not any(c['R_m'] >= 2 * b['R_m'] and
                           (b['x'] - c['x']) ** 2 + (b['y'] - c['y']) ** 2 < (rim_excl * c['R_m'] / px) ** 2
                           for c in craters)]
    return craters + boulders


def shadow_height_m(shadow_len_m, meta):
    """影の長さ → ボルダーの高さ  h = L / tan(i)"""
    return shadow_len_m / np.tan(np.deg2rad(float(meta['Incidence Angle'])))


# ---------------------------------------------------------------------------
# 4. SR と GT の比較
# ---------------------------------------------------------------------------
def match(dets_sr, dets_gt, px, tol=0.5):
    """同じ種類どうしを、中心距離 < tol*半径 で対応付ける。
    戻り値: (SRにだけある=ハルシネーション, GTにだけある=見落とし, 一致したペア数)"""
    hallu, missed, n_match = [], [], 0
    for kind in ('crater', 'boulder'):
        A = [d for d in dets_sr if d['type'] == kind]
        B = [d for d in dets_gt if d['type'] == kind]
        usedB = set()
        for a in sorted(A, key=lambda d: -d['score']):
            best, bd = None, None
            for j, b in enumerate(B):
                if j in usedB:
                    continue
                lim = tol * max(a['R_m'], b['R_m']) / px + 1.5
                dist = np.hypot(a['x'] - b['x'], a['y'] - b['y'])
                if dist < lim and (bd is None or dist < bd):
                    best, bd = j, dist
            if best is None:
                hallu.append(a)
            else:
                usedB.add(best)
                n_match += 1
        missed += [b for j, b in enumerate(B) if j not in usedB]
    return hallu, missed, n_match


def draw(img, dets, px, color, thickness=1):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for d in dets:
        r = max(int(round(d['R_m'] / px)), 2)
        if d['type'] == 'crater':
            cv2.circle(out, (d['x'], d['y']), r, color, thickness)
        else:
            cv2.rectangle(out, (d['x'] - r, d['y'] - r), (d['x'] + r, d['y'] + r), color, thickness)
    return out


# ---------------------------------------------------------------------------
# 5. 実行例(元のフォルダ構成のまま。このリポジトリでは評価スクリプトの --detector filter を使う)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    META_XLSX = sorted(glob.glob('./meta/*.xlsx'))      # QuickMapのxlsxを全部ここに入れる
    GEOMETRY = 'raw'                                    # 北が上に投影した画像なら 'north_up'
    SR_DIRS = {
        'gan5000': './result/Real-ESRGAN5000',
        'gan3000': './result/Real-ESRGAN3000',
        'net5000': './result/Real-ESRNet5000',
        'edsr':    './result/edsr',
    }
    OUT = './result/mapping_filter'
    os.makedirs(OUT, exist_ok=True)

    table = load_meta_table(META_XLSX)
    rows = []
    for p_gt in sorted(glob.glob('./input_ground_truth/*.tif')):
        name = os.path.basename(p_gt)
        try:
            meta = meta_for(p_gt, table)
        except KeyError as e:
            print('スキップ:', e)
            continue
        img_gt = cv2.imread(p_gt, cv2.IMREAD_GRAYSCALE)
        dets_gt = detect(img_gt, meta, scale=1.0, geometry=GEOMETRY)
        px = float(meta['Resolution'])

        for model, d in SR_DIRS.items():
            p_sr = os.path.join(d, name)                # GTと同じファイル名を想定
            if not os.path.exists(p_sr):
                continue
            img_sr = cv2.imread(p_sr, cv2.IMREAD_GRAYSCALE)
            if img_sr.shape != img_gt.shape:
                img_sr = cv2.resize(img_sr, img_gt.shape[::-1], interpolation=cv2.INTER_CUBIC)
            dets_sr = detect(img_sr, meta, scale=1.0, geometry=GEOMETRY)
            hallu, missed, n = match(dets_sr, dets_gt, px)

            vis = draw(img_sr, hallu, px, (0, 0, 255))       # 赤: SRにだけある
            vis = draw(vis, missed, px, (0, 255, 0))         # 緑: GTにだけある
            cv2.imwrite(os.path.join(OUT, f'{model}_{name}'), vis)

            for kind in ('crater', 'boulder'):
                ng = sum(x['type'] == kind for x in dets_gt)
                ns = sum(x['type'] == kind for x in dets_sr)
                nh = sum(x['type'] == kind for x in hallu)
                nm = sum(x['type'] == kind for x in missed)
                rows.append(dict(image=name, model=model, kind=kind,
                                 product_id=meta['Product ID'],
                                 incidence=meta['Incidence Angle'],
                                 n_gt=ng, n_sr=ns, hallucinated=nh, missed=nm,
                                 precision=(ns - nh) / ns if ns else np.nan,
                                 recall=(ng - nm) / ng if ng else np.nan))
        print('done', name)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'summary.csv'), index=False, encoding='utf-8-sig')
