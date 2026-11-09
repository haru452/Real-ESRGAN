"""障害物（クレーター・ボルダー）検出と、2 つのマスクの差分を調べる関数。

検出器は 2 つ: OriginalCNN（先行研究）と FilterDetector（lroc_filters.py。太陽の向きを使う）。
評価スクリプトでは --detector で選ぶ。

もとは Downloads/evaluate.py（Colab 版）。import しただけでは何も実行されないように、関数だけを取り出した。
検出の処理（original_cnn）は元のまま。差分の抽出は、結果が同じになるように速くしてある。
"""
import cv2
import numpy as np


class OriginalCNN:
    """先行研究の障害物検出（元の名前は original_cnn）。グレースケール画像からマスクを作る。"""

    def conv1(self, img):
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        _, sobel_mask = cv2.threshold(magnitude, 80.0, 255, cv2.THRESH_BINARY)
        sobel_mask = sobel_mask.astype(np.uint8)
        img_and = cv2.bitwise_and(th3, sobel_mask)
        return cv2.GaussianBlur(img_and, (5, 5), 0)

    def conv2(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def forward(self, img):
        return self.conv2(self.conv1(img))

    def mask(self, img, stem=None, gt_shape=None, outline=False):
        """FilterDetector と同じ呼び方にそろえるためのもの。stem などは使わない。"""
        return self.forward(img)


class FilterDetector:
    """太陽の向きに合わせたフィルターで検出し（lroc_filters.detect）、結果をマスクに直す。

    メタデータ（QuickMap の xlsx）は、画像名に含まれる Product ID で引く。
    gt_px_scale: GT の 1 画素が、xlsx の Resolution の何画素分か（GT が元の NAC と同じ解像度なら 1）。
    """

    def __init__(self, xlsx_paths, geometry='raw', gt_px_scale=1.0, k_sigma=6.0, rim_excl=1.6):
        from lroc_filters import load_meta_table  # pandas が要るので、使うときだけ読み込む
        if not xlsx_paths:
            raise SystemExit('メタデータの xlsx が見つかりません（--meta で指定してください）')
        self.table = load_meta_table(xlsx_paths)
        self.geometry, self.gt_px_scale = geometry, gt_px_scale
        self.k_sigma, self.rim_excl = k_sigma, rim_excl
        self.last_dets = []

    def detect(self, img, stem, gt_shape=None):
        from lroc_filters import detect, meta_for
        meta = meta_for(stem, self.table)  # 見つからなければ KeyError
        scale = self.gt_px_scale * (gt_shape[1] / img.shape[1] if gt_shape else 1.0)
        dets = detect(img, meta, scale=scale, geometry=self.geometry,
                      k_sigma=self.k_sigma, rim_excl=self.rim_excl)
        return dets, float(meta['Resolution']) * scale

    def mask(self, img, stem, gt_shape=None, outline=False):
        """クレーターは円、ボルダーは正方形を塗ったマスク（0/255）。

        outline=True のときは、クレーターを縁（太さ 2 px の円周）だけにする。縁のシャープさを測るため。
        """
        dets, px = self.detect(img, stem, gt_shape)
        self.last_dets = dets
        m = np.zeros(img.shape[:2], np.uint8)
        for d in dets:
            r = max(int(round(d['R_m'] / px)), 1)
            if d['type'] == 'crater':
                cv2.circle(m, (d['x'], d['y']), r, 255, 2 if outline else -1)
            else:
                cv2.rectangle(m, (d['x'] - r, d['y'] - r), (d['x'] + r, d['y'] + r), 255, -1)
        return m


def add_detector_args(p):
    """評価スクリプト共通の、検出器を選ぶ引数。"""
    p.add_argument('--detector', choices=['original', 'filter'], default='original',
                   help='original: 先行研究（adaptive threshold + Sobel）, filter: 太陽の向きに合わせたフィルター（lroc_filters.py）')
    p.add_argument('--meta', nargs='*', default=None,
                   help='filter で使う QuickMap の xlsx。省略時は --lq_dir と --gt_dir の中の *.xlsx を全部使う')
    p.add_argument('--geometry', choices=['raw', 'north_up'], default='raw',
                   help='filter: raw = 投影していない切り出し, north_up = 北が上に投影した画像')
    p.add_argument('--gt_px_scale', type=float, default=1.0,
                   help='filter: GT の 1 画素が xlsx の Resolution の何画素分か（GT が元の解像度なら 1）')
    p.add_argument('--k_sigma', type=float, default=6.0, help='filter: 検出の閾値（MAD の何倍）')
    p.add_argument('--rim_excl', type=float, default=1.6, help='filter: クレーター内のボルダー候補を消す範囲（半径の何倍）')


def build_detector(args):
    if args.detector == 'original':
        return OriginalCNN()
    import glob
    import os
    paths = args.meta
    if not paths:
        dirs = [getattr(args, 'lq_dir', None), getattr(args, 'gt_dir', None)]
        paths = sorted({p for d in dirs if d for p in glob.glob(os.path.join(d, '*.xlsx'))})
    print('メタデータ:', paths)
    return FilterDetector(paths, args.geometry, args.gt_px_scale, args.k_sigma, args.rim_excl)


def extract_unique_craters_by_ratio(img_a, img_b, ratio_thresh=0.3):
    """img_a の各クレーターのうち、img_b との重なり割合が ratio_thresh 以下のものだけを残したマスクを返す。

    元の関数と同じ結果（ラベルごとにループせず、まとめて数えて速くしてある）。
    """
    num_labels, labels = cv2.connectedComponents(img_a)
    if num_labels <= 1:
        return np.zeros_like(img_a)
    area = np.bincount(labels.ravel(), minlength=num_labels)
    overlap = np.bincount(labels[img_b > 0], minlength=num_labels)
    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = overlap[1:] <= ratio_thresh * area[1:]  # 0 番は背景
    return (keep[labels] * 255).astype(np.uint8)


def overlay_mask_on_image(img, mask, color=(0, 0, 255), alpha=0.15):
    """画像に、マスクの部分だけ指定色（BGR）を薄く重ねたカラー画像を返す。"""
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    overlay = color_img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, color_img, 1 - alpha, 0)


def count_components(mask):
    """マスクに含まれる塊（クレーター）の数。"""
    return cv2.connectedComponents(mask)[0] - 1


def compare_masks(sr_mask, gt_mask, ratio_thresh=0.3):
    """SR のマスクと GT のマスクの違いを、数値と画像の材料にして返す。

    sr_only: SR にだけある塊（GT と重なりが少ない）。= 元のコードの赤
    gt_only: GT にだけある塊（SR と重なりが少ない）。= 元のコードの緑
    """
    sr_only = extract_unique_craters_by_ratio(sr_mask, gt_mask, ratio_thresh)
    gt_only = extract_unique_craters_by_ratio(gt_mask, sr_mask, ratio_thresh)
    a, b = sr_mask > 0, gt_mask > 0
    union = np.count_nonzero(a | b)
    return {
        'sr_only_count': count_components(sr_only),
        'gt_only_count': count_components(gt_only),
        'sr_only_px': int(np.count_nonzero(sr_only)),
        'gt_only_px': int(np.count_nonzero(gt_only)),
        'mask_iou': float(np.count_nonzero(a & b) / union) if union else 1.0,
        'sr_only_mask': sr_only,
        'gt_only_mask': gt_only,
    }
