"""先行研究の障害物（クレーター・ボルダー）検出と、2 つのマスクの差分を調べる関数。

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
