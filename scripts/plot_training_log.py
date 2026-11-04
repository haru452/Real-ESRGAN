"""学習ログ（experiments/<name>/train_*.log）から、PSNR などの評価値と損失をグラフにするスクリプト。

TensorBoard が使えない環境（画面のないサーバーなど）向け。学習中に実行しても、その時点までの分を描く。
途中から再開して train_*.log が複数ある場合は、まとめて 1 本の線にする（同じ iter は新しいログを優先）。

使い方:
    python scripts/plot_training_log.py                                         # x2 の実験フォルダを読む
    python scripts/plot_training_log.py experiments/finetune_RealESRGANx4plus_400k
    python scripts/plot_training_log.py experiments/xxx/train_xxx.log --smooth 50

出力（既定では実験フォルダの中）:
    training_curves.png   評価値・損失・学習率のグラフ
    val_metrics.csv       検証ごとの評価値（iter, psnr, ...）
"""
import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')  # 画面がなくても画像として保存できるようにする
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# 学習の行: [finet..][epoch:  3, iter:   9,900, lr:(1.000e-04,)] [eta: ...] l_g_pix: 1.2e-02 l_g_gan: ...
TRAIN_RE = re.compile(r'\[epoch:\s*(\d+), iter:\s*([\d,]+), lr:\(([^)]*)\)\]')
VALUE_RE = re.compile(r'\b(l_\w+|out_d_\w+): ([-+\d.eE]+)')
# 検証の行: "Validation validation" の後に "\t # psnr: 28.1234\tBest: ..." が続く
VAL_HEAD_RE = re.compile(r'Validation (\S+)')
METRIC_RE = re.compile(r'#\s*(\w+): ([-+\d.eE]+)')


def find_logs(path):
    if os.path.isfile(path):
        return [path]
    logs = sorted(glob.glob(os.path.join(path, 'train_*.log')))  # ファイル名の日時順 = 古い順
    if not logs:
        raise SystemExit(f'学習ログ（train_*.log）が見つかりません: {path}')
    return logs


def parse_logs(paths):
    """ログを読み、(損失 {iter: {名前: 値}}, 学習率 {iter: lr}, 評価値 {(データ名, 指標): {iter: 値}}) を返す。"""
    losses, lrs, metrics = {}, {}, {}
    for path in paths:
        last_iter = None
        val_name = None
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                m = TRAIN_RE.search(line)
                if m:
                    last_iter = int(m.group(2).replace(',', ''))
                    lrs[last_iter] = float(m.group(3).split(',')[0])
                    losses[last_iter] = {k: float(v) for k, v in VALUE_RE.findall(line)}
                    val_name = None
                    continue
                m = VAL_HEAD_RE.search(line)
                if m:
                    val_name = m.group(1)
                    continue
                # 検証は、同じ iter の学習の行を書いた直後に行われるので、直前の iter を使う
                m = METRIC_RE.search(line)
                if m and val_name and last_iter is not None:
                    metrics.setdefault((val_name, m.group(1)), {})[last_iter] = float(m.group(2))
    return losses, lrs, metrics


def moving_average(y, window):
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    head = [np.mean(y[:i + 1]) for i in range(window - 1)]  # 最初の部分は、ある分だけで平均する
    return np.concatenate([head, np.convolve(y, kernel, mode='valid')])


def plot(losses, lrs, metrics, out_png, smooth, title):
    # 生成器の損失 → 識別器の損失 → 識別器の出力 の順に並べる
    order = {'l_g': 0, 'l_d': 1, 'out': 2}
    loss_names = sorted({k for v in losses.values() for k in v}, key=lambda k: (order.get(k[:3], 3), k))
    panels = len(metrics) + len(loss_names) + 1  # +1 は学習率
    cols = 3
    rows = (panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    axes = axes.ravel()
    i = 0

    # 評価値（PSNR など）: 点と線で描き、最高値に印を付ける
    for (val_name, metric), data in sorted(metrics.items()):
        x = np.array(sorted(data))
        y = np.array([data[k] for k in x])
        ax = axes[i]
        ax.plot(x, y, 'o-', color='#d1495b')
        best = int(np.argmin(y)) if metric.lower() in ('niqe', 'lpips') else int(np.argmax(y))
        ax.plot(x[best], y[best], '*', color='gold', markersize=16, markeredgecolor='black')
        ax.set_title(f'{metric} ({val_name})  best {y[best]:.4f} @ {x[best]:,} iter')
        i += 1

    # 損失: 揺れが大きいので、薄い線（そのまま）と濃い線（移動平均）を重ねる
    iters = np.array(sorted(losses))
    for name in loss_names:
        x = np.array([k for k in iters if name in losses[k]])
        y = np.array([losses[k][name] for k in x])
        ax = axes[i]
        ax.plot(x, y, color='#3b6ea5', alpha=0.25, linewidth=0.8)
        ax.plot(x, moving_average(y, smooth), color='#3b6ea5', linewidth=1.6)
        ax.set_title(name)
        i += 1

    ax = axes[i]
    ax.plot(iters, [lrs[k] for k in iters], color='#555555')
    ax.set_title('learning rate')
    ax.set_yscale('log')
    i += 1

    for ax in axes[:i]:
        ax.set_xlabel('iter')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))
        ax.grid(alpha=0.3)
    for ax in axes[i:]:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_csv(metrics, out_csv):
    keys = sorted(metrics)
    all_iters = sorted({it for data in metrics.values() for it in data})
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['iter'] + [f'{name}/{metric}' for name, metric in keys])
        for it in all_iters:
            w.writerow([it] + [metrics[k].get(it, '') for k in keys])


def main():
    parser = argparse.ArgumentParser(description='学習ログから評価値と損失のグラフを作る')
    parser.add_argument('path', nargs='?', default='experiments/finetune_RealESRGANx2plus',
                        help='実験フォルダ（中の train_*.log を全部読む）か、ログファイル')
    parser.add_argument('-o', '--out', default=None, help='出力先フォルダ（省略時は実験フォルダ）')
    parser.add_argument('--smooth', type=int, default=20, help='損失の移動平均の幅（ログの行数）。1 で平均しない')
    args = parser.parse_args()

    logs = find_logs(args.path)
    losses, lrs, metrics = parse_logs(logs)
    if not losses:
        raise SystemExit('学習の行が見つかりません。学習が始まって print_freq（100 iter）経つまで待ってください。')

    out_dir = args.out or (args.path if os.path.isdir(args.path) else os.path.dirname(args.path) or '.')
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'training_curves.png')
    title = os.path.basename(os.path.normpath(out_dir if os.path.isdir(args.path) else os.path.dirname(logs[0])))
    plot(losses, lrs, metrics, out_png, args.smooth, title)

    print(f'読んだログ: {len(logs)} 個, 学習の記録: {len(losses)} 点 (最後 {max(losses):,} iter)')
    print(f'グラフ: {out_png}')
    if metrics:
        out_csv = os.path.join(out_dir, 'val_metrics.csv')
        write_csv(metrics, out_csv)
        print(f'評価値: {out_csv}')
        for (name, metric), data in sorted(metrics.items()):
            last = max(data)
            print(f'  {metric} ({name}): 最新 {data[last]:.4f} @ {last:,} iter / 検証 {len(data)} 回')
    else:
        print('評価値（PSNR など）はまだありません。最初の検証は val_freq（10,000 iter）の後です。')


if __name__ == '__main__':
    main()
