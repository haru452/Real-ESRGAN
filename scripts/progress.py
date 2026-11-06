"""長い処理の「進み具合と残り時間」を表示する小さな道具。"""
import time


def format_seconds(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{h}時間{m:02d}分' if h else f'{m}分{s:02d}秒'


class Progress:
    """全体で total 個の作業のうち、何個終わったかと残り時間の目安を出す。

    残り時間 = ここまでの 1 個あたりの平均 × 残りの個数。
    """

    def __init__(self, total, name=''):
        self.total, self.name, self.done = total, name, 0
        self.start = time.time()

    def step(self, message=''):
        self.done += 1
        elapsed = time.time() - self.start
        remain = elapsed / self.done * (self.total - self.done)
        finish = time.strftime('%H:%M', time.localtime(time.time() + remain))
        print(f'[{self.name} {self.done}/{self.total}] {message}'
              f'  経過 {format_seconds(elapsed)} / 残り約 {format_seconds(remain)} (終了予定 {finish})', flush=True)
