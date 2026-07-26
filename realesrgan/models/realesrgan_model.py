# NumPyライブラリを数値計算（ランダム数値の生成など）のためにインポート
import numpy as np
# Python標準のランダム処理ライブラリ（確率的な分岐選択など）をインポート
import random
# PyTorchライブラリ（テンソル操作やディープラーニング機能）をインポート
import torch
# BasicSRからガウスノイズおよびポアソンノイズをPyTorchテンソルに追加する関数をインポート
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
# 高解像度（GT）と低解像度（LQ）の画像ペアを同期してランダムクロップする関数をインポート
from basicsr.data.transforms import paired_random_crop
# ベースとなるGANモデル（SRGANModel）をインポートし、これを継承して拡張する
from basicsr.models.srgan_model import SRGANModel
# 差別化可能なJPEG圧縮モジュールと、USM（アンシャープマスキング）研磨モジュールをインポート
from basicsr.utils import DiffJPEG, USMSharp
# 2次元フィルタ（畳み込み処理によるボカシなど）を適用する汎用関数をインポート
from basicsr.utils.img_process_util import filter2D
# BasicSRのモデルレジストリ（モデル名を登録して文字列から動的呼び出し可能にする仕組み）をインポート
from basicsr.utils.registry import MODEL_REGISTRY
# 挿入順序を保持する辞書型データ構造（損失の記録などに使用）をインポート
from collections import OrderedDict
# PyTorchのニューラルネットワーク関数群（補間関数 F.interpolate など）をインポート
from torch.nn import functional as F


# MODEL_REGISTRYにRealESRGANModelクラスを登録するデコレータ
@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    # 初期化メソッド。設定辞書（opt）を受け取ってモデルをセットアップする
    def __init__(self, opt):
        # 親クラスであるSRGANModelの初期化処理（ネットワーク構造や最適化手法の定義など）を呼び出す
        super(RealESRGANModel, self).__init__(opt)
        # JPEG圧縮の劣化を疑似再現するDiffJPEGインスタンスを作成しGPUに転送（勾配計算は不要なためdifferentiable=False）
        #self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        # 高解像度（GT）画像の輪郭をくっきりさせるアンシャープマスキングの処理モジュールをGPUに転送
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        # 劣化の多様性を生み出すための「トレーニングペアプール（キュー）」のサイズを取得（デフォルトは180）
        self.queue_size = opt.get('queue_size', 180)

    # 勾配（グラディエント）追跡をオフにして、メモリ消費と計算コストを抑えるデコレータ
    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # 初回実行時：低解像度（lq）テンソルからバッチサイズ(b)、チャンネル(c)、高さ(h)、幅(w)を取得
        b, c, h, w = self.lq.size()
        # キュー用のテンソル（queue_lr）がまだ定義されていないかチェック
        if not hasattr(self, 'queue_lr'):
            # キューのサイズがバッチサイズで割り切れることを確認（割り切れない場合はエラーを出す）
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            # 低解像度画像用のキューバッファをゼロで初期化してGPU上に保持
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            # 高解像度（gt）画像のチャンネル、高さ、幅を取得
            _, c, h, w = self.gt.size()
            # 高解像度画像用のキューバッファをゼロで初期化してGPU上に保持
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            # キューの現在位置を指すポインタを0に初期化
            self.queue_ptr = 0
        # キューにデータが満たされた（ポインタがキューサイズに達した）場合の処理
        if self.queue_ptr == self.queue_size:  # the pool is full
            # デキュー（取り出し）とエンキュー（追加）を同時に行う
            # キュー内のデータをランダムに並び替えるためのインデックス（シャッフル）を作成
            idx = torch.randperm(self.queue_size)
            # 低解像度キューの順序をシャッフル
            self.queue_lr = self.queue_lr[idx]
            # 高解像度キューの順序をシャッフル
            self.queue_gt = self.queue_gt[idx]
            # シャッフルされたキューの先頭からバッチサイズ(b)分の画像を取り出す（クローンして参照を切る）
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # 取り出した先頭の位置に、今回新しく合成した画像(lq, gt)を入れてキューを更新
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            # 今回の学習ステップで実際に使用するlqとgtを、キューから取り出した画像に置き換える
            self.lq = lq_dequeue
            self.gt = gt_dequeue
        # キューがまだ一杯になっていない場合の処理
        else:
            # キューの空き位置（queue_ptr）に現在のバッチ(lq, gt)を追加するだけ
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            # キューのポインタをバッチサイズ分だけ進める
            self.queue_ptr = self.queue_ptr + b

    # 勾配追跡をオフにして、学習データから低解像度劣化画像を生成する
    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # 学習モードかつ「高次劣化（2段階の劣化処理）」が有効になっている場合の処理
        if self.is_train and self.opt.get('high_order_degradation', True):
            # データローダーから高解像度（GT）画像を取得し、対象デバイス（GPU）へ送る
            self.gt = data['gt'].to(self.device)
            # GT画像に対してアンシャープマスキングを適用し、輪郭を際立たせた画像を作成
            self.gt_usm = self.usm_sharpener(self.gt)

            # データローダーから1段階目用のボケカーネルを取得してGPUへ送る
            self.kernel1 = data['kernel1'].to(self.device)
            # データローダーから2段階目用のボケカーネルを取得してGPUへ送る
            self.kernel2 = data['kernel2'].to(self.device)
            # データローダーからSincフィルタ（リング効果を再現するフィルタ）を取得してGPUへ送る
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            # 元のGT画像の高さと幅を取得
            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- 1回目の劣化プロセス ----------------------- #
            # 1. ぼかし処理：GT(USM適用済)画像に対して1つ目のボケカーネルを畳み込む
            out = filter2D(self.gt_usm, self.kernel1)
            # 2. ランダムリサイズ：拡大・縮小・そのままの確率設定に基づきリサイズ形式を選択
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            # 拡大（up）が選ばれた場合：1〜最大スケール範囲の倍率をランダム選定
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            # 縮小（down）が選ばれた場合：最小スケール〜1範囲の倍率をランダム選定
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            # そのまま（keep）の場合：スケール変数は1
            else:
                scale = 1
            # 補間アルゴリズム（エリア、バイリニア、バイキュービック）をランダムで選択
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            # 設定したスケールと補間モードで画像をリサイズする
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # 3. ノイズ付加：グレースケールノイズにするかどうかの確率を設定
            gray_noise_prob = self.opt['gray_noise_prob']
            # 設定確率に基づいて「ガウスノイズ」または「ポアソンノイズ」を選択して追加する
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # 4. JPEG圧縮：バッチごとに設定範囲からランダムなクオリティ値を生成
            #jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            # 画素値を[0, 1]の範囲内に安全にクリップ（溢れるとDiffJPEGで異常値が出るため）
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            # DiffJPEGを呼び出してJPEG圧縮のノイズ（ブロックノイズなど）を再現
           # out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- 2回目の劣化プロセス ----------------------- #
            # 1. 2度目のぼかし処理：設定した確率で2つ目のボケカーネルを適用する
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # 2. 2度目のランダムリサイズ：確率に基づいて拡大・縮小・そのままを選択
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            # 補間アルゴリズムを選択
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            # 最終的な目標解像度（元サイズ / 倍率）に合わせたスケールでリサイズ処理を行う
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # 3. 2度目のノイズ付加：設定に従ってガウスノイズまたはポアソンノイズを追加
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # 4. 2度目のJPEG圧縮 ＋ 最終Sincフィルタ処理
            # 順序によって不自然な線が出ないよう、50%の確率で順番を入れ替えて処理する
            if np.random.uniform() < 0.5:
                # パターンA：[目標サイズへのリサイズ + Sincフィルタ] ➔ [JPEG圧縮]
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                # 画像のサイズを目標の低解像度サイズ（元サイズ ÷ 倍率）に整える
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                # Sincフィルタを適用（リンギング・高周波劣化の付加）
                out = filter2D(out, self.sinc_kernel)
                # JPEGクオリティパラメータの生成
                #jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                # 値を[0, 1]にクリップ
                out = torch.clamp(out, 0, 1)
                # JPEG圧縮を適用
                #out = self.jpeger(out, quality=jpeg_p)
            else:
                # パターンB：[JPEG圧縮] ➔ [目標サイズへのリサイズ + Sincフィルタ]
                #jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                #out = self.jpeger(out, quality=jpeg_p)
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # 画像を[0, 255]に引き延ばして丸め処理を行い、再度[0, 1]に正規化することで実際の8bit画像化を模倣
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # 学習用に特定のパッチサイズへランダムに切り出し（クロップ）を行う
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # キュープールに作成した画像を渡し、バッチ間でのデータの多様性を担保する
            self._dequeue_and_enqueue()
            # キューから新しいGT画像に置き換わった可能性があるため、USM研磨画像を再計算する
            self.gt_usm = self.usm_sharpener(self.gt)
            # メモリレイアウトの不連続警告を防ぐため、メモリを連続（contiguous）に配置する
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        # 検証時または高次劣化を使用しない標準的なデータ読み込み時の処理
        else:
            # 低解像度（lq）画像をそのまま取得してGPUへ転送
            self.lq = data['lq'].to(self.device)
            # データ内に高解像度（gt）画像が含まれている場合
            if 'gt' in data:
                # gt画像をGPUへ転送
                self.gt = data['gt'].to(self.device)
                # gt画像にUSM研磨処理を適用
                self.gt_usm = self.usm_sharpener(self.gt)

    # 分散処理を行わない検証（バリデーション）実行関数
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 検証中に誤って合成劣化（feed_dataのランダム処理）が走らないよう一時的に学習フラグをオフにする
        self.is_train = False
        # 親クラスの検証処理（推論と指標計算）を実行
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        # 検証が終わったら学習フラグをオンに戻す
        self.is_train = True

    # 1ステップ分のパラメータ更新（ネットワークの最適化）を実行する関数
    def optimize_parameters(self, current_iter):
        # ターゲット（正解データ）としてUSM適用画像を使うかどうかをフラグ設定から決定する初期化
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        # L1（ピクセル）損失でクリーンなGTを使う設定の場合
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        # 感覚（Perceptual）損失でクリーンなGTを使う設定の場合
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        # GAN（識別器）の正解データでクリーンなGTを使う設定の場合
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # ------------ 1. 生成器（Generator: net_g）の最適化 ------------ #
        # 識別器（Discriminator）のパラメータを固定（勾配計算を停止）
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 生成器のオプティマイザ（勾配）をリセット
        self.optimizer_g.zero_grad()
        # 低解像度画像（lq）から高解像度画像（output）を復元生成する
        self.output = self.net_g(self.lq)

        # 生成器の合計損失を保持する変数をリセット
        l_g_total = 0
        # 損失のログ記録用辞書を作成
        loss_dict = OrderedDict()
        # 識別器の更新間隔（net_d_iters）や初期ウォームアップ（net_d_init_iters）の条件をチェック
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # ピクセルレベルのL1損失計算（画像全体の粗い色や構造の一致）
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # 感覚損失・スタイル損失の計算（VGG等を用いた人間が感じる質感の一致）
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # 識別器に生成画像を通し、「本物」と誤認させるための敵対的損失（GAN Loss）を計算
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            # 生成器の損失に対して逆伝播を実行し、勾配を算出
            l_g_total.backward()
            # 生成器の重みを更新
            self.optimizer_g.step()

        # ------------ 2. 識別器（Discriminator: net_d）の最適化 ------------ #
        # 識別器のパラメータ固定を解除（勾配計算を有効化）
        for p in self.net_d.parameters():
            p.requires_grad = True

        # 識別器のオプティマイザ（勾配）をリセット
        self.optimizer_d.zero_grad()
        # 本物画像（gan_gt）に対する識別器の予測と、本物として識別するための損失計算
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        # 本物画像側の損失を逆伝播
        l_d_real.backward()
        # 偽物画像（生成画像 output）に対する識別器の予測と、偽物として識別するための損失計算（計算グラフ分離のためclone().detach()を使用）
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        # 偽物画像側の損失を逆伝播
        l_d_fake.backward()
        # 識別器の重みを更新
        self.optimizer_d.step()

        # EMA（指数移動平均）が設定されている場合は、モデルパラメータの移動平均を計算して安定化を図る
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # 全プロセスの損失数値をロギング用にまとめる
        self.log_dict = self.reduce_loss_dict(loss_dict)
