# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# 作成したEncoder, Decoderクラスをインポート
from encoder import Encoder

# 作成した初期化関数をインポート
from initialize import lecun_initialization

import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyE2EModel(nn.Module):
    ''' Attention RNN によるEnd-to-Endモデルの定義
    dim_in:            入力次元数
    dim_enc_hid:       エンコーダの隠れ層の次元数
    dim_enc_proj:      エンコーダのProjection層の次元数
                       (これがエンコーダの出力次元数になる)
    dim_dec_hid:       デコーダのRNNの次元数
    dim_out:           出力の次元数(sosとeosを含む全トークン数)
    dim_att:           Attention機構の次元数
    att_filter_size:   LocationAwareAttentionのフィルタサイズ
    att_filter_num:    LocationAwareAttentionのフィルタ数
    sos_id:            <sos>トークンの番号
    enc_bidirectional: Trueにすると，エンコーダに
                       bidirectional RNNを用いる
    enc_sub_sample:    エンコーダにおいてレイヤーごとに設定する，
                       フレームの間引き率
    enc_rnn_type:      エンコーダRNNの種類．'LSTM'か'GRU'を選択する
    '''
    def __init__(self, dim_in, dim_out,
                 conv_layers, conv_in_channels, conv_out_channels, conv_kernel_size, conv_stride, conv_dropout_rate,
                 enc_num_layers, enc_att_hidden_dim, enc_num_heads, enc_input_maxlen,  enc_feedfoward_dim, enc_dropout_rate,
                 sos_id, enc_pad_id, pad_id
                 ):
        super(MyE2EModel, self).__init__()
        #print( "dim_out:{}".format( dim_out ))
        # エンコーダを作成

        self.encoder = Encoder(
            conv_layers = conv_layers,
            conv_in_channels = conv_in_channels,
            conv_out_channels = conv_out_channels,
            conv_kernel_size = conv_kernel_size,
            conv_stride = conv_stride,
            conv_dropout_rate = conv_dropout_rate,
            num_enc_layers = enc_num_layers,
            enc_hidden_dim = enc_att_hidden_dim,
            enc_num_heads = enc_num_heads,
            enc_input_maxlen = enc_input_maxlen,
            enc_feedfoward_dim = enc_feedfoward_dim,
            enc_dropout_rate = enc_dropout_rate,
        )
        
        self.classifier = nn.Linear( enc_att_hidden_dim, dim_out, bias=False )
        self.sos_id = sos_id
        self.enc_pad_id = enc_pad_id
        self.pad_id = pad_id

        # LeCunのパラメータ初期化を実行
        #lecun_initialization(self)

    def forward(self,
                input_sequence,
                input_lengths):
                #label_sequence=None,
                #label_lengths=10):
                #dec_input,
                #dec_input_lens):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
        label_sequence: 各発話の正解ラベル系列(学習時に用いる) [B x Tout]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        device = input_sequence.device
        # エンコーダに入力する
        mask = torch.all( torch.eq( input_sequence, self.enc_pad_id ) , dim = 2 )[:,:,None].float()
        enc_out, src_key_padding_mask = self.encoder(input_sequence, input_lengths, mask )

        outputs = self.classifier( enc_out )

        # デコーダ出力とエンコーダ出力系列長を出力する
        return outputs

    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(utt, filename)
        

