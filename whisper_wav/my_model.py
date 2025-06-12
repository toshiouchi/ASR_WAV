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
from feature_extractor import FeatureExtractor
from encoder import Encoder
from decoder import Decoder

# 作成した初期化関数をインポート
from initialize import lecun_initialization

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                 fe_conv_layer, fe_conv_channel, fe_conv_kernel, fe_conv_stride, fe_conv_dropout_rate, fe_out_dim,
                 conv_layers, conv_in_channels, conv_out_channels, conv_kernel_size, conv_stride, conv_dropout_rate,
                 enc_num_layers, enc_att_hidden_dim, enc_num_heads, enc_input_maxlen,  enc_att_kernel_size, enc_att_filter_size, enc_dropout_rate,
                 ds_rate,
                 dec_num_layers, dec_att_hidden_dim, dec_num_heads, dec_target_maxlen, dec_att_kernel_size, dec_att_filter_size, dec_dropout_rate,
                 sos_id, enc_pad_id,
                 ):
        super(MyE2EModel, self).__init__()
        #print( "dim_out:{}".format( dim_out ))
        # エンコーダを作成
        #self.encoder = Encoder(dim_in=dim_in, 
        #                       dim_hidden=dim_enc_hid, 
        #                       dim_proj=dim_enc_proj, 
        #                       num_layers=enc_num_layers, 
        #                       bidirectional=enc_bidirectional, 
        #                       sub_sample=enc_sub_sample, 
        #                       rnn_type=enc_rnn_type)
        #self.enc_input_max = enc_input_max
        #print( "in my_model enc_input_maxlen:{}".format( enc_input_maxlen ))

        self.fe = FeatureExtractor(
            fe_conv_layer=fe_conv_layer,
            fe_conv_channel=fe_conv_channel,
            fe_conv_kernel=fe_conv_kernel,
            fe_conv_stride=fe_conv_stride,
            fe_conv_dropout_rate = fe_conv_dropout_rate,
            fe_out_dim=fe_out_dim,
        )
        
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
            enc_kernel_size = enc_att_kernel_size,
            enc_filter_size = enc_att_filter_size,
            enc_dropout_rate = enc_dropout_rate,
            enc_pad_id = enc_pad_id,
        )
        
        
        # デコーダを作成
        self.decoder = Decoder(
            dec_num_layers = dec_num_layers,
            dec_input_maxlen = dec_target_maxlen,
            decoder_hidden_dim = dec_att_hidden_dim,
            dec_num_heads = dec_num_heads,
            dec_kernel_size = dec_att_kernel_size,
            dec_filter_size = dec_att_filter_size,
            dec_dropout_rate = dec_dropout_rate,
            dim_out = dim_out
        )

        #　デコーダーのあとに、n * t * hidden を n * t * num_vocab にする線形層。
        self.classifier = nn.Linear( dec_att_hidden_dim, dim_out, bias=False )
        
        self.dec_target_maxlen = dec_target_maxlen
        self.sos_id = sos_id
        self.ds_rate = ds_rate
        self.enc_pad_id = enc_pad_id        

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
        # feture_extractor
        #print( "input_sequence size:", input_sequence.size() )
        mask = torch.all( torch.eq( input_sequence, self.enc_pad_id ), dim = 2 ).float() * -1e9
        #print( "0 mask size:", mask.size() )
        input_sequence, input_lengths = self.fe( input_sequence, input_lengths )
        #print( "fe out size:", input_sequence.size() )
        # エンコーダに入力する
        enc_out, src_key_padding_mask = self.encoder(input_sequence, input_lengths, mask )
        #print( "enc_out size:", enc_out.size() )
        
        #dec_input, outputs_lens = self.downsample( enc_out, input_lengths )
        dec_input = F.interpolate( enc_out.transpose(1,2), scale_factor = self.ds_rate ).transpose(1,2)
        #print( "enc_out size():", enc_out.size() )
        #print( "dec_input size():", dec_input.size() )
        mask0 = mask
        mask = F.interpolate( mask0[:,None,:], size=enc_out.size(1) )[:,0,:]
        tgt_padding_mask = F.interpolate( mask0[:,None,:], size= dec_input.size(1) )[:,0,:]
        
        # デコーダに入力する
        #print( "mask size:", mask.size() )
        #print( "tgt_padding_mask:", tgt_padding_mask.size() )
        dec_out = self.decoder(enc_out, dec_input, mask, tgt_padding_mask )
        #print( "dec_out size:", dec_out.size() )

        # n * T * hidden → n * T * num_vocab 
        outputs = self.classifier( dec_out )
        #sys.exit()

        # デコーダ出力とエンコーダ出力系列長を出力する
        return outputs

    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(utt, filename)
        
        
    def inference(self,
                input_sequence,
                input_lengths,
                #labels
                 ):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        # feture_extractor
        mask = torch.eq( input_sequence, 0 ).float()
        input_sequence, input_lengths = self.fe( input_sequence, input_lengths )
        
        # エンコーダに入力する
        enc_out, src_key_padding_mask = self.encoder(input_sequence, input_lengths, mask )
        
        dec_input = torch.ones( (input_sequence.size(0), 1)  ).long() * self.sos_id
        dec_input = dec_input.to(device)
        #print( "self.sos_id:{}".format( self.sos_id ))
        #print( "dec_input:{}".format( dec_input ))
        
        #length = []
        for i in range( self.dec_target_maxlen ):
        # デコーダに入力する
            seq = dec_input.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=seq, dtype=torch.bool, device = device)
            tgt_key_padding_mask  = torch.eq( dec_input, 0 ).bool()
            dec_out = self.decoder(enc_out, dec_input, tgt_mask, tgt_key_padding_mask, src_key_padding_mask  )
            outputs = self.classifier( dec_out )
            #print( "size of dec_out:{}".format( dec_out.size() ))
            #print("dec_out:{}".format( dec_out ))
            logits = torch.argmax( outputs, dim = 2 ).long()
            #print("logits[0]:{}".format( logits[0] ))
            last_logit = torch.unsqueeze(logits[:, -1], axis= 1 )
            #print("last_logit:{}".format( last_logit ))
            #if last_logit == self.sos_id:
            #    length.append( i )
            #    break
            dec_input = torch.cat([dec_input, last_logit], axis=1)
            #print( "dec_input:", dec_input )

        #sys.exit()
        # デコーダ出力とエンコーダ出力系列長を出力する
        #print("dec_input:{}".format( dec_input ))
        return dec_input
