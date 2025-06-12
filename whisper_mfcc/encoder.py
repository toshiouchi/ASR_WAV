# -*- coding: utf-8 -*-

#
# Transformer エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import numpy as np
from attention import ResidualAttentionBlock
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(
        self,
        conv_layers=3,
        conv_in_channels=[13,256,512],
        conv_out_channels=[256.512,512],
        conv_kernel_size=[5,5,5],
        conv_stride=[2,1,1],
        conv_dropout_rate = 0.1,
        enc_hidden_dim = 512,
        num_enc_layers = 6,
        enc_num_heads = 4,
        enc_kernel_size = [5,1],
        enc_filter_size = 2048,
        enc_input_maxlen = 3000,
        enc_dropout_rate = 0.1,
        dim_out = 2680,
        enc_pad_id = 0,
    ):
        super(Encoder, self).__init__()

        self.pos_emb = nn.Embedding(enc_input_maxlen, enc_hidden_dim)
        # 1 次元畳み込みの重ね合わせ：局所的な時間依存関係のモデル化
        convs = nn.ModuleList()
        for layer in range(conv_layers):
            #print( "in_channel:", conv_in_channels[layer])
            #print( "out_channel:", conv_out_channels[layer] )
            convs += [
                nn.Conv1d(
                    in_channels=conv_in_channels[layer],
                    out_channels=conv_out_channels[layer],
                    kernel_size=conv_kernel_size[layer],
                    stride = conv_stride[layer],
                    padding=(conv_kernel_size[layer] - 1) // 2,
                    bias=False,  # この bias は不要です
                ),
                nn.BatchNorm1d(conv_out_channels[layer]),
                nn.ReLU(),
                nn.Dropout(conv_dropout_rate),
            ]
        self.convs = nn.Sequential(*convs)
        # Attention Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(enc_hidden_dim, enc_num_heads, cross_attention = False, kernel_size = enc_kernel_size, filter_size = enc_filter_size ) for _ in range(num_enc_layers)]
        )
        self.dropout = nn.Dropout(p=enc_dropout_rate)
        self.input_maxlen = enc_input_maxlen
        self.enc_pad_id = enc_pad_id
        self.enc_num_heads = enc_num_heads
        
    def forward(self, x, in_lens, mask ):
        device = x.device
        #print( "x size:", x.size() )
        #sprint( "mask size:", mask.size() )

        # conv 層
        out = self.convs(x.transpose(1, 2)).transpose(1, 2)
        #print("size of out:{}".format( out.size()) )
        
        # position embbeding
        maxlen = out.size()[1]
        positions = torch.arange(start=0, end=self.input_maxlen, step=1).to(torch.long).to(device)
        #print( "size of positions:{}".format( positions.size() ))
        positions = self.pos_emb(positions.to(device))[:maxlen,:]
        #print( "size of positions:{}".format( positions.size() ))
        x = out.to(device) + positions.to(device)
        #x = self.dropout( x )
        #print( "x size:", x.size() )
        
        mask = F.interpolate( mask[:,None,:], size = x.size(1) )[:,0,:]
        mask = mask[:,None,:,None]
        mask = mask.expand( (-1, self.enc_num_heads, -1, x.size(1)))
        #print( "mask size:", mask.size() )
        
        # attention block
        attention_weights = []
        #print( "in encoder" )
        for i, block in enumerate( self.blocks ):
            x, attn1, attn2 = block(x, x, self_mask = mask)
            attention_weights.append( attn1 )
            attention_weights.append( attn2 )
        
       
        return x, mask  # (batch_size, input_seq_len, d_model)
