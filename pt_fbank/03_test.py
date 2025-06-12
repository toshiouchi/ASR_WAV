# -*- coding: utf-8 -*-

#
# RNN Attention Encoder-Decoderによるデコーディングを行います
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
from torch.utils.data import DataLoader

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# モデルの定義をインポート
from my_model import MyE2EModel

# json形式の入出力を行うモジュールをインポート
import json

# os, sysモジュールをインポート
import os
import sys


pad_id = 0
blank_id = 1
enc_pad_id = 0

def ctc_simple_decode(int_vector, token_list):
    ''' 以下の手順で，フレーム単位のCTC出力をトークン列に変換する
        1. 同じ文字が連続して出現する場合は削除
        2. blank を削除
    int_vector: フレーム単位のCTC出力(整数値列)
    token_list: トークンリスト
    output:     トークン列
    '''
    # 出力文字列
    output = []
    # 一つ前フレームの文字番号
    prev_token = -1
    # フレーム毎の出力文字系列を前から順番にチェックしていく
    for n in int_vector:
        #print( " n:{}".format( n ))
        #print( " prev_token:{}".format( prev_token ))
        if n != prev_token:
            # 1. 前フレームと同じトークンではない
            if token_list[n] != '<blank>':
                # 2. かつ，blank(番号=0)ではない
                # --> token_listから対応する文字を抽出し，
                #     出力文字列に加える
                output.append(token_list[n])
                if token_list[n] == '<eos>':
                    break
            # 前フレームのトークンを更新
            prev_token = n
    return output

#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'char'

    # 実験ディレクトリ
    exp_dir = './exp_train_large'

    # 評価データの特徴量(feats.scp)が存在するディレクトリ
    feat_dir_test = '../01compute_features/fbank/test'
    wav_dir_test = '../data/label/test'

    # 評価データの特徴量リストファイル
    feat_scp_test = os.path.join(feat_dir_test, 'feats.scp')
    wav_scp_test = os.path.join(wav_dir_test, 'wav2.scp')

    # 評価データのラベルファイル
    label_test = os.path.join(exp_dir, 'data', unit, 'label_test')

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')

    # 学習済みモデルが格納されているディレクトリ
    model_dir = os.path.join(exp_dir, unit+'_model_trf_nar')

    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(model_dir, 'mean_std.txt')

    # 学習済みのモデルファイル
    model_file = os.path.join(model_dir, 'best_model.pt')

    # デコード結果を出力するディレクトリ
    output_dir = os.path.join(model_dir, 'decode_test')

    # デコード結果および正解文の出力ファイル
    hypothesis_file = os.path.join(output_dir, 'hypothesis.txt')
    reference_file = os.path.join(output_dir, 'reference.txt')

    # 学習時に出力した設定ファイル
    config_file = os.path.join(model_dir, 'config.json')

    # ミニバッチに含める発話数
    #batch_size = 10
    batch_size = 1
    
    #
    # 設定ここまで
    #

    # 設定ファイルを読み込む
    with open(config_file, mode='r') as f:
        config = json.load(f)

    # 読み込んだ設定を反映する
    # Encoderの設定
    # 中間層のレイヤー数
    #enc_num_layers = config['enc_num_layers']
    # 層ごとのsub sampling設定
    #enc_sub_sample = config['enc_sub_sample']
    # RNNのタイプ(LSTM or GRU)
    #enc_rnn_type = config['enc_rnn_type']
    # 中間層の次元数
    #enc_hidden_dim = config['enc_hidden_dim']
    # Projection層の次元数
    #enc_projection_dim = config['enc_projection_dim']
    # bidirectional を用いるか(Trueなら用いる)
    #enc_bidirectional = config['enc_bidirectional']


    conv_layers = config['conv_layers']
    conv_in_channels = config['conv_in_channels']
    conv_out_channels = config['conv_out_channels']
    conv_kernel_size = config['conv_kernel_size']
    conv_stride = config['conv_stride']
    conv_dropout = config['conv_dropout_rate']
    enc_num_layers = config['enc_num_layers']
    enc_num_heads = config['enc_num_heads']
    enc_input_maxlen = config['enc_input_maxlen']
    enc_att_hidden_dim = config['enc_att_hidden_dim']
    enc_feedfoward_dim = config['enc_feedfoward_dim']
    enc_dropout = config['enc_dropout_rate']
    enc_pad_id = config['enc_pad_id']
    blank_id = config['blank_id']
    #enc_pad_id = 0
    #blank_id = 1
    batch_size = config['batch_size']
    max_num_epoch = config['max_num_epoch']
    clip_grad_threshold = config['clip_grad_threshold']
    initial_learning_rate = config['initial_learning_rate']
    lr_decay_start_epoch = config['lr_decay_start_epoch']
    lr_decay_factor = config['lr_decay_factor']
    early_stop_threshold = config['early_stop_threshold']

    batch_size = 1

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 特徴量の平均/標準偏差ファイルを読み込む
    with open(mean_std_file, mode='r') as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        # 3行目が標準偏差ベクトル(std)
        mean_line = lines[1]
        std_line = lines[3]
        # スペース区切りのリストに変換
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy arrayに変換
        feat_mean = np.array(feat_mean, 
                                dtype=np.float32)
        feat_std = np.array(feat_std, 
                               dtype=np.float32)

    # 次元数の情報を得る
    feat_dim = np.size(feat_mean)
    #feat_dim = 1

    # トークンリストをdictionary型で読み込む
    # このとき，0番目は blank と定義する
    # (ただし，このプログラムではblankは使われない)
    #token_list = {0: '<blank>'}
    token_list = {}
    with open(token_list_path, mode='r', encoding='UTF-8') as f:
        # 1行ずつ読み込む
        for line in f: 
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            parts = line.split()
            # 0番目の要素がトークン，1番目の要素がID
            token_list[int(parts[1])] = parts[0]

    pad_id = 0
    blank_id = 1

    # <eos>トークンをユニットリストの末尾に追加
    eos_id = len(token_list)
    token_list[eos_id] = '<eos>'
    # 本プログラムでは、<sos>と<eos>を
    # 同じトークンとして扱う
    #sos_id = eos_id
    sos_id = len(token_list)
    token_list[sos_id] = '<sos>'

    # トークン数(blankを含む)
    num_tokens = len(token_list)
    
    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = MyE2EModel(dim_in=feat_dim,
                       dim_out=num_tokens,
                       conv_layers=conv_layers,
                       conv_in_channels=conv_in_channels,
                       conv_out_channels=conv_out_channels,
                       conv_kernel_size=conv_kernel_size,
                       conv_stride=conv_stride,
                       conv_dropout_rate=conv_dropout,
                       enc_num_layers = enc_num_layers,
                       enc_att_hidden_dim=enc_att_hidden_dim,
                       enc_num_heads = enc_num_heads,
                       enc_input_maxlen = enc_input_maxlen, 
                       enc_feedfoward_dim=enc_feedfoward_dim,
                       enc_dropout_rate = enc_dropout,
                       sos_id=sos_id, 
                       enc_pad_id=enc_pad_id,
                       pad_id=pad_id,
                       )
    print(model)
    
    # モデルのパラメータを読み込む
    model.load_state_dict(torch.load(model_file))

    # 訓練/開発データのデータセットを作成する
    test_dataset = SequenceDataset(feat_scp_test,
                                   label_test,
                                   feat_mean,
                                   feat_std,
                                   enc_pad_id)

    # 評価データのDataLoaderを呼び出す
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # モデルを評価モードに設定する
    model.eval()

    # デコード結果および正解ラベルをファイルに書き込みながら
    # 以下の処理を行う
    with open(hypothesis_file, mode='w') as hyp_file, \
         open(reference_file, mode='w') as ref_file:
        # 評価データのDataLoaderから1ミニバッチ
        # ずつ取り出して処理する．
        # これを全ミニバッチ処理が終わるまで繰り返す．
        # ミニバッチに含まれるデータは，
        # 音声特徴量，ラベル，フレーム数，
        # ラベル長，発話ID
        for (features, labels, feat_lens,
             label_lens, utt_ids) in test_loader:

            
            #
            # ラベルの末尾に<eos>を付与する最初に<sos>
            #
            # ゼロ埋めにより全体の長さを1増やす
            labels = torch.cat( ( torch.zeros(labels.size()[0],1,dtype=torch.long), labels), dim = 1 )
            labels = torch.cat((labels,
                                torch.zeros(labels.size()[0],
                                1,
                                dtype=torch.long)), dim=1)
            # 末尾に<eos>追加、最初に<sos> を追加
            for m, length in enumerate(label_lens):
                labels[m][0] = sos_id
                labels[m][length+1] = eos_id
            label_lens += 2        
        

            # CUDAが使える場合はデータをGPUに，
            # そうでなければCPUに配置する
            features = features.to(device)

            # モデルの出力を計算(フォワード処理)
            logits = model(features, feat_lens)
            #print( "size of logits:{}".format( logits.size() ))

            # バッチ内の1発話ごとに以下の処理を行う
            for n in range(logits.size(0)):
                # 出力はフレーム長でソートされている
                # 元のデータ並びに戻すため，
                # 対応する要素番号を取得する
                #idx = torch.nonzero(indices==n, 
                #                    as_tuple=False).view(-1)[0]
                idx = n
                #print( "idx:{}".format( idx ) )

                # 各ステップのデコーダ出力を得る
                #_, hyp_per_step = torch.max(outputs[idx], 1)
                preds = torch.argmax( logits, dim = 2 )
                hyp_per_step = preds[idx]
                # numpy.array型に変換
                #hyp_per_step = hyp_per_step.cpu().numpy()
                hyp_per_step = hyp_per_step.cpu().numpy()
                # 認識結果の文字列を取得
                #hypothesis = []
                ##for m in hyp_per_step[:out_lens[idx]]:
                #for m in hyp_per_step[:dec_target_maxlen]:
                #    #print( "m:{}".format( m ) )
                #    hypothesis.append(token_list[m])
                #    if m == eos_id:
                #        break
                hypothesis = ctc_simple_decode( hyp_per_step, token_list )

                # 正解の文字列を取得
                reference = []
                for m in labels[n][:label_lens[n]].cpu().numpy():
                    reference.append(token_list[m])
                    if m == eos_id:
                        break
                    
                print("hypo:", ' '.join(hypothesis) )
                print("refe:", ' '.join(reference))
             
                # 結果を書き込む
                # (' '.join() は，リスト形式のデータを
                # スペース区切りで文字列に変換している)
                hyp_file.write('%s %s\n' \
                    % (utt_ids[n], ' '.join(hypothesis)))
                ref_file.write('%s %s\n' \
                    % (utt_ids[n], ' '.join(reference)))
            #sys.exit()


