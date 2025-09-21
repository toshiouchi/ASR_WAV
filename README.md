# Accuracy measurement of speech recognition module using Wav input

## Motivation for making measurements

In automatic speech recognition (ASR), fbank or mfcc is generally used for acoustic features. On the other 
hand, in image classification using Transformer, feature extractors using CNNs are often used to analyze 
features. Therefore, we measured the accuracy of the ASR module using a feature extractor that directly 
inputs a wav file and calculates acoustic features using Conv1d. For comparison, we also measured 
the accuracy using fbank and mfcc.

## Voice recognition module

Speech recognition was performed using two non-autoregressive speech recognition algorithms. 
One was 

a 24-layer PyTorch Transformer Encoder + CTCloss

The other was 

a 12-layer whisper-like Transformer Encoder + 12-layer whisper-like Transformer Decoder + CTCLoss.


Both include a simple conv1d in the Encoder. The input to the whisper-like Transformer Decoder
is the Encoder Outs as the memory input and the Encoder Outs downsampled by 0.25 in the sequence
direction as the target input.


For speech recognition using Wav input, we used a feature extractor 

7 layers: (conv1d + BatchNorm1d + GELU) + LayerNorm + Linear + Dropout

before the encoder in both of the above cases.

## Learning results

<table>
<caption> Nar-ASR program inference WER with test data.
<thread>
<th>model<th>WER
<tbody>
<tr><td style="text-align:left;"> PT fbank<td>25.8
<tr><td style="text-align:left;"> PT mfcc<td>26.7
<tr><td style="text-align:left;"> PT wav<td>26.1
<tr><td style="text-align:left;"> Whisper like fbank<td>19.8
<tr><td style="text-align:left;"> Whisper like mfcc<td>20.3
<tr><td style="text-align:left;"> Whisper like wav<td>21.7
</table>

## Training data

The training data used was 3,000 utterances of BASIC5000 and 35,000 utterances of 
common voice Japanese. The development data used was 1,000 utterances of BASIC5000, 
and the test data also used 1,000 utterances of BASIC5000.

## Measurement

### fbank + PyTorch Transformer Encoder + CTCLoss

The graphs of loss and CER during measurement are shown below.

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/cc2ec3c8-bcc7-464e-9f7d-98da03a902ea" />

<img width="640" height="480" alt="error" src="https://github.com/user-attachments/assets/a225cb48-3dbd-4e09-9818-bc22b02fa42a" />


CER 25.8% in test

### mfcc + PyTorcn Transformer Encoder + CTCLoss

The graphs of loss and CER during measurement are shown below.

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/7b9acb54-7100-4d14-a891-4cc6e1736e5e" />

<img width="640" height="480" alt="error" src="https://github.com/user-attachments/assets/0fd4a116-924f-4a32-937a-857d15b1dbec" />

CER in test 26.7%

### Wav + Feature Extractor + PyTorch Transformer Encoder + CTCLoss

The graphs of loss and CER during measurement are shown below.

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/0581f90f-0c48-4f79-8a26-ce01af7af887" />

<img width="640" height="480" alt="error" src="https://github.com/user-attachments/assets/8631bc25-e8f2-42ba-924c-e14e520497a2" />

CER 26.1% in test

### fbank + whisper like Transformer Encoder + downsampling + Transformer Decoder + CTCLoss

The graphs of loss and CER during measurement are shown below.

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/df2fa43d-5f8c-4363-8e67-36a13d7bf027" />

<img width="640" height="480" alt="error" src="https://github.com/user-attachments/assets/082e0a96-4e12-4c1c-a2ca-54b1a34257d9" />

CER in test 19.8%

### mfcc + whisper like Transformer Encoder + downsampling + Transformer Decoder + CTCLoss

The graphs of loss and CER during measurement are shown below.

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/ebf7177f-07a2-4451-a1b5-5992075fcc13" />

<img width="640" height="480" alt="error" src="https://github.com/user-attachments/assets/3ebfa1f4-c8cd-47fb-8649-e132dd7db750" />

CER 20.3% in test

### Wav + Feature Extractor + whisper like Transformer Encoder + downsampling + Transformer Decoder + CTCLoss

The graphs of loss and CER during measurement are shown below.

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/93b6b147-77fe-44ec-860c-fd3297654aa6" />

<img width="640" height="480" alt="error" src="https://github.com/user-attachments/assets/15b33750-ac79-4fa8-a901-df9f0658503d" />

CER 21.7% in test
