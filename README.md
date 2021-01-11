# SER (Speech Emotion Recognition)

## 음성 감정 인식 인공지능 모델 제작
음성을 통해 화자의 감정을 인식할 수 있는 모델을 제작

## Dataset
- RAVDESS Dataset

## Libraries
- torch : 1.6.0
- torchaudio : 0.6.0

## Conditions
- 모든 음원은 고정길이를 가지고 있음

## Functions
### utils/features.py
feature를 추출할때 사용할 기능들을 담고 있음
#### extract_spectrogram
torchaudio 라이브러리를 사용해서 음원의 spectrogram을 추출
- source
- sample_rate
- n_fft : None (win_length와 동일)
- window_size : 0.025
- window_stride : 0.01
##### return type
- Dimension (…, freq, time)
