# Расчет метрик разделения аудио на стемы
# Маркин Иван, 2026

# Импорты
import museval
import torch
import torchaudio
import numpy as np

def load_audio(path):
    '''
    Корректная подгрузка аудио-файла
    '''
    wav, sr = torchaudio.load(path)
    return wav

# Подгрузка файлов-таргетов
ref_vocals = load_audio("songs/4/vocals.wav")
ref_drums = load_audio("songs/4/drums.wav")
ref_bass = load_audio("songs/4/bass.wav")
ref_other = load_audio("songs/4/other.wav")

# Подгрузка файлов-предсказаний
est_vocals = load_audio("stems/2c5c0cee-f2e8-460d-8ea3-de9edfca3a87/vocals.wav")
est_drums = load_audio("stems/2c5c0cee-f2e8-460d-8ea3-de9edfca3a87/drums.wav")
est_bass = load_audio("stems/2c5c0cee-f2e8-460d-8ea3-de9edfca3a87/bass.wav")
est_other = load_audio("stems/2c5c0cee-f2e8-460d-8ea3-de9edfca3a87/other.wav")

# Приведение данных к формату
references = torch.stack([ref_drums, ref_bass, ref_other, ref_vocals]).permute(0, 2, 1).numpy()
estimates = torch.stack([est_drums, est_bass, est_other, est_vocals]).permute(0, 2, 1).numpy()

# Расчет метрик
print("start")
scores = museval.evaluate(references, estimates)

# Визуализация
names = ['Drums', 'Bass', 'Other', 'Vocals']
for i, name in enumerate(names):
    # Среднее по медиане
    sdr = np.nanmedian(scores[0][i])
    sir = np.nanmedian(scores[1][i])
    sar = np.nanmedian(scores[2][i])
    # Вывод
    print(f"{name}")
    print(f"SDR: {sdr:.2f} dB")
    print(f"SIR: {sir:.2f} dB")
    print(f"SAR: {sar:.2f} dB\n")