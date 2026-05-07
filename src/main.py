# Веб-бекэнд для разделения аудио на стемы
# Маркин Иван, 2026

# --- Импорты ---  
# Работа с аудио
import torch
import librosa
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
# Бэкенд
import asyncio
import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.staticfiles import StaticFiles

# --- Конфиг ---
MAX_SIZE = 1024 * 1024
MAX_TIME = 3600
TIME_OUT = 3600

# --- Веб-приложение ---
app = FastAPI()

# Загрузка модели
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model("htdemucs").to(device)
model.eval()

# Папка для выходных стемов
OUTPUT_DIRECTORY = "stems"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
app.mount("/download_stems", StaticFiles(directory=OUTPUT_DIRECTORY), name="/download_stems")

# Ручка разделения входного аудио-файла на стемы
@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    # Генерация отдельной сессии
    session_id = str(uuid.uuid4())
    session_directory = os.path.join(OUTPUT_DIRECTORY, session_id)
    os.makedirs(session_directory, exist_ok=True)
    # Обработка входного файла
    input_path = os.path.join(session_directory, file.filename)
    try:
        # Проверка на слишком большой файл
        with open(input_path, "wb") as input:
            f = await file.read()
            if len(f) > MAX_SIZE:
                raise HTTPException(detail="File is too big!!")
            input.write(f)
    # Разделение на стемы
        # Проверка librosa
        try:
            wav, sr = librosa.load(input_path, sr=model.samplerate, mono=False)
        except Exception as e:
            raise HTTPException(detail=f"{str(e)}")
        # Проверка длины аудио
        t = librosa.get_duration(y=wav, sr=sr)
        if t > MAX_TIME:
            raise HTTPException(detail="File is too long!!")
        # Подготовка тензора
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
        # Тайм-Аут
        try:
            with torch.no_grad():
                srcs = await asyncio.wait_for(asyncio.to_thread(apply_model, model, wav_t, split=True, segment=5), timeout = TIME_OUT)
                srcs = srcs[0]
        except asyncio.TimeoutError:
            raise HTTPException(detail="Timeout error!!")
        # Сохранение стемов
        links = {}
        main_url = "http://localhost:8000/download_stems"
        for i, name in enumerate(model.sources):
            stem_path = os.path.join(session_directory, f"{name}.wav")
            torchaudio.save(stem_path, srcs[i].cpu(), model.samplerate)
            links[name] = f"{main_url}_{session_id}_{name}.wav"
        return {'session_id': session_id, "stems": links}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(detail=f"{str(e)}")
