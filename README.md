# Audio Splitter
A prototype of an ML-pipeline for an audio separation service based on Demucs model. 
## Overview
* A simple inference of Demucs model that can split an audio file (song) into its stems: vocals, drums, bass, other; 
* REST-endpoint for audio separation;
* Scripts for metrics calculations and building spectrograms of audio files;
* Analysis of results and ideas for improving the Demucs. 

## Evaluation
|   Stem   | SIR (dB) | SAR (dB) | SDR (dB) |
|:---------|:--------:|:--------:|:--------:|
|  vocals  | 15 ± 4   | 11 ± 9   | 8  ± 2   |
|  drums   | 14 ± 7   | 10 ± 12  | 6  ± 5   |
|  bass    | 8  ± 15  | 20 ± 6   | 12 ± 3   |
|  other   | 12 ± 6   | 3  ± 13  | 6  ± 3   |

## Stack
### Audio
* torchaudio — models loading, working with demucs;
* librosa    — provides stft, working with wav, mp3;
* museval    — metrics SIR, SAR, SDR. 
### Web
* FastAPI — backend, documentation; 
* uuid    — unique sessions. 
### ML
* PyTorch — ML framework;
* demucs  — native working with demucs. 
### Data Analysis
* numpy, pandas, matplotlib — basic DA-stack;
* jupyter notebook — programming environment. 

## Get started
### Backend
After you have cloned the repository 
``` bash
git clone "https://github.com/14imarkin/audio_splitter"
pip install -r requirements.txt
```

You can type into your terminal this for opening the website on your local host:
```bash
uvicorn main:app --reload
```

Then you can visit a swagger (special tool of FastAPI that allows to make good documentation automatically) by typing "/docs" after the URL. 
http://127.0.0.1:8000/docs

Then follow swagger recommendations for trying the endpoint "/separate".
- In the Swagger UI, find the /separate endpoint.
- Click Try it out.
- Upload an audio file (MP3, WAV).
- Click Execute.
The results will be stored in the folder "stems" in the root of your project as wav files. 

### Metrics and spectrograms
To create metrics and spectrograms you should open the "make_*.py" file in your code editor and type in the path for your audio file into the special section. Build the project and you are good to go.
