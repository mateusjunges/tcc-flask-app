from flask import Flask
from flask import render_template
from flask import session
from flask import request
from flask import redirect
from flask import flash
from flask import Response
import os
from SER.speech_emotion_recognition import SpeechEmotionRecognition
from AudioExtraction.audio_extraction import extract_video_audio



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    step = 1
    sample_rate = 16000

    # Upload do arquivo de vídeo selecionado
    file = request.files['file']
    file_extension = file.filename.split('.')[-1]
    video_path = 'uploads' + '/' + 'video.' + file_extension
    file.save(video_path)


    # Extrair o áudio do vídeo enviado:
    audio_path = extract_video_audio(video_path, 'extracted-audios/')

    SER = SpeechEmotionRecognition('models/audio-emotion-recognition-model.hdf5')
    emotions, timestamp = SER.predict_emotion_from_file(audio_path, predict_proba=True)

    predictions_from_audio = {
        'angry': '{:f}'.format(emotions[0][0]),
        'disgust': '{:f}'.format(emotions[0][1]),
        'fear': '{:f}'.format(emotions[0][2]),
        'happy': '{:f}'.format(emotions[0][3]),
        'neutral': '{:f}'.format(emotions[0][4]),
        'sad': '{:f}'.format(emotions[0][5]),
        'surprise': '{:f}'.format(emotions[0][6])
    }

    os.remove(video_path)
    os.remove(audio_path)

    return predictions_from_audio


if __name__ == "__main__":
    app.run()
