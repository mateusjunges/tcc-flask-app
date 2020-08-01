from flask import Flask
from flask import render_template
from flask import session
from flask import request
from flask import redirect
from flask import flash
from flask import Response
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

    # Analise das emoções através do audio

    # Extrair o áudio do vídeo enviado:
    # extract_video_audio()

    SER = SpeechEmotionRecognition('models/audio-emotion-recognition-model.hdf5')
    # emotions, timestamp = SER.predict_emotion_from_file('./test/03-01-03-02-01-02-07.wav', predict_proba=True)
    emotions, timestamp = SER.predict_emotion_from_file('./test/03-01-02-02-01-02-21.wav', predict_proba=True)

    predictions_from_audio = {
        'angry': '{:f}'.format(emotions[0][0]),
        'disgust': '{:f}'.format(emotions[0][1]),
        'fear': '{:f}'.format(emotions[0][2]),
        'happy': '{:f}'.format(emotions[0][3]),
        'neutral': '{:f}'.format(emotions[0][4]),
        'sad': '{:f}'.format(emotions[0][5]),
        'surprise': '{:f}'.format(emotions[0][6])
    }




    return predict


if __name__ == "__main__":
    app.run()
