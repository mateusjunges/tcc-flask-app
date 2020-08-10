from __future__ import division
from flask import Flask
from flask import render_template
from flask import session
from flask import request
from flask import redirect
from flask import flash
from flask import Response
import os
from SER.speech_emotion_recognition import SpeechEmotionRecognition
from FER.face_emotion_recognition import FaceEmotionRecognition
from AudioExtraction.audio_extraction import extract_video_audio
from FrameExtraction.image_frame_extraction import extract_video_frames
import numpy as np
import cv2
from keras.models import model_from_json


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

    # Extrair os frames do vídeo enviado:
    frames_path = extract_video_frames(video_path, 'extracted-frames')

    # Analise do audio extraido
    SER = SpeechEmotionRecognition('models/audio-emotion-recognition-model.hdf5')

    emotions, timestamp = SER.predict_emotion_from_file(audio_path, predict_proba=True)

    emotion = np.argmax(emotions)

    predictions_from_audio = {
        'angry': "{:.2%}".format(emotions[0][0]),
        'disgust': "{:.2%}".format(emotions[0][1]),
        'fear': "{:.2%}".format(emotions[0][2]),
        'happy': "{:.2%}".format(emotions[0][3]),
        'neutral': "{:.2%}".format(emotions[0][4]),
        'sad': "{:.2%}".format(emotions[0][5]),
        'surprise': "{:.2%}".format(emotions[0][6])
    }

    # Análise dos frames extraídos
    with open('models/model-fer.json', 'r') as file:
        json = file.read()
    model = model_from_json(json)
    model.load_weights('models/model-fer-weights.h5')

    # Arrays pra salvar as probabilidades de cada emoção por frame analisado
    angry, disgust, fear, happy, sad, surprise, neutral = [], [], [], [], [], [], []

    for frame in os.listdir(frames_path):

        FER = FaceEmotionRecognition()

        image_to_predict = frames_path + '/' + frame

        face = cv2.imread(image_to_predict)
        predictions = []

        for face in FER.extract_features_from_face(FER.face_detector(face)):
            to_predict = np.reshape(face.flatten(), (1, 48, 48, 1))
            emotions = model.predict(to_predict)

            angry.append(float('{:f}'.format(emotions[0][0])))
            disgust.append(float('{:f}'.format(emotions[0][1])))
            fear.append(float('{:f}'.format(emotions[0][2])))
            happy.append(float('{:f}'.format(emotions[0][3])))
            sad.append(float('{:f}'.format(emotions[0][4])))
            surprise.append(float('{:f}'.format(emotions[0][5])))
            neutral.append(float('{:f}'.format(emotions[0][6])))

            prediction = np.argmax(emotions)
            predictions.append(prediction)

            emotion = np.argmax(emotions)

    predictions_from_frames = {
        'angry': "{:.2%}".format(sum(angry) / len(angry)),
        'disgust': "{:.2%}".format(sum(disgust) / len(disgust)),
        'fear': "{:.2%}".format(sum(fear) / len(fear)),
        'happy': "{:.2%}".format(sum(happy) / len(happy)),
        'neutral': "{:.2%}".format(sum(neutral) / len(neutral)),
        'sad': "{:.2%}".format(sum(sad) / len(sad)),
        'surprise': "{:.2%}".format(sum(surprise) / len(surprise))
    }

    os.remove(video_path)
    os.remove(audio_path)
    # for frame in os.listdir('extracted-frames/'):
    #     os.remove('extracted-frames/' + frame)

    prediction_results = {
        'predictions_from_audio': predictions_from_audio,
        'predictions_from_frames': predictions_from_frames,
    }

    # return prediction_results

    return render_template('report.html', data=prediction_results)


@app.route('/report', methods=['GET'])
def report():
    return render_template('report.html')


if __name__ == "__main__":
    app.run()
