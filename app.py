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
    print("\n\n\n\n" + audio_path)
    emotions, timestamp = SER.predict_emotion_from_file(audio_path, predict_proba=True)

    emotion = np.argmax(emotions)
    print("Emoção audio: " + str(SER.get_emotion_label(emotion)))

    predictions_from_audio = {
        'angry': '{:f}'.format(emotions[0][0]),
        'disgust': '{:f}'.format(emotions[0][1]),
        'fear': '{:f}'.format(emotions[0][2]),
        'happy': '{:f}'.format(emotions[0][3]),
        'neutral': '{:f}'.format(emotions[0][4]),
        'sad': '{:f}'.format(emotions[0][5]),
        'surprise': '{:f}'.format(emotions[0][6])
    }

    # Análise dos frames extraídos
    with open('models/model-fer.json', 'r') as file:
        json = file.read()
    model = model_from_json(json)
    model.load_weights('models/model-fer-weights.h5')

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

            print("Emoção encontrada: " + FER.get_emotion_name(emotion))

    predictions_from_frames = {
        'angry': sum(angry) / 100,
        'disgust': sum(disgust) / 100,
        'fear': sum(fear) / 100,
        'happy': sum(happy) / 100,
        'neutral': sum(neutral) / 100,
        'sad': sum(sad) / 100,
        'surprise': sum(surprise) / 100
    }

    os.remove(video_path)
    os.remove(audio_path)
    for frame in os.listdir('extracted-frames/'):
        os.remove('extracted-frames/' + frame)

    return {
        'prediction_from_audio': predictions_from_audio,
        'prediction_from_frames': predictions_from_frames,
    }


if __name__ == "__main__":
    app.run()
