from __future__ import division
from flask import Flask
from flask import render_template
from flask import session
from flask import request
from flask import redirect
from flask import flash
from flask import Response
import os
# from SER.speech_emotion_recognition import SpeechEmotionRecognition
from SER.speech_emotion_recognition import SpeechEmotionRecognition
from FER.face_emotion_recognition import FaceEmotionRecognition
from AudioExtraction.audio_extraction import extract_video_audio
from FrameExtraction.image_frame_extraction import extract_video_frames
import numpy as np
import cv2
from keras.models import model_from_json
import operator

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
    filename = file.filename
    file_extension = file.filename.split('.')[-1]
    video_path = 'uploads' + '/' + 'video.' + file_extension
    file.save(video_path)

    # Extrair o áudio do vídeo enviado:
    audio_path = extract_video_audio(video_path, 'extracted-audios/')

    # Extrair os frames do vídeo enviado:
    frames_path = extract_video_frames(video_path, 'extracted-frames')

    # Analise do audio extraido
    SER = SpeechEmotionRecognition('models/ser/best_model.hdf5', 'models/ser/model-weights.h5')

    emotions, timestamp = SER.predict_emotion_from_file(audio_path, predict_proba=True, chunk_step=step*sample_rate)


    emotion = np.argmax(emotions)

    predictions_from_audio_float = {
        'angry': emotions[0][0],
        'disgust': emotions[0][1],
        'fear': emotions[0][2],
        'happy': emotions[0][3],
        'neutral': emotions[0][4],
        'sad': emotions[0][5],
        'surprise': emotions[0][6]
    }

    predictions_from_audio = {
        'angry': "{:.2%}".format(emotions[0][0]),
        'disgust': "{:.2%}".format(emotions[0][1]),
        'fear': "{:.2%}".format(emotions[0][2]),
        'happy': "{:.2%}".format(emotions[0][3]),
        'neutral': "{:.2%}".format(emotions[0][4]),
        'sad': "{:.2%}".format(emotions[0][5]),
        'surprise': "{:.2%}".format(emotions[0][6])
    }

    print(predictions_from_audio)

    # Análise dos frames extraídos
    with open('models/model1-fer.json', 'r') as file:
        json = file.read()
    model = model_from_json(json)
    # model = create_model1()
    model.load_weights('models/model1-fer-weights.h5')

    # Arrays pra salvar as probabilidades de cada emoção por frame analisado
    angry, disgust, fear, happy, sad, surprise, neutral = [], [], [], [], [], [], []

    for frame in os.listdir(frames_path):
        if frame not in ['.gitkeep']:
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

    predictions_from_frames_float = {
        'angry': sum(angry) / len(angry),
        'disgust': sum(disgust) / len(disgust),
        'fear': sum(fear) / len(fear),
        'happy': sum(happy) / len(happy),
        'neutral': sum(neutral) / len(neutral),
        'sad': sum(sad) / len(sad),
        'surprise': sum(surprise) / len(surprise)
    }

    predictions_with_sum = {
        'angry': predictions_from_frames_float['angry'] + predictions_from_audio_float['angry'],
        'disgust': predictions_from_frames_float['disgust'] + predictions_from_audio_float['disgust'],
        'fear': predictions_from_frames_float['fear'] + predictions_from_audio_float['fear'],
        'happy': predictions_from_frames_float['happy'] + predictions_from_audio_float['happy'],
        'neutral': predictions_from_frames_float['neutral'] + predictions_from_audio_float['neutral'],
        'sad': predictions_from_frames_float['sad'] + predictions_from_audio_float['sad'],
        'surprise': predictions_from_frames_float['surprise'] + predictions_from_audio_float['surprise'],
    }

    predictions_with_product = {
        'angry': predictions_from_frames_float['angry'] * predictions_from_audio_float['angry'],
        'disgust': predictions_from_frames_float['disgust'] * predictions_from_audio_float['disgust'],
        'fear': predictions_from_frames_float['fear'] * predictions_from_audio_float['fear'],
        'happy': predictions_from_frames_float['happy'] * predictions_from_audio_float['happy'],
        'neutral': predictions_from_frames_float['neutral'] * predictions_from_audio_float['neutral'],
        'sad': predictions_from_frames_float['sad'] * predictions_from_audio_float['sad'],
        'surprise': predictions_from_frames_float['surprise'] * predictions_from_audio_float['surprise'],
    }

    emotions_ptbr = {
        'angry': 'Raiva',
        'disgust': 'Desgosto',
        'fear': 'Medo',
        'happy': 'Feliz',
        'neutral': 'Calmo',
        'sad': 'Triste',
        'surprise': 'Surpresa'
    }

    os.remove(video_path)
    os.remove(audio_path)

    emotion_sum = emotions_ptbr[max(predictions_with_sum.items(), key=operator.itemgetter(1))[0]]
    emotion_product = emotions_ptbr[max(predictions_with_product.items(), key=operator.itemgetter(1))[0]]

    prediction_results = {
        'predictions_from_audio': predictions_from_audio,
        'predictions_from_frames': predictions_from_frames,
        'predictions_with_sum': predictions_with_sum,
        'predictions_with_product': predictions_with_product,
        'emotion_sum': emotion_sum,
        'emotion_product': emotion_product,
        'filename': filename,
    }

    # return prediction_results

    return render_template('report.html', data=prediction_results)


@app.route('/report', methods=['GET'])
def report():
    return render_template('report.html')


from tensorflow.keras.models import Model as ModelTF2
from tensorflow.keras.layers import Input as InputTF2
from tensorflow.keras.layers import Dropout as DropoutTF2
from tensorflow.keras.layers import Conv2D as Conv2DTF2
from tensorflow.keras.layers import MaxPooling2D as MaxPooling2DTF2
from tensorflow.keras.layers import BatchNormalization as BatchNormalizationTF2
from tensorflow.keras.layers import Flatten as FlattenTF2
from tensorflow.keras.layers import Dense as DenseTF2
from tensorflow.keras.regularizers import l2

def create_model1():
    num_features = 64
    width = height = 48
    input_shape = InputTF2(shape=(48, 48, 1))

    x = Conv2DTF2(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01))(input_shape)
    x = Conv2DTF2(num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = Conv2DTF2(2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = Conv2DTF2(2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = Conv2DTF2(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = Conv2DTF2(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = Conv2DTF2(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = Conv2DTF2(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalizationTF2()(x)
    x = MaxPooling2DTF2(pool_size=(2, 2), strides=(2, 2))(x)
    x = DropoutTF2(0.5)(x)

    x = FlattenTF2()(x)

    x = DenseTF2(2*2*2*num_features, activation='relu')(x)
    x = DropoutTF2(0.4)(x)
    x = DenseTF2(2*2*num_features, activation='relu')(x)
    x = DropoutTF2(0.4)(x)
    x = DenseTF2(2*num_features, activation='relu')(x)
    x = DropoutTF2(0.5)(x)

    x = DenseTF2(7, activation='softmax')(x)

    return ModelTF2(inputs=input_shape, outputs=x)



if __name__ == "__main__":
    app.run()
