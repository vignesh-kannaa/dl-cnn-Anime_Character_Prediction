
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os


app = Flask(__name__)
model=load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    imagefile= request.files['imagefile'];
    image_path='./static/'+imagefile.filename;
    imagefile.save(image_path)
    anime_characters=['Brook', 'Chopper']
    test_image = tf.keras.preprocessing.image.load_img(image_path, target_size = (256, 256))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)
    print('prediction value:',prediction)
    character=anime_characters[np.argmax(prediction)]
    print(character)
    # os.remove(image_path)
    return render_template('index.html',prediction=character,image=image_path)

# python -m flask run