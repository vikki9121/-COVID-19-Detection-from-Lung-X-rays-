import re
import numpy as np
import os

from flask import Flask, render_template, request, app, url_for
from keras import models
from keras.models import load_model
from keras.utils import load_img, img_to_array
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input
import requests

model = load_model("../IBM/InceptionV3-covid.h5")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def res():
    if request.method == "POST":
        f = request.files['file']
        name = request.values['pname']
        basepath = os.path.dirname(__file__)
        print(basepath)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        print(filepath)
        f.save(filepath)
        img = load_img(filepath, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        prediction = np.argmax(model.predict(img_data), axis=1)
        index = ['COVID', 'Lung_Capacity', 'Normal', 'Viral Pneumonia']
        result = str(index[prediction[0]])
        print(result)
        return render_template('predict.html', name=name, prediction=result)


if __name__ == '__main__':
    app.run(debug=True)
