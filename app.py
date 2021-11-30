import os
from flask_ngrok import run_with_ngrok
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import load
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# load the dataset
data = load('/content/drive/MyDrive/Colab Notebooks/van_gogh_and_faces.npz')
dataA, dataB = data['arr_0'], data['arr_1']

# This generator applies style transfer
van_gogh_model_AtoB_path = "/content/drive/MyDrive/Colab Notebooks/van_gogh_g_model_AtoB_002007.h5"
van_gogh_model_BtoA_path = "/content/drive/MyDrive/Colab Notebooks/van_gogh_g_model_BtoA_002007.h5"
monet_model_AtoB_path = "/content/drive/MyDrive/Colab Notebooks/monet_g_model_AtoB_002000.h5"
monet_model_BtoA_path = "/content/drive/MyDrive/Colab Notebooks/monet_g_model_BtoA_002000.h5"

# define custome layer to load in model
layer = InstanceNormalization(axis=-1)

# Load in art transfer model
van_gogh_art_transfer_model_art_to_real = load_model(van_gogh_model_AtoB_path, custom_objects={"InstanceNormalization": layer}, compile=False)
van_gogh_art_transfer_model_real_to_art = load_model(van_gogh_model_BtoA_path, custom_objects={"InstanceNormalization": layer}, compile=False)
monet_art_transfer_model_art_to_real = load_model(monet_model_AtoB_path, custom_objects={"InstanceNormalization": layer}, compile=False)
monet_art_transfer_model_real_to_art = load_model(monet_model_BtoA_path, custom_objects={"InstanceNormalization": layer}, compile=False)


app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def apply_art_transfer(model, filename):
  img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  # Process image
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (256, 256))

  # apply art transfer to image
  X = (model.predict(np.expand_dims(img, axis=0)) + 1) / 2.0 # unnormalize


  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
  # Plot original image
  ax1.imshow(img.astype('uint8'))
  ax1.axis('off')

  # Plot art transfer image
  ax2.imshow(np.squeeze(X))
  ax2.axis('off')

  fig.savefig(os.path.join(app.config['UPLOAD_FOLDER'], filename))




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return '''
    <!doctype html>
    <title>Art by GANs Inc</title>
    <h1>Art by GANs Inc</h1>
    <a rel="nofollow" target="_blank", href="https://github.com/rmoreira/csce5215-colab-flaskapp">Link to download images</a><br><br>
    <a rel="nofollow" target="_blank", href="/van_gogh_art_to_real">van_gogh_art_to_real</a><br>
    <a rel="nofollow" target="_blank", href="/van_gogh_real_to_art">van_gogh_real_to_art</a><br>
    <a rel="nofollow" target="_blank", href="/monet_art_to_real">monet_art_to_real</a><br>
    <a rel="nofollow" target="_blank", href="/monet_real_to_art">monet_real_to_art</a><br>
    '''

@app.route('/van_gogh_art_to_real', methods=['GET', 'POST'])
def van_gogh_art_to_real():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            apply_art_transfer(van_gogh_art_transfer_model_art_to_real, filename)
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>/van_gogh_art_to_real - Upload new File</title>
    <h1>/van_gogh_art_to_real - Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/van_gogh_real_to_art', methods=['GET', 'POST'])
def van_gogh_real_to_art():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            apply_art_transfer(van_gogh_art_transfer_model_real_to_art, filename)
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>/van_gogh_real_to_art - Upload new File</title>
    <h1>/van_gogh_real_to_art - Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/monet_art_to_real', methods=['GET', 'POST'])
def monet_art_to_real():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            apply_art_transfer(monet_art_transfer_model_art_to_real, filename)
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>/monet_art_to_real - Upload new File</title>
    <h1>/monet_art_to_real - Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/monet_real_to_art', methods=['GET', 'POST'])
def monet_real_to_art():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            apply_art_transfer(monet_art_transfer_model_real_to_art, filename)
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>/monet_real_to_art - Upload new File</title>
    <h1>/monet_real_to_art - Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)



if __name__ == '__main__':
   app.run()

