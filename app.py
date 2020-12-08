from flask import redirect, url_for, session, send_file
import scipy.ndimage as ndimage
from werkzeug.utils import secure_filename
import numpy as np
from flask import Flask, flash, render_template, request
from keras.models import load_model
import io
import os
import tensorflow as tf
from vis.visualization import visualize_saliency
from vis.utils import utils
import matplotlib.pyplot as plt
from keras import activations
from keras.preprocessing import image
graph = tf.get_default_graph()
plt.switch_backend('Agg')

def get_saliency_map(model,img_path):
    with graph.as_default():
        layer_index = utils.find_layer_idx(model, 'visualized_layer')
        model.layers[layer_index].activation = activations.linear
        model = utils.apply_modifications(model)
        test_image1 = image.load_img(img_path, target_size=(128, 128))
        test_image = image.img_to_array(test_image1)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        fig, axes = plt.subplots(1, 2)
        # Generate visualization
        visualization = visualize_saliency(model, layer_index, filter_indices=None, seed_input=test_image,
                                           backprop_modifier=None, grad_modifier="absolute")
        plt.imshow(visualization)
        gaus = ndimage.gaussian_filter(visualization, sigma=5)
        plt.imshow(test_image1)
        plt.imshow(gaus, alpha=.7)
        axes[0].imshow(test_image1)
        axes[0].set_title('Original image')
        axes[1].set_title('Saliency map')
        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format='png')
        bytes_image.seek(0)
        return bytes_image
    # plt.show()

def return_prediction(model, img_path):
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    test_image = image.load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    with graph.as_default():
        prediction = model.predict(test_image)
        return prediction[0][1]


app = Flask(__name__)
# Configure a secret SECRET_KEY
app.config["SECRET_KEY"] = "1234abcdege"
# Loading the model and scaler
tb_model = load_model('resnet_model.h5')
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        name = request.form['Name']
        if request.form.get('cough'):
            cough = request.form.get('cough')
            session['cough'] = cough
        if request.form.get('fever'):
            fever = request.form.get('fever')
            session['fever'] = fever
        if request.form.get('weightloss'):
            weightloss = request.form.get('weightloss')
            session['weightloss']  = weightloss
        if request.form.get('chestpain'):
            chestpain = request.form.get('chestpain')
            session['chestpain'] = chestpain
        session['name'] = name

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(loc)
            session['filename'] = filename
            return redirect(url_for('uploaded_file'))
    return render_template("home.html")

@app.route('/uploaded')
def uploaded_file():
    # return send_from_directory(app.config['UPLOAD_FOLDER'],
    #                            filename)
    # loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    filename = session['filename']
    loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    tb_prob = return_prediction(tb_model, loc)
    name = session['name']
    return render_template("prediction.html", tb_prob=tb_prob, name=name,filename=filename)


@app.route('/plot/saliencymap', methods=['GET'])
def saliency_map():
    filename = session['filename']
    loc = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    bytes_obj = get_saliency_map(tb_model, loc)
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
