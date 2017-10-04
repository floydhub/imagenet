"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Look for a Image and then process it to be ImageNet compliant
    - Returns the evaluation

POST req:
    parameter:
        - file, required, a image to classify

"""
import os
import torch
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from imagenet_models import ConvNet

ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])

MODEL_PATH = '/model'
print('Loading model from path: %s' % MODEL_PATH)

EVAL_PATH = '/eval'
TRAIN_PATH = '/input/train'
MODEL = "resnet18"

# Is there the EVAL_PATH?
try:
    os.makedirs(EVAL_PATH)
except OSError:
    pass

app = Flask('ImageNet-Classifier')

# Build the model before to improve performance
checkpoint = os.path.join(MODEL_PATH, "model_best.pth.tar") # FIX to
Model = ConvNet(ckp=checkpoint, train_dir=TRAIN_PATH, arch=MODEL)
Model.build_model()

# Return an Image
@app.route('/<path:path>', methods=['POST'])
def geneator_handler(path):
    """Upload an image file, then
    preprocess and classify"""
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    filename = secure_filename(file.filename)
    image_folder = os.path.join(EVAL_PATH, "images")
    # Create dir /eval/images
    try:
        os.makedirs(image_folder)
    except OSError:
        pass
    # Save Image to process
    input_filepath = os.path.join(image_folder, filename)
    file.save(input_filepath)
    # Preprocess and Evaluate
    Model.image_preprocessing()
    pred = Model.classify()
    # Return classification and remove uploaded file
    output = "Images: {file}, Classified as {pred}\n".format(file=file.filename,
        pred=pred)
    os.remove(input_filepath)
    return output


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
