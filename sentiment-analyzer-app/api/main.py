import os
from flask import Flask, request, send_from_directory
from flask_cors import CORS, cross_origin
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np


app = Flask(__name__)
cors = CORS(app)

MODEL_FOLDER = 'models/'
UPLOAD_FOLDER = 'uploads/'

@app.before_first_request
def load__model():
    """
    Load model
    :return: model (global variable)
    """
    print('[INFO] Model Loading ........')
    print(os.system("ls"))
    global model
    model = load_model(MODEL_FOLDER + 'best_model.h5')
    print('[INFO] : Model loaded')


def predict(fullpath):
    data = image.load_img(fullpath, target_size=(150, 150, 3))
    # (150,150,3) ==> (1,150,150,3)
    data = np.expand_dims(data, axis=0)
    # Scaling
    data = data.astype('float') / 255

    # Prediction:
    result = model.predict(data)

    return result

# Home Page
@app.route('/predict', methods=['POST', 'GET'])
def index():

    if request.method == 'POST':
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)

        pred_prob = result.item()

        label = ""
        accuracy = 0

        if pred_prob > .5:
            label = 'Recyclable'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Not Recyclable'
            accuracy = round((1 - pred_prob) * 100, 2)
        
        return label
    
    else:
        return "0"   
    


if __name__ == '__main__':
    PORT = os.environ['PORT'] or "3000"
    print(PORT)
    app.run(host="0.0.0.0", port=PORT, debug=True)
