'''import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('BrainTumor.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor Detected :)"
	elif classNo==1:
		return "Brain Tumor Detected :("


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Limit TensorFlow memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the trained model
model = load_model('BrainTumor.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


# Function to classify the result
def get_class_name(class_no):
    return "No Brain Tumor Detected :)" if class_no == 0 else "Brain Tumor Detected :("


# Function to preprocess the image and make a prediction
def get_result(img_path):
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((64, 64))  # Resize to match model input
        img_array = np.array(img) / 255.0  # Normalize

        # Expand dimensions to match model input shape
        input_img = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(input_img)
        class_no = int(prediction[0][0] > 0.5)  # Convert to binary class
        return get_class_name(class_no)

    except Exception as e:
        return f"Error in processing: {str(e)}"


# Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Secure and save the uploaded file
    file_path = os.path.join("uploads", secure_filename(f.filename))
    os.makedirs("uploads", exist_ok=True)  # Ensure directory exists
    f.save(file_path)

    # Get prediction result
    result = get_result(file_path)
    return jsonify({"prediction": result})


if __name__ == '__main__':
    app.run(debug=True)
