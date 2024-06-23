from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model (ensure you have the trained model file in the same directory)
prediction_model = tf.keras.models.load_model('prediction_model.h5')

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    if w < 32:
        add_zeros = np.full((32 - w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128 - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    img = img / 255

    return img

def levenshtein_distance(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    matrix = [[0 for _ in range(len_str2)] for _ in range(len_str1)]
    for i in range(len_str1):
        matrix[i][0] = i
    for j in range(len_str2):
        matrix[0][j] = j

    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,  
                               matrix[i][j - 1] + 1,  
                               matrix[i - 1][j - 1] + cost) 

    return matrix[len_str1 - 1][len_str2 - 1]

def similarity_score(text1, text2):
    lev_distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 100.0
    score = (1 - lev_distance / max_len) * 100
    return round(score)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Read and process the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return jsonify({'error': 'Error reading image'})

        processed_image = process_image(image)
        
        
        # Make a prediction
        prediction = prediction_model.predict(processed_image.reshape(1, 32, 128, 1))
        
        
        decoded = K.ctc_decode(prediction,
                               input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0]
        out = K.get_value(decoded)
    

        predicted_text = ''
        for p in out[0]:
            if int(p) != -1:
                predicted_text += char_list[int(p)]
        
        print(f"Predicted text: {predicted_text}")

        # Example reference text (Jawaban)
        # In practice, this can be received via another form field or parameter
        reference_text = request.form.get('jawaban', '')

        if not reference_text:
            return jsonify({'error': 'No reference text provided'})

        # Calculate similarity score
        similarity = similarity_score(reference_text, predicted_text)
        
        return jsonify({
            "Total Nilai Jawaban": similarity, 
            "Kunci Jawaban": reference_text, 
            "Hasil Ocr": predicted_text
        }), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000)
