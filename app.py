import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained models
models = {
    'Model 1': tf.keras.models.load_model('models/xception_medicinal_plants_batch_1.h5'),
    'Model 2': tf.keras.models.load_model('models/xception_medicinal_plants_batch_2.h5'),
    'Model 3': tf.keras.models.load_model('models/xception_medicinal_plants_batch_3.h5'),
    'Model 4': tf.keras.models.load_model('models/xception_medicinal_plants_batch_4.h5'),
    'Model 5': tf.keras.models.load_model('models/xception_medicinal_plants_batch_5.h5')
}

# Define the class mappings for each model
class_mappings = {
    'Model 1': {
        0: {"name": "Abelmoschus sagittifolius", "benefits": ["Treats inflammation", "Relieves pain", "Improves digestion"]},
        1: {"name": "Abrus precatorius", "benefits": ["Helps with cough", "Aids in skin conditions", "Boosts immunity"]},
        2: {"name": "Abutilon indicum", "benefits": ["Acts as a diuretic", "Promotes wound healing", "Reduces fever"]},
        3: {"name": "Acanthus integrifolius", "benefits": ["Used for healing wounds", "Anti-inflammatory", "Treats gastrointestinal issues"]},
        4: {"name": "Acorus tatarinowii", "benefits": ["Helps with digestion", "Reduces stress", "Promotes blood circulation"]},
        5: {"name": "Agave americana", "benefits": ["Aids in digestion", "Promotes skin healing", "Reduces inflammation"]},
        6: {"name": "Ageratum conyzoides", "benefits": ["Reduces fever", "Treats wounds", "Acts as an antimicrobial"]},
        7: {"name": "Allium ramosum", "benefits": ["Reduces blood pressure", "Improves heart health", "Boosts immunity"]},
        8: {"name": "Alocasia macrorrhizos", "benefits": ["Reduces fever", "Treats skin diseases", "Improves appetite"]},
        9: {"name": "Aloe vera", "benefits": ["Heals burns", "Moisturizes skin", "Boosts digestive health"]}
    },
    'Model 2': {
        0: {"name": "Alpinia officinarum", "benefits": ["Reduces nausea", "Helps with digestion", "Treats respiratory problems"]},
        1: {"name": "Amomum longiligulare", "benefits": ["Aids digestion", "Boosts immunity", "Treats nausea"]},
        2: {"name": "Ampelopsis cantoniensis", "benefits": ["Improves circulation", "Reduces inflammation", "Helps with joint pain"]},
        3: {"name": "Andrographis paniculata", "benefits": ["Boosts immune system", "Fights infections", "Reduces fever"]},
        4: {"name": "Angelica dahurica", "benefits": ["Helps with cough", "Relieves pain", "Improves blood circulation"]},
        5: {"name": "Ardisia sylvestris", "benefits": ["Improves skin health", "Relieves fatigue", "Boosts immunity"]},
        6: {"name": "Artemisia vulgaris", "benefits": ["Treats digestive disorders", "Relieves menstrual cramps", "Acts as an antimicrobial"]},
        7: {"name": "Artocarpus altilis", "benefits": ["Boosts immune system", "Helps in wound healing", "Improves digestion"]},
        8: {"name": "Artocarpus heterophyllus", "benefits": ["Aids in digestion", "Improves skin health", "Boosts immunity"]},
        9: {"name": "Artocarpus lakoocha", "benefits": ["Improves blood circulation", "Relieves fever", "Reduces inflammation"]}
    },
    'Model 3': {
        0: {"name": "Asparagus cochinchinensis", "benefits": ["Reduces cholesterol", "Improves digestion", "Promotes kidney health"]},
        1: {"name": "Asparagus officinalis", "benefits": ["Supports heart health", "Rich in antioxidants", "Boosts immunity"]},
        2: {"name": "Averrhoa carambola", "benefits": ["Promotes liver health", "Rich in vitamin C", "Helps reduce high blood pressure"]},
        3: {"name": "Baccaurea sp", "benefits": ["Anti-inflammatory", "Aids digestion", "Treats respiratory issues"]},
        4: {"name": "Barleria lupulina", "benefits": ["Relieves pain", "Improves skin health", "Boosts immunity"]},
        5: {"name": "Bengal Arum", "benefits": ["Treats respiratory problems", "Reduces fever", "Improves digestive health"]},
        6: {"name": "Berchemia lineata", "benefits": ["Aids in wound healing", "Treats diarrhea", "Reduces inflammation"]},
        7: {"name": "Bidens pilosa", "benefits": ["Treats fever", "Boosts immune function", "Acts as an anti-inflammatory"]},
        8: {"name": "Bischofia trifoliata", "benefits": ["Improves blood circulation", "Relieves pain", "Reduces fever"]},
        9: {"name": "Blackberry Lily", "benefits": ["Boosts immune system", "Treats digestive issues", "Improves skin health"]}
    },
    'Model 4': {
        0: {"name": "Blumea balsamifera", "benefits": ["Reduces fever", "Promotes liver health", "Treats headaches"]},
        1: {"name": "Boehmeria nivea", "benefits": ["Treats inflammation", "Relieves pain", "Promotes wound healing"]},
        2: {"name": "Breynia vitis", "benefits": ["Improves digestion", "Reduces fever", "Treats skin conditions"]},
        3: {"name": "Caesalpinia sappan", "benefits": ["Improves circulation", "Acts as an antimicrobial", "Helps reduce blood sugar"]},
        4: {"name": "Callerya speciosa", "benefits": ["Boosts immune system", "Promotes digestion", "Reduces inflammation"]},
        5: {"name": "Callisia fragrans", "benefits": ["Relieves pain", "Helps with respiratory problems", "Improves skin health"]},
        6: {"name": "Calophyllum inophyllum", "benefits": ["Reduces inflammation", "Aids in wound healing", "Improves skin health"]},
        7: {"name": "Calotropis gigantea", "benefits": ["Reduces pain", "Acts as a natural diuretic", "Treats respiratory issues"]},
        8: {"name": "Camellia chrysantha", "benefits": ["Rich in antioxidants", "Promotes heart health", "Aids digestion"]},
        9: {"name": "Caprifoliaceae", "benefits": ["Treats colds", "Reduces inflammation", "Boosts immune system"]}
    },
    'Model 5': {
        0: {"name": "Capsicum annuum", "benefits": ["Reduces pain", "Boosts metabolism", "Rich in vitamins"]},
        1: {"name": "Carica papaya", "benefits": ["Improves digestion", "Reduces inflammation", "Boosts skin health"]},
        2: {"name": "Catharanthus roseus", "benefits": ["Treats diabetes", "Improves blood circulation", "Boosts immunity"]},
        3: {"name": "Celastrus hindsii", "benefits": ["Helps with arthritis", "Improves memory", "Reduces stress"]},
        4: {"name": "Celosia argentea", "benefits": ["Rich in vitamins", "Improves digestion", "Aids in wound healing"]},
        5: {"name": "Centella asiatica", "benefits": ["Boosts cognitive function", "Aids in wound healing", "Improves circulation"]},
        6: {"name": "Citrus aurantifolia", "benefits": ["Boosts immune system", "Aids digestion", "Rich in antioxidants"]},
        7: {"name": "Citrus hystrix", "benefits": ["Helps with digestion", "Relieves nausea", "Reduces inflammation"]},
        8: {"name": "Clausena indica", "benefits": ["Relieves fever", "Promotes digestion", "Reduces inflammation"]},
        9: {"name": "Cleistocalyx operculatus", "benefits": ["Boosts immune system", "Reduces pain", "Improves digestion"]}
    }
}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to load and preprocess the image
def load_image(img_path, image_size=(299, 299)):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files or 'model' not in request.form:
            return redirect(request.url)

        file = request.files['file']
        selected_model = request.form['model']

        if file.filename == '' or selected_model not in models:
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            model = models[selected_model]
            class_mapping = class_mappings[selected_model]
            img_array = load_image(file_path)
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds, axis=1)[0]
            confidence = preds[0][predicted_class]

            predicted_label = class_mapping[predicted_class]["name"]
            benefits = class_mapping[predicted_class]["benefits"]

            return render_template(
                'index.html',
                filename=filename,
                prediction=predicted_label,
                confidence=f"{confidence * 100:.2f}%",
                model_name=selected_model,
                benefits=benefits
            )
    
    return render_template('index.html', models=models.keys())

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
