from flask import Flask, request, jsonify
import h5py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
import json
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Full file paths for the models and feature columns
metadata_model_path = r"C:\Users\Will\Documents\models\SkinSense\skin_cancer_metadata_model.h5"
image_model_path = r"C:\Users\Will\Documents\models\SkinSense\skin_cancer_ResNet50Xfer.h5"
feature_columns_path = r"C:\Users\Will\Documents\models\SkinSense\feature_columns.json"

# Create a temporary directory for uploaded files
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels for predictions
CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def load_metadata_model():
    # Open the HDF5 file and read the stored model array
    with h5py.File(metadata_model_path, "r") as hf:
        model_array = hf["model"][()]
    # Convert the NumPy array back into bytes and unpickle to load the model
    metadata_model = pickle.loads(model_array.tobytes())
    return metadata_model

def load_image_model():
    # Load the image model (ResNet50 transfer learning model) saved as a Keras model
    image_model = tf.keras.models.load_model(image_model_path)
    return image_model

# Load feature columns for metadata model preprocessing
with open(feature_columns_path, 'r') as f:
    FEATURE_COLUMNS = json.load(f)
logger.info(f"Feature columns loaded: {FEATURE_COLUMNS[:5]} ...")  # Log first 5 for brevity

# Load both models at startup
try:
    metadata_model = load_metadata_model()
    image_model = load_image_model()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

@app.route('/metadata_model_info', methods=['GET'])
def metadata_model_info():
    try:
        expected_features = metadata_model.n_features_in_ if hasattr(metadata_model, 'n_features_in_') else None
        return jsonify({
            'expected_features': int(expected_features) if expected_features is not None else None,
            'model_type': str(type(metadata_model).__name__),
            'class_labels': CLASS_LABELS,
            'feature_columns': FEATURE_COLUMNS
        })
    except Exception as e:
        logger.error(f"Error getting metadata model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_metadata', methods=['POST'])
def predict_metadata():
    try:
        data = request.get_json()
        logger.info(f"Received metadata prediction request: {data}")
        
        if 'features' not in data or not isinstance(data['features'], dict):
            return jsonify({'error': 'Missing or invalid features in request; expected a dictionary'}), 400
        
        raw_df = pd.DataFrame([data['features']])
        logger.info(f"Raw input DataFrame:\n{raw_df}")
        
        processed_df = pd.get_dummies(raw_df, columns=['sex', 'localization'], drop_first=True)
        logger.info(f"DataFrame after one-hot encoding:\n{processed_df}")
        
        processed_df = processed_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        logger.info(f"Final features DataFrame shape: {processed_df.shape}")
        
        features = processed_df.values.reshape(1, -1)
        
        expected_features = metadata_model.n_features_in_ if hasattr(metadata_model, 'n_features_in_') else None
        if expected_features and features.shape[1] != expected_features:
            logger.error(f"Feature count mismatch. Expected {expected_features}, got {features.shape[1]}")
            return jsonify({
                'error': f'Feature count mismatch. Expected {expected_features}, got {features.shape[1]}',
                'expected_features': int(expected_features)
            }), 400
        
        prediction = metadata_model.predict(features)
        # prediction[0] is expected to be a string like "nv"
        predicted_class = prediction[0]
        try:
            predicted_class_index = CLASS_LABELS.index(predicted_class)
        except ValueError:
            predicted_class_index = -1
        
        logger.info(f"Metadata prediction: {predicted_class} (index: {predicted_class_index})")
        
        return jsonify({
            'prediction': predicted_class_index,
            'class': predicted_class
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in metadata prediction: {e}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get the actual diagnosis if provided (optional)
        actual_diagnosis = request.form.get('actual_diagnosis', None)
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        
        # Process the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        
        # Make prediction
        preds = image_model.predict(img_array)
        predicted_class_index = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds) * 100)
        predicted_class = CLASS_LABELS[predicted_class_index] if 0 <= predicted_class_index < len(CLASS_LABELS) else "unknown"
        
        logger.info(f"Image prediction: {predicted_class_index} ({predicted_class}) with confidence {confidence:.2f}%")
        
        is_correct = None
        if actual_diagnosis:
            actual_diagnosis = actual_diagnosis.lower()
            is_correct = (actual_diagnosis == predicted_class.lower())
            logger.info(f"Prediction correctness: {is_correct} (Actual: {actual_diagnosis}, Predicted: {predicted_class.lower()})")
        
        # Clean up temporary file
        os.remove(filepath)
        
        response = {
            'prediction': predicted_class_index,
            'class': predicted_class,
            'confidence': confidence
        }
        if actual_diagnosis:
            response.update({
                'actual_diagnosis': actual_diagnosis,
                'is_correct': is_correct
            })
        return jsonify(response)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in image prediction: {e}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/predict_fusion', methods=['POST'])
def predict_fusion():
    try:
        # --- Process the image input ---
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file for image'}), 400
        
        # Save the file temporarily
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        logger.info(f"Processing fusion image: {filename}")
        
        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224)) # 224 * 224 size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        
        # Obtain image probabilities from the image model
        image_probs = image_model.predict(img_array)  # shape: (1, n_classes)
        
        # --- Process the metadata input ---
        metadata_json = request.form.get('metadata', None)
        if not metadata_json:
            return jsonify({'error': 'No metadata provided'}), 400
        
        try:
            metadata_input = json.loads(metadata_json)
        except Exception as e:
            return jsonify({'error': 'Invalid metadata JSON', 'details': str(e)}), 400
        
        if not isinstance(metadata_input, dict):
            return jsonify({'error': 'Metadata must be a JSON object (dictionary)'}), 400
        
        raw_df = pd.DataFrame([metadata_input])
        processed_df = pd.get_dummies(raw_df, columns=['sex', 'localization'], drop_first=True)
        processed_df = processed_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        features = processed_df.values.reshape(1, -1)
        
        expected_features = metadata_model.n_features_in_ if hasattr(metadata_model, 'n_features_in_') else None
        if expected_features and features.shape[1] != expected_features:
            logger.error(f"Metadata feature count mismatch. Expected {expected_features}, got {features.shape[1]}")
            return jsonify({
                'error': f'Feature count mismatch. Expected {expected_features}, got {features.shape[1]}'
            }), 400
        
        # Obtain metadata probabilities. Assumes predict_proba method exists.
        try:
            metadata_probs = metadata_model.predict_proba(features)  # shape: (1, n_classes)
        except Exception as e:
            logger.warning("predict_proba not available on metadata model, using one-hot encoding of predict() result.")
            prediction = metadata_model.predict(features)
            metadata_probs = np.zeros((1, len(CLASS_LABELS)))
            try:
                idx = CLASS_LABELS.index(prediction[0])
                metadata_probs[0, idx] = 1.0
            except ValueError:
                pass
        
        # --- Late Fusion: Combine predictions ---
        weight_image = 0.6
        weight_metadata = 0.4
        final_probs = weight_image * image_probs + weight_metadata * metadata_probs
        
        final_prediction_index = int(np.argmax(final_probs, axis=1)[0])
        final_class = CLASS_LABELS[final_prediction_index] if 0 <= final_prediction_index < len(CLASS_LABELS) else "unknown"
        
        # Clean up temporary image file
        os.remove(filepath)
        
        logger.info(f"Fusion prediction: {final_prediction_index} ({final_class})")
        
        return jsonify({
            'fusion_prediction': final_prediction_index,
            'class': final_class,
            'image_probs': image_probs.tolist(),
            'metadata_probs': metadata_probs.tolist(),
            'final_probs': final_probs.tolist()
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error in fusion prediction: {e}\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'metadata_model_type': str(type(metadata_model).__name__),
        'image_model_type': str(type(image_model).__name__)
    })

if __name__ == '__main__':
    logger.info("Starting Flask API server")
    app.run(host='0.0.0.0', port=5001, debug=True)
