#flask applicaition
from flask import Flask, jsonify, request
from training.models import train_and_evaluate_model
import os
import joblib
import numpy as np

app = Flask(__name__)
# model file is stored under model/models/linear_model.pkl relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'linear_model.pkl')

@app.route('/train', methods=['POST'])
def train():
    try:
        train_and_evaluate_model()
        return jsonify({"message": "Model trained and saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}

        # If client supplied raw feature vector, use it directly
        if 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
        else:
            # Accept either combined `date` in YYYYMMDD (int or string), or separate year/month/day
            if 'date' in data:
                # allow int or string like 20240119
                date_raw = str(data['date'])
                # normalize to YYYYMMDD integer
                try:
                    date_int = int(date_raw)
                except Exception:
                    raise ValueError('Invalid `date` format. Use YYYYMMDD (e.g. 20240119) or provide `year`, `month`, `day`.')
            else:
                y = int(data.get('year'))
                m = int(data.get('month'))
                d = int(data.get('day'))
                date_int = y * 10000 + m * 100 + d

            # required numeric fields
            try:
                volume = float(data['volume'])
                open_p = float(data['open'])
                high_p = float(data['high'])
                low_p = float(data['low'])
            except KeyError as ke:
                raise KeyError(f'Missing required field: {ke.args[0]}')
            except Exception:
                raise ValueError('Numeric fields `volume`, `open`, `high`, `low` must be numeric')

            # Build features in the order expected by the model: [date_int, volume, open, high, low]
            features = np.array([date_int, volume, open_p, high_p, low_p]).reshape(1, -1)

        # load model and predict
        model = joblib.load(MODEL_PATH)
        prediction = model.predict(features)

        # return scalar when possible
        pred_value = prediction.tolist()
        if isinstance(pred_value, list) and len(pred_value) == 1:
            pred_value = pred_value[0]

        return jsonify({"prediction": pred_value}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500  
        
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)