from flask import Flask, request, jsonify
import json
import numpy as np
from sklearn.covariance import fast_mcd


app = Flask(__name__)

# Create a JSON Encoder class
class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/detect_outliers', methods=['POST'])
def detect_outliers_api():
    try:
        data = json.loads(request.get_data())
        if not data or not data['features']:
            return jsonify({'error': 'Invalid JSON input. Expected "features" key with data points.'}), 400
	
        data_points = np.array(data['features']) # Assume 'features' is a list of lists (data points)
        if data_points.ndim == 1: # Handle single data point case
            data_points = data_points.reshape(1, -1)
        print(data_points)
        support_mask = fast_mcd(data_points, support_fraction=0.9)
        outlier_indices = np.where(~support_mask[2]) # Convert to list for JSON

        results = {
            'outlier_indices': list(outlier_indices),
            'is_outlier': ~support_mask[2] # Boolean list of outlier status for each input point
        }
        return json.dumps(results, cls=json_serialize), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # Accessible on network (adjust host/port as needed)