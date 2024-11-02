from classifier import Classifier
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/set_params', methods=['POST'])
def set_params():
    try:
        params = request.json
        print(f'Run with params {params}')
        return jsonify(Classifier(**params).api_model.to_dict()), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Wystąpił błąd serwera {str(e)}"}), 500

@app.route('/images/<path:filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(os.getcwd(), filename)
    except Exception as e:
        return jsonify({"error": f"Image not found: {str(e)}"}), 404