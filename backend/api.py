from classifier import Classifier
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/set_params', methods=['POST'])
def set_params():
    try:
        params = request.json
        return jsonify(Classifier(**params).api_model.to_dict()), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Wystąpił błąd serwera {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)