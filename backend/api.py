from classifier import Classifier
from flask import Flask, request, jsonify

app = Flask(__name__)
classifier = Classifier()

@app.route('/set_params', methods=['POST'])
def set_params():
    try:
        params = request.json
        classifier.set_params(**params)
        return 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Wystąpił błąd serwera"}), 500

if __name__ == '__main__':
    app.run(debug=True)