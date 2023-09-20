from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='../frontend/build')
CORS(app,resources={r"/*":{"origins":"*"}})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


def run_app(port=5001, debug=False):
    app.run(port=port, debug=debug)    

if __name__ == '__main__':
    run_app(debug=True)
