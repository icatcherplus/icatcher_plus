from flask import Flask, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

REACT_BUILD_FOLDER = str(
    Path(Path(__file__).parent.parent, "frontend", "build").absolute()
)
REACT_APP_FILE = "index.html"

app = Flask(__name__, static_folder=REACT_BUILD_FOLDER)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, REACT_APP_FILE)


def run_app(port=5001, debug=False):
    app.run(port=port, debug=debug)


if __name__ == "__main__":
    run_app(debug=True)
