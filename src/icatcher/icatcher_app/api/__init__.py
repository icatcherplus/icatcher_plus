import webbrowser
from .api import run_app
from .builder import build_app

DEFAULT_PORT = 5001


def run_icatcher_app():
    build_app(force=False, debug=False, info=False)
    webbrowser.open(f"http://localhost:{DEFAULT_PORT}")
    run_app(port=DEFAULT_PORT, debug=False)
