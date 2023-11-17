import webbrowser
from .api import run_app
from .builder import build_app


def run_icatcher_app():
    DEFAULT_PORT = 5001
    success = build_app(force=False)
    if success:
        webbrowser.open(f"http://localhost:{DEFAULT_PORT}")
        run_app(port=DEFAULT_PORT, debug=False)
