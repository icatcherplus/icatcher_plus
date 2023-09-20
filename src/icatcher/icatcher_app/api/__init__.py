import webbrowser
from .api import run_app as run_icatcher_app

default_port = 5001


def open_app():
    webbrowser.open(f"http://localhost:{default_port}")
    run_icatcher_app(port=default_port)
