"""
Start the local rename-risk web application and open it in the default browser.

    python run.py                 # http://127.0.0.1:8000, opens the browser once the server answers /health
    python run.py --no-browser    # start the server only
    python run.py --port 8010
    python run.py --review-cutoff 0.6   # user-chosen review cutoff (default 0.70; not validated)

The server binds to 127.0.0.1 (this machine only). Stop it with Ctrl+C.
"""
import argparse
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def wait_until_ready(url, timeout=60.0, interval=0.2):
    """Poll /health until it answers (no fixed sleep); returns True when ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(interval)
    return False


def port_is_free(host, port):
    """True if nothing is listening on host:port (so /health cannot be answered by another server)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--review-cutoff", type=float, default=None,
                    help="user-chosen review cutoff in (0, 1); default 0.70 (not a validated threshold)")
    args = ap.parse_args(argv)
    if not port_is_free(args.host, args.port):
        print(f"Port {args.port} is already in use (an earlier server may still be running). Stop it, or start "
              f"this one with --port {args.port + 1}.")
        return 1
    import uvicorn
    from src.app.server import create_app

    from src.app.inference import REVIEW_CUTOFF
    app = create_app(review_cutoff=REVIEW_CUTOFF if args.review_cutoff is None else args.review_cutoff)  # models load once
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="info", reload=False))
    home = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"

    def open_browser():
        if wait_until_ready(home + "/health"):
            print(f"Opening {home}")
            webbrowser.open(home, new=2)                  # once: no auto-reload, so no duplicate tabs
        else:
            print(f"Server did not become ready; open {home} manually.")

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()
    server.run()                                          # blocks until Ctrl+C
    return 0


if __name__ == "__main__":
    sys.exit(main())
