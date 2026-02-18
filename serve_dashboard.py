"""
Tiny dashboard server - serves the dashboard HTML and the JSON data files.
Double-click open-dashboard.bat to start this.
"""
import http.server
import socketserver
import os
import sys
import webbrowser
import threading

PORT = 8888
DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def do_GET(self):
        # Redirect / to dashboard.html
        if self.path == '/' or self.path == '':
            self.send_response(302)
            self.send_header('Location', '/dashboard.html')
            self.end_headers()
            return
        return super().do_GET()

    def end_headers(self):
        # Allow all CORS and disable caching so dashboard always gets fresh data
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress request logs to keep console clean
        pass

def open_browser():
    """Open dashboard in browser (skip on headless servers)."""
    try:
        # Skip on headless Linux servers (no DISPLAY)
        if sys.platform != "win32" and not os.environ.get("DISPLAY"):
            return
        webbrowser.open(f'http://localhost:{PORT}/dashboard.html')
    except Exception:
        pass  # Headless server, no browser available

if __name__ == '__main__':
    # Kill any existing server on this port
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"========================================")
            print(f"  Polymarket Bot Dashboard")
            print(f"  http://localhost:{PORT}/dashboard.html")
            print(f"  Press Ctrl+C to stop")
            print(f"========================================")
            # Open browser after a short delay
            threading.Timer(1.0, open_browser).start()
            httpd.serve_forever()
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print(f"Port {PORT} already in use. Trying {PORT+1}...")
            PORT += 1
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"Dashboard running at http://localhost:{PORT}/dashboard.html")
                threading.Timer(1.0, open_browser).start()
                httpd.serve_forever()
        else:
            raise
