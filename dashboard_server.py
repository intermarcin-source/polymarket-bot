"""
Lightweight HTTP server for the BTC Sniper Dashboard.
Serves dashboard.html and JSON data files from the data/ directory.

Usage:
    python dashboard_server.py              # Port 8080
    python dashboard_server.py --port 9090  # Custom port
"""

import os
import sys
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve dashboard + JSON data with proper CORS and cache headers."""

    def do_GET(self):
        # Strip query string for routing
        path = self.path.split("?")[0]

        if path == "/" or path == "/dashboard" or path == "/index.html":
            self._serve_file(ROOT_DIR / "dashboard.html", "text/html")
        elif path.startswith("/data/") and path.endswith(".json"):
            filename = path.split("/")[-1]
            self._serve_file(DATA_DIR / filename, "application/json")
        else:
            self.send_error(404, "Not found")

    def _serve_file(self, filepath: Path, content_type: str):
        if not filepath.exists():
            # Return empty JSON for missing data files
            if content_type == "application/json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            self.send_error(404, "Not found")
            return

        try:
            content = filepath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def log_message(self, format, *args):
        # Quieter logging - only errors
        if args and "200" not in str(args[0]):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="BTC Sniper Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard server running at http://{args.host}:{args.port}")
    print(f"Serving data from: {DATA_DIR}")
    print(f"Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
