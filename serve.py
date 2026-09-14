import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal dependency-free health endpoint")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    HTTPServer(("0.0.0.0", args.port), HealthHandler).serve_forever()


if __name__ == "__main__":
    main()
