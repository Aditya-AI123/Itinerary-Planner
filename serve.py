"""
HTTP server for the Itinerary Planner frontend.
- Serves all static files from the project root
- Exposes /api/config with the Google API key (for photo fetching)

Run from project root:  python3 serve.py
Then open:  http://localhost:8000/frontend/index.html
"""
import http.server
import socketserver
import json
import re
import os

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def read_env_key(key_name):
    try:
        with open('.env') as f:
            for line in f:
                m = re.match(rf'{key_name}\s*=\s*(.+)', line.strip())
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return ''


class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/api/config':
            body = json.dumps({
                'googleApiKey': read_env_key('GOOGLE_API_KEY'),
            }).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        # Suppress noisy request logs; only show startup message
        pass


print(f'Server running at http://localhost:{PORT}/frontend/index.html')
print('Press Ctrl+C to stop.')
with socketserver.TCPServer(('', PORT), Handler) as httpd:
    httpd.serve_forever()
