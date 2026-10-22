"""Real HTTP loopback verifies batching/retry without external service access."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch

from flex_gspo.comet22 import ServiceCOMET22


class LoopbackCOMET22Tests(unittest.TestCase):
    def test_order_payload_and_retry_over_real_http(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, data))
                retry = len(seen) == 1
                self.send_response(503 if retry else 200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"score": [0.7, 0.3]}).encode("utf-8"))

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            scorer = ServiceCOMET22({"service_url": f"http://127.0.0.1:{server.server_port}/22",
                                    "response_key": "score"})
            with patch("flex_gspo.comet22.time.sleep"):
                result = scorer.score_batch("Source", ["译文甲", "译文乙"], "参考")
            self.assertEqual(result, [0.7, 0.3])
            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[0], seen[1])
            self.assertEqual(seen[0], ("/22", {"srcs": ["Source", "Source"],
                                             "mts": ["译文甲", "译文乙"], "refs": ["参考", "参考"]}))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
