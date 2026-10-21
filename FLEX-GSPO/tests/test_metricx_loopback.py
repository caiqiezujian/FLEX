"""Real HTTP wire check against a local fixture, never the user's service."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
from unittest.mock import patch

from flex_gspo.metricx import ServiceMetricX


class MetricXLoopbackTests(unittest.TestCase):
    def test_both_endpoints_send_ordered_text_batches_without_proxy(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, payload))
                body = json.dumps({"scores": [0.5, 1.25]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        settings = {"backend": "service", "service_url": f"http://127.0.0.1:{server.server_port}",
                    "service_id": "local-http-fixture-not-a-metricx-model"}
        source, candidates, reference = " A meeting. ", [" 一场会议。", "开会。"], "一次会议。"
        try:
            with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1",
                                            "http_proxy": "http://127.0.0.1:1",
                                            "ALL_PROXY": "http://127.0.0.1:1",
                                            "all_proxy": "http://127.0.0.1:1",
                                            "NO_PROXY": "", "no_proxy": ""}):
                for mode in ("qe", "reference"):
                    self.assertEqual(ServiceMetricX(dict(settings, mode=mode)).score_batch(
                        source, candidates, reference), [0.5, 1.25])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        self.assertEqual(received, [
            ("/metricx24/qe", {"srcs": [source, source], "mts": candidates}),
            ("/metricx24", {"srcs": [source, source], "mts": candidates,
                            "refs": [reference, reference]}),
        ])


if __name__ == "__main__":
    unittest.main()
