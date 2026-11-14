"""Fake OpenAI-compatible server (stand-in for vLLM or xAI) for testing the runbook locally.

usage: python fake_openai.py <served-model-id> [--port N] [any vLLM flags, ignored]
Replies are placeholders, never results.
"""
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = sys.argv[1]
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._send({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "fake"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if req.get("model") != MODEL:
            return self._send({"error": {"message": f"model {req.get('model')} not served"}}, 404)
        h = int(hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:8], 16)
        p = round(0.05 + 0.9 * (h % 1000) / 1000, 2)
        logprobs = None
        if req.get("logprobs"):
            text = "Yes"
            logprobs = {"content": [{"token": "Yes", "logprob": -0.3, "bytes": [89, 101, 115],
                                     "top_logprobs": [{"token": "Yes", "logprob": -0.3, "bytes": [89, 101, 115]},
                                                      {"token": "No", "logprob": -1.4, "bytes": [78, 111]}]}]}
        elif "assess how decision" in json.dumps(req):
            text = f"PROBABILITY_YES: {p:.2f}"
        else:
            system = req["messages"][0]["content"] if req["messages"][0]["role"] == "system" else ""
            head = ("EVIDENCE_TIER: 2\nPROBABILITY_RANGE: 0.40 to 0.70\nWOULD_CHANGE_IF: a trial\n\n"
                    if "EVIDENCE_TIER" in system else "")
            text = f"{head}[FAKE SERVER placeholder reply]\n\nPROBABILITY_YES: {p:.2f}"
        n_in = sum(len(m["content"]) for m in req["messages"]) // 4
        self._send({"id": f"chatcmpl-{h}", "object": "chat.completion", "created": int(time.time()), "model": MODEL,
                    "choices": [{"index": 0, "finish_reason": "stop", "logprobs": logprobs,
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": n_in, "completion_tokens": len(text) // 4,
                              "total_tokens": n_in + len(text) // 4}})


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
