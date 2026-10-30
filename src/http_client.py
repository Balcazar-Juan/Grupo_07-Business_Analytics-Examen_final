from __future__ import annotations
import time
import requests

class HttpClient:
    def __init__(self, timeout: int = 30, retries: int = 4):
        self.session = requests.Session()
        self.timeout = timeout
        self.retries = retries

    def request(self, method: str, url: str, **kwargs):
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    time.sleep(min(2 ** attempt, 16))
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                last = exc
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 16))
        raise last or RuntimeError("HTTP error")
