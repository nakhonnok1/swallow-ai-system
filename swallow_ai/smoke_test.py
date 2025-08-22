#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick smoke test for swallow_ai.app_working
- Imports the Flask app without running the server
- Pings key JSON endpoints using Flask test_client
Run:
  python -m swallow_ai.smoke_test
"""
import sys
import traceback

try:
    from . import app_working as m
except Exception:
    # allow running as a script from repo root
    sys.path.append('C:/Nakhonnok')
    from swallow_ai import app_working as m  # type: ignore


def ping(client, path: str) -> int:
    try:
        r = client.get(path)
        print(f"PING {path} -> {r.status_code}")
        if path.endswith('/api/system-health'):
            print(r.json)
        return r.status_code
    except Exception:
        traceback.print_exc()
        return -1


def main():
    app = m.app
    with app.test_client() as c:
        overall = 0
        for p in ['/api/system-health', '/api/object-detection/status', '/api/insights', '/']:
            code = ping(c, p)
            overall += 0 if code == 200 else 1
        if overall == 0:
            print('\n✅ Smoke test passed')
            raise SystemExit(0)
        else:
            print(f"\n⚠️ Smoke test completed with {overall} failure(s)")
            raise SystemExit(1)


if __name__ == '__main__':
    main()
