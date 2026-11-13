#!/bin/bash
set -e
cd -- "$(dirname -- "$0")"
exec python3 server.py --port 8765
