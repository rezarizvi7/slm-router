#!/bin/bash
# Double-click to stop SLM Router (macOS). Downloaded models are kept for next time.
cd "$(dirname "$0")" || exit 1
echo "Stopping SLM Router..."
docker compose down && echo "Stopped."
echo
read -r -p "Press Enter to close this window..." _
