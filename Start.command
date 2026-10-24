#!/bin/bash
# Double-click to start SLM Router (macOS). On Linux, run it from your file manager or terminal.
cd "$(dirname "$0")" || exit 1

pause() { echo; read -r -p "Press Enter to close this window..." _; }
open_url() { if command -v open >/dev/null; then open "$1"; else xdg-open "$1" >/dev/null 2>&1; fi; }

echo "=== SLM Router ==="
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed."
  echo "Install Docker Desktop from https://www.docker.com/products/docker-desktop"
  echo "then double-click Start again."
  open_url "https://www.docker.com/products/docker-desktop"
  pause; exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Starting Docker Desktop..."
  [ "$(uname)" = "Darwin" ] && open -a Docker
  for _ in $(seq 1 60); do
    docker info >/dev/null 2>&1 && break
    sleep 2
  done
  if ! docker info >/dev/null 2>&1; then
    echo "Docker didn't start. Open Docker Desktop yourself, wait until it says it's running,"
    echo "then double-click Start again."
    pause; exit 1
  fi
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "First-time setup."
  echo "Paste your Anthropic API key so hard questions can go to Claude."
  echo "(Get one at https://console.anthropic.com - or just press Enter to skip;"
  echo " everything will then be answered by the free local model.)"
  read -r -p "API key: " KEY
  if [ -n "$KEY" ]; then
    awk -v k="$KEY" '/^ANTHROPIC_API_KEY=/{print "ANTHROPIC_API_KEY=" k; next} {print}' .env > .env.tmp && mv .env.tmp .env
    echo "Saved."
  fi
  echo
fi

echo "Starting everything. The FIRST time this downloads about 1.5 GB and can take 5-15 minutes."
echo
if ! docker compose up -d --build; then
  echo
  echo "Something went wrong starting the containers (see the messages above)."
  pause; exit 1
fi

echo
echo "Waiting for the router to be ready..."
for _ in $(seq 1 450); do
  curl -s http://localhost:8080/healthz >/dev/null 2>&1 && break
  sleep 2
done
if ! curl -s http://localhost:8080/healthz >/dev/null 2>&1; then
  echo "The router didn't come up in time. Recent log lines:"
  docker compose logs --tail 30 router
  pause; exit 1
fi

echo
echo "Ready!"
echo "  Chat:      http://localhost:8080"
echo "  Dashboard: http://localhost:3000"
open_url "http://localhost:8080"
echo
echo "It keeps running in the background. Double-click Stop when you're done."
pause
