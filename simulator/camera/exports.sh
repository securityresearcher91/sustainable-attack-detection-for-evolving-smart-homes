# Tell the camera where to push “cloud recording” traffic and which video file to use
export CAMERA_CLOUD_ENABLED=1
export CAMERA_CLOUD_BASE_URL="https://cloud-discard.kushal-ramkumar.workers.dev"
export CAMERA_CLOUD_INGEST_PATH="/v1/ingest"
export CAMERA_CLOUD_HEARTBEAT_PATH="/v1/heartbeat"
export CAMERA_VIDEO="/home/sas2/dev/applications/camera/videos/video.mp4"

# Optional: tune cadence/bitrate
export CAMERA_TARGET_FPS=3.0
export CAMERA_FRAMES_PER_SEGMENT=6
export CAMERA_UPLOAD_INTERVAL_S=2.0
export CAMERA_JPEG_QUALITY=60

# Expose the camera endpoints via a Cloudflare tunnel for remote viewing
export CAMERA_TUNNEL=1
