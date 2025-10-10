#!/usr/bin/env python3
import argparse
import random
import subprocess
import time
from datetime import datetime, timedelta

MIN_STREAM_SEC = 30
MAX_STREAM_SEC = 120
MIN_WAIT_SEC = 60
MAX_WAIT_SEC = 300

def run_ffmpeg_session(url, duration):
    # Read from HTTP and discard output (-f null -)
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-re",                # read input at native rate for realism
        "-i", url,
        "-t", str(duration),  # cap session duration
        "-f", "null", "-"     # discard output
    ]
    print(f"[{datetime.now()}] Streaming {url} for {duration}s ...")
    subprocess.run(cmd)
    print(f"[{datetime.now()}] Session completed.")

def sporadic_stream(urls, total_minutes=240):
    end = datetime.now() + timedelta(minutes=total_minutes)
    while datetime.now() < end:
        url = random.choice(urls)
        duration = random.randint(MIN_STREAM_SEC, MAX_STREAM_SEC)
        run_ffmpeg_session(url, duration)
        if datetime.now() < end:
            sleep_s = random.randint(MIN_WAIT_SEC, MAX_WAIT_SEC)
            print(f"[{datetime.now()}] Idle {sleep_s}s before next session...")
            time.sleep(sleep_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera streaming scenario simulator")
    parser.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3],
                       help="1: LAN only, 2: Public only, 3: Both")
    parser.add_argument("--hours", type=float, default=1.0,
                       help="Duration in hours (default: 1.0)")
    parser.add_argument("--lan-url", type=str, required=True,
                       help="LAN camera URL (e.g., http://10.42.0.42:41277/video_feed)")
    parser.add_argument("--public-url", type=str, required=True,
                       help="Public camera URL (e.g., https://example.trycloudflare.com/video_feed)")
    
    args = parser.parse_args()

    # Construct URLs based on scenario
    if args.scenario == 1:
        urls = [args.lan_url]
        print(f"Scenario 1: LAN only ({args.lan_url})")
    elif args.scenario == 2:
        urls = [args.public_url]
        print(f"Scenario 2: Public only ({args.public_url})")
    else:
        urls = [args.lan_url, args.public_url]
        print(f"Scenario 3: LAN + Public")
        print(f"  LAN: {args.lan_url}")
        print(f"  Public: {args.public_url}")

    print(f"Running for {args.hours} hour(s)...")
    sporadic_stream(urls, total_minutes=int(args.hours * 60))
    print("Done.")

