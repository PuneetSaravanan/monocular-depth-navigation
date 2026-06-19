#!/usr/bin/env bash
# record_demo.sh — capture a demo of a navigation run and make a GIF for the README.
#
# Runs a course in Webots WITH rendering, records the 3D view to mp4 via the
# controller's Supervisor movie hook (NAV_RECORD), then converts to a GIF.
#
# Usage:  scripts/record_demo.sh [course] [controller]
#   course     : single | slalom | cluttered | dense   (default: slalom)
#   controller : depth_nav | classical_cv_nav | sonar_nav (default: depth_nav)
#
# Requires: Webots, and ffmpeg for the GIF conversion (brew install ffmpeg).
set -euo pipefail

COURSE="${1:-slalom}"
CONTROLLER="${2:-depth_nav}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export WEBOTS_HOME="${WEBOTS_HOME:-/Applications/Webots.app/Contents}"
WEBOTS="$WEBOTS_HOME/MacOS/webots"

REC_DIR="$ROOT/docs/recordings"
mkdir -p "$REC_DIR"
MP4="$REC_DIR/${CONTROLLER}_${COURSE}.mp4"
GIF="$ROOT/docs/demo.gif"

# Build a world that uses the requested controller.
TMP="$ROOT/worlds/_demo_tmp.wbt"
sed "s/controller \"depth_nav\"/controller \"$CONTROLLER\"/" \
    "$ROOT/worlds/course_${COURSE}.wbt" > "$TMP"

echo "Recording $CONTROLLER on course '$COURSE' -> $MP4"
# Rendering ON (no --no-rendering) so the movie has frames; realtime looks best.
NAV_RECORD="$MP4" NAV_GOAL="5,0" NAV_MAX_TIME="120" \
  "$WEBOTS" --batch --mode=realtime --stdout --stderr "$TMP" || true
rm -f "$TMP"

if [ ! -f "$MP4" ]; then
  echo "No mp4 produced — was rendering enabled / did the run finish?" >&2
  exit 1
fi

if command -v ffmpeg >/dev/null 2>&1; then
  echo "Converting to GIF -> $GIF"
  PAL="$REC_DIR/palette.png"
  ffmpeg -y -i "$MP4" -vf "fps=12,scale=640:-1:flags=lanczos,palettegen" "$PAL" >/dev/null 2>&1
  ffmpeg -y -i "$MP4" -i "$PAL" -lavfi "fps=12,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" "$GIF" >/dev/null 2>&1
  echo "Done: $GIF"
else
  echo "ffmpeg not found — mp4 saved at $MP4 (install ffmpeg to make the GIF)."
fi
