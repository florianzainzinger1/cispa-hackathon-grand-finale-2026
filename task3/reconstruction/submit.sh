#!/bin/bash
# Submit the built .pt to the organizer. YOU run this with your API key in an
# env var so the key is never pasted into chat, stored, or echoed:
#
#   FL_API_KEY='your-key-here' bash submit.sh
#
# Optional first arg = path to a different .pt file.
set -e
: "${FL_API_KEY:?Set FL_API_KEY to your organizer API key: FL_API_KEY=xxx bash submit.sh}"
FILE="${1:-/p/scratch/training2625/zainzinger1/1337/recon_work/outputs/submission.pt}"

if [ ! -f "$FILE" ]; then
  echo "submission file not found: $FILE" >&2
  exit 1
fi

echo "Uploading $FILE ($(du -h "$FILE" | cut -f1)) to 21-fl-audit ..."
curl -sS -X POST "http://<SUBMIT_SERVER>/submit/21-fl-audit" \
  -H "X-API-Key: $FL_API_KEY" \
  -F "file=@$FILE"
echo
