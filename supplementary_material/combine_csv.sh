#!/bin/bash

CSV_DIR="$1"
if [[ -z "$CSV_DIR" ]]; then
  echo "Usage: $0 <path_to_csv_directory>"
  exit 1
fi

OUTPUT_FILE="$CSV_DIR/combined.csv"
rm -f "$OUTPUT_FILE"

csv_files=$(find "$CSV_DIR" -maxdepth 1 -type f -name '*_[0-9]*.pcap.csv' | sort)

first=1
for csv in $csv_files; do
  if [ "$first" -eq 1 ]; then
    cat "$csv" > "$OUTPUT_FILE"
    first=0
  else
    tail -n +2 "$csv" >> "$OUTPUT_FILE"
  fi
done

echo "Combined CSV written to: $OUTPUT_FILE"

