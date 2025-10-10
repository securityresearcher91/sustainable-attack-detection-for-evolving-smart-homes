#!/bin/bash

PCAP_DIR="./"

cd "$PCAP_DIR" || exit 1

for f in *.pcap*; do
    # Match filenames like xyz.pcap10, xyz.pcap3, etc.
    if [[ "$f" =~ ^(.*\.pcap)([0-9]+)$ ]]; then
        base="${BASH_REMATCH[1]}"
        num="${BASH_REMATCH[2]}"
        padded_num=$(printf "%02d" "$num")
        
        new_name="${base%.pcap}_$padded_num.pcap"
        echo "Renaming $f → $new_name"
        mv "$f" "$new_name"
    fi
done

