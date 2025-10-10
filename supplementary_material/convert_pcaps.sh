source ./cicflowmeter/.venv/bin/activate
mkdir -p <PCAP-DIR>/cic_output
for f in <PCAP-DIR>/*.pcap; do
    cicflowmeter -f "$f" -c "<PCAP-DIR>/cic_output/$(basename "$f").csv"
done
deactivate
