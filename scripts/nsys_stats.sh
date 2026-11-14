#!/usr/bin/env bash
# Kernel summary of existing Nsight Systems reports. Usage: scripts/nsys_stats.sh naive sdpa_flash
export PATH=/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64:$PATH
for m in "$@"; do
  echo "== $m"
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true "/tmp/prof/$m.nsys-rep" 2>&1 \
    | python3 -c "
import sys,csv
rows=[l for l in sys.stdin if l.strip()]
start=next((i for i,l in enumerate(rows) if l.startswith('Time (%)') or l.startswith('\"Time (%)')),None)
if start is None:
    print(''.join(rows[-8:])); sys.exit()
for r in csv.DictReader(rows[start:]):
    print(f\"{float(r['Time (%)']):6.1f}%  total {float(r['Total Time (ns)'])/1e3:10.1f} us  n={r['Instances']:>3s}  {r['Name'][:90]}\")
"
done
