#!/usr/bin/env python3
import argparse, csv, hashlib, os
from collections import defaultdict

# Baleen op-code constants
GET_TEMP, GET_PERM, PUT_TEMP, PUT_PERM, GET_NOT_INIT, PUT_NOT_INIT = 1, 2, 3, 4, 5, 6
GETS = {GET_TEMP, GET_PERM, GET_NOT_INIT}

def ts_to_float_seconds(ts_str: str) -> float:
    ts = int(ts_str)
    if ts >= 10**15:   # ns
        return ts / 1e9
    elif ts >= 10**12: # us
        return ts / 1e6
    else:              # s
        return float(ts)

def split_across_segments(abs_off: int, size: int, seg_bytes: int):
    end = abs_off + size
    cur = abs_off
    while cur < end:
        seg_start = (cur // seg_bytes) * seg_bytes
        within = cur - seg_start
        can = min(end - cur, seg_bytes - within)
        yield (cur // seg_bytes, within, can)
        cur += can

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_csv_row(row):
    # Expect: ts,dev,cpu,op,offset,size,duration
    if not row or len(row) < 7: return None
    ts_str, dev, cpu, op_str, off_str, size_str, _dur = row[:7]
    try:
        abs_off = int(off_str); size = int(size_str)
    except ValueError:
        return None
    t = ts_to_float_seconds(ts_str)
    op_low = op_str.strip().lower()
    if op_low.startswith("read"):  op_code = GET_PERM
    elif op_low.startswith("write"): op_code = PUT_PERM
    else: op_code = GET_PERM
    try:
        user_name = int(cpu)
    except ValueError:
        user_name = int(hashlib.sha1(cpu.encode()).hexdigest()[:8], 16)
    user_ns = int(hashlib.sha1(dev.encode()).hexdigest()[:8], 16)
    return t, abs_off, size, op_code, user_ns, user_name

def main():
    ap = argparse.ArgumentParser(description="Convert CSV I/O trace to Baleen sampled files.")
    ap.add_argument("--input", required=True,
                    help="CSV: ts,dev,cpu,op,offset,size,duration (no header).")
      ap.add_argument("--baleen-root", required=True,
                    help="Path to Baleen-FAST24 repo root (e.g., ~/Baleen-FAST24).")
    ap.add_argument("--group", required=True, help="Trace group (e.g., 202508).")
    ap.add_argument("--region", required=True, help="Region name (e.g., RegionMine).")
    ap.add_argument("--segment-mib", type=float, default=8.0,
                    help="Logical segment size in MiB (default 8).")
    ap.add_argument("--samples", nargs="+", default=["0:0.1"],
                    help="One or more START:RATIO pairs (fractions of total time). Default 0:0.1")
    args = ap.parse_args()

    seg_bytes = int(args.segment_mib * 1024 * 1024)
    root = os.path.expanduser(args.baleen_root)
    outdir = os.path.join(root, "data", "tectonic", args.group, args.region)
    os.makedirs(outdir, exist_ok=True)

    # Always (re)write header
    path_header = os.path.join(outdir, "full.header")
    with open(path_header, "w") as fh:
        fh.write("block_id io_byte_offset io_size time op user_namespace user_name\n")

    # Pass 1: find time range
    tmin = float("inf"); tmax = float("-inf")
    with open(args.input, newline="") as fin:
        rdr = csv.reader(fin)
        for row in rdr:
            p = parse_csv_row(row)
            if not p: continue
            t = p[0]
            if t < tmin: tmin = t
            if t > tmax: tmax = t
    if not (tmin < tmax):
        raise SystemExit("Could not determine valid time range from CSV")

    # Prepare checksum file (append mode, to accumulate)
    path_cksum = os.path.join(outdir, "checksums.sha1")
    ck_lines = []

    # For each requested sample window, generate trace & keys
    for tag in args.samples:
        try:
            start_s, ratio_s = tag.split(":")
            start = float(start_s); ratio = float(ratio_s)
        except Exception:
            raise SystemExit(f"Bad --samples entry: {tag} (use START:RATIO, e.g., 0:0.1)")
        if start < 0 or ratio <= 0 or start + ratio > 1.0:
            raise SystemExit(f"Invalid window {tag}; require 0<=start, 0<ratio, start+ratio<=1")

        win_start = tmin + start * (tmax - tmin)
        win_end   = tmin + (start + ratio) * (tmax - tmin)

        trace_name = f"full_{int(start)}_{ratio}.trace"
        keys_name  = f"full_{int(start)}_{ratio}.keys"
        path_trace = os.path.join(outdir, trace_name)
        path_keys  = os.path.join(outdir, keys_name)

        totals = defaultdict(int); gets = defaultdict(int)
        kept = 0; rows = 0
      
        with open(args.input, newline="") as fin, open(path_trace, "w") as ft:
            rdr = csv.reader(fin)
            for row in rdr:
                parsed = parse_csv_row(row)
                if not parsed: continue
                t, abs_off, size, op_code, user_ns, user_name = parsed
                rows += 1
                if t < win_start or t > win_end:
                    continue
                for blk_id, within, chunk in split_across_segments(abs_off, size, seg_bytes):
                    ft.write(f"{blk_id} {within} {chunk} {t:.6f} {op_code} {user_ns} {user_name}\n")
                    totals[blk_id] += 1
                    if op_code in GETS: gets[blk_id] += 1
                    kept += 1

        with open(path_keys, "w") as fk:
            for blk_id, tot in totals.items():
                fk.write(f"{blk_id} {tot} {gets.get(blk_id, 0)}\n")

        # queue checksums
        for p in (path_header, path_keys, path_trace):
            ck_lines.append(f"{sha1_of_file(p)}  {os.path.basename(p)}\n")

        print(f"[OK] {trace_name}: kept {kept} segment-ops from {rows} CSV rows "
              f"for window [{win_start:.6f},{win_end:.6f}] of [{tmin:.6f},{tmax:.6f}]")

    # Append checksums at the end
    with open(path_cksum, "a") as fc:
        fc.writelines(ck_lines)
    print(f"[OK] Wrote checksums: {path_cksum}")

if __name__ == "__main__":
    main()
