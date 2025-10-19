export RUNS_DIR=~/Baleen-FAST24/runs/example/baleen/example/202508_RegionMine_0_0.1
ls "$RUNS_DIR/offline_analysis_ea_5892.86.csv"
export TMP_DIR=~/Baleen-FAST24/tmp/example/202508_RegionMine_0_0.1
ls "$TMP_DIR"
export OUT_DIR=~/baleen_bundle
mkdir -p "$OUT_DIR"

export CUTOFF=$(python3 - <<'PY'
import csv, os, sys
csv_path = os.path.expanduser(os.environ['RUNS_DIR'] + '/offline_analysis_ea_5892.86.csv')
with open(csv_path) as f:
    rows = list(csv.DictReader(f))
rows = [r for r in rows if (r.get('Cutoff score') or '').strip()]
row = next((r for r in rows if (r.get('Target') or '').strip().lower() == 'cache size'), rows[0])
print(row['Cutoff score'])
PY
)
echo "CUTOFF=$CUTOFF"

export SEG_BYTES=2097152
echo "SEG_BYTES=$SEG_BYTES"

python3 - <<'PY'
import os, lightgbm as lgb
tmp = os.path.expanduser(os.environ['TMP_DIR'])
models = [p for p in os.listdir(tmp) if p.endswith('_admit_threshold_binary.model')]
assert models, "No *admit_threshold_binary.model found in TMP_DIR"
bst = lgb.Booster(model_file=os.path.join(tmp, models[0]))
names = bst.feature_name()
out = os.path.join(os.environ['OUT_DIR'], 'baleen_admit.features.txt')
open(out,'w').write("\n".join(names))
print("wrote", len(names), "features to", out)
PY

cat > "$OUT_DIR/baleen_admit.meta.json" <<EOF
{
  "cutoff": $CUTOFF,
  "segment_size": $SEG_BYTES
}
EOF

cat "$OUT_DIR/baleen_admit.meta.json"
 python3 export_admit_model.py
tar czf baleen_bundle_202508_RegionMine.tgz \
  baleen_admit.so \
  Baleen-FAST24/tmp/example/202508_RegionMine_0_0.1/ea_5892.86_wr_35.599_admit_threshold_binary.model \
  baleen_bundle/baleen_admit.features.txt \
  baleen_bundle/baleen_admit.meta.json

mkdir -p baleen_bundle && tar xzf baleen_bundle_202508_RegionMine.tgz -C baleen_bundle
cp baleen_bundle_202508_RegionMine.tgz baleen_bundle/
