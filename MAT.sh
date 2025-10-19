python3 -m pip install --user lightgbm pandas pyyaml numpy
git clone https://github.com/yangdsh/CacheLib_2023  
cd CacheLib_2023  
./contrib/build.sh  
cd ..  
cd MAT_always/build  
unset MAT_ML_CONFIG  
export MAT_TRAIN_MODE=1  
export MAT_TRAIN_OUT=/proj/cac101-PG0/mat_train.csv
./trace_replay

python3 - << 'PY'
import pandas as pd, numpy as np, lightgbm as lgb, os
csv = "/proj/cac101-PG0/mat_train.csv"
df  = pd.read_csv(csv)

# features: 32-bit recency window + age
X   = df[[*(f"b{i}" for i in range(32)), "age_ops", "age_ns"]].astype("float32")

# label: prefer time-to-next-access (ns). Fall back to ops if empty.
y_ns = df["ttl_ns"].to_numpy()
if (y_ns == 0).all():
    print("[train] ttl_ns all zero; falling back to ttl_ops")
    y = np.log1p(df["ttl_ops"].to_numpy(dtype=np.float64))
else:
    y = np.log1p(y_ns.astype(np.float64))

dtrain = lgb.Dataset(X, label=y)
params = dict(objective="regression_l2", metric="l2",
              num_leaves=31, learning_rate=0.05,
              feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=200, verbosity=-1)
bst = lgb.train(params, dtrain, num_boost_round=400)

out = "/proj/cac101-PG0/mat_model.txt"     
bst.save_model(out)
print("Saved:", out)
PY

# DO NOT collect training rows now
unset MAT_TRAIN_MODE
unset MAT_TRAIN_OUT


export MAT_ML_CONFIG='{
  "modelType": "lgbm_text",
  "modelPath": "/proj/cac101-PG0/mat_model.txt",
  "windowCnt": 32,
  "asyncMode": false,
  "predictionBatchSize": 64,
  "useFIFO": false,
  "debugMode": 1
}'

./trace_replay
