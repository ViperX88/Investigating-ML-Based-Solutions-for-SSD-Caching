# Investigating ML-Based Solutions for SSD Caching

This repository reproduces results on **CacheLib/Navy** with heuristic policies (LRU, TinyLFU, Reject-First) and ML-based policies (**Baleen** admission, **MAT** eviction).
I will add the paper later.

## 1) Quick start (install & build CacheLib)

```bash
git clone https://github.com/ViperX88/Investigating-ML-Based-Solutions-for-SSD-Caching
cd Investigating-ML-Based-Solutions-for-SSD-Caching
chmod +x install.sh
./install.sh
```

> Tip: to customize (e.g., cache file size or device mounts), open `install.sh` and tweak the variables at the top.

## 2) Traces

A small `trace.csv` is included for tests.
To reproduce results, download the MSR Cambridge traces from:
[http://dsn.ce.sharif.edu/ftp/IO/Data-Center/](http://dsn.ce.sharif.edu/ftp/IO/Data-Center/)

---

## 3) Run a baseline (example: LRU + always-admit)

```bash
cd LRU_always
mkdir -p build && cd build
cmake ..
make -j
./trace_replay
```

---

## 4) Run MAT (ML-guided eviction)

Build once:

```bash
cd MAT_always
mkdir -p build && cd build
cmake ..
make -j
```

Then run the provided helper script:

```bash
bash ../../MAT.sh
```

(If you prefer manual steps, the script does: run once to emit training rows → train LightGBM → run again with `MAT_ML_CONFIG`.)

---

## 5) Run Baleen (ML-guided admission)

Baleen uses its own simulator to train a model, then you can export that model for CacheLib.

### 5.1 Train Baleen with its simulator

```bash
git clone --recurse-submodules https://github.com/wonglkd/Baleen-FAST24.git
cd Baleen-FAST24
python3 -m pip install --user -r BCacheSim/install/requirements.txt
cd data
bash get-tectonic.sh
cd ../..
```

Convert your trace for Baleen:

```bash
python3 convert_to_baleen.py \
  --input trace.csv \
  --baleen-root Baleen-FAST24 \
  --group 202508 \
  --region RegionMine \
  --segment-mib 8
```

Run Baleen training:

```bash
cd Baleen-FAST24
./BCacheSim/run_py.sh py -B -m BCacheSim.episodic_analysis.train \
  --exp example \
  --policy PolicyUtilityServiceTimeSize2 \
  --region RegionMine \
  --sample-ratio 0.1 \
  --sample-start 0 \
  --trace-group 202508 \
  --supplied-ea physical \
  --target-wrs 34 50 100 75 20 10 60 90 30 \
  --target-csizes 366.475 \
  --output-base-dir runs/example/baleen \
  --eviction-age 5892.856 \
  --rl-init-kwargs filter_=prefetch \
  --train-target-wr 35.599 \
  --train-models admit prefetch \
  --train-split-secs-start 0 \
  --train-split-secs-end 86400 \
  --ap-acc-cutoff 15 \
  --ap-feat-subset meta+block+chunk
```

Export the model (Treelite/tl2cgen toolchain):

```bash
conda create -y -n tl2cgen python=3.10
conda activate tl2cgen
conda install -y -c conda-forge treelite tl2cgen lightgbm
```

### 5.2 Use the trained Baleen model in CacheLib

Set the environment variables to your exported model/meta:

```bash
export BALEEN_MODEL_PATH=/proj/cac101-PG0/baleen/Baleen-FAST24/tmp/example/202508_RegionMine_0_0.1/ea_5892.86_wr_35.599_admit_threshold_binary.model
export BALEEN_META_JSON=/proj/cac101-PG0/baleen/baleen_bundle/baleen_admit.meta.json
```

Build and run the CacheLib experiment:

```bash
cd LRU_Baleen
mkdir -p build && cd build
cmake ..
make -j
./trace_replay
```

---

## 6) Notes & troubleshooting

* **Permissions / mounts:** Some scripts create and mount `/mnt/origin` and `/mnt/cache-ssd`. You can adjust those manually if needed.
* **CMake version:** We use **CMake 3.29.6** via Kitware APT to match MAT’s build expectations.
* **Python deps:** MAT training uses `lightgbm`, `pandas`, `numpy`, `pyyaml`. The scripts install them via `pip --user`.

---

## 7) Citation

If you use this code or results, please cite:

## References

[**Baleen: ML Admission & Prefetching for Flash Caches**](https://www.usenix.org/system/files/fast24-wong.pdf)  
Daniel Lin-Kit Wong, Hao Wu, Carson Molder, Sathya Gunasekar, Jimmy Lu, Snehal Khandkar, Abhinav Sharma, Daniel S. Berger, Nathan Beckmann, Gregory R. Ganger  
*USENIX FAST 2024*

[**MAT: Machine Learning Assisted Eviction Policy for Cache**](https://arxiv.org/pdf/2301.11886)  
Zhen Yang, Sudarsun Kannan, Arjun Kashyap, et al.  
*arXiv 2023*

[**CacheLib / Navy**](https://github.com/facebook/CacheLib)  
Meta Open Source  
*Project repository*

[**TinyLFU: A Highly Efficient Cache Admission Policy**](https://arxiv.org/pdf/1512.00727)  
Gil Einziger, Roy Friedman, Ben Manes  
*arXiv 2015*


