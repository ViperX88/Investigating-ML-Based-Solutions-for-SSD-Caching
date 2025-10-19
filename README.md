# Investigating ML-Based Solutions for SSD-Caching
This repository is for reproducing results using CacheLib, Baleen and MAT. This repository includes code used in the paper: I will upload the paper here later.
You can run the repository the following way:

git clone https://github.com/ViperX88/Investigating-ML-Based-Solutions-for-SSD-Caching   
cd Investigating-ML-Based-Solutions-for-SSD-Caching  
chmod +x install.sh  
./install.sh  

Tip: If you wish to change the setup, look into the script and change it accordingly, for example for the space allocated for the cache.

The current trace.csv is a small trace upon which you can run test results. However, to reproduce the results from the paper, please download the according traces from this source: http://dsn.ce.sharif.edu/ftp/IO/Data-Center/.
Through this approach you can install the basic Cachelib and run the basic Cachelib experiments.

To run an experiement using a specific algorithm (e.g: LRU_always) , do this:

cd LRU_always  
mkdir -p build && cd build   
cmake ..   
make -j   
./trace_replay    
  


To run MAT:
cd MAT_always  
mkdir build  
cd build  
cmake ..  
make -j  

Now MAT is ready, and you can simply run the script: MAT.sh to get results.


To run Baleen:
Baleen is a little more complex to run. First, on your own you have to clone the Baleen simulator repository, convert the trace and save the results in a file.

First pull Baleen:
git clone --recurse-submodules https://github.com/wonglkd/Baleen-FAST24.git  
cd Baleen-FAST24  
python3 -m pip install --user -r BCacheSim/install/requirements.txt  
cd data  
bash get-tectonic.sh  
cd ..  
cd ..  

  python3 convert_to_baleen.py \  
  --input trace.csv \  
  --baleen-root Baleen-FAST24 \  
  --group 202508 \  
  --region RegionMine \  
  --segment-mib 8  

cd Baleen-FAST24  
./BCacheSim/run_py.sh py -B -m BCacheSim.episodic_analysis.train --exp example --policy PolicyUtilityServiceTimeSize2 --region RegionMine --sample-ratio 0.1 --sample-start 0 --trace-group 202508 --supplied-ea physical --target-wrs 34 50 100 75 20 10 60 90 30 --target-csizes 366.475 --output-base-dir runs/example/baleen --eviction-age 5892.856 --rl-init-kwargs filter_=prefetch --train-target-wr 35.599 --train-models admit prefetch --train-split-secs-start 0 --train-split-secs-end 86400 --ap-acc-cutoff 15 --ap-feat-subset meta+block+chunk

conda create -y -n tl2cgen python=3.10  
conda activate tl2cgen  
conda install -y -c conda-forge treelite tl2cgen lightgbm  

The run the script: bash BALEEN.sh

Export the model and parameters to use for Cachelib:

export BALEEN_MODEL_PATH=/proj/cac101-PG0/baleen/Baleen-FAST24/tmp/example/202508_RegionMine_0_0.1/ea_5892.86_wr_35.599_admit_threshold_binary.model
export BALEEN_META_JSON=/proj/cac101-PG0/baleen/baleen_bundle/baleen_admit.meta.json


Now you can run the Baleen file:
cd LRU_Baleen  
mkdir build && cd build  
cmake ..  
make -j  
./trace_replay  



