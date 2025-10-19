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
COMING




