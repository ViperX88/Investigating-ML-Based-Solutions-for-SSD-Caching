sudo apt update && sudo apt install -y \
  build-essential cmake git curl wget unzip \
  python3 python3-pip libtool autoconf pkg-config \
  libssl-dev libevent-dev

git clone https://github.com/facebook/CacheLib.git
cd CacheLib
git checkout tags/v20240320_stable -b stable-local
git submodule update --init --recursive
./contrib/build.sh -j -T
sudo apt install -y git cmake build-essential libboost-all-dev libnuma-dev libgflags-dev libgoogle-glog-dev \
    libevent-dev libdouble-conversion-dev libssl-dev libzstd-dev liblz4-dev libjemalloc-dev libunwind-dev \
    libsnappy-dev pkg-config curl

python3 -m pip uninstall -y cmake || true

sudo apt-get remove -y cmake cmake-data
sudo apt-get autoremove -y
sudo apt-get update
sudo apt-get install -y gpg software-properties-common lsb-release
wget -O - https://apt.kitware.com/kitware-archive.sh | sudo bash
sudo apt-get update
VER=3.29.6-0kitware1ubuntu22.04.1
sudo apt-get update
sudo apt-get install -y --allow-downgrades \
  cmake=$VER cmake-data=$VER
sudo apt-mark hold cmake cmake-data
cmake --version

# This is for mounting, if you prefer to do this yourself, please remove this part
ORIGIN_MNT="/mnt/origin"
CACHE_MNT="/mnt/cache-ssd"

read -p "Origin device (e.g., /dev/sdb) [Enter to skip]: " ORIGIN_DEV
read -p "Cache  device (e.g., /dev/sdc) [Enter to skip]: " CACHE_DEV

if [[ -n "$ORIGIN_DEV" && -n "$CACHE_DEV" ]]; then
  echo "About to FORMAT $ORIGIN_DEV and $CACHE_DEV (ext4) and mount them."
  read -p "Type YES to continue, anything else to skip mkfs: " ACK
  if [[ "$ACK" == "YES" ]]; then
    sudo mkfs.ext4 -F "$ORIGIN_DEV"
    sudo mkfs.ext4 -F "$CACHE_DEV"
  else
    echo "Skipping mkfs."
  fi

  sudo mkdir -p "$ORIGIN_MNT" "$CACHE_MNT"
  mountpoint -q "$ORIGIN_MNT" || sudo mount "$ORIGIN_DEV" "$ORIGIN_MNT"
  mountpoint -q "$CACHE_MNT"  || sudo mount "$CACHE_DEV"  "$CACHE_MNT"
  sudo chown -R "$USER":"$USER" "$ORIGIN_MNT" "$CACHE_MNT"

  # Optional: preallocate Navy file (ok if it already exists)
  fallocate -l 100G "$CACHE_MNT/nvm.dat" || true
else
  echo "Disk setup skipped (no devices provided)."
fi
