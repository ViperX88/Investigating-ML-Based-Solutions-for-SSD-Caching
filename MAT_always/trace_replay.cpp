#include <atomic>
#include <bitset>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdlib>

#include <folly/init/Init.h>
#include "cachelib/allocator/CacheAllocator.h"
#include "cachelib/allocator/MemoryTierCacheConfig.h"
#include "cachelib/allocator/CacheTraits.h"

using Cache      = facebook::cachelib::LruAllocator;
using ObjectInfo = facebook::cachelib::ObjectInfo;

constexpr const char* kTracePath   = "/proj/cac101-PG0/trace.csv";
constexpr const char* kOriginPath  = "/mnt/origin/origin.bin";
constexpr const char* kNvmFile     = "/mnt/cache-ssd/nvm.dat";
constexpr const char* kPersistBase = "/proj/cac101-PG0/cachelib_state";

constexpr uint64_t kDramMB    = 1024;
constexpr uint64_t kNvmBytes  = 100ULL * 1024 * 1024 * 1024;
constexpr size_t   kBlockSize = 4096;
constexpr size_t   kAlign     = 4096;

static constexpr int      kFeatWindows  = 32;
static constexpr uint64_t kOpsPerWindow = 10000;

inline bool isAligned(uint64_t off, size_t sz) { return (off % kAlign == 0) && (sz % kAlign == 0); }
inline uint64_t nowNs() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void ensureDir(const char* dir) {
  if (::mkdir(dir, 0755) != 0 && errno != EEXIST) {
    throw std::runtime_error(std::string("mkdir failed for ") + dir + ": " + std::strerror(errno));
  }
}
inline void ensureParentDirForFile(const char* filePath) {
  std::string p(filePath);
  auto slash = p.find_last_of('/');
  if (slash == std::string::npos) return;
  std::string dir = p.substr(0, slash);
  if (!dir.empty()) ensureDir(dir.c_str());
}
inline uint64_t getFileSize(const char* path) {
  struct stat st{}; if (::stat(path, &st) == 0) return static_cast<uint64_t>(st.st_size);
  return 0;
}
inline void ensureRegularFileSizedAtLeast(const char* path, uint64_t bytes) {
  ensureParentDirForFile(path);
  int fd = ::open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) throw std::runtime_error(std::string("open/create failed for ") + path + ": " + std::strerror(errno));
  uint64_t cur = getFileSize(path);
  if (cur < bytes) {
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      int e = errno; ::close(fd);
      throw std::runtime_error(std::string("ftruncate failed for ") + path + ": " + std::strerror(e));
    }
  }
  ::close(fd);
}
inline void ensureRegularFileExact(const char* path, uint64_t bytes) {
  ensureParentDirForFile(path);
  int fd = ::open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) throw std::runtime_error(std::string("open/create failed for ") + path + ": " + std::strerror(errno));
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    int e = errno; ::close(fd);
    throw std::runtime_error(std::string("ftruncate failed for ") + path + ": " + std::strerror(e));
  }
  ::close(fd);
}
inline void logFsType(const char* dir) {
  struct statfs s{}; if (::statfs(dir, &s) == 0) {
    std::cout << "[FS] " << dir << " f_type=0x" << std::hex << s.f_type << std::dec << "\n";
  } else {
    std::cout << "[FS] statfs failed for " << dir << ": " << std::strerror(errno) << "\n";
  }
}

struct TraceEntry { std::string op; uint64_t offset; uint32_t size; };
std::vector<TraceEntry> loadTrace(const std::string& filename, uint64_t& originNeededBytes) {
  std::vector<TraceEntry> trace;
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Failed to open trace: " + filename);
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    std::istringstream iss(line);
    std::string ts, dev, cpu, op, offsetStr, sizeStr, duration;
    if (!std::getline(iss, ts, ',')) continue;
    if (!std::getline(iss, dev, ',')) continue;
    if (!std::getline(iss, cpu, ',')) continue;
    if (!std::getline(iss, op, ',')) continue;
    if (!std::getline(iss, offsetStr, ',')) continue;
    if (!std::getline(iss, sizeStr, ',')) continue;
    std::getline(iss, duration);

    TraceEntry e;
    e.op     = op;
    e.offset = static_cast<uint64_t>(std::stoll(offsetStr));
    e.size   = static_cast<uint32_t>(std::stoul(sizeStr));
    trace.push_back(e);

    const uint64_t endOff = e.offset + e.size;
    if (endOff > originNeededBytes) originNeededBytes = endOff;
  }
  return trace;
}

struct OriginFDs { int fd_direct{-1}; int fd_buf{-1}; };
static OriginFDs g_origin;

OriginFDs openOrigin(const char* path) {
  OriginFDs fds;
  fds.fd_buf = ::open(path, O_RDWR);
  if (fds.fd_buf < 0) {
    throw std::runtime_error(std::string("open buffered failed: ") + path + ": " + std::strerror(errno));
  }
  fds.fd_direct = ::open(path, O_RDWR | O_DIRECT);
  if (fds.fd_direct < 0) {
    std::cerr << "[warn] O_DIRECT open failed for origin: " << std::strerror(errno)
              << " — using buffered fallback only.\n";
  }
  return fds;
}
inline void closeOrigin(OriginFDs& fds) { if (fds.fd_direct >= 0) ::close(fds.fd_direct); if (fds.fd_buf >= 0) ::close(fds.fd_buf); }

inline void* alignedAlloc(size_t sz) {
  void* p = nullptr;
  size_t cap = ((sz + kAlign - 1) / kAlign) * kAlign;
  if (posix_memalign(&p, kAlign, cap) != 0) throw std::bad_alloc();
  return p;
}

ssize_t direct_read_unaligned(const OriginFDs& fds, void* dst, size_t sz, uint64_t off) {
  const uint64_t a_off = (off / kAlign) * kAlign;
  const size_t   head  = static_cast<size_t>(off - a_off);
  const size_t   a_sz  = ((head + sz + kAlign - 1) / kAlign) * kAlign;

  const int fd = (fds.fd_direct >= 0) ? fds.fd_direct : fds.fd_buf;

  void* tmp = alignedAlloc(a_sz);
  ssize_t n = ::pread(fd, tmp, a_sz, static_cast<off_t>(a_off));
  if (n < 0) { std::free(tmp); return n; }

  size_t canCopy = 0;
  if (static_cast<size_t>(n) > head) {
    canCopy = std::min(sz, static_cast<size_t>(n) - head);
    std::memcpy(dst, static_cast<char*>(tmp) + head, canCopy);
  }
  std::free(tmp);
  return static_cast<ssize_t>(canCopy);
}

ssize_t direct_write_block(const OriginFDs& fds, const void* src, size_t sz, uint64_t off) {
  const int fd = (fds.fd_direct >= 0) ? fds.fd_direct : fds.fd_buf;
  if (fd == fds.fd_direct && isAligned(off, sz) && (reinterpret_cast<uintptr_t>(src) % kAlign == 0)) {
    return ::pwrite(fd, src, sz, static_cast<off_t>(off));
  }
  void* tmp = alignedAlloc(sz);
  std::memcpy(tmp, src, sz);
  ssize_t w = ::pwrite(fd, tmp, sz, static_cast<off_t>(off));
  std::free(tmp);
  return w;
}

static int  g_payload_fd = -1;
static bool g_have_payload = false;
void openPayloadIfAny() {
  if (const char* p = std::getenv("TRACE_PAYLOAD_PATH")) {
    int fd = ::open(p, O_RDONLY);
    if (fd >= 0) { g_payload_fd = fd; g_have_payload = true; std::cout << "[Payload] Using " << p << "\n"; }
    else { std::cerr << "[Payload] open failed: " << std::strerror(errno) << " — will use 0xAA filler.\n"; }
  } else {
    std::cout << "[Payload] TRACE_PAYLOAD_PATH not set — using 0xAA filler for writes.\n";
  }
}
inline void closePayloadIfAny() { if (g_payload_fd >= 0) ::close(g_payload_fd); }
ssize_t readWritePayload(void* dst, size_t len, uint64_t off) {
  if (!g_have_payload) { std::memset(dst, 0xAA, len); return static_cast<ssize_t>(len); }
  ssize_t n = ::pread(g_payload_fd, dst, len, static_cast<off_t>(off));
  if (n < 0) { std::memset(dst, 0xAA, len); return static_cast<ssize_t>(len); }
  if (static_cast<size_t>(n) < len) std::memset(static_cast<char*>(dst) + n, 0xAA, len - n);
  return static_cast<ssize_t>(len);
}

std::unique_ptr<Cache> cache;
facebook::cachelib::PoolId defaultPool;
static std::unordered_set<std::string> g_dirty;
static std::mutex g_dirty_mu;

struct Counters {
  size_t writes{0}, writeBytes{0};
  size_t reads{0},  readBytes{0}, misses{0};
  size_t missBytes{0};
  size_t deviceReadBytes{0};
  size_t writeBackBytes{0};
  uint64_t featBuildNs{0};
  uint64_t featRefreshNs{0};
  uint64_t trainEmitNs{0};
  size_t   nvmEvictions{0};
} total;

inline void markDirty(const std::string& key) {
  std::lock_guard<std::mutex> g(g_dirty_mu);
  g_dirty.insert(key);
}
inline bool consumeDirtyIfPresent(const std::string& key) {
  std::lock_guard<std::mutex> g(g_dirty_mu);
  auto it = g_dirty.find(key);
  if (it == g_dirty.end()) return false;
  g_dirty.erase(it);
  return true;
}

struct Meta {
  uint64_t lastOp{0};
  uint64_t lastWindow{0};
  std::bitset<kFeatWindows> bits{};
};
static std::unordered_map<std::string, Meta> g_meta;
static std::atomic<uint64_t> g_opSeq{0};

static bool g_training_mode = false;
static std::string g_train_path;
static std::ofstream g_trainCsv;
static std::mutex g_trainMu;

inline void buildObjectInfoForKey(const std::string& key, ObjectInfo& oi) {
  const uint64_t t0 = nowNs();
  auto& m = g_meta[key];
  const uint64_t op = ++g_opSeq;
  const uint64_t curWin = op / kOpsPerWindow;

  if (curWin > m.lastWindow) {
    const uint64_t delta = std::min<uint64_t>(curWin - m.lastWindow, kFeatWindows);
    m.bits <<= delta;
    m.lastWindow = curWin;
  }
  m.bits.set(0);

  oi.past_timestamp = m.lastOp;
  oi.feat = static_cast<uint32_t>(m.bits.to_ulong());

  m.lastOp = op;
  total.featBuildNs += (nowNs() - t0);
}

inline void refreshFeaturesByReinsert(const std::string& key, const Cache::ReadHandle& cur) {
  const uint64_t t0 = nowNs();
  ObjectInfo oi{}, oiRet{};
  buildObjectInfoForKey(key, oi);

  auto hNew = cache->allocate(defaultPool, key, kBlockSize, oi, oiRet);
  if (!hNew) { total.featRefreshNs += (nowNs() - t0); return; }
  std::memcpy(hNew->getMemory(), cur->getMemory(), kBlockSize);
  cache->insertOrReplace(hNew);
  total.featRefreshNs += (nowNs() - t0);
}

inline void initTraining() {
  if (!g_training_mode) return;
  if (g_train_path.empty()) {
    std::cerr << "[MAT] MAT_TRAIN_OUT not set; training rows will be dropped.\n";
    return;
  }
  ensureParentDirForFile(g_train_path.c_str());
  g_trainCsv.open(g_train_path, std::ios::out | std::ios::trunc);
  if (!g_trainCsv) {
    std::cerr << "[MAT] Failed to open training CSV: " << g_train_path << "\n";
    return;
  }
  std::cout << "[MAT] Training CSV -> " << g_train_path << "\n";
  g_trainCsv << "b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,"
                "b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,"
                "age_ops,age_ns,ttl_ops,ttl_ns\n";
}

inline void emitTrainingSample(const std::string& key, uint64_t nowOpIdx, uint64_t nowNsTs) {
  if (!g_training_mode || !g_trainCsv) return;
  const uint64_t t0 = nowNs();
  const auto it = g_meta.find(key);
  uint32_t bits = 0; uint64_t lastOp = 0;
  if (it != g_meta.end()) { bits = static_cast<uint32_t>(it->second.bits.to_ulong()); lastOp = it->second.lastOp; }
  const uint64_t ttl_ops = (lastOp == 0 || nowOpIdx <= lastOp) ? 0 : (nowOpIdx - lastOp);
  const uint64_t age_ops = ttl_ops;
  const uint64_t age_ns  = 0;
  const uint64_t ttl_ns  = 0;
  for (int i = 0; i < 32; ++i) {
    g_trainCsv << ((bits >> i) & 1);
    g_trainCsv << (i == 31 ? ',' : ',');
  }
  g_trainCsv << age_ops << ',' << age_ns << ',' << ttl_ops << ',' << ttl_ns << "\n";
  total.trainEmitNs += (nowNs() - t0);
}

void initializeCacheWithWriteBack() {
  Cache::Config config;

  const pid_t pid = ::getpid();
  std::string cacheName = "ReplayCacheWB-" + std::to_string(pid);
  config.setCacheName(cacheName);
  ensureDir(kPersistBase);
  std::string stateDir = std::string(kPersistBase) + "/run-" + std::to_string(pid);
  ensureDir(stateDir.c_str());
  config.enableCachePersistence(stateDir);
  config.setDropNvmCacheOnShmNew(true);

  config.useEvictionControl = true;
  if (const char* ml = std::getenv("MAT_ML_CONFIG")) {
    struct stat st{};
    if (::stat(ml, &st) == 0 && S_ISREG(st.st_mode)) {
      std::ifstream in(ml);
      if (!in) {
        std::cerr << "[MAT] WARN: can't open ML config file: " << ml
                  << " — falling back to LRU\n";
      } else {
        std::stringstream buf; buf << in.rdbuf();
        config.MLConfig = buf.str();
        std::cout << "[MAT] MLConfig loaded (" << config.MLConfig.size() << " bytes)\n";
      }
    } else {
      config.MLConfig = ml;
      std::cout << "[MAT] Using inline ML config (" << config.MLConfig.size()
                << " chars)\n";
    }
  } else {
    std::cout << "[MAT] MLConfig not set; eviction falls back to LRU.\n";
  }

  config.setCacheSize(kDramMB * 1024ULL * 1024ULL)
        .setAccessConfig({20, 10})
        .configureMemoryTiers({ facebook::cachelib::MemoryTierCacheConfig::fromShm().setRatio(1) });

  ensureDir("/mnt/cache-ssd");
  ensureDir("/mnt/origin");
  ensureRegularFileExact(kNvmFile, kNvmBytes);
  logFsType("/mnt/cache-ssd");

  Cache::NvmCacheConfig nvmConfig;
  nvmConfig.navyConfig.setSimpleFile(kNvmFile, kNvmBytes, true);
  nvmConfig.navyConfig.enableRandomAdmPolicy().setAdmProbability(1.0);
  config.enableNvmCache(nvmConfig);

  auto itemDestructor = [&](const Cache::DestructorData& data) {
    using DC = facebook::cachelib::DestructorContext;
    if (data.context != DC::kEvictedFromNVM) return;
    total.nvmEvictions++;

    auto keySP = data.item.getKey();
    std::string key(keySP.data(), keySP.size());
    if (!consumeDirtyIfPresent(key)) {
      if (g_training_mode) emitTrainingSample(key, g_opSeq.load(), nowNs());
      return;
    }

    uint64_t blockOff = 0;
    try { blockOff = std::stoull(key); } catch (...) {
      if (g_training_mode) emitTrainingSample(key, g_opSeq.load(), nowNs());
      return;
    }

    const void* src = data.item.getMemory();
    ssize_t w = direct_write_block(g_origin, src, kBlockSize, blockOff);
    if (w >= 0) total.writeBackBytes += static_cast<size_t>(w);

    if (g_training_mode) emitTrainingSample(key, g_opSeq.load(), nowNs());
  };
  config.setItemDestructor(itemDestructor);

  config.validate();
  cache = std::make_unique<Cache>(Cache::SharedMemNew, config);
  defaultPool = cache->addPool("default", cache->getCacheMemoryStats().ramCacheSize);

  std::cout << "[Config] LRU(+MAT), write-back | DRAM_MB=" << kDramMB
            << " NVM_bytes=" << kNvmBytes
            << " blockSize=" << kBlockSize
            << " name=" << cacheName << "\n";
}

void processEntry(const TraceEntry& e) {
  if (e.size == 0) return;

  uint64_t lastByte;
  if (__builtin_add_overflow(e.offset, static_cast<uint64_t>(e.size), &lastByte)) return;
  lastByte -= 1;

  const uint64_t startBlock = e.offset / kBlockSize;
  const uint64_t endBlock   = lastByte   / kBlockSize;

  std::vector<char> blk(kBlockSize);
  std::vector<char> payload;

  for (uint64_t b = startBlock; b <= endBlock; ++b) {
    const uint64_t blockOff = b * kBlockSize;
    const std::string key   = std::to_string(blockOff);

    if (e.op == "Read") {
      const uint64_t rStart   = std::max<uint64_t>(blockOff, e.offset);
      const uint64_t rEnd     = std::min<uint64_t>(blockOff + kBlockSize, e.offset + e.size);
      if (rEnd <= rStart) continue;
      const size_t   len      = static_cast<size_t>(rEnd - rStart);
      const size_t   offInBlk = static_cast<size_t>(rStart - blockOff);

      auto h = cache->find(key);
      if (h) {
        refreshFeaturesByReinsert(key, h);
      } else {
        ssize_t got = direct_read_unaligned(g_origin, blk.data(), kBlockSize, blockOff);
        if (got < 0) {
          std::cerr << "pread failed: " << std::strerror(errno) << "\n";
          std::memset(blk.data(), 0, kBlockSize);
        } else {
          total.deviceReadBytes += static_cast<size_t>(got);
          if (static_cast<size_t>(got) < kBlockSize) {
            std::memset(blk.data() + got, 0, kBlockSize - static_cast<size_t>(got));
          }
        }

        ObjectInfo oi{}, oiRet{};
        buildObjectInfoForKey(key, oi);
        auto hNew = cache->allocate(defaultPool, key, kBlockSize, oi, oiRet);
        if (hNew) { std::memcpy(hNew->getMemory(), blk.data(), kBlockSize); cache->insertOrReplace(hNew); }

        total.misses++; total.missBytes += len;
      }
      total.reads++; total.readBytes += len;
      continue;
    }

    if (e.op == "Write") {
      const uint64_t wStart = std::max<uint64_t>(blockOff, e.offset);
      const uint64_t wEnd   = std::min<uint64_t>(blockOff + kBlockSize, e.offset + e.size);
      if (wEnd <= wStart) continue;
      const size_t len      = static_cast<size_t>(wEnd - wStart);
      const size_t offInBlk = static_cast<size_t>(wStart - blockOff);
      const bool   full     = (len == kBlockSize) && (offInBlk == 0);

      if (full) {
        if (payload.size() < len) payload.resize(len);
        readWritePayload(payload.data(), len, wStart);
        std::memcpy(blk.data(), payload.data(), kBlockSize);
      } else {
        ssize_t got = direct_read_unaligned(g_origin, blk.data(), kBlockSize, blockOff);
        if (got < 0) { std::memset(blk.data(), 0, kBlockSize); }
        else {
          total.deviceReadBytes += static_cast<size_t>(got);
          if (static_cast<size_t>(got) < kBlockSize) {
            std::memset(blk.data() + got, 0, kBlockSize - static_cast<size_t>(got));
          }
        }
        if (payload.size() < len) payload.resize(len);
        readWritePayload(payload.data(), len, wStart);
        std::memcpy(blk.data() + offInBlk, payload.data(), len);
      }

      ObjectInfo oi{}, oiRet{};
      buildObjectInfoForKey(key, oi);
      auto hNew = cache->allocate(defaultPool, key, kBlockSize, oi, oiRet);
      if (hNew) {
        std::memcpy(hNew->getMemory(), blk.data(), kBlockSize);
        cache->insertOrReplace(hNew);
        markDirty(key);
      }

      total.writes++; total.writeBytes += len;
    }
  }
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  try {
    std::cout << "[SMART] Logical Sectors Written (before):\n";
    (void)std::system("sudo smartctl -x /dev/sdc | grep \"Logical Sectors Written\"");

    if (const char* m = std::getenv("MAT_TRAIN_MODE")) {
      g_training_mode = (std::string(m) == "1" || std::string(m) == "true");
    }
    if (const char* p = std::getenv("MAT_TRAIN_OUT")) {
      g_train_path = p;
    }

    uint64_t originNeedBytes = 0;
    auto trace = loadTrace(kTracePath, originNeedBytes);
    std::cout << "Trace entries: " << trace.size() << "\n";
    ensureRegularFileSizedAtLeast(kOriginPath, (originNeedBytes == 0 ? kBlockSize : originNeedBytes));

    g_origin = openOrigin(kOriginPath);
    openPayloadIfAny();

    initTraining();
    initializeCacheWithWriteBack();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (const auto& e : trace) {
      processEntry(e);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    auto mb = [](size_t b){ return b / 1024.0 / 1024.0; };
    auto iops = [seconds](size_t ops){ return seconds > 0 ? (ops / seconds) : 0.0; };

    const double reqPerSec = seconds > 0 ? (double)(total.reads + total.writes) / seconds : 0.0;
    const double MissRatio = (total.readBytes > 0) ? (100.0 * (double)total.missBytes / (double)total.readBytes) : 0.0;

    const double build_ms   = total.featBuildNs   / 1e6;
    const double refresh_ms = total.featRefreshNs / 1e6;
    const double train_ms   = total.trainEmitNs   / 1e6;
    const double perEvict_us = (total.nvmEvictions > 0)
                                 ? ( (total.featBuildNs + total.featRefreshNs) / (double)total.nvmEvictions / 1e3 )
                                 : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Writes:  " << total.writes << " (" << total.writeBytes << " bytes)\n";
    std::cout << "Reads:   " << total.reads  << " (" << total.readBytes  << " bytes)\n";
    std::cout << "Misses:  " << MissRatio << "%\n";
    std::cout << "Time:    " << (seconds * 1000.0) << " ms\n";
    std::cout << "IOPS W:  " << iops(total.writes) << "\n";
    std::cout << "IOPS R:  " << iops(total.reads)  << "\n";

    if (g_trainCsv.is_open()) g_trainCsv.flush();
    closePayloadIfAny();
    closeOrigin(g_origin);
    cache.reset();
    return 0;

  } catch (const std::exception& ex) {
    std::cerr << "Fatal: " << ex.what() << "\n";
    return 1;
  }
}
