#include <atomic>
#include <array>
#include <bitset>
#include <chrono>
#include <cstring>
#include <cerrno>
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
#include <algorithm>
#include <cmath>

#include <jsoncpp/json/json.h>
#include <LightGBM/c_api.h>

#include <folly/init/Init.h>
#include "cachelib/allocator/CacheAllocator.h"
#include "cachelib/allocator/MemoryTierCacheConfig.h"
#include "cachelib/allocator/CacheTraits.h"

using Cache = facebook::cachelib::LruAllocator;

constexpr const char* kTracePath   = "/proj/cac101-PG0/trace.csv";
constexpr const char* kOriginPath  = "/mnt/origin/origin.bin";
constexpr const char* kNvmFile     = "/mnt/cache-ssd/nvm.dat";
constexpr const char* kPersistBase = "/proj/cac101-PG0/cachelib_state";

static size_t   kBlockSize = 2 * 1024 * 1024;
constexpr uint64_t kDramMB   = 128;
constexpr uint64_t kNvmBytes = 100ULL * 1024 * 1024 * 1024;
constexpr size_t   kAlign    = 4096;
static const bool  kUseODirectReads = true;

static bool g_baleen_enabled = []{
  if (const char* v = std::getenv("BALEEN_ENABLE")) return std::string(v) != "0";
  return true;
}();

struct Counters {
  size_t writes{0}, writeBytes{0};
  size_t reads{0},  readBytes{0}, misses{0};
  size_t deviceReadBytes{0};
  size_t writeBackBytes{0};
  size_t r4kBlocks{0};
  size_t r4kHits{0};
  size_t r4kMisses{0};
} total;

static double g_seekCostSec = []{
  if (const char* s = std::getenv("BALEEN_SEEK_COST_MS")) {
    try { return std::stod(s) / 1000.0; } catch (...) {}
  }
  return 0.005;
}();

static double g_byteCostSec = []{
  if (const char* s = std::getenv("BALEEN_BYTE_COST_NS")) {
    try { return std::stod(s) * 1e-9; } catch (...) {}
  }
  return 5e-9;
}();

static double g_dtWindowSec = []{
  if (const char* s = std::getenv("BALEEN_DT_WINDOW_SEC")) {
    try { return std::stod(s); } catch (...) {}
  }
  return 600.0;
}();

static std::chrono::steady_clock::time_point g_runStart;
static std::vector<double> g_dtWindows;

inline void initDT() {
  g_runStart = std::chrono::steady_clock::now();
  g_dtWindows.clear();
}

inline void accountDT(size_t bytes) {
  double dt = g_seekCostSec + g_byteCostSec * static_cast<double>(bytes);
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - g_runStart).count();
  size_t windowIndex = static_cast<size_t>(elapsed / g_dtWindowSec);
  if (windowIndex >= g_dtWindows.size()) {
    g_dtWindows.resize(windowIndex + 1, 0.0);
  }
  g_dtWindows[windowIndex] += dt;
}

static size_t g_admit = 0, g_reject = 0;
static size_t g_w_admit = 0, g_w_reject = 0;
static std::atomic<size_t> g_topUps{0};

inline bool isAligned(uint64_t off, size_t sz) { return (off % kAlign == 0) && (sz % kAlign == 0); }
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
  if (kUseODirectReads) {
    fds.fd_direct = ::open(path, O_RDWR | O_DIRECT);
    if (fds.fd_direct < 0) {
      std::cerr << "[warn] O_DIRECT open failed for origin: " << std::strerror(errno)
                << " — using buffered fallback only.\n";
    }
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

// ======================= Baleen Admission =======================
struct BaleenAdmission {
  BoosterHandle booster{nullptr};
  double cutoff = 0.5;
  size_t segSize = 2 * 1024 * 1024;
  bool useGovernor = false;
  double targetWriteMBps = 0.0;
  double ewmaWriteMBps = 0.0;
  mutable double lastThr = 0.0;
  const double alpha = 0.1;
  std::chrono::steady_clock::time_point lastTick = std::chrono::steady_clock::now();
  uint64_t bytesSinceTick = 0;

  ~BaleenAdmission() {
    if (booster) { LGBM_BoosterFree(booster); booster = nullptr; }
  }

  static std::string getenvOrEmpty(const char* k) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : std::string();
  }

  void maybeLoadMeta(const std::string& metaPath) {
    if (metaPath.empty()) return;
    std::ifstream in(metaPath);
    if (!in) {
      std::cerr << "[Baleen] WARN: cannot open meta json: " << metaPath << "\n";
      return;
    }
    Json::Value root;
    in >> root;
    if (root.isMember("segment_size") && root["segment_size"].isUInt64()) {
      segSize = static_cast<size_t>(root["segment_size"].asUInt64());
    }
    if (root.isMember("cutoff") && root["cutoff"].isDouble()) {
      cutoff = root["cutoff"].asDouble();
    }
  }

  void init() {
    std::string meta = getenvOrEmpty("BALEEN_META_JSON");
    maybeLoadMeta(meta);

    std::string modelPath = getenvOrEmpty("BALEEN_MODEL_PATH");
    if (g_baleen_enabled) {
      if (modelPath.empty()) throw std::runtime_error("BALEEN_MODEL_PATH not set");
      int out_n_models = 0;
      int err = LGBM_BoosterCreateFromModelfile(modelPath.c_str(), &out_n_models, &booster);
      if (err != 0 || booster == nullptr) {
        throw std::runtime_error("LightGBM load failed for model: " + modelPath);
      }
      int num_feature = 0;
      if (LGBM_BoosterGetNumFeature(booster, &num_feature) == 0) {
        std::cerr << "[Baleen] Model features: " << num_feature << "\n";
      } else {
        std::cerr << "[Baleen] WARN: unable to read model feature count\n";
      }
    }

    if (const char* t = std::getenv("BALEEN_TARGET_WRITE_MBPS")) {
      try { targetWriteMBps = std::stod(t); } catch (...) { targetWriteMBps = 0.0; }
      useGovernor = (targetWriteMBps > 0.0);
    }

    std::cerr << "[Baleen] " << (g_baleen_enabled ? "enabled" : "disabled")
              << ", seg=" << segSize << "B, cutoff=" << cutoff
              << (useGovernor ? (", target MB/s=" + std::to_string(targetWriteMBps)) : "")
              << "\n";
  }

  void buildFeatures(bool isRead,
                     uint64_t ns,
                     uint64_t user,
                     uint64_t f3,
                     uint64_t f4,
                     uint64_t f5,
                     std::array<double,18>& fv) const {
    fv[0] = isRead ? 0.0 : 1.0;
    fv[1] = static_cast<double>(ns);
    fv[2] = static_cast<double>(user);
    fv[3] = static_cast<double>(f3);
    fv[4] = static_cast<double>(f4);
    fv[5] = static_cast<double>(f5);
    for (int i=0;i<6;i++) fv[6+i]  = 0.0;
    for (int i=0;i<6;i++) fv[12+i] = 0.0;
  }

  double predictProb(const std::array<double,18>& fv) const {
    if (!g_baleen_enabled) return 1.0;

    const double* data = fv.data();
    int32_t nrow = 1, ncol = 18;
    int is_row_major = 1;
    int predict_type = C_API_PREDICT_NORMAL;
    int start_iteration = 0;
    int num_iteration   = -1;
    const char* param   = "";

    std::vector<double> out(1, 0.0);
    int64_t out_len = 0;

    int err = LGBM_BoosterPredictForMat(
        booster, data, C_API_DTYPE_FLOAT64,
        nrow, ncol, is_row_major,
        predict_type, start_iteration, num_iteration,
        param, &out_len, out.data());
    if (err != 0 || out_len < 1) return 0.0;

    return std::min(std::max(out[0], 0.0), 1.0);
  }

  bool admit(double prob) {
    if (!g_baleen_enabled) return true;
    double thr = cutoff;
    if (useGovernor && ewmaWriteMBps > 0.0 && targetWriteMBps > 0.0) {
      double ratio = ewmaWriteMBps / targetWriteMBps;
      double delta = std::clamp((ratio - 1.0) * 0.02, -0.05, 0.05);
      thr = std::clamp(cutoff + delta, 0.05, 0.95);
    }
    lastThr = thr;
    return prob >= thr;
  }

  void accountAdmitBytes(size_t bytes) {
    bytesSinceTick += bytes;
    auto now = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTick).count();
    if (ms >= 1000) {
      double mbps = (bytesSinceTick / 1.0e6) * (1000.0 / (double)ms);
      if (ewmaWriteMBps == 0.0) ewmaWriteMBps = mbps;
      else ewmaWriteMBps = alpha * mbps + (1.0 - alpha) * ewmaWriteMBps;
      bytesSinceTick = 0;
      lastTick = now;
    }
  }
};

static std::unique_ptr<BaleenAdmission> g_baleen;

static constexpr size_t kSubBlock = 4096;
static std::unordered_map<std::string, std::vector<uint8_t>> g_cov;
inline size_t subsPerSeg() { return kBlockSize / kSubBlock; }
inline std::vector<uint8_t>& covFor(const std::string& key) {
  auto it = g_cov.find(key);
  if (it == g_cov.end()) {
    auto [insIt, _] = g_cov.emplace(key, std::vector<uint8_t>(subsPerSeg(), 0));
    return insIt->second;
  }
  if (it->second.size() != subsPerSeg()) it->second.assign(subsPerSeg(), 0);
  return it->second;
}
inline void markCoveredRange(const std::string& key, size_t offInSeg, size_t len) {
  auto& bm = covFor(key);
  const size_t first = offInSeg / kSubBlock;
  const size_t last  = std::min(subsPerSeg(), (offInSeg + len + kSubBlock - 1) / kSubBlock);
  for (size_t i = first; i < last; ++i) bm[i] = 1;
}
inline size_t count4kBlocks(size_t offInSeg, size_t len) {
  const size_t first = offInSeg / kSubBlock;
  const size_t last  = std::min(subsPerSeg(), (offInSeg + len + kSubBlock - 1) / kSubBlock);
  return (last > first) ? (last - first) : 0;
}
inline std::pair<size_t,size_t> count4kHitMiss(const std::string& key, size_t offInSeg, size_t len) {
  auto& bm = covFor(key);
  const size_t first = offInSeg / kSubBlock;
  const size_t last  = std::min(subsPerSeg(), (offInSeg + len + kSubBlock - 1) / kSubBlock);
  size_t hits = 0, misses = 0;
  for (size_t i = first; i < last; ++i) {
    if (bm[i]) ++hits; else ++misses;
  }
  return {hits, misses};
}

inline void ensureCovered(const std::string& key,
                          const OriginFDs& fds,
                          void* segMem,
                          uint64_t segOff,
                          size_t offInSeg,
                          size_t len) {
  auto& bm = covFor(key);
  const size_t first = offInSeg / kSubBlock;
  const size_t last  = std::min(subsPerSeg(), (offInSeg + len + kSubBlock - 1) / kSubBlock);

  size_t i = first;
  while (i < last) {
    if (bm[i]) { ++i; continue; }
    size_t j = i + 1;
    while (j < last && !bm[j]) ++j;

    size_t runOff = i * kSubBlock;
    size_t runLen = (j - i) * kSubBlock;
    if (runOff + runLen > kBlockSize) runLen = kBlockSize - runOff;

    ssize_t got = direct_read_unaligned(
        fds,
        static_cast<char*>(segMem) + runOff,
        runLen, segOff + runOff);
    if (got > 0) {
      g_topUps++;
      total.deviceReadBytes += static_cast<size_t>(got);
      accountDT(static_cast<size_t>(got));
      size_t blocks = std::min(runLen, static_cast<size_t>(got)) / kSubBlock;
      for (size_t z = 0; z < blocks && (i + z) < subsPerSeg(); ++z) bm[i + z] = 1;
    }
    i = j;
  }
}

std::unique_ptr<Cache> cache;
facebook::cachelib::PoolId defaultPool;
static std::unordered_set<std::string> g_dirty;

inline void markDirty(const std::string& key) { g_dirty.insert(key); }
inline bool consumeDirtyIfPresent(const std::string& key) {
  auto it = g_dirty.find(key);
  if (it == g_dirty.end()) return false;
  g_dirty.erase(it);
  return true;
}

void initializeCacheWithWriteBack() {
  g_baleen = std::make_unique<BaleenAdmission>();
  g_baleen->init();
  kBlockSize = g_baleen->segSize;
  std::cout << "[Config] Segment size = " << kBlockSize << " bytes (from meta)\n";

  Cache::Config config;
  const pid_t pid = ::getpid();
  std::string cacheName = "ReplayCacheWB-" + std::to_string(pid);
  config.setCacheName(cacheName);
  ensureDir(kPersistBase);
  std::string stateDir = std::string(kPersistBase) + "/run-" + std::to_string(pid);
  ensureDir(stateDir.c_str());
  config.enableCachePersistence(stateDir);
  config.setDropNvmCacheOnShmNew(true);

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
    if (data.context != DC::kEvictedFromNVM) {
      return;
    }

    auto keySP = data.item.getKey();
    std::string key(keySP.data(), keySP.size());

    if (!consumeDirtyIfPresent(key)) {
      g_cov.erase(key);
      return;
    }

    uint64_t segOff = 0; try { segOff = std::stoull(key); } catch (...) { g_cov.erase(key); return; }

    ensureCovered(key, g_origin, data.item.getMemory(), segOff, 0, kBlockSize);

    const void* src = data.item.getMemory();
    ssize_t w = direct_write_block(g_origin, src, kBlockSize, segOff);
    if (w < 0) {
      std::cerr << "[wb] destructor: pwrite failed: " << std::strerror(errno) << "\n";
    } else {
      total.writeBackBytes += static_cast<size_t>(w);
      accountDT(static_cast<size_t>(w));
    }

    g_cov.erase(key);
  };
  config.setItemDestructor(itemDestructor);

  config.validate();
  cache = std::make_unique<Cache>(Cache::SharedMemNew, config);
  defaultPool = cache->addPool("default", cache->getCacheMemoryStats().ramCacheSize);

  std::cout << "[Config] LRU, write-back | DRAM_MB=" << kDramMB
            << " NVM_bytes=" << kNvmBytes << " name=" << cacheName << "\n";
}

void processEntry(const TraceEntry& e) {
  if (e.size == 0) return;

  static thread_local std::vector<char> smallBuf;

  const uint64_t startSeg = e.offset / kBlockSize;
  const uint64_t endSeg   = (e.offset + e.size - 1) / kBlockSize;

  for (uint64_t b = startSeg; b <= endSeg; ++b) {
    const uint64_t segOff   = b * kBlockSize;
    const std::string key   = std::to_string(segOff);

    const uint64_t reqStart = std::max<uint64_t>(segOff, e.offset);
    const uint64_t reqEnd   = std::min<uint64_t>(segOff + kBlockSize, e.offset + e.size);
    const size_t   len      = (reqEnd > reqStart) ? static_cast<size_t>(reqEnd - reqStart) : 0;
    const size_t   offInSeg = static_cast<size_t>(reqStart - segOff);
    if (len == 0) continue;

    const std::string& op = e.op;
    const bool isRead  = (!op.empty() && (op[0] == 'R' || op[0] == 'r'));
    const bool isWrite = (!op.empty() && (op[0] == 'W' || op[0] == 'w'));

    if (isRead) {
      if (auto h = cache->find(key)) {
        {
          auto [hits4k, miss4k] = count4kHitMiss(key, offInSeg, len);
          total.r4kHits   += hits4k;
          total.r4kMisses += miss4k;
          total.r4kBlocks += (hits4k + miss4k);
        }

        if (auto wh = cache->findToWrite(key)) {
          ensureCovered(key, g_origin, wh->getMemory(), segOff, offInSeg, len);
        } else {
          if (smallBuf.size() < len) smallBuf.resize(len);
          ssize_t got = direct_read_unaligned(g_origin, smallBuf.data(), len, reqStart);
          if (got > 0) {
            total.deviceReadBytes += static_cast<size_t>(got);
            accountDT(static_cast<size_t>(got));
          }
        }
        total.reads++;
        total.readBytes += len;
        continue;
      }

      {
        size_t blocks = count4kBlocks(offInSeg, len);
        total.r4kMisses += blocks;
        total.r4kBlocks += blocks;
      }

      std::array<double,18> fv;
      const uint64_t ns = 0, user = 0;
      const uint64_t blk_id = b;
      const uint64_t within = offInSeg;
      const uint64_t chunk  = len;
      g_baleen->buildFeatures(true, ns, user, blk_id, within, chunk, fv);
      double p = g_baleen->predictProb(fv);
static double p_min = 1.0, p_max = 0.0;
static size_t buckets[10] = {0};
p_min = std::min(p_min, p);
p_max = std::max(p_max, p);
int bkt = std::clamp(int(p * 10), 0, 9);
buckets[bkt]++;
if (((total.reads + total.writes) % 200000) == 0) {
  std::cerr << "[Baleen] p in [" << p_min << "," << p_max << "] hist:";
  for (int i=0;i<10;i++) std::cerr << " " << buckets[i];
  std::cerr << " thr=" << g_baleen->lastThr << "\n";
}

      if (g_baleen->admit(p)) {
        g_admit++;
        if (auto handle = cache->allocate(defaultPool, key, kBlockSize)) {
          ssize_t got = direct_read_unaligned(
              g_origin,
              static_cast<char*>(handle->getMemory()) + offInSeg,
              len, reqStart);
          if (got > 0) {
            total.deviceReadBytes += static_cast<size_t>(got);
            accountDT(static_cast<size_t>(got));
          }
          cache->insertOrReplace(handle);
          g_baleen->accountAdmitBytes(kBlockSize);
          if (got > 0) {
            markCoveredRange(key, offInSeg, static_cast<size_t>(got));
          }
        } else {
          if (smallBuf.size() < len) smallBuf.resize(len);
          ssize_t got = direct_read_unaligned(g_origin, smallBuf.data(), len, reqStart);
          if (got > 0) {
            total.deviceReadBytes += static_cast<size_t>(got);
            accountDT(static_cast<size_t>(got));
          }
        }
      } else {
        g_reject++;
        if (smallBuf.size() < len) smallBuf.resize(len);
        ssize_t got = direct_read_unaligned(g_origin, smallBuf.data(), len, reqStart);
        if (got > 0) {
          total.deviceReadBytes += static_cast<size_t>(got);
          accountDT(static_cast<size_t>(got));
        }
      }

      total.reads++;
      total.readBytes += len;
      total.misses++;
      continue;
    }

    if (isWrite) {
      auto wh = cache->findToWrite(key);
      if (!wh) {
        std::array<double,18> fvW;
        const uint64_t ns = 0, user = 0;
        const uint64_t blk_id = b;
        const uint64_t within = offInSeg;
        const uint64_t chunk  = len;
        g_baleen->buildFeatures(false, ns, user, blk_id, within, chunk, fvW);
        double pw = g_baleen->predictProb(fvW);
static double p_min = 1.0, p_max = 0.0;
static size_t buckets[10] = {0};
p_min = std::min(p_min, pw);
p_max = std::max(p_max, pw);
int bkt = std::clamp(int(pw * 10), 0, 9);
buckets[bkt]++;
if (((total.reads + total.writes) % 200000) == 0) {
  std::cerr << "[Baleen] p in [" << p_min << "," << p_max << "] hist:";
  for (int i=0;i<10;i++) std::cerr << " " << buckets[i];
  std::cerr << " thr=" << g_baleen->lastThr << "\n";
}

        if (!g_baleen->admit(pw)) {
          g_w_reject++;
          static thread_local std::vector<char> tmpWT;
          if (tmpWT.size() < len) tmpWT.resize(len);
          readWritePayload(tmpWT.data(), len, reqStart);
          ssize_t w = direct_write_block(g_origin, tmpWT.data(), len, reqStart);
          if (w < 0) {
            std::cerr << "[write-through] pwrite failed: " << std::strerror(errno) << "\n";
          }
          if (w > 0) {
            total.writeBytes += static_cast<size_t>(w);
            accountDT(static_cast<size_t>(w));
          }
          total.writes++;
          continue;
        }

        if (auto h = cache->allocate(defaultPool, key, kBlockSize)) {
          cache->insertOrReplace(h);
          wh = cache->findToWrite(key);
          g_w_admit++;
          g_baleen->accountAdmitBytes(kBlockSize);
        } else {
          static thread_local std::vector<char> tmpWT;
          if (tmpWT.size() < len) tmpWT.resize(len);
          readWritePayload(tmpWT.data(), len, reqStart);
          ssize_t w = direct_write_block(g_origin, tmpWT.data(), len, reqStart);
          if (w < 0) {
            std::cerr << "[write-through] pwrite failed: " << std::strerror(errno) << "\n";
          }
          if (w > 0) {
            total.writeBytes += static_cast<size_t>(w);
            accountDT(static_cast<size_t>(w));
          }
          total.writes++;
          continue;
        }
      }

      static thread_local std::vector<char> tmpWriteBuf;
      if (tmpWriteBuf.size() < len) tmpWriteBuf.resize(len);
      readWritePayload(tmpWriteBuf.data(), len, reqStart);
      std::memcpy(static_cast<char*>(wh->getMemory()) + offInSeg, tmpWriteBuf.data(), len);

      markCoveredRange(key, offInSeg, len);
      markDirty(key);

      total.writeBytes += len;
      total.writes++;
      continue;
    }
  }
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  try {
    initDT();
    std::cout << "[SMART] Logical Sectors Written (before):\n";
    int smart_rc = std::system("sudo smartctl -x /dev/sdc | grep \"Logical Sectors Written\"");
    if (smart_rc == -1) std::cerr << "[SMART] system() failed to invoke smartctl\n";

    uint64_t originNeedBytes = 0;
    auto trace = loadTrace(kTracePath, originNeedBytes);
    std::cout << "Trace entries: " << trace.size() << "\n";
    ensureRegularFileSizedAtLeast(kOriginPath, (originNeedBytes == 0 ? kBlockSize : originNeedBytes));

    g_origin = openOrigin(kOriginPath);
    openPayloadIfAny();

    initializeCacheWithWriteBack();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (const auto& e : trace) {
      processEntry(e);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    auto mb = [](size_t b){ return b / 1024.0 / 1024.0; };
    auto iops = [seconds](size_t ops){ return seconds > 0 ? (ops / seconds) : 0.0; };
    const double r4kHitRatio = (total.r4kBlocks ? (100.0 * double(total.r4kHits) / double(total.r4kBlocks)) : 0.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Writes:  " << total.writes << " (" << total.writeBytes << " bytes)\n";
    std::cout << "Reads:   " << total.reads  << " (" << total.readBytes  << " bytes)\n";
    std::cout << "Misses:  " << total.misses << "  (miss rate "
              << (total.reads ? (100.0 * total.misses / total.reads) : 0.0) << "%)\n";
    std::cout << "Time:    " << (seconds * 1000.0) << " ms\n";
    std::cout << "IOPS W:  " << iops(total.writes) << "\n";
    std::cout << "IOPS R:  " << iops(total.reads)  << "\n";
    std::cout << "4KiB blocks read:       " << total.r4kBlocks << "\n";
    std::cout << "4KiB hits / misses:     " << total.r4kHits << " / " << total.r4kMisses
              << "  (hit ratio " << r4kHitRatio << "%)\n";

    double peakDT = 0.0;
    double sumDT  = 0.0;
    for (double wdt : g_dtWindows) {
      if (wdt > peakDT) peakDT = wdt;
      sumDT += wdt;
    }
    double avgDT = (!g_dtWindows.empty() ? (sumDT / g_dtWindows.size()) : 0.0);
    std::cout << "DT windows:            " << g_dtWindows.size() << "\n";
    std::cout << "Peak DT (max window):  " << peakDT << " s\n";

    closePayloadIfAny();
    closeOrigin(g_origin);
    cache.reset();
    return 0;

  } catch (const std::exception& ex) {
    std::cerr << "Fatal: " << ex.what() << "\n";
    return 1;
  }
}
