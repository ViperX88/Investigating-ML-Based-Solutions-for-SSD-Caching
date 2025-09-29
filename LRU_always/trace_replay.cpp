#include <atomic>
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
#include <unordered_set>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include <folly/init/Init.h>
#include "cachelib/allocator/CacheAllocator.h"
#include "cachelib/allocator/MemoryTierCacheConfig.h"
#include "cachelib/allocator/CacheTraits.h"

using Cache = facebook::cachelib::LruAllocator;

constexpr const char* kTracePath   = "/proj/cac101-PG0/trace.csv";
constexpr const char* kOriginPath  = "/mnt/origin/origin.bin";
constexpr const char* kNvmFile     = "/mnt/cache-ssd/nvm.dat";
constexpr const char* kPersistBase = "/proj/cac101-PG0/cachelib_state";

constexpr uint64_t kDramMB    = 128;
constexpr uint64_t kNvmBytes  = 100ULL * 1024 * 1024 * 1024;
constexpr size_t   kBlockSize = 4096;
constexpr size_t   kAlign     = 4096;

inline bool isAligned(uint64_t off, size_t sz) { return (off % kAlign == 0) && (sz % kAlign == 0); }

inline void ensureDir(const char* dir) {
  if (::mkdir(dir, 0755) != 0 && errno != EEXIST) {
    throw std::runtime_error(std::strerror(errno));
  }
}

inline void ensureParentDirForFile(const char* filePath) {
  std::string p(filePath);
  auto slash = p.find_last_of('/');
  if (slash == std::string::npos) return;
  std::string dir = p.substr(0, slash);
  if (!dir.empty()) ensureDir(dir.c_str());
}

inline void ensureRegularFileExact(const char* path, uint64_t bytes) {
  ensureParentDirForFile(path);
  int fd = ::open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) throw std::runtime_error(std::strerror(errno));
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    int e = errno; ::close(fd);
    throw std::runtime_error(std::strerror(e));
  }
  ::close(fd);
}
inline void logFsType(const char* dir) {
  struct statfs s{}; if (::statfs(dir, &s) == 0) {
    std::cout << "[FS] " << dir << " f_type=0x" << std::hex << s.f_type << std::dec << "\n";
  } else {
    std::cout << std::strerror(errno) << "\n";
  }
}

inline uint64_t getFileSize(const char* path) {
  struct stat st{}; if (::stat(path, &st) == 0) return static_cast<uint64_t>(st.st_size);
  return 0;
}

inline void ensureRegularFileSizedAtLeast(const char* path, uint64_t bytes) {
  ensureParentDirForFile(path);
  int fd = ::open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) throw std::runtime_error(std::strerror(errno));
  uint64_t cur = getFileSize(path);
  if (cur < bytes) {
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      int e = errno; ::close(fd);
      throw std::runtime_error(std::strerror(e));
    }
  }
  ::close(fd);
}

// --- Disk-head time (DT) tracking ---
static double g_seekCostSec = []{
  if (const char* s = std::getenv("BALEEN_SEEK_COST_MS")) {
    try { return std::stod(s) / 1000.0; } catch (...) {}
  }
  return 0.005; // default 5 ms per I/O
}();

static double g_byteCostSec = []{
  if (const char* s = std::getenv("BALEEN_BYTE_COST_NS")) {
    try { return std::stod(s) * 1e-9; } catch (...) {}
  }
  return 5e-9;  // default ~200 MB/s => 5 ns/byte
}();

static double g_dtWindowSec = []{
  if (const char* s = std::getenv("BALEEN_DT_WINDOW_SEC")) {
    try { return std::stod(s); } catch (...) {}
  }
  return 600.0; // 10 minutes
}();

static std::chrono::steady_clock::time_point g_runStart;
static std::vector<double> g_dtWindows;

inline void initDT() {
  g_runStart = std::chrono::steady_clock::now();
  g_dtWindows.clear();
}
inline void accountDT(size_t bytes) {
  const double dt = g_seekCostSec + g_byteCostSec * static_cast<double>(bytes);
  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - g_runStart).count();
  const size_t idx = static_cast<size_t>(elapsed / g_dtWindowSec);
  if (idx >= g_dtWindows.size()) g_dtWindows.resize(idx + 1, 0.0);
  g_dtWindows[idx] += dt;
}


struct TraceEntry { std::string op; uint64_t offset; uint32_t size; };
std::vector<TraceEntry> loadTrace(const std::string& filename, uint64_t& originNeededBytes) {
  std::vector<TraceEntry> trace;
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Failed to open trace");
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
    throw std::runtime_error(std::strerror(errno));
  }
  fds.fd_direct = ::open(path, O_RDWR | O_DIRECT);
  if (fds.fd_direct < 0) {
    std::cerr << "O_DIRECT open failed " << std::strerror(errno)
              << "safeback\n";
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

struct Counters {
  size_t writes{0}, writeBytes{0};
  size_t reads{0},  readBytes{0}, misses{0};
  size_t deviceReadBytes{0};
  size_t writeBackBytes{0};
} total;

inline void markDirty(const std::string& key) { g_dirty.insert(key); }
inline bool consumeDirtyIfPresent(const std::string& key) {
  auto it = g_dirty.find(key);
  if (it == g_dirty.end()) return false;
  g_dirty.erase(it);
  return true;
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

  config.setCacheSize(kDramMB * 1024ULL * 1024ULL)
        .setAccessConfig({20, 10})
        .configureMemoryTiers({ facebook::cachelib::MemoryTierCacheConfig::fromShm().setRatio(1) });

  ensureDir("/mnt/cache-ssd");
  ensureDir("/mnt/origin");
  ensureRegularFileExact(kNvmFile, kNvmBytes);
  logFsType("/mnt/cache-ssd");

  Cache::NvmCacheConfig nvmConfig;
  nvmConfig.navyConfig.setSimpleFile(kNvmFile, kNvmBytes, /*useDirectIO*/ true);
  nvmConfig.navyConfig.enableRandomAdmPolicy().setAdmProbability(1.0);
  config.enableNvmCache(nvmConfig);

  auto itemDestructor = [&](const Cache::DestructorData& data) {
    using DC = facebook::cachelib::DestructorContext;
    if (data.context != DC::kEvictedFromRAM && data.context != DC::kEvictedFromNVM) {
      return;
    }
    auto keySP = data.item.getKey();
    std::string key(keySP.data(), keySP.size());
    if (!consumeDirtyIfPresent(key)) return;

    uint64_t blockOff = 0; try { blockOff = std::stoull(key); } catch (...) { std::cerr << "[wb] destructor: bad key\n"; return; }
    const void* src = data.item.getMemory();
    ssize_t w = direct_write_block(g_origin, src, kBlockSize, blockOff);
    if (w < 0) std::cerr << "pwrite failed " << std::strerror(errno) << "\n";
    else {
      total.writeBackBytes += static_cast<size_t>(w);
      accountDT(static_cast<size_t>(w));
    }
  };
  config.setItemDestructor(itemDestructor);

  config.validate();
  cache = std::make_unique<Cache>(Cache::SharedMemNew, config);
  defaultPool = cache->addPool("default", cache->getCacheMemoryStats().ramCacheSize);

}

void processEntry(const TraceEntry& e) {
  std::vector<char> blkBuf(kBlockSize);
  std::vector<char> tmpWriteBuf;

  const uint64_t startBlock = e.offset / kBlockSize;
  const uint64_t endBlock   = (e.offset + e.size - 1) / kBlockSize;

  for (uint64_t b = startBlock; b <= endBlock; ++b) {
    const uint64_t blockOff = b * kBlockSize;
    const std::string key = std::to_string(blockOff);

    if (e.op == "Read") {
      auto h = cache->find(key);
      const uint64_t rStart = std::max<uint64_t>(blockOff, e.offset);
      const uint64_t rEnd   = std::min<uint64_t>(blockOff + kBlockSize, e.offset + e.size);
      const size_t   len    = (rEnd > rStart) ? static_cast<size_t>(rEnd - rStart) : 0;
      const size_t   offInBlk = static_cast<size_t>(rStart - blockOff);
      if (len == 0) continue;

      if (h) {
        total.reads++;
        total.readBytes += len;
        continue;
      }
      ssize_t got = direct_read_unaligned(g_origin, blkBuf.data(), kBlockSize, blockOff);
      if (got < 0) { std::cerr << "pread failed: " << std::strerror(errno) << "\n"; continue; }
      total.deviceReadBytes += static_cast<size_t>(got);

      if (auto handle = cache->allocate(defaultPool, key, kBlockSize)) {
        std::memcpy(handle->getMemory(), blkBuf.data(), kBlockSize);
        cache->insertOrReplace(handle);
      } else {
        std::cerr << "alloc fail\n";
      }
      total.reads++;
      total.readBytes += len;
      total.misses++;

    } else if (e.op == "Write") {
      const uint64_t wStart = std::max<uint64_t>(blockOff, e.offset);
      const uint64_t wEnd   = std::min<uint64_t>(blockOff + kBlockSize, e.offset + e.size);
      if (wEnd <= wStart) {
        continue;
      }
      const size_t len      = static_cast<size_t>(wEnd - wStart);
      const size_t offInBlk = static_cast<size_t>(wStart - blockOff);
      const bool fullBlockWrite = (len == kBlockSize) && (offInBlk == 0);

      auto wh = cache->findToWrite(key);
      if (!wh) {
        auto h = cache->allocate(defaultPool, key, kBlockSize);
        if (!h) { std::cerr << "alloc fail\n"; continue; }

        if (!fullBlockWrite) {
          ssize_t got = direct_read_unaligned(g_origin, h->getMemory(), kBlockSize, blockOff);
          if (got < 0) {
            std::cerr << std::strerror(errno);
            std::memset(h->getMemory(), 0, kBlockSize);
          } else {
            total.deviceReadBytes += static_cast<size_t>(got);
            if (static_cast<size_t>(got) < kBlockSize) {
              std::memset(static_cast<char*>(h->getMemory()) + got, 0, kBlockSize - static_cast<size_t>(got));
              accountDT(static_cast<size_t>(got)); 
            }
          }
        }

        cache->insertOrReplace(h);
        wh = cache->findToWrite(key);
        if (!wh) { std::cerr << "Warning: findToWrite failed after insert\n"; continue; }
      }

      if (tmpWriteBuf.size() < len) tmpWriteBuf.resize(len);
      readWritePayload(tmpWriteBuf.data(), len, wStart);
      std::memcpy(static_cast<char*>(wh->getMemory()) + offInBlk, tmpWriteBuf.data(), len);

      markDirty(key);
      total.writeBytes += len;
      total.writes++;
    }
  }
}

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  try {

    std::cout << "[SMART] Logical Sectors Written (before):\n";
    int smart_rc = std::system("sudo smartctl -x /dev/sdc | grep \"Logical Sectors Written\"");

    uint64_t originNeedBytes = 0;
    auto trace = loadTrace(kTracePath, originNeedBytes);
    std::cout << "Trace entries: " << trace.size() << "\n";

    ensureParentDirForFile(kOriginPath);
    ensureRegularFileSizedAtLeast(kOriginPath, originNeedBytes ? originNeedBytes : kBlockSize);

    g_origin = openOrigin(kOriginPath);
    openPayloadIfAny();

    initializeCacheWithWriteBack();
    initDT();

    const auto t0 = std::chrono::high_resolution_clock::now();

    for (const auto& e : trace) {
      processEntry(e);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();

    {
    int mnt = ::open("/mnt/cache-ssd", O_RDONLY | O_DIRECTORY);
    if (mnt >= 0) { ::syncfs(mnt); ::close(mnt); }
    else { ::sync(); }
    }

    auto mb = [](size_t b){ return b / 1024.0 / 1024.0; };
    auto iops = [seconds](size_t ops){ return seconds > 0 ? (ops / seconds) : 0.0; };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Writes:  " << total.writes << " (" << total.writeBytes << " bytes)\n";
    std::cout << "Reads:   " << total.reads  << " (" << total.readBytes  << " bytes)\n";
    std::cout << "Misses:  " << total.misses << "  (miss rate "
              << (total.reads ? (100.0 * total.misses / total.reads) : 0.0) << "%)\n";
    std::cout << "Time:    " << (seconds * 1000.0) << " ms\n";
    std::cout << "IOPS W:  " << iops(total.writes) << "\n";
    std::cout << "IOPS R:  " << iops(total.reads)  << "\n";
    double peakDT = 0.0, sumDT = 0.0;
    for (double wdt : g_dtWindows) { if (wdt > peakDT) peakDT = wdt; sumDT += wdt; }
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

