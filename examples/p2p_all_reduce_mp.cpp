// Multi-process P2P all-reduce reference (world = 2, fork + socketpair).
//
// Demonstrates the integration-owned rendezvous the library deliberately does
// not ship: each process allocates its own [signal | staging] region, exports
// it as a dma_buf fd (quixicore/xpu/ipc.hpp), sends the fd to the peer over a
// Unix socketpair with SCM_RIGHTS (an L0 fd is process-local — it cannot be
// passed by value), opens the peer's region, and runs ops::all_reduce. Both
// processes verify the fp32 fixed-rank-order reference bitwise.
//
// Needs >= 2 GPUs and Level Zero IPC; exits 0 with a skip message otherwise.

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/ipc.hpp"
#include "quixicore/xpu/ops.hpp"

namespace {

// Send/receive one fd over a Unix socket (SCM_RIGHTS).
bool send_fd(int sock, int fd) {
  char byte = 'f';
  iovec iov{&byte, 1};
  char ctrl[CMSG_SPACE(sizeof(int))] = {};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl;
  msg.msg_controllen = sizeof(ctrl);
  cmsghdr* cm = CMSG_FIRSTHDR(&msg);
  cm->cmsg_level = SOL_SOCKET;
  cm->cmsg_type = SCM_RIGHTS;
  cm->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cm), &fd, sizeof(int));
  return sendmsg(sock, &msg, 0) == 1;
}

int recv_fd(int sock) {
  char byte = 0;
  iovec iov{&byte, 1};
  char ctrl[CMSG_SPACE(sizeof(int))] = {};
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl;
  msg.msg_controllen = sizeof(ctrl);
  if (recvmsg(sock, &msg, 0) != 1) return -1;
  cmsghdr* cm = CMSG_FIRSTHDR(&msg);
  if (cm == nullptr || cm->cmsg_type != SCM_RIGHTS) return -1;
  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(cm), sizeof(int));
  return fd;
}

int run_rank(int rank, int sock, const sycl::device& dev, std::size_t count) {
  using namespace quixicore::xpu;
  sycl::queue q(dev);
  if (!ipc::available(q)) {
    std::printf("rank %d: Level Zero IPC unavailable (skip)\n", rank);
    return 0;
  }

  const std::size_t bytes = count * sizeof(float);
  const std::size_t region_bytes = ops::all_reduce_region_bytes(bytes);
  void* self_region = sycl::malloc_device(region_bytes, q);
  float* inp = sycl::malloc_device<float>(count, q);
  float* out = sycl::malloc_device<float>(count, q);
  q.memset(self_region, 0, region_bytes).wait();

  std::vector<float> host_in(count);
  for (std::size_t i = 0; i < count; ++i)
    host_in[i] = static_cast<float>((i * 31 + rank * 7) % 101) * 0.25f;
  q.memcpy(inp, host_in.data(), bytes).wait();

  const int self_fd = ipc::export_region_fd(q, self_region);
  if (self_fd < 0 || !send_fd(sock, self_fd)) {
    std::printf("rank %d: fd export/send failed\n", rank);
    return 1;
  }
  const int peer_fd = recv_fd(sock);
  void* peer_region = ipc::open_region_fd(q, peer_fd);
  if (peer_region == nullptr) {
    std::printf("rank %d: peer open failed\n", rank);
    return 1;
  }

  void* regions[2];
  regions[rank] = self_region;
  regions[1 - rank] = peer_region;

  // Sync so neither rank launches before both regions are zeroed and mapped.
  char token = 's';
  (void)!write(sock, &token, 1);
  (void)!read(sock, &token, 1);

  ops::all_reduce(q, inp, out, regions, rank, /*world=*/2, count,
                  quixicore::xpu::DType::f32);

  // Reference: fp32 fixed rank order (both ranks' inputs are derivable).
  std::vector<float> got(count), ref(count);
  q.memcpy(got.data(), out, bytes).wait();
  for (std::size_t i = 0; i < count; ++i) {
    float acc = 0.0f;
    for (int r = 0; r < 2; ++r)
      acc += static_cast<float>((i * 31 + r * 7) % 101) * 0.25f;
    ref[i] = acc;
  }
  const bool ok = std::memcmp(got.data(), ref.data(), bytes) == 0;
  std::printf("rank %d: %s\n", rank, ok ? "bitwise OK" : "MISMATCH");

  ipc::close_region(q, peer_region);
  sycl::free(self_region, q);
  sycl::free(inp, q);
  sycl::free(out, q);
  return ok ? 0 : 1;
}

}  // namespace

int main() {
  std::vector<sycl::device> devices;
  for (const auto& p : sycl::platform::get_platforms()) {
    std::vector<sycl::device> gpus;
    for (auto& d : p.get_devices())
      if (d.is_gpu()) gpus.push_back(d);
    if (gpus.size() > devices.size()) devices = std::move(gpus);
  }
  if (devices.size() < 2) {
    std::printf("p2p_all_reduce_mp: <2 GPUs (skip)\n");
    return 0;
  }
  quixicore::xpu::ipc::enable_peer_access(devices[0], devices[1]);
  quixicore::xpu::ipc::enable_peer_access(devices[1], devices[0]);

  int socks[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, socks) != 0) {
    std::perror("socketpair");
    return 1;
  }
  const pid_t child = fork();
  if (child == 0) {
    close(socks[0]);
    const int rc = run_rank(1, socks[1], devices[1], 8192);
    close(socks[1]);
    _exit(rc);
  }
  close(socks[1]);
  const int rc0 = run_rank(0, socks[0], devices[0], 8192);
  close(socks[0]);
  int status = 0;
  waitpid(child, &status, 0);
  const int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  std::printf("p2p_all_reduce_mp: %s\n",
              (rc0 == 0 && rc1 == 0) ? "PASS" : "FAIL");
  return rc0 == 0 && rc1 == 0 ? 0 : 1;
}
