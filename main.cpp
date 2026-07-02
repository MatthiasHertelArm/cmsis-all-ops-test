// Copyright 2026 Arm Limited and/or its affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// All-ops consumer runner, built entirely from the CMSIS Pack's ExecuTorch
// runtime + kernel components.
//
// Dual purpose:
//   * Build/link coverage -- all_ops.cproject.yml selects every operator
//     component, so linking this image exercises every op's generated
//     registration, forward declaration and kernel source.
//   * Execution coverage -- every model produced by generate_test_models.py
//     is embedded directly into the ELF (.rodata) via embedded_models.S, so
//     this image self-tests on bare metal without semihosting access to the
//     host filesystem. Iterates over every model.pte / input_*.bin /
//     expected_*.bin, prints "PASS"/"FAIL" per op, and a final aggregate.
//
// API usage mirrors examples/arm/executor_runner/arm_executor_runner.cpp.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include "embedded_models.h"
#include "arm_embedded_module.hpp"

using executorch::aten::Tensor;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

using namespace arm::embedded;

namespace {

constexpr size_t kMethodPoolSize = 16 * 1024 * 1024;
constexpr size_t kTempPoolSize = 4 * 1024 * 1024;

alignas(16) uint8_t g_method_pool[kMethodPoolSize];
alignas(16) uint8_t g_temp_pool[kTempPoolSize];

// Cortex-M55 DWT cycle counter, sampled around method->execute() to report a
// rough inference cost per op.
inline volatile uint32_t& reg32(uintptr_t addr) {
  return *reinterpret_cast<volatile uint32_t*>(addr);
}
constexpr uintptr_t kDemcr = 0xE000EDFC; // CoreDebug DEMCR, TRCENA = bit 24
constexpr uintptr_t kDwtCtrl = 0xE0001000; // DWT_CTRL, CYCCNTENA = bit 0
constexpr uintptr_t kDwtCyccnt = 0xE0001004;

// Per-op stats accumulated during the run and dumped as a table at the end.
struct RowStat {
  const EmbeddedModel* m;
  bool pass;
  size_t in_bytes;
  size_t out_bytes;
  uint32_t cycles;
};
constexpr size_t kMaxRows = 512;
RowStat g_rows[kMaxRows];

// Minimal in-memory DataLoader so we do not depend on the extension/ tree
// (the pack ships only the runtime/ + kernels).
class BufferLoader final : public DataLoader {
 public:
  BufferLoader(const void* data, size_t size) : data_(data), size_(size) {}

  Result<FreeableBuffer> load(
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo&) const override {
    if (offset + size > size_) {
      return Error::InvalidArgument;
    }
    return FreeableBuffer(
        static_cast<const uint8_t*>(data_) + offset, size, nullptr);
  }

  Result<size_t> size() const override {
    return size_;
  }

 private:
  const void* data_;
  size_t size_;
};

bool allclose(
    const Tensor& a,
    const Tensor& b,
    double rtol = 1e-6,
    double atol = 1e-6) 
{

  int a_bytes = a.numel() * a.element_size();
  int b_bytes = b.numel() * b.element_size();

  if (a_bytes != b_bytes)
  {
     printf(
        "  size mismatch: got %u vs expected %u bytes\n",
        static_cast<unsigned>(a_bytes),
        static_cast<unsigned>(b_bytes));
    return false;
  }

  const float* pa = a.const_data_ptr<float>();
  const float* pb = b.const_data_ptr<float>();

  float max_abs = 0.f;
  size_t worst = 0;

  bool ok = true;
  size_t n = a.numel();
  for (size_t i = 0; i < n; ++i) {
    float d = std::fabs(pa[i] - pb[i]);
    if (d > max_abs) {
        max_abs = d;
        worst = i;
    }
    if (d > atol + rtol * std::abs(pb[i])) {
      ok = false;
    }
  }
  printf(
        "  [cmp] %u float(s): max|err|=%g at [%u] (got %f vs exp %f),"
        " atol=%g rtol=%g -> %s\n",
        static_cast<unsigned>(n),
        max_abs,
        static_cast<unsigned>(worst),
        pa[worst],
        pb[worst],
        atol,
        rtol,
        ok ? "within tol" : "OUT OF TOL");
  return ok;
}

bool tensor_equal(
    const Tensor& a,
    const Tensor& b) {

    if (a.scalar_type() != b.scalar_type())
        return false;

    if (a.dim() != b.dim())
        return false;

    for (int i = 0; i < a.dim(); ++i) {
        if (a.size(i) != b.size(i))
            return false;
    }

    size_t nbytes = a.nbytes();  // if available
    bool ok = std::memcmp(a.const_data_ptr(), b.const_data_ptr(), nbytes) == 0;
    printf(
        "  [cmp] %u byte(s) exact compare -> %s\n",
        static_cast<unsigned>(nbytes),
        ok ? "match" : "MISMATCH");
    return ok;
}

bool tensors_match(
    const Tensor& got,
    const Tensor& expected,
    float atol,
    float rtol) {
 
  const auto dtype = got.scalar_type();
  if (dtype == executorch::aten::ScalarType::Float) 
  {
    return allclose(got, expected, rtol, atol);
  }
  else
  {
    return tensor_equal(got, expected);
  }
}

// Runs one embedded model end-to-end, returning true if it produced outputs
// matching the embedded expected_*.bin within (atol, rtol). Prints
// "Test_result: <op> PASS" / "FAIL (<stage>)" so a host log parser can
// aggregate results across the full run.
bool run_one_model(const EmbeddedModel& m, RowStat& row) {
  row.m = &m;
  row.pass = false;
  row.in_bytes = 0;
  row.out_bytes = 0;
  row.cycles = 0;
 

  auto method_allocator = std::make_unique<MemoryAllocator>(kMethodPoolSize,
                                                            g_method_pool);
  auto temp_allocator = std::make_unique<MemoryAllocator>(kTempPoolSize,
                                                          g_temp_pool);

  auto loader = std::make_unique<BufferLoader>(m.pte_data, m.pte_size);
  EmbeddedModule module_(m.pte_data,
                        m.pte_size,
                        std::move(loader),
                        std::move(method_allocator),
                        std::move(temp_allocator));

  
  size_t num_inputs = 0;
  auto nb_inputs_ = module_.execute("nb_inputs");
  if (nb_inputs_.ok())
     num_inputs = nb_inputs_.get()[0].toInt();

  size_t num_outputs = 0;
  auto num_outputs_ = module_.execute("nb_outputs");
  if (!num_outputs_.ok())
     return false;

  num_outputs = num_outputs_.get()[0].toInt();

  printf(
      "Test_exec: %s (dir=%s) pte=%u bytes, %u input(s), %u expected\n",
      m.op,
      m.dir,
      static_cast<unsigned>(m.pte_size),
      static_cast<unsigned>(num_inputs),
      static_cast<unsigned>(num_outputs));

  char method_name[256];

  for(size_t i = 0; i < num_inputs; ++i) 
  {
    sprintf(method_name, "input_%zu", i);
    auto input_ = module_.execute(method_name);
    if (input_.ok())
    {
      auto input = input_.get()[0];
      auto error = module_.set_input(input, i);
      if (error != Error::Ok) {
        printf("  input %u FAIL (set_input)\n", static_cast<unsigned>(i));
        return false;
      }
    }
  }
 
  printf("  [run] %s execute() ...\n", m.op);
  uint32_t c0 = reg32(kDwtCyccnt);
  const auto result = module_.forward();
  row.cycles = reg32(kDwtCyccnt) - c0;
  if (!result.ok())
  {
    printf("Test_result: %s FAIL (execute)\n", m.op);
    return false;
  }

 

  printf(
      "  [run] %s execute() ok, %u output(s)\n",
      m.op,
      static_cast<unsigned>(num_outputs));
  bool pass = true;

  float atol = 0;
  auto atol_ = module_.execute("atol");
  if (!atol_.ok())
     return false;
  atol = static_cast<float>(atol_.get()[0].toDouble());

  float rtol = 0;
  auto rtol_ = module_.execute("rtol");
  if (!rtol_.ok())
     return false;
  rtol = static_cast<float>(rtol_.get()[0].toDouble());

  for (size_t i = 0; i < num_outputs; ++i) 
  {
    const auto got = result->at(i).toTensor();
    
    sprintf(method_name, "output_%zu", i);
    auto output_ = module_.execute(method_name);
    if (!output_.ok())
    {
      printf("  output %u FAIL (get_output)\n", static_cast<unsigned>(i));
      pass = false;
      continue;
    }
    const auto expected = output_.get()[0].toTensor();
    
    if (!tensors_match(got, expected, atol, rtol)) {
      printf("  output %u mismatch\n", static_cast<unsigned>(i));
      pass = false;
    }
  }

  row.pass = pass;
  printf("Test_result: %s %s\n", m.op, pass ? "PASS" : "FAIL");
  return pass;
}

} // namespace

int main(void) {
  // Unbuffered stdout: the bare-metal startup may loop after main() returns
  // (never calling exit()), so block-buffered output would never flush. Make
  // every printf reach the semihosting console immediately.
  setvbuf(stdout, nullptr, _IONBF, 0);

  executorch::runtime::runtime_init();

  // Enable the DWT cycle counter so run_one_model can time each inference.
  reg32(kDemcr) |= (1u << 24);
  reg32(kDwtCyccnt) = 0;
  reg32(kDwtCtrl) |= 1u;

  size_t passed = 0;
  const size_t total = g_embedded_models_count;
  for (size_t mi = 0; mi < total; ++mi) {
    RowStat& row = g_rows[mi < kMaxRows ? mi : kMaxRows - 1];
    if (run_one_model(g_embedded_models[mi], row)) {
      ++passed;
    }
  }

  printf(
      "\n==== Per-op results (%u models) ====\n", static_cast<unsigned>(total));
  printf(
      "%-30s %-6s %10s %8s %8s %12s\n",
      "op",
      "result",
      "model(B)",
      "in(B)",
      "out(B)",
      "cycles");
  for (size_t mi = 0; mi < total && mi < kMaxRows; ++mi) {
    const RowStat& r = g_rows[mi];
    printf(
        "%-30s %-6s %10u %8u %8u %12u\n",
        r.m->op,
        r.pass ? "PASS" : "FAIL",
        static_cast<unsigned>(r.m->pte_size),
        static_cast<unsigned>(r.in_bytes),
        static_cast<unsigned>(r.out_bytes),
        static_cast<unsigned>(r.cycles));
  }

  printf(
      "Test_result: SUMMARY %u/%u PASS\n",
      static_cast<unsigned>(passed),
      static_cast<unsigned>(total));
  return passed == total ? 0 : 1;
}
