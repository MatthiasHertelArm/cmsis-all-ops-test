# ExecuTorch all-ops CMSIS projects

Standalone copy of the ExecuTorch CMSIS-Pack all-ops test, exported from
`backends/arm/cmsis_pack/test/all_ops/`. Four build contexts share one solution:

| Context | Target | Device | CMSIS-NN path | Models |
|---|---|---|---|---|
| `all_ops_cpu.Debug+SSE-315` | Corstone-315 FVP | `ARM::SSE-315-FVP` | MVE (Cortex-M85) | 167 CPU `.pte` |
| `all_ops_ethos_u.Debug+SSE-320` | Corstone-320 FVP | `ARM::SSE-320-FVP` | MVE + Ethos-U85 NPU | 143 Vela-delegated `.pte` (118 on NPU) |
| `all_ops_h7b3.Release+H7B3` | STM32H7B3I-DK board | `STMicroelectronics::STM32H7B3LIHxQ` | DSP (Cortex-M7) | 167 CPU `.pte` |
| `all_ops_m0plus.Release+CM0plus` | MPS2 FVP | `ARM::ARMCM0P` | SCALAR (Cortex-M0+, no DSP/MVE) | 167 CPU `.pte` |

The three CPU contexts cover all three CMSIS-NN scratch-sizing regimes
(MVE / DSP / SCALAR). Models 166-167 are scratch-buffer danger shapes
(`quantized_depthwise_conv2d_deep` with 384 channels, `quantized_conv2d_1xn_pad`)
whose DSP scratch requirement exceeds the MVE-sized buffer.

**Per-core scratch sizing:** each context embeds models whose AoT scratch is
sized for that context's core — the CPU/Corstone contexts use the M55/MVE
default, and the H7B3 (Cortex-M7) context embeds its own models exported with
`generate_test_models.py --target-core m7` so the scratch is DSP-sized at
compile time (e.g. `quantized_depthwise_conv2d_deep`: 19,200 B for DSP vs
12,400 B for MVE). No runtime scratch handling is needed — the danger shapes
verify that the per-core AoT sizing is correct; a model exported for the wrong
path would under-allocate on the DSP kernel, whose only guard is a NULL check.

## Notes on this export

The models were exported with executorch installed **from the checkout** (an
isolated venv, editable install), not a stale site-packages copy. That fixed a
signature skew that made the earlier export's quantized cortex_m ops fail on
device with a KernelCall arity mismatch:

- `quantized_add` now carries the fused `activation_min/max` args (13, was 11).
- `quantized_conv2d` / `quantized_depthwise_conv2d` now carry the AoT `scratch`
  tensor (13 / 14, were 12 / 13).

CPU export is now **165 exported, 0 failed** (was 159 / 6 failed). Ethos-U
delegates **118** ops to the NPU (was 106). `generate_test_models.py` also gains
two export-time guards that abort if executorch is imported from outside the
source tree, or if an exported cortex_m node's arity does not match the shipped
schema — so this class of skew can no longer reach the FVP silently.

## Layout

```
all_ops.csolution.yml         four contexts (SSE-315 / SSE-320 / H7B3 / CM0plus)
vcpkg-configuration.json      toolchain pins (cmsis-toolbox 2.14.1, GCC 14.3.1)
cpu/        Corstone-315 (Cortex-M85 MVE)   — 167 CPU models embedded
ethos_u/    Corstone-320 (Ethos-U85)        — 143 delegated models embedded
h7b3/       STM32H7B3I-DK (Cortex-M7 DSP)    — CPU models, UART1 + flash layout
m0plus/     MPS2 (Cortex-M0+ SCALAR)        — CPU models, no DSP/MVE
gen_cproject.py, gen_embedded_models.py   generators (for reference / regen)
```

The `PyTorch::ExecuTorch` pack is resolved by name from the pack index (not
bundled); build inside the AVH-MLOps environment where it is installed.

## Build & run

Requires CMSIS-Toolbox 2.14.1 (`vcpkg activate` in the AVH-MLOps image; run
`vcpkg x-update-registry --all` first so 2.14.1 resolves).

```sh
cbuild all_ops.csolution.yml --packs --update-rte --context all_ops_cpu.Debug+SSE-315
cbuild all_ops.csolution.yml --packs --update-rte --context all_ops_ethos_u.Debug+SSE-320
```

```sh
# Corstone-315
FVP_Corstone_SSE-315 -C mps4_board.subsystem.cpu0.semihosting-enable=1 \
  -C mps4_board.subsystem.cpu0.INITSVTOR=0x90000000 \
  -C mps4_board.subsystem.iotss3_systemcontrol.INITSVTOR_RST=0x90000000 \
  -C mps4_board.visualisation.disable-visualisation=1 \
  -a out/all_ops_cpu/SSE-315/Debug/all_ops_cpu.elf --timelimit 900

# Corstone-320 (Ethos-U85; num_macs must match the ethos-u85-256 AOT target)
FVP_Corstone_SSE-320 -C mps4_board.subsystem.cpu0.semihosting-enable=1 \
  -C mps4_board.subsystem.cpu0.INITSVTOR=0x90000000 \
  -C mps4_board.subsystem.iotss3_systemcontrol.INITSVTOR_RST=0x90000000 \
  -C mps4_board.subsystem.ethosu.num_macs=256 \
  -C mps4_board.visualisation.disable-visualisation=1 \
  -a out/all_ops_ethos_u/SSE-320/Debug/all_ops_ethos_u.elf --timelimit 900
```

Each run prints a per-op table (result, model/in/out bytes, cycles, `max|err|`),
a memory high-water-mark report (`MemReport:`), and a final
`Test_result: SUMMARY <pass>/<total> PASS`. On the Ethos-U build, ops that run
on the NPU but land outside tolerance are marked `PASS*` with the delta
reported. Ethos-U models use Vela memory_mode `Sram_Only` (weights co-located
with the DDR-loaded blob; other modes fail with `invalid_weight_stream`).
