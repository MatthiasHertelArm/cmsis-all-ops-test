# ExecuTorch CMSIS-Pack — all-ops coverage test

A self-contained [CMSIS-Toolbox](https://github.com/Open-CMSIS-Pack/cmsis-toolbox)
solution that exercises the **entire operator surface** of the ExecuTorch
CMSIS-Pack on a Cortex-M55 target (Arm Corstone-300 / `ARMCM55`).

It gives every operator the pack ships two kinds of coverage:

- **Build / link** — [all_ops.cproject.yml](all_ops.cproject.yml) selects every
  ExecuTorch operator component, so linking the firmware exercises each op's
  generated registration, forward declaration and kernel source.
- **Execution** — 165 per-operator test models (`model.pte` + reference
  `input_*.bin` / `expected_*.bin`) are embedded directly into the ELF `.rodata`
  via [embedded_models_blob.S](embedded_models_blob.S). The firmware runs every
  model on the ExecuTorch runtime, compares against the expected output, and
  prints `PASS`/`FAIL` per op plus a final aggregate — no semihosting filesystem
  access needed at run time.

Models covered: **149 Portable + 16 Cortex-M = 165**. Operators that are
build/link-covered but not executed are skipped with a recorded reason, so there
are no silent coverage gaps.

## The ExecuTorch CMSIS-Pack

`PyTorch::ExecuTorch` is a **source** CMSIS-Pack that ships the ExecuTorch
runtime, kernel utilities, kernel registration, and **one selectable CMSIS
component per operator** (Portable / Quantized / Cortex-M categories), plus an
Ethos-U backend component.

- **No Docker for the build.** The firmware builds host-natively with the
  CMSIS-Toolbox (`csolution`/`cbuild`) and `arm-none-eabi-gcc` / Arm Compiler 6.
  Docker is only used by the pack's own consumer-validation smoke test and to
  *export* the test models with PyTorch — not to build or link this firmware.
- **Code size = pick the operators you need.** Each operator component is guarded
  by `#ifdef RTE_ML_EXECUTORCH_OP_<CATEGORY>_<NAME>`; only the operator components
  a project selects compile their registration / forward declaration / kernel in.
  Unselected operators are not linked. **This project deliberately selects *every*
  operator for coverage** — a real application selects only the subset its model
  uses, which is how you keep the image small.

## Prerequisites

- [CMSIS-Toolbox](https://github.com/Open-CMSIS-Pack/cmsis-toolbox) ≥ 2.14
  (`cbuild`, `csolution`, `cpackget`)
- A toolchain: `arm-none-eabi-gcc` 14.x (GCC), Arm Compiler 6 (AC6), or Arm
  Toolchain for Embedded / LLVM (CLANG)
- CLANG requires CMSIS-Toolbox ≥ 2.14
- These packs installed (declared in [all_ops.csolution.yml](all_ops.csolution.yml)):
  `PyTorch::ExecuTorch`, `ARM::CMSIS`, `ARM::Cortex_DFP`, `ARM::CMSIS-NN`,
  **`ARM::CMSIS-Compiler`**:

  ```bash
  cpackget add PyTorch.ExecuTorch.<version>.pack
  ```

## Build

```bash
cbuild all_ops.csolution.yml                    # GCC (default)
cbuild all_ops.csolution.yml --toolchain AC6    # Arm Compiler 6
```

The image is written to `out/all_ops/ARMCM55/Debug/all_ops.elf`. GCC and CLANG
link with `ARMCM55_large.ld`; AC6 uses the scatter file `ARMCM55_large.sct`.

## Run

On the **Arm FVP** (Corstone-300):

```bash
FVP_Corstone_SSE-300 \
  -C cpu0.semihosting-enable=1 \
  -C cpu0.INITSVTOR=0x70000000 \
  -C mps3_board.sse300.iotss3_systemcontrol.INITSVTOR_RST=0x70000000 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.shutdown_on_eot=1 \
  -a out/all_ops/ARMCM55/Debug/all_ops.elf --timelimit 600
```

> On Windows with FVP 11.31.x, add your Python install directory to `PATH` so the
> FVP can load `python.dll`.

stdout/stderr are routed over **Arm semihosting** by the
`ARM::CMSIS-Compiler:STDOUT:Custom` / `STDERR:Custom` components plus
[stdout_semihosting.c](stdout_semihosting.c) (`SYS_WRITEC`, `BKPT 0xAB`).
CMSIS-Compiler:CORE supplies the toolchain `_write`/retarget glue, so **no
`rdimon.specs` or `--specs=` flag is needed** — and you must *not* add one.

On **hardware** via a CMSIS-DAP probe (pyOCD): use the `CMSIS Load+Run` task in
[.vscode/tasks.json](.vscode/tasks.json). `.vscode/` ships ready-made FVP
(`fvp-gdb`) and pyOCD debug tasks/launch configs.

## Regenerating the embedded models

The committed `embedded_models_blob.S` + `embedded_models.{cpp,h}` are produced
from `models/` by:

```bash
python3 gen_embedded_models.py
```

Re-run it whenever `models/manifest.json` or any model directory changes. The
`.incbin` paths are emitted **relative** to the project directory; the cproject
carries both `add-path: .` (C/C++) **and `add-path-asm: .` (assembler)** so the
assembler can resolve them regardless of the build working directory.

## Layout

| Path | Purpose |
|------|---------|
| `all_ops.csolution.yml` / `all_ops.cproject.yml` | CMSIS solution selecting every operator component |
| `main.cpp` | All-ops runner: loads, runs and checks each embedded model |
| `models/` | Per-op `.pte` + reference input/expected vectors (`manifest.json` indexes them) |
| `embedded_models_blob.S` / `embedded_models.{cpp,h}` | Models embedded into `.rodata`; generated by `gen_embedded_models.py` |
| `stdout_semihosting.c` | CMSIS-Compiler STDOUT/STDERR:Custom backend (semihosting) |
| `et_pal_override.cpp` | Routes ExecuTorch `ET_LOG()` to `printf` |
| `ARMCM55_large.ld` / `ARMCM55_large.sct` | Linker script (GCC/CLANG) / scatter file (AC6), sized for the model blob |
| `random_device_shim.h` | `-include`d for CLANG/AtFE so the RNG ops build against libc++ |

## License

BSD-3-Clause — see [LICENSE](LICENSE). This is a test harness for the upstream
[ExecuTorch](https://github.com/pytorch/executorch) Arm/CMSIS backend.
