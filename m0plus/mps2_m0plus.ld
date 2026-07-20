/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Derived from the ARM Cortex-M85 GCC linker script, retargeted to the
 * Corstone-315 (SSE-315) FVP's DDR regions.
 *
 * The all-ops coverage build links EVERY ExecuTorch operator into one image
 * (~5-6 MB of operator code; runtime pools right-sized to ~360 KB), which does not fit the small on-chip
 * regions (ITCM/DTCM, ISRAM 2 MB @0x21000000, 2 MB FPGA SRAM). The MPS4 board
 * backs large DDR4 windows, so everything is placed in DDR. This image exists
 * only to exercise the full compile+link+execute surface; no real device runs
 * all operators at once.
 *
 * Boot: the vector table is the first thing in FLASH (0x90000000), so run the
 * FVP with INITSVTOR pinned to that base (see run.sh / fvp_config.txt) so reset
 * fetches SP/PC from DDR.
 */

/* MPS2 (FVP_MPS2_Cortex-M0plus) memory map:
     ZBT SSRAM1  4 MB @ 0x00000000  -- code + rodata (boot alias)
     ZBT SSRAM2/3 4 MB @ 0x20000000 -- data/bss/heap/stack
     PSRAM      16 MB @ 0x21000000  -- embedded test models (.rodata blob)
   The Armv6-M core boots from address 0 (vector table at ROM base). */
__ROM_BASE = 0x00000000;
__ROM_SIZE = 0x00400000;   /* 4 MB ZBT SSRAM1 */

__RAM_BASE = 0x20000000;
__RAM_SIZE = 0x00400000;   /* 4 MB ZBT SSRAM2/3 */

/*--------------------- Stack / Heap Configuration ----------------------------*/
__STACK_SIZE = 0x00010000; /* 64 KB */
__HEAP_SIZE  = 0x00100000; /* 1 MB  */

__STACKSEAL_SIZE = 0;


MEMORY
{
  FLASH (rx)  : ORIGIN = __ROM_BASE, LENGTH = __ROM_SIZE
  RAM   (rwx) : ORIGIN = __RAM_BASE, LENGTH = __RAM_SIZE
  PSRAM (rx)  : ORIGIN = 0x21000000, LENGTH = 0x01000000
}

ENTRY(Reset_Handler)

SECTIONS
{
  /* Embedded test models (~1 MB) exceed what fits alongside v6-M code in the
     4 MB ZBT; keep the blob in PSRAM. Placed before .text so this specific
     pattern claims the section ahead of the *(.rodata*) catch-all. */
  .embedded_models :
  {
    . = ALIGN(16);
    *(.rodata.embedded_models)
  } > PSRAM

  .text :
  {
    KEEP(*(.vectors))
    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)

    KEEP(*(.eh_frame*))
  } > FLASH

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > FLASH

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > FLASH
  __exidx_end = .;

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;

    LONG (LOADADDR(.data))
    LONG (ADDR(.data))
    LONG (SIZEOF(.data) / 4)

    __copy_table_end__ = .;
  } > FLASH

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    __zero_table_end__ = .;
  } > FLASH

  /*
   * This __etext variable is kept for backward compatibility with older,
   * ASM based startup files.
   */
  PROVIDE(__etext = LOADADDR(.data));

  .data : ALIGN(4)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    /* preinit data.  picolibc's __libc_init_array walks a single combined
       array bounded by __bothinit_array_start / __bothinit_array_end (rather
       than GNU's separate preinit and init arrays), so mark that span here --
       otherwise those bounds are weak-undefined (0/0) and NO static ctors run
       under CLANG/picolibc. Harmless for GCC/newlib, which uses the init array. */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    PROVIDE_HIDDEN (__bothinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    /* init data -- kept contiguous with preinit for __bothinit_array */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
    PROVIDE_HIDDEN (__bothinit_array_end = .);

    . = ALIGN(4);
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(4);
    /* All data end */
    __data_end__ = .;

  } > RAM AT > FLASH

  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > RAM AT > RAM

  /* Ethos-U NPU scratch/cache buffer (ethos_setup.cpp, Corstone-320 only).
     NOLOAD: the driver fills it at runtime, so it need not be zeroed at boot.
     Empty on Corstone-315, which links no Ethos-U sources. */
  .ethos_cache_buf (NOLOAD) : ALIGN(32)
  {
    *(ethos_cache_buf)
  } > RAM

  /* TLS template: picolibc places errno (and other __thread vars) here. */
  .tdata : ALIGN(8)
  {
    __tdata_start = .;
    *(.tdata .tdata.* .gnu.linkonce.td.*)
    __tdata_end = .;
  } > RAM AT > FLASH

  .tbss (NOLOAD) : ALIGN(8)
  {
    __tbss_start = .;
    *(.tbss .tbss.* .gnu.linkonce.tb.* .tcommon)
    __tbss_end = .;
  } > RAM

  .heap (NOLOAD) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > RAM

  .stack (ORIGIN(RAM) + LENGTH(RAM) - __STACK_SIZE - __STACKSEAL_SIZE) (NOLOAD) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > RAM
  PROVIDE(__stack = __StackTop);

  /* picolibc crt0 (Arm Toolchain for Embedded) expects these symbols. They
     are unused by GCC/newlib and AC6, so PROVIDE keeps this script portable:
     the aliases only materialise when picolibc's startup references them. */
  PROVIDE(__data_start  = __data_start__);
  PROVIDE(__data_end    = __data_end__);
  PROVIDE(__data_source = LOADADDR(.data));
  PROVIDE(__data_size   = __data_end__ - __data_start__);
  PROVIDE(__bss_start   = __bss_start__);
  PROVIDE(__bss_end     = __bss_end__);
  PROVIDE(__bss_size    = __bss_end__ - __bss_start__);
  PROVIDE(__tls_base     = __tdata_start);
  PROVIDE(__tls_end      = __tbss_end);
  PROVIDE(__tdata_source = LOADADDR(.tdata));
  PROVIDE(__tdata_size   = __tdata_end - __tdata_start);
  PROVIDE(__tbss_size    = __tbss_end - __tbss_start);
  PROVIDE(__tbss_offset  = __tbss_start - __tdata_start);
  PROVIDE(__tls_size     = __tbss_end - __tdata_start);
  PROVIDE(__tls_align    = 8);
  PROVIDE(__arm32_tls_tcb_offset = 8);
  PROVIDE(__stack_limit = __StackLimit);
  PROVIDE(__heap_start  = __end__);
  PROVIDE(__heap_end    = __HeapLimit);

  /* Check if data + heap + stack exceeds RAM limit */
  ASSERT(__StackLimit >= __HeapLimit, "region RAM overflowed with stack")
}
