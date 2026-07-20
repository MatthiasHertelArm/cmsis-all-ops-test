/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * STM32H7B3I-DK board support for the all-ops runner.
 *
 * CMSIS-Compiler STDOUT/STDERR:Custom backend routed to USART1 (PA9/PA10,
 * AF7), which the on-board STLINK-V3E exposes as a Virtual COM Port. The
 * board runs on the reset clock tree (HSI, 64 MHz, all prescalers /1), so the
 * USART1 kernel clock (rcc_pclk2 at reset) is 64 MHz.
 *
 * The constructor also enables the I/D caches and unlocks the DWT (the
 * Cortex-M7 gates DWT_CTRL writes behind DWT_LAR, unlike the Cortex-M85 the
 * FVP build targets), so main()'s cycle counter enable takes effect.
 */

#include "stm32h7b3xxq.h"

#include "retarget_stdout.h"
#include "retarget_stderr.h"

#define UART_BAUD 115200u
#define UART_KERNEL_CLK 64000000u

static void uart1_init(void) {
  RCC->AHB4ENR |= RCC_AHB4ENR_GPIOAEN;
  RCC->APB2ENR |= RCC_APB2ENR_USART1EN;
  (void)RCC->APB2ENR;

  /* PA9/PA10 -> alternate function 7 (USART1 TX/RX), push-pull, no pull. */
  /* Canonical MODER names: the legacy GPIO_MODER_MODE10_* aliases in ST's
     stm32h7b3xxq.h are truncated (_Po/_Ms) and do not compile. */
  GPIOA->MODER = (GPIOA->MODER & ~(GPIO_MODER_MODER9 | GPIO_MODER_MODER10)) |
                 (2u << GPIO_MODER_MODER9_Pos) | (2u << GPIO_MODER_MODER10_Pos);
  GPIOA->AFR[1] = (GPIOA->AFR[1] & ~(GPIO_AFRH_AFSEL9 | GPIO_AFRH_AFSEL10)) |
                  (7u << GPIO_AFRH_AFSEL9_Pos) | (7u << GPIO_AFRH_AFSEL10_Pos);

  USART1->BRR = (UART_KERNEL_CLK + UART_BAUD / 2u) / UART_BAUD;
  USART1->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

__attribute__((constructor)) static void board_init(void) {
  SCB_EnableICache();
  SCB_EnableDCache();
  /* DWT_LAR (not in the CMSIS DWT_Type): unlock so main() can enable CYCCNT */
  *(volatile uint32_t *)0xE0001FB0u = 0xC5ACCE55u;
  uart1_init();
}

static int uart1_putc(int ch) {
  while ((USART1->ISR & USART_ISR_TXE_TXFNF) == 0u) {
  }
  USART1->TDR = (uint8_t)ch;
  return ch;
}

int stdout_putchar(int ch) { return uart1_putc(ch); }
int stderr_putchar(int ch) { return uart1_putc(ch); }
