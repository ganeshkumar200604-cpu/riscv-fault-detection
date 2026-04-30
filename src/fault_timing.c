#include <stdint.h>

#define UART0       0x10000000
#define TOTAL_RUNS  1000

void uart_putc(char c) { *(volatile uint8_t *)UART0 = c; }
void uart_print(const char *s) { while (*s) uart_putc(*s++); }
void uart_print_uint32(uint32_t value) {
    char buffer[11]; int i = 10; buffer[i] = '\0';
    if (value == 0) { uart_putc('0'); return; }
    while (value > 0 && i > 0) { buffer[--i] = '0' + (value % 10); value /= 10; }
    uart_print(&buffer[i]);
}

uint32_t read_cycle()   { uint32_t v; asm volatile("csrr %0, mcycle"  : "=r"(v)); return v; }
uint32_t read_instret() { uint32_t v; asm volatile("csrr %0, minstret": "=r"(v)); return v; }
uint32_t read_sp()      { uint32_t v; asm volatile("mv %0, sp"        : "=r"(v)); return v; }
uint32_t read_ra()      { uint32_t v; asm volatile("mv %0, ra"        : "=r"(v)); return v; }

void workload() {
    volatile uint32_t sum = 0;
    for (uint32_t i = 0; i < 100000; i++) sum += i;
}

void inject_delay() {
    volatile uint32_t waste = 0;
    for (uint32_t i = 0; i < 500000; i++) waste += i;
}

int main() {
    uart_print("run,cycles,instructions,cpi,sp,ra,exception_flag,label\n");
    for (uint32_t run = 1; run <= TOTAL_RUNS; run++) {
        uint32_t c0 = read_cycle();
        uint32_t i0 = read_instret();
        uint32_t sp = read_sp();
        uint32_t ra = read_ra();
        workload();
        inject_delay();
        uint32_t cycles = read_cycle()   - c0;
        uint32_t insts  = read_instret() - i0;
        uart_print_uint32(run);    uart_putc(',');
        uart_print_uint32(cycles); uart_putc(',');
        uart_print_uint32(insts);  uart_putc(',');
        if (insts) {
            uart_print_uint32(cycles / insts); uart_putc('.');
            uint32_t frac = (cycles % insts) * 100 / insts;
            if (frac < 10) uart_putc('0');
            uart_print_uint32(frac);
        }
        uart_putc(',');
        uart_print_uint32(sp); uart_putc(',');
        uart_print_uint32(ra); uart_putc(',');
        uart_putc('0');        uart_putc(',');
        uart_print("timing_fault\n");
    }
    *((volatile uint32_t*)0x100000) = 0x5555;
}
