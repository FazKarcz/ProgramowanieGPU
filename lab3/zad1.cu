#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Definicja rozmiaru wektora
// UWAGA: Dla konfiguracji 1 bloku, N nie może przekroczyć limitu wątków na blok (zazwyczaj 1024)
#define N 1024 

// Kernel: Suma wektorów + pomiar cykli
__global__ void vectorAddWithCycles(const int *a, const int *b, int *c, long long *cycles) {
    // Ponieważ mamy tylko 1 blok, globalny indeks to po prostu threadIdx.x
    int i = threadIdx.x;

    if (i < N) {
        // b. Zbieranie miar: Liczba cykli (start)
        long long start_clock = clock64();

        // Właściwa operacja
        c[i] = a[i] + b[i];

        // b. Zbieranie miar: Liczba cykli (stop)
        long long end_clock = clock64();

        // Zapisz zużyte cykle dla tego wątku
        cycles[i] = end_clock - start_clock;
    }
}

int main() {
    // Alokacja pamięci hosta
    int *h_a, *h_b, *h_c;
    long long *h_cycles;
    size_t bytes = N * sizeof(int);
    size_t bytes_cycles = N * sizeof(long long);

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    h_cycles = (long long*)malloc(bytes_cycles);

    // Inicjalizacja danych
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Alokacja pamięci urządzenia (Device)
    int *d_a, *d_b, *d_c;
    long long *d_cycles;
    
    // b. Zbieranie miar: Wykorzystanie pamięci (przed alokacją na GPU)
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Pamiec GPU przed alokacja: Wolne: %zu B, Calkowite: %zu B\n", free_mem, total_mem);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_cycles, bytes_cycles);

    // Kopiowanie danych Host -> Device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Przygotowanie zdarzeń do pomiaru czasu (najdokładniejsza metoda w kodzie)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // a. Uruchomienie: 1 blok, N wątków
    printf("Uruchamianie kernela z <<<1, %d>>>...\n", N);
    
    cudaEventRecord(start);
    vectorAddWithCycles<<<1, N>>>(d_a, d_b, d_c, d_cycles);
    cudaEventRecord(stop);

    // Czekamy na zakończenie GPU
    cudaEventSynchronize(stop);

    // Obliczanie czasu wykonania
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Kopiowanie wyników Device -> Host
    cudaMemcpy(h_c, h_c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cycles, h_cycles, bytes_cycles, cudaMemcpyDeviceToHost);

    // Raportowanie wyników
    printf("\n--- WYNIKI PROFILOWANIA ---\n");
    printf("Czas wykonania kernela: %.5f ms\n", milliseconds);
    
    // Obliczanie średniej liczby cykli
    long long total_cycles = 0;
    for (int i = 0; i < N; i++) {
        total_cycles += h_cycles[i];
    }
    printf("Srednia liczba cykli na watek: %lld\n", total_cycles / N);
    
    // Sprawdzenie poprawności dla pierwszego elementu
    printf("Weryfikacja (indeks 0): %d + %d = %d\n", h_a[0], h_b[0], h_c[0]);

    // Sprzątanie
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_cycles);
    free(h_a); free(h_b); free(h_c); free(h_cycles);

    return 0;
}
