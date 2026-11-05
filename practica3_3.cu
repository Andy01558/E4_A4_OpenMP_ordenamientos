#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// =====================================================
// KERNELS
// =====================================================
__global__ void sumarArreglos(int* A, int* B, int* C, int N, bool verbose) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < N) {
        C[globalIdx] = A[globalIdx] + B[globalIdx];
        if (verbose && globalIdx % 5 == 0) {
            printf("[DEVICE] [Equipo: COLAB] [Bloque CUDA: %d] [Hilo CUDA: %d] [Indice Global: %d] [Operacion: Suma] -> %d\n", blockIdx.x, threadIdx.x, globalIdx, C[globalIdx]);
        }
    }
}

__global__ void reduccionSumatoria(int* A, int* resultadosParciales, int N, bool verbose) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? A[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        resultadosParciales[blockIdx.x] = sdata[0];
        if (verbose) {
            printf("[DEVICE] [Equipo: COLAB] [Bloque CUDA: %d] [Hilo CUDA: %d] [Operacion: Reduccion Suma Local] -> Parcial: %d\n", blockIdx.x, tid, sdata[0]);
        }
    }
}

__global__ void reduccionMaximo(int* A, int* resultadosParciales, int N, bool verbose) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? A[i] : -999999999;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        resultadosParciales[blockIdx.x] = sdata[0];
        if (verbose) {
            printf("[DEVICE] [Equipo: COLAB] [Bloque CUDA: %d] [Hilo CUDA: %d] [Operacion: Reduccion Maximo Local] -> Parcial: %d\n", blockIdx.x, tid, sdata[0]);
        }
    }
}

__global__ void reduccionMinimo(int* A, int* resultadosParciales, int N, bool verbose) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? A[i] : 999999999;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        resultadosParciales[blockIdx.x] = sdata[0];
        if (verbose) {
            printf("[DEVICE] [Equipo: COLAB] [Bloque CUDA: %d] [Hilo CUDA: %d] [Operacion: Reduccion Minimo Local] -> Parcial: %d\n", blockIdx.x, tid, sdata[0]);
        }
    }
}

// =====================================================
// MAIN
// =====================================================
int main() {
    FILE* logFile = fopen("log_ejecucion_cuda.txt", "w");
    fprintf(logFile, "=====================================================\n");
    fprintf(logFile, "INICIO - PRACTICA 3.3 REDUCE CUDA\n");
    fprintf(logFile, "[ZARATE CASTILLO ANDREA ITZEL]\n");
    fprintf(logFile, "[MARIN PUGA ULISES ISRAEL]\n");
    fprintf(logFile, "=====================================================\n");
    
    printf("=====================================================\n");
    printf("INICIO - PRACTICA 3.3 REDUCE CUDA\n");
    printf("[ZARATE CASTILLO ANDREA ITZEL]\n");
    printf("[MARIN PUGA ULISES ISRAEL]\n");
    printf("=====================================================\n");

    int N = 4000000; 
    int threadsPerBlock = 256; 
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t bytes = N * sizeof(int);

    int *h_A = new int[N], *h_B = new int[N];
    for (int i = 0; i < N; i++) { h_A[i] = i + 1; h_B[i] = (i + 1) * 2; }

    int *d_A, *d_B, *d_C, *d_parciales;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_parciales, numBlocks * sizeof(int));
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    printf("[HOST] [Equipo: COLAB] Iniciando operaciones con N=%d...\n\n", N);

    // 1. Suma
    printf("--- EJECUTANDO SUMA ---\n");
    sumarArreglos<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, false);
    cudaDeviceSynchronize();

    // 2. Reduccion Sumatoria
    printf("\n--- EJECUTANDO REDUCCION (SUMATORIA) ---\n");
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduccionSumatoria<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_A, d_parciales, N, false);
    cudaDeviceSynchronize();
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    int* h_parciales = new int[numBlocks];
    cudaMemcpy(h_parciales, d_parciales, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int sumaGlobal = 0;
    for (int i = 0; i < numBlocks; i++) sumaGlobal += h_parciales[i];
    printf("[HOST] [Equipo: COLAB] Tiempo de Reduccion (Kernel): %.4f ms\n", ms);
    printf("[HOST] [Equipo: COLAB] Resultado global (Sumatoria): %d\n", sumaGlobal);

    // 3. Reduccion Maximo
    printf("\n--- EJECUTANDO REDUCCION (MAXIMO) ---\n");
    reduccionMaximo<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_A, d_parciales, N, false);
    cudaDeviceSynchronize();
    cudaMemcpy(h_parciales, d_parciales, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int maxGlobal = -999999999;
    for (int i = 0; i < numBlocks; i++) if (h_parciales[i] > maxGlobal) maxGlobal = h_parciales[i];
    printf("[HOST] [Equipo: COLAB] Resultado global (Maximo): %d\n", maxGlobal);

    // 4. Reduccion Minimo
    printf("\n--- EJECUTANDO REDUCCION (MINIMO) ---\n");
    reduccionMinimo<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_A, d_parciales, N, false);
    cudaDeviceSynchronize();
    cudaMemcpy(h_parciales, d_parciales, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    int minGlobal = 999999999;
    for (int i = 0; i < numBlocks; i++) if (h_parciales[i] < minGlobal) minGlobal = h_parciales[i];
    printf("[HOST] [Equipo: COLAB] Resultado global (Minimo): %d\n", minGlobal);

    fprintf(logFile, "=====================================================\n");
    fprintf(logFile, "FIN - PRACTICA 3.3 REDUCE CUDA\n");
    fprintf(logFile, "[ZARATE CASTILLO ANDREA ITZEL]\n");
    fprintf(logFile, "[MARIN PUGA ULISES ISRAEL]\n");
    fprintf(logFile, "=====================================================\n");
    fclose(logFile);

    printf("\n=====================================================\n");
    printf("FIN - PRACTICA 3.3 REDUCE CUDA\n");
    printf("[ZARATE CASTILLO ANDREA ITZEL]\n");
    printf("[MARIN PUGA ULISES ISRAEL]\n");
    printf("=====================================================\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_parciales);
    delete[] h_A; delete[] h_B; delete[] h_parciales;

    return 0;
}
