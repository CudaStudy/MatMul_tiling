#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAT_MUL_NSH
#define MAT_MUL_SH

// Matrix Spec
#define ROW_SIZE 1024
#define K_SIZE 1024
#define COL_SIZE 2048
#define WORK_LOAD (ROW_SIZE * COL_SIZE)

#define MAT_SIZE_A (ROW_SIZE * K_SIZE)
#define MAT_SIZE_B (K_SIZE * COL_SIZE)
#define MAT_SIZE_C (ROW_SIZE * COL_SIZE)

#define BLOCK_SIZE 16

// Macro
#define INDEX2ROW(_index, _width) (int)((_index) / (_width))
#define INDEX2COL(_index, _width) ((_index) % (_width))
#define ID2INDEX(_row, _col, _width) (((_row) * (_width)) + (_col))

#define CPU 0

#ifdef MAT_MUL_SH
#define GPU 0
#define CPU2GPU 1
#define GPU2CPU 2
#endif

#ifdef MAT_MUL_NSH
#define NSH_GPU 0
#define NSH_CPU2GPU 1
#define NSH_GPU2CPU 2
#endif

#ifdef MAT_MUL_SH
__global__ void matMul_kernel_shared(float *_A, float *_B, float *_C, int _m, int _n, int _k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= _m || col >= _n)
        return;

    int row_offset = threadIdx.x;
    int col_offset = threadIdx.y;

    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE]; // 16 * 16 * 8B = 2048B
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE]; // 16 * 16 * 8B = 2048B

    for (int i = 0; i < ceil((float)_k / BLOCK_SIZE); i++)
    {
        int block_offset = i * BLOCK_SIZE;

        if (row >= _m || block_offset + col_offset >= _k)
            sA[row_offset][col_offset] = 0;
        else
            sA[row_offset][col_offset] = _A[ID2INDEX(row, block_offset + col_offset, _k)];

        if (col >= _m || block_offset + row_offset >= _k)
            sB[row_offset][col_offset] = 0;
        else
            sB[row_offset][col_offset] = _B[ID2INDEX(block_offset + row_offset, col, _n)];
    }
    __syncthreads(); // wait until all thread load the matrix

    // matrix multiplication
    int val = 0;

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        val += __fmul_rn(sA[row_offset][i], sB[col_offset][i]);
    }

    __syncthreads(); // wait until all thread load the matrix

    _C[ID2INDEX(row, col, _n)] = val;
}
#endif

#ifdef MAT_MUL_NSH
__global__ void matMul_kernel(float *_matA, float *_matB, float *_matC, int _m, int _n, int _k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= _m || col >= _n)
        return;

    float val = 0;
    for (int i = 0; i < _k; i++)
    {
        val += __fmul_rn(_matA[ID2INDEX(row, i, _k)], _matB[ID2INDEX(i, col, _n)]);
    }
    _matC[ID2INDEX(row, col, _n)] = val;
}
#endif

int main(void)
{
    DS_timer timer_cpu(1);
    timer_cpu.setTimerName(CPU, (char *)"[CPU]");

#ifdef MAT_MUL_SH
    DS_timer timer_sh(3);
    timer_sh.setTimerName(GPU, (char *)"[GPU_SHARED] : Multilplication");
    timer_sh.setTimerName(CPU2GPU, (char *)"[GPU_SHARED] : Host->Device");
    timer_sh.setTimerName(GPU2CPU, (char *)"[GPU_SHARED] : Device->Host");
#endif

#ifdef MAT_MUL_NSH
    DS_timer timer_nsh(3);
    timer_nsh.setTimerName(NSH_GPU, (char *)"[GPU_NOT_SHARED] : Multiplication");
    timer_nsh.setTimerName(NSH_CPU2GPU, (char *)"[GPU_NOT_SHARED] : Host->Device");
    timer_nsh.setTimerName(NSH_GPU2CPU, (char *)"[GPU_NOT_SHARED] : Device->Host");
#endif

    printf("Step1: Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", ROW_SIZE, K_SIZE, K_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);

    // host input matrix
    float *A = new float[MAT_SIZE_A];
    float *B = new float[MAT_SIZE_B];
    float *hostC = new float[MAT_SIZE_C]; // host result

    memset(A, 0, sizeof(float) * MAT_SIZE_A);
    memset(B, 0, sizeof(float) * MAT_SIZE_B);
    memset(hostC, 0, sizeof(float) * MAT_SIZE_C);

    // generate input matrices
    for (int i = 0; i < MAT_SIZE_A; i++)
    {
        A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    for (int i = 0; i < MAT_SIZE_B; i++)
    {
        B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }

    printf("Step2: CPU Matrix Multiplication\n");
    timer_cpu.onTimer(CPU);
    for (int row = 0; row < ROW_SIZE; row++)
    {
        for (int col = 0; col < COL_SIZE; col++)
        {
            int matC_idx = ID2INDEX(row, col, COL_SIZE);
            hostC[matC_idx] = 0;
            for (int i = 0; i < K_SIZE; i++)
            {
                hostC[matC_idx] += A[ID2INDEX(row, i, K_SIZE)] * B[ID2INDEX(i, col, COL_SIZE)];
            }
        }
    }
    timer_cpu.offTimer(CPU);

    printf("--------------------CPU MULTIPLICATION TOP--------------------\n");
    timer_cpu.printTimer();

    // check the results
    bool isCorrect = true;
    printf("--------------------CPU MULTIPLICATION BOT--------------------\n");

#ifdef MAT_MUL_SH

    // device result
    float *deviceC = new float[MAT_SIZE_C];
    memset(deviceC, 0, sizeof(float) * MAT_SIZE_C);

    // device I/O matrix
    float *dA, *dB, *dC;
    dA = dB = dC = NULL;

    // device memory allocaiton
    cudaMalloc(&dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&dC, sizeof(float) * MAT_SIZE_C);

    // Copy input matrices : H -> D
    printf("Step3: CPU -> GPU \n");
    timer_sh.onTimer(CPU2GPU);
    cudaMemcpy(dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
    timer_sh.offTimer(CPU2GPU);

    //// Kernel call (shared memory)
    dim3 gridDim_sh(ceil((float)ROW_SIZE / BLOCK_SIZE), ceil((float)COL_SIZE / BLOCK_SIZE));
    dim3 blockDim_sh(BLOCK_SIZE, BLOCK_SIZE);

    printf("Step4: GPU Matrix Mulatiplication\n");
    printf("Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", ROW_SIZE, K_SIZE, K_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);
    printf("Dim  : Grid = (%d, %d), BLock = (%d x %d)\n", gridDim_sh.x, gridDim_sh.y, blockDim_sh.x, blockDim_sh.y);

    timer_sh.onTimer(GPU);
    matMul_kernel_shared<<<gridDim_sh, blockDim_sh>>>(dA, dB, dC, ROW_SIZE, COL_SIZE, K_SIZE);
    cudaDeviceSynchronize();
    timer_sh.offTimer(GPU);

    // Get back result : D -> H
    printf("Step5: GPU -> CPU \n");
    timer_sh.onTimer(GPU2CPU);
    cudaMemcpy(deviceC, dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
    timer_sh.offTimer(GPU2CPU);

    float *pHostC = &hostC[0];
    float *pDeviceC = &deviceC[0];

    for (int i = 0; i < MAT_SIZE_C; i++)
    {
        if (pHostC[i] != pDeviceC[i])
        {
            printf("[%d] %.2f, %.2f\n", i, pHostC[i], pDeviceC[i]);
            isCorrect = false;
            break;
        }
    }

    if (isCorrect)
        printf("SHARED Result is correct!\n");
    else
        printf("SHARED Result is not correct!!!!!!\n");

    printf("--------------------GPU_SH MULTIPLICATION TOP--------------------\n");
    timer_sh.printTimer();
    printf("--------------------GPU_SH MULTIPLICATION BOT--------------------\n");

#endif

    // check the results
    isCorrect = true;

#ifdef MAT_MUL_NSH

    // device result
    float *nsh_deviceC = new float[MAT_SIZE_C];
    memset(nsh_deviceC, 0, sizeof(float) * MAT_SIZE_C);

    // device I/O matrix
    float *nsh_dA, *nsh_dB, *nsh_dC;
    nsh_dA = nsh_dB = nsh_dC = NULL;

    // device memory allocation
    cudaMalloc(&nsh_dA, sizeof(float) * MAT_SIZE_A);
    cudaMalloc(&nsh_dB, sizeof(float) * MAT_SIZE_B);
    cudaMalloc(&nsh_dC, sizeof(float) * MAT_SIZE_C);

    timer_nsh.onTimer(NSH_CPU2GPU);
    cudaMemcpy(nsh_dA, A, sizeof(float) * MAT_SIZE_A, cudaMemcpyHostToDevice);
    cudaMemcpy(nsh_dB, B, sizeof(float) * MAT_SIZE_B, cudaMemcpyHostToDevice);
    timer_nsh.offTimer(NSH_CPU2GPU);

    dim3 gridDim_nsh(ceil((float)ROW_SIZE / BLOCK_SIZE), ceil((float)COL_SIZE / BLOCK_SIZE));
    dim3 blockDim_nsh(BLOCK_SIZE, BLOCK_SIZE);
    printf("Step6: not_shared gpu multiplication start!\n");
    printf("Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", ROW_SIZE, K_SIZE, K_SIZE, COL_SIZE, ROW_SIZE, COL_SIZE);
    printf("Dim  : Grid = (%d, %d), BLock = (%d x %d)\n", gridDim_nsh.x, gridDim_nsh.y, blockDim_nsh.x, blockDim_nsh.y);

    //// Kernel call (not shared memory)
    timer_nsh.onTimer(NSH_GPU);
    matMul_kernel<<<gridDim_nsh, blockDim_nsh>>>(nsh_dA, nsh_dB, nsh_dC, ROW_SIZE, COL_SIZE, K_SIZE);
    cudaDeviceSynchronize();
    timer_nsh.offTimer(NSH_GPU);

    timer_nsh.onTimer(NSH_GPU2CPU);
    cudaMemcpy(nsh_deviceC, nsh_dC, sizeof(float) * MAT_SIZE_C, cudaMemcpyDeviceToHost);
    timer_nsh.offTimer(NSH_GPU2CPU);

    float *nsh_pHostC = &hostC[0];
    float *nsh_pDeviceC = &nsh_deviceC[0];
    for (int i = 0; i < MAT_SIZE_C; i++)
    {
        if (nsh_pHostC[i] != nsh_pDeviceC[i])
        {
            printf("[%d] %.2f, %.2f\n", i, nsh_pHostC[i], nsh_pDeviceC[i]);
            isCorrect = false;
        }
    }

    if (isCorrect)
        printf("NOT SHARED Result is correct!\n");
    else
        printf("NOT SHARED Result is not correct!!!!!!\n");

    printf("--------------------GPU_NSH MULTIPLICATION TOP--------------------\n");
    timer_nsh.printTimer();
    printf("--------------------GPU_NSH MULTIPLICATION BOT--------------------\n");
#endif

    return 0;
}
