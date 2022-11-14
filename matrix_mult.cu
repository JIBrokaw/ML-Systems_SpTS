/*****************************************************************************
 * File:        matrix_mult.cu
 *
 * Run:         ./matrix_mult
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "common_string.h"
#include "support.h"
#include "kernel.cu"


void triangular_solve_matrix(float* matrix_L, float* b, float* x, int size){
  for (int row = 0; row < size; row++){
    float left_sum = 0;
    for (int col = 0; col < row; col++){
      left_sum += matrix_L[size*row + col]*x[col];
    }
    x[row] = (b[row] - left_sum);
  }
}

void triangular_solve_csr(CSR* matrix_L, float* b, float* x, int size){
  for (int row = 0; row < size; row++){
    float left_sum = 0;
    for (int index = matrix_L->row_pointer[row]; index < matrix_L->row_pointer[row+1]; index++){
      left_sum += matrix_L->elements[index]*x[matrix_L->col_indices[index]];
    }
    x[row] = (b[row] - left_sum);
  }
}

bool solve(int matrix_size, float sparsity)
{
    unsigned int bytes = matrix_size * matrix_size * sizeof(float);
    float* h_L, *h_b, *h_x;
    float* d_L, *d_b, *d_x;
    double throughput;

    // allocate host memory
    // h_L = (float*)malloc(bytes);
    h_b = (float*)malloc(matrix_size * sizeof(float));
    h_x = (float*)malloc(matrix_size * sizeof(float));

    printf("allocated\n");

    // // initialize L and b
    for (int i = 0; i < matrix_size; i++){
    //     for (int j = 0; j <= i; j++){
    //         if(j==i){
    //           h_L[i*matrix_size + j] = 1;
    //         }
    //         else{
    //           h_L[i*matrix_size + j] = rand()/(float)RAND_MAX < sparsity ? rand() / (float)RAND_MAX : 0;
    //         }
    //     }
        h_b[i] = rand()/(float)RAND_MAX;
    }
    // printf("matrix 1 initialized\n");
    CSR csr;
    // convert_matrix_to_csr(&csr, h_L, matrix_size);
    create_random_csr(&csr, matrix_size, sparsity);
    printf("Matrices initialized!\n");

    // allocate device memory
    // cudaMalloc((void**)&d_L, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_x, bytes);
    // cudaMemcpy(d_L, h_L, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, matrix_size*sizeof(float), cudaMemcpyHostToDevice);

    printf("\n %s\n", "Launch Kernel....");

    struct timeval startTime;
    struct timeval endTime;

    // dim3 dimBlock(16, 16, 1);
    // dim3 dimGrid(1, 1, 1);
    // printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    // printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    // printf("Matrix size: %d*%d\n", matrix_size, matrix_size);
    //
    // //todo: warmup
    // vanilla_gpu_mult<<<dimBlock,dimGrid>>>(d_A, d_B, d_out, matrix_size);

    printf("\nbasic_solve:\n");
    gettimeofday(&startTime, NULL);
    // basic_mult<<<dimBlock, dimGrid>>>(d_A, d_B, d_out, matrix_size);
    triangular_solve_csr(&csr, h_b, h_x, matrix_size);
    // cudaDeviceSynchronize();

    gettimeofday(&endTime, NULL);
    printf("Time elapsed: %f seconds", (float) ((endTime.tv_sec - startTime.tv_sec)  + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }

    // float* gpu_out = (float*)malloc(bytes);
    // cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);

    if(matrix_size <= 15){
        printf("\nMatrix L:\n ");
        printCsr(&csr, matrix_size);
        printf("Vector b:\n ");
        printVector(h_b, matrix_size);
        printf("Result:\n ");
        printVector(h_x, matrix_size);
    }

    // printf("\nvanilla_gpu_mult:\n");
    // gettimeofday(&startTime, NULL);
    // vanilla_gpu_mult<<<dimBlock, dimGrid>>>(d_A, d_B, d_out, matrix_size);
    // cudaDeviceSynchronize();
    // gettimeofday(&endTime, NULL);
    // printf("Time elapsed: %f seconds", (float) ((endTime.tv_sec - startTime.tv_sec)  + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }

    // cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // if(matrix_size <= 15){
    //     //printf("\nMatrix A:\n ");
    //     //printMatrix(h_A, matrix_size);
    //     //printf("Matrix B:\n ");
    //     //printMatrix(h_B, matrix_size);
    //     printf("\nResult:\n ");
    //     printMatrix(gpu_out, matrix_size);
    // }

    // //todo: getting result
    // printf("[Kernel %s] Throughput = %.4f GB/s, Time = %.5f ms\n",
    //     kernel_type, throughput, elapsedTime(timer)*1000);
    // printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    // free memory (both device and host mem)

    // cudaFree(d_L);
    cudaFree(d_b);
    cudaFree(d_x);
    // free(h_L);
    free(h_b);
    free(h_x);
    delete(csr.row_pointer);
    delete(csr.col_indices);
    delete(csr.elements);

    return true;
}

bool multiply(int matrix_size, float sparsity)
{
    unsigned int bytes = matrix_size * matrix_size * sizeof(float);
    float* h_A, *h_B, *h_out;
    float* d_A, *d_B, *d_out;
    double throughput;

    // allocate host memory
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);

    // initialize two sparse lower triangular matrices
    for (int i = 0; i < matrix_size; i++){
        for (int j = 0; j <= i; j++){
            if(j==i){
              h_A[i*matrix_size + j] = 1;
              h_B[i*matrix_size + j] = 1;
            }
            else{
              h_A[i*matrix_size + j] = rand()/(float)RAND_MAX < sparsity ? rand() / (float)RAND_MAX : 0;
              h_B[i*matrix_size + j] = rand()/(float)RAND_MAX < sparsity ? rand() / (float)RAND_MAX : 0;
            }
        }
    }
    printf("Matrices initialized!\n");

    // allocate device memory
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_out, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    printf("\n %s\n", "Launch Kernel....");

    struct timeval startTime;
    struct timeval endTime;

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(1, 1, 1);
    printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Matrix size: %d*%d\n", matrix_size, matrix_size);

    //todo: warmup
    vanilla_gpu_mult<<<dimBlock,dimGrid>>>(d_A, d_B, d_out, matrix_size);

    printf("\nbasic_mult:\n");
    gettimeofday(&startTime, NULL);
    basic_mult<<<dimBlock, dimGrid>>>(d_A, d_B, d_out, matrix_size);
    cudaDeviceSynchronize();
    gettimeofday(&endTime, NULL);
    printf("Time elapsed: %f seconds", (float) ((endTime.tv_sec - startTime.tv_sec)  + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }

    float* gpu_out = (float*)malloc(bytes);
    cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);

    if(matrix_size <= 15){
        printf("\nMatrix A:\n ");
        printMatrix(h_A, matrix_size);
        printf("Matrix B:\n ");
        printMatrix(h_B, matrix_size);
        printf("Result:\n ");
        printMatrix(gpu_out, matrix_size);
    }

    printf("\nvanilla_gpu_mult:\n");
    gettimeofday(&startTime, NULL);
    vanilla_gpu_mult<<<dimBlock, dimGrid>>>(d_A, d_B, d_out, matrix_size);
    cudaDeviceSynchronize();
    gettimeofday(&endTime, NULL);
    printf("Time elapsed: %f seconds", (float) ((endTime.tv_sec - startTime.tv_sec)  + (endTime.tv_usec - startTime.tv_usec)/1.0e6));
    //     double flopsPerMatrixMul = 2.0 * static_cast<double>(kernel_width) * \
    //                                 static_cast<double>(kernel_width) * static_cast<double>(kernel_width); \
    //     double numMatrixMul = width*height*channels;
    //     throughput = (numMatrixMul*flopsPerMatrixMul * 1.0e-9f) / (elapsedTime(timer) / 1000.0f);
    // }

    cudaMemcpy(gpu_out, d_out, bytes, cudaMemcpyDeviceToHost);

    if(matrix_size <= 15){
        //printf("\nMatrix A:\n ");
        //printMatrix(h_A, matrix_size);
        //printf("Matrix B:\n ");
        //printMatrix(h_B, matrix_size);
        printf("\nResult:\n ");
        printMatrix(gpu_out, matrix_size);
    }

    // //todo: getting result
    // printf("[Kernel %s] Throughput = %.4f GB/s, Time = %.5f ms\n",
    //     kernel_type, throughput, elapsedTime(timer)*1000);
    // printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    // free memory (both device and host mem)

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    free(h_A);
    free(h_B);
    free(h_out);
    // free(cpu_out);
    free(gpu_out);

    return true;
}

/*
 * Argument:
 *      "--width=<N>"       : Specify the number of width of input image (default: 1024)
 *      "--height=<N>"      : Specify the number of height of input image (default: 2048)
 *      "--channel=<N>"     : Specify the number of channels of input image (default: 1, <= 3)
 *      "--filter=<N>"      : Specify the number of filter width for convolution (default: 5)
*/

int main(int argc, char** argv)
{
    //printf("[Multiplying matrices...]\n\n");

    int matrix_size = 10;
    float sparsity = 0.7;

    if (checkCmdLineFlag(argc, (const char **)argv, "size")) {
        matrix_size = getCmdLineArgumentInt(argc, (const char **)argv, "size");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "sparsity")) {
        sparsity = getCmdLineArgumentInt(argc, (const char **)argv, "sparsity");
    }

    int dev = 0;
    cudaSetDevice(dev);

    solve(matrix_size, sparsity);

    // bool result;
    // result = compute(width, height, channels, kernel_width, "vanilla");
    // printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    //
    //
    // result = compute(width, height, channels, kernel_width, "shared_mem");
    // printf(result ? "Test PASSED\n" : "Test FAILED!\n");


    //todo: bonus
    //result = compute(width, height, channels, kernel_width, "optimized");
    //printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    cudaDeviceReset();

    return 0;
}
