#ifndef __FILEH__
#define __FILEH__

typedef struct {
    int* row_pointer;
    int* col_indices;
    double* elements;
} CSR;

#ifdef __cplusplus
extern "C" {
#endif
//Helper functions to print data to console
void printMatrix(float* matrix, int matrix_size);
void printCsr(CSR* matrix, int matrix_size);
void printVector(float* vector, int size);
//Converts a sparse triangular matrix in an array form to csr format
void create_random_csr(CSR* csr, int size, float sparsity);
void convert_matrix_to_csr(CSR* csr, float* matrix, int size);
#ifdef __cplusplus
}
#endif

#endif
