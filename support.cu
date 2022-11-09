#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"


//Helper function to print a matrix in readable form
void printMatrix(float* matrix, int matrix_size){
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            printf(" %f", matrix[matrix_size*i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printCsr(CSR* matrix, int matrix_size){
  for(int row = 0; row < matrix_size; row++){
      int prev_element = -1;
      for(int index = matrix->row_pointer[row]; index < matrix->row_pointer[row+1]; index++){
        for(int empty_spaces = prev_element+1; empty_spaces < matrix->col_indices[index]; empty_spaces++){
          printf(" 0.000000");
        }
        prev_element = matrix->col_indices[index];
        printf(" %f", matrix->elements[index]);
      }
      for(int empty_spaces = prev_element+1; empty_spaces < matrix_size; empty_spaces++){
        printf(" 0.000000");
      }
      printf("\n");
  }
  printf("\n");
}

void printVector(float* vector, int size){
  for(int i = 0; i < size; i++){
    printf(" %f", vector[i]);
  }
  printf("\n");
}

void convert_matrix_to_csr(CSR* csr, float* matrix, int size){
  csr->row_pointer = (int*)malloc((size+1)*sizeof(int));

  //set up row pointer array
  for(int row = 0; row < size; row++){
    int values_seen = 0;
    csr->row_pointer[0] = 0;
    for(int col = 0; col <= row; col++){
      if(matrix[row*size + col] != 0) values_seen++;
    }
    csr->row_pointer[row+1] = csr->row_pointer[row] + values_seen;
  }
  //Fill values arrays
  csr->col_indices = (int*)malloc((csr->row_pointer[size])*sizeof(int)); //new int(csr->row_pointer[size]);
  csr->elements = (double*)malloc((csr->row_pointer[size])*sizeof(double));//new double(csr->row_pointer[size]);
  int index = 0;
  for(int row = 0; row < size; row++){
    for(int col = 0; col <= row; col++){
      if(matrix[row*size + col] != 0){
        csr->col_indices[index] = col;
        csr->elements[index] = matrix[row*size + col];
        index++;
      }
    }
  }
}
