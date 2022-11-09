
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I..
EXE	        = matrix_mult
OBJ	        = matrix_mult.o support.o

default: $(EXE)

matrix_mult.o: matrix_mult.cu
	$(NVCC) --gpu-architecture=sm_50 -c -o $@ matrix_mult.cu $(NVCC_FLAGS)

support.o: support.cu support.h
		$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
