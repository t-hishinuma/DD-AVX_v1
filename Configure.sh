./configure --enable-omp --enable-quad --enable-avx CC=icc 
#./configure --enable-omp --enable-quad --enable-avx --enable-mpi
make clean
make
#./configure --enable-omp --enable-quad --enable-avx CC=gcc CFLAGS=-mavx
