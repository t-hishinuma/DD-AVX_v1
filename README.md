# DD-AVX\_v1
DD-AVX [beta] README (BSD License)
written by Toshiaki Hishinuma (hishinuma@slis.tsukuba.ac.jp)

DD-AVX: Library of high-precision operations accelerated by AVX.
This library has Double-Double (DD) precision operations accelerated by AVX and AVX2.
This library needs to merge "Lis" for DD iterative solver.
Official support "Lis" version is "lis-1.4.58".
After merge "lis" and "DD-AVX", the interface is same as "lis".

## What is this library

### What is "DD-AVX ver.1"?
This library has Double-Double (DD) precision operations accelerated by AVX and AVX2.

### This library includes...
C-code of DD precision operation kernels using AVX and AVX2 (vector operations and SpMV) [1] [2].
Matrix storage format "BSR" (Block CSR) for DD precision SpMV.
block size 4x1 is suitable for AVX [3].
Auto selection of OpenMP loop Scheduling for DD precision SpMV [4].

## Installing

### Merging DD-AVX and lis

1. download "lis-1.4.58" from web site.
2. put "DD-AVX" and "lis-1.4.58" on same dir.
3. sh ./merge.sh [lis dir] [DD-AVX dir] [output dir]


### Build
please type "Configure" and "make." example of configure and make is Configure.sh

1. configure 
        
```
[command]$ ./configure [options]
	options
	 	CC=[compiler]
		CFLAGS=[compile options]
			[--enable-mpi]
			[--enable-omp]
			[--enable-sse2]
			[--enable-avx]
			[--enable-avx2]
```
	
2. make


```
$ make
```

## Sample code (output_dir/test)

1. dont use MPI: `test/test1 [options]`
2. use MPI: `mpirun -np [proc]  test/test1 [options]`

## compile your code

please type following command.

```
$gcc a.c -L [output_dir]/src/.libs -I [output]/include
```
