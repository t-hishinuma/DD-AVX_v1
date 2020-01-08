/* Copyright (C) 2014~ DD-AVX Project. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   3. Neither the name of the project nor the names of its contributors 
      may be used to endorse or promote products derived from this software 
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE SCALABLE SOFTWARE INFRASTRUCTURE PROJECT
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE SCALABLE SOFTWARE INFRASTRUCTURE
   PROJECT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
   OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
	#include "lis_config.h"
#else
#ifdef HAVE_CONFIG_WIN32_H
	#include "config_win32.h"
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
//#ifdef USE_SSE2
//	#include <emmintrin.h>
//#endif
#ifdef _OPENMP
	#include <omp.h>
#endif
#ifdef USE_MPI
	#include <mpi.h>
#endif
#include "lislib.h"

#ifdef USE_QUAD_PRECISION

//TODO put to lis_precision.h
#include <stdint.h>

#ifdef USE_AVX
void lis_matvec_bsr_mp2_4x1(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,n,nr,ii;
	LIS_INT				is,ie;
	LIS_INT				j0,j1;
	LIS_INT 				j2,j3,ij;
	LIS_INT				jj0;

	LIS_SCALAR			*vv0;
	LIS_SCALAR			*x,*y,*xl,*yl;

	LIS_AVX_TYPE			tt_hi, tt_lo;
	LIS_SCALAR			s_hi, s_lo;
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvec_bsr_4x1");
	#endif
	n     = A->n;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	nr    = A->nr;

	vv0 = A->value;
	__m256d yv,ylv,xv,xlv,av,tmpv;

	#ifdef _OPENMP
	#pragma omp parallel for private(i)
	#endif
	for(i=0; i<n; i++)
	{
		y[i] = 0.0;
	}

#ifdef _OPENMP
	#pragma omp parallel for schedule(guided) private(i,jj0,j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,yv,ylv,tt_hi,tt_lo,s_hi,s_lo, tmpv)
#endif
	for(i=0; i<nr; i++)
	{
		yv = _mm256_loadu_pd(&(y[i*4]));
		ylv = _mm256_loadu_pd(&(yl[i*4]));

		is = A->bptr[i];
		ie = A->bptr[i+1];

		for(j=is;j<ie;j++)
		{
			jj0 = A->bindex[j];

			bh  = _mm256_broadcast_sd(&( x[ jj0 ] )); 
			bl  = _mm256_broadcast_sd(&( xl[ jj0 ] )); 
			ch  = _mm256_loadu_pd(&(vv0[j*4]));

			LIS_QUAD_MULDN_AVX_CORE; 
			t1  = LIS_AVX_FUNC(sub_pd)(ch,p1);
			p2  = LIS_AVX_FUNC(sub_pd)(p2,t1);
			bh  = LIS_AVX_FUNC(loadu_pd)((double*)&(yv));
			bl  = LIS_AVX_FUNC(loadu_pd)((double*)&(ylv));
			LIS_QUAD_ADDN_AVX_CORE; 
			LIS_AVX_FUNC(storeu_pd)((double*)&(yv),sh); 
			sh = LIS_AVX_FUNC(sub_pd)(sh,th); 
			bh = LIS_AVX_FUNC(sub_pd)(bh,sh);
			LIS_AVX_FUNC(storeu_pd)((double*)&(ylv),bh);
		}                                             
			_mm256_storeu_pd(&(y[i*4]),yv);
			_mm256_storeu_pd(&(yl[i*4]),ylv);
	}
}

void lis_matvec_bsr_mp2_1x4(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,n;
	LIS_INT				is,ie;
	LIS_INT				j0,j1;
	LIS_INT 				j2,j3,ij;
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;

	LIS_AVX_TYPE	tt_hi, tt_lo;
		LIS_SCALAR		s_hi, s_lo;
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvec_bsr_1x4");
	#endif
	n     = A->n;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;

	jj0 = A->bindex;
	vv0 = A->value;
	__m256d yv,ylv,xv,xlv,av,tmpv;

	#ifdef _OPENMP
	#pragma omp parallel for private(i)
	#endif
	for(i=0; i<n; i++)
	{
		y[i] = 0.0;
	}


#ifdef _OPENMP
	#pragma omp parallel for schedule(guided) private(i,j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,yv,ylv,s_hi,s_lo, tmpv)
#endif
	for(i=0; i<n; i++)
	{
		yv = _mm256_broadcast_sd(&(y[i]));
		ylv = _mm256_broadcast_sd(&(yl[i]));

		is = A->bptr[i];
		ie = A->bptr[i+1];

		for(j=is;j<ie;j++)
		{
			j0 = jj0[j]*4+0; j1 = jj0[j]*4+1;
			j2 = jj0[j]*4+2; j3 = jj0[j]*4+3;

			bh  = _mm256_loadu_pd(&(x[jj0[j]*4])); 
			bl  = _mm256_loadu_pd(&(xl[jj0[j]*4])); 
			ch  = _mm256_loadu_pd(&(vv0[j*4]));

			LIS_QUAD_MULDN_AVX_CORE; 
			t1  = LIS_AVX_FUNC(sub_pd)(ch,p1);
			p2  = LIS_AVX_FUNC(sub_pd)(p2,t1);
			bh  = LIS_AVX_FUNC(loadu_pd)((double*)&(yv));
			bl  = LIS_AVX_FUNC(loadu_pd)((double*)&(ylv));
			LIS_QUAD_ADDN_AVX_CORE; 
			LIS_AVX_FUNC(storeu_pd)((double*)&(yv),sh); 
			sh = LIS_AVX_FUNC(sub_pd)(sh,th); 
			bh = LIS_AVX_FUNC(sub_pd)(bh,sh);
			LIS_AVX_FUNC(storeu_pd)((double*)&(ylv),bh);
		}

		LIS_QUAD_HADDALL_AVX(y[i],yl[i],yv,ylv);
	}
}
#endif

void lis_matvec_bsr_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,k,n;
	LIS_INT				bi=0, bc = 0, bj = 0;
	LIS_INT				nr=0, nc = 0;
	LIS_INT				bnr=0, bnc = 0;
	LIS_INT				bs = 0;

	LIS_INT				is,ie;
	LIS_INT				tmp1,tmp2;
	LIS_INT				j0,j1;
	LIS_INT 				j2,j3,ij;
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;

	LIS_SCALAR		s_hi, s_lo;
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvec_bsr");
	#endif
	n     = A->n;
	nr    = A->nr;
	nc    = A->nc;
	bnr   = A->bnr;
	bnc   = A->bnc;
	bs		= bnr*bnc;

	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;

	jj0 = A->bindex;
	vv0 = A->value;
	#ifdef _OPENMP
	#pragma omp parallel for private(i)
	#endif
	for(i=0; i<n; i++)
	{
		y[i] = 0.0;
	}

	#ifdef _OPENMP
	#pragma omp parallel for schedule(guided) private(i,j,k,tmp1, tmp2,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,s_hi,s_lo,bi, bj, bc)
	#endif
	for(bi=0; bi<nr; bi++)
	{
		for(bc = A->bptr[bi];bc<A->bptr[bi+1];bc++)
		{
			bj = A->bindex[bc] * bnc;
			k	= bc * bs;
			for(j = 0; j < bnc ; j ++)
			{
					tmp2 = bj + j;
				for(i = 0 ; i < bnr ; i++)
				{
					tmp1 = bi * bnr + i;
					LIS_QUAD_FMAD_SSE2(y[tmp1],yl[tmp1],y[tmp1],yl[tmp1],x[tmp2],xl[tmp2],A->value[k]);
					k++;
				}
			}
		}
	}
}

void lis_matvect_bsr_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,k,n,np;
	LIS_INT				bi=0, bc = 0, bj = 0;
	LIS_INT				nr=0, nc = 0;
	LIS_INT				bnr=0, bnc = 0;
	LIS_INT				bs = 0;

	LIS_INT				is,ie;
	LIS_INT				tmp1,tmp2;
	LIS_INT				j0,j1;
	LIS_INT 				j2,j3,ij;
	LIS_INT				*jj0;
	LIS_INT				nprocs;
	LIS_INT				my_rank;
	LIS_SCALAR			*vv0;
	LIS_SCALAR			*x,*y,*xl,*yl;
	LIS_SCALAR			*ww,*wwl;

	LIS_SCALAR		s_hi, s_lo;
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvec_bsr");
	#endif
	n     = A->n;
	np     = A->np;
	nr    = A->nr;
	nc    = A->nc;
	bnr   = A->bnr;
	bnc   = A->bnc;
	bs		= bnr*bnc;

	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;

	jj0 = A->bindex;
	vv0 = A->value;

	//#ifdef _OPENMP
//		nprocs = omp_get_max_threads();
//	#else 
		nprocs = 1;
//	#endif
	ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR), "lis_matvect_bsr_mp2::ww" );
	wwl = &ww[nprocs*np];

	//#ifdef _OPENMP
	//#pragma omp parallel private(my_rank,i,j,k,tmp1, tmp2,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,s_hi,s_lo,bi, bj, bc)
	//#endif
	{
//#ifdef _OPENMP
		//my_rank = omp_get_thread_num();
//		#pragma omp for
//#else 
		my_rank = 0;
//#endif
		for(j=0;j<nprocs;j++)
		{
			memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
			memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
		}

//#ifdef _OPENMP
//		#pragma omp for
//#endif
		for(bi=0; bi<nr; bi++)
		{
			for(bc = A->bptr[bi];bc<A->bptr[bi+1];bc++)
			{
				bj = my_rank * np + A->bindex[bc] * bnc;
				k	= bc * bs;
				for(j = 0; j < bnc ; j++)
				{
					tmp1 = bj + j;
					for(i = 0 ; i < bnr ; i++)
					{
						tmp2 = bi * bnr + i;
						LIS_QUAD_FMAD_SSE2(ww[tmp1],wwl[tmp1],ww[tmp1],wwl[tmp1],x[tmp2],xl[tmp2],A->value[k]);
						k++;
					}
				}
			}
		}
	}
	for(i=0;i<np;i++)
	{
		y[i] = yl[i] = 0.0;
	}
//#ifdef _OPENMP
//		#pragma omp parallel for private(j,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
//#endif
	for(i=0;i<np;i++)//nprocs=threads_num,np=N
	{
		for(j=0;j<nprocs;j++)
		{
			LIS_QUAD_ADD_SSE2
				(y[i],yl[i],
				 y[i],yl[i],
				 ww[np*j+i],wwl[np*j+i]);
		}
	}
	lis_free(ww);
}

#endif
