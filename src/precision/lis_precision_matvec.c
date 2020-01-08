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
#ifdef _OPENMP
	#include <omp.h>
#endif
#ifdef USE_MPI
	#include <mpi.h>
#endif
#include "lislib.h"

#ifdef USE_QUAD_PRECISION

#include <stdint.h>

void lis_matvec_csr_mp(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,n;
	LIS_INT				is,ie,jj,j0;
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	LIS_QUAD_DECLAR;


	n     = A->n;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	if( A->is_splited )
	{
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(i,j,is,ie,j0,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<n;i++)
		{
			#ifndef USE_SSE2
				LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#else
				LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#endif
			is = A->L->ptr[i];
			ie = A->L->ptr[i+1];
			for(j=is;j<ie-0;j+=1)
			{
				j0 = A->L->index[j+0];
				#ifndef USE_SSE2
					LIS_QUAD_FMAD(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
				#else
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
				#endif
			}
			is = A->U->ptr[i];
			ie = A->U->ptr[i+1];
			for(j=is;j<ie-0;j+=1)
			{
				j0 = A->U->index[j+0];
				#ifndef USE_SSE2
					LIS_QUAD_FMAD(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
				#else
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
				#endif
			}
		}
	}
	else
	{
		jj0 = A->index;
		vv0 = A->value;
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(i,j,is,ie,j0,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<n;i++)
		{
			y[i] = yl[i] = 0.0;

			is = A->ptr[i];
			ie = A->ptr[i+1];
			for(j=is;j<ie-0;j+=1)
			{
				j0 = jj0[j+0];
				#ifndef USE_SSE2
					LIS_QUAD_FMAD(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				#else
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				#endif
			}
		}
	}
}

void lis_matvec_csr_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,n;
	LIS_INT				is,ie;
	LIS_INT				j0,j1;
	#if defined(USE_AVX)
		LIS_INT j2,j3,ij;
	#endif
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#if defined(USE_AVX)
		LIS_AVX_TYPE	tt_hi, tt_lo;
		LIS_SCALAR		s_hi, s_lo;
		LIS_AVX_TYPE	tmph,tmpl;
	#else
		LIS_QUAD_PD		tt;
	#endif
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvec_csr_mp2");
	#endif
	n     = A->n;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	if( A->is_splited )
	{
		printf("error\n");
	
		#ifdef _OPENMP	
		#if defined(USE_AVX)
			#pragma omp parallel for private(j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,tt_hi,tt_lo,s_hi,s_lo)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(j,is,ie,j0,j1,tt,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(j,is,ie,j0,j1,tt,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<n;i++)
		{
			#ifndef USE_SSE2
				LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#else
				LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#endif

			#if defined(USE_AVX)
				tt_hi = LIS_AVX_FUNC(setzero_pd)();
				tt_lo = LIS_AVX_FUNC(setzero_pd)();

				is = A->L->ptr[i];
				ie = A->L->ptr[i+1];
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					j0 = A->L->index[j+0]; j1 = A->L->index[j+1];
					j2 = A->L->index[j+2]; j3 = A->L->index[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],A->L->value[j]);
					#else
						#error no implementation for this vector size
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->L->index[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
				}
				is = A->U->ptr[i];
				ie = A->U->ptr[i+1];
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					j0 = A->U->index[j+0]; j1 = A->U->index[j+1];
					j2 = A->U->index[j+2]; j3 = A->U->index[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],A->U->value[j]);
					#else
						#error no implementation for this vector size
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->U->index[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
				}
				LIS_QUAD_HADDALL_AVX(s_hi,s_lo,tt_hi,tt_lo);
				LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],s_hi,s_lo);
			#elif defined(USE_SSE2)
				tt.hi[0] = tt.hi[1] = tt.lo[0] = tt.lo[1] = 0.0;
				is = A->L->ptr[i];
				ie = A->L->ptr[i+1];
				for(j=is;j<ie-1;j+=2)
				{
					j0 = A->L->index[j+0];
					j1 = A->L->index[j+1];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],A->L->value[j]);
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->L->index[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
					#endif
				}
				is = A->U->ptr[i];
				ie = A->U->ptr[i+1];
				for(j=is;j<ie-1;j+=2)
				{
					j0 = A->U->index[j+0];
					j1 = A->U->index[j+1];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],A->U->value[j]);
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->U->index[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
					#endif
				}
				#ifdef USE_SSE2
					LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],tt.hi[0],tt.lo[0]);
					LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],tt.hi[1],tt.lo[1]);
				#endif
			#else
				#error no implementation
			#endif
		}
	}
	else
	{
	   //mp3_matvec
		jj0 = A->index;
		vv0 = A->value;

#define THREAD_DETAIL 0
#if THREAD_DETAIL==1
			LIS_INT a[4]={0,0,0,0};
			LIS_INT b[4]={0,0,0,0};
			LIS_INT c[4]={0,0,0,0};
			LIS_INT d[4]={0,0,0,0};
#endif
		
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for schedule(guided) private(j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,tt_hi,tt_lo,s_hi,s_lo)
		#elif defined(USE_SSE2)
			#pragma omp parallel for schedule(guided) private(j,is,ie,j0,j1,tt,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for schedule(guided)  private(j,is,ie,j0,j1,tt,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<n;i++)
		{
			#if THREAD_DETAIL==1
				LIS_INT thnum = omp_get_thread_num();
			#endif

			#if defined(USE_AVX)
				LIS_AVX_TYPE tt_hi = LIS_AVX_FUNC(setzero_pd)();
				LIS_AVX_TYPE tt_lo = LIS_AVX_FUNC(setzero_pd)();
			
				is = A->ptr[i];
				ie = A->ptr[i+1];
				#if THREAD_DETAIL==1
					a[thnum] += ie - is;
				#endif
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					#if 1
						j0 = jj0[j+0]; j1 = jj0[j+1];
						j2 = jj0[j+2]; j3 = jj0[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],vv0[j]);
						#else
						#error no implementation for this vector size
						#endif
					#else
						j0 = jj0[j];
						LIS_QUAD_FMADN_AVX(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],vv0[j]);
					#endif
				}
				

				#define padding 1
				//padding in executon (use AVX)
				#if padding == 1
				if(j==ie-3)
				{
					#if THREAD_DETAIL==1
						b[thnum]++;
					#endif
				   j0 = jj0[j+0]; j1 = jj0[j+1];
					j2 = jj0[j+2]; 
					LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],0,0,vv0[j]);
					j+=3;
				}
				else if(j==ie-2)
				{
					#if THREAD_DETAIL==1
						c[thnum]++;
					#endif
					j0 = jj0[j+0]; j1 = jj0[j+1];
					LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],0,0,0,0,vv0[j]);
					j+=2;
				}
				else if(j==ie-1)
				{
					#if THREAD_DETAIL==1
						d[thnum]++;
					#endif
					j0 = jj0[j+0];
					LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],0,0,0,0,0,0,vv0[j]);
					j+=1;
				}

				LIS_QUAD_HADDALL_AVX(y[i],yl[i],tt_hi,tt_lo);
				#endif
				
				//Scalar
				#if padding == 2
				LIS_QUAD_HADDALL_AVX(y[i],yl[i],tt_hi,tt_lo);
				for(;j<ie;j++)
				{
					j0 = jj0[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				}
				#endif

				//combinations of SSE2+Scalar
				#if padding == 3
				LIS_QUAD_HADDALL_AVX(y[i],yl[i],tt_hi,tt_lo);
				for(;j<ie-1;j=j+2)
			   	{	   
					j0 = jj0[j+0];
					j1 = jj0[j+1];
					LIS_QUAD_FMAD2_SSE2_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],vv0[j]);
				}
				LIS_QUAD_ADD_SSE2(y[i],yl[i],tt.hi[0],tt.lo[0],tt.hi[1],tt.lo[1]);
				for(;j<ie;j++)
				{
					j0 = jj0[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				}
				#endif

			#elif defined(USE_SSE2)
				tt.hi[0] = tt.hi[1] = tt.lo[0] = tt.lo[1] = 0.0;

				is = A->ptr[i];
				ie = A->ptr[i+1];

				for(j=is;j<ie-1;j+=2)
				{
					#if 1
						j0 = jj0[j+0];
						j1 = jj0[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],vv0[j]);
						#endif
					#else
						j0 = jj0[j];
						LIS_QUAD_FMAD2_SSE2(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],vv0[j]);
					#endif
				}
				#ifdef USE_SSE2
					LIS_QUAD_ADD_SSE2(y[i],yl[i],tt.hi[0],tt.lo[0],tt.hi[1],tt.lo[1]);
				#endif
				for(;j<ie;j++)
				{
					j0 = jj0[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
					#endif
				}
			#else
				#error no implementation
			#endif
	}
			#if THREAD_DETAIL==1
			for(i=0;i<4;i++)
			{
				printf("%d\t",a[i]+b[i]+c[i]+d[i]);
			}		
				printf("\t");
			#endif

	}
}

void lis_matvect_csr_mp(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,js,je,jj;
	LIS_INT				n,np;
	LIS_QUAD_PTR	tt0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#ifdef _OPENMP
		LIS_INT k,nprocs;
		LIS_SCALAR		*ww,*wwl;
	#endif
	LIS_QUAD_DECLAR;

	n    = A->n;
	np   = A->np;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	tt0.hi = &X->work[0];
	tt0.lo = &X->work[1];
	if( A->is_splited )
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR),"lis_matvect_csr_mp::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,jj,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for 
				for(i=0; i<n; i++)
				{
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					for(j=js;j<je;j++)
					{
						jj  = k*np+A->L->index[j];
						#ifndef USE_SSE2
							LIS_QUAD_FMAD(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->L->value[j]);
						#else
							LIS_QUAD_FMAD_SSE2(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->L->value[j]);
						#endif
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					for(j=js;j<je;j++)
					{
						jj  = k*np+A->U->index[j];
						#ifndef USE_SSE2
							LIS_QUAD_FMAD(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->U->value[j]);
						#else
							LIS_QUAD_FMAD_SSE2(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->U->value[j]);
						#endif
					}
				}
				#pragma omp for 
				for(i=0;i<np;i++)
				{
					#ifndef USE_SSE2
						LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
					#else
						LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
					#endif
					for(j=0;j<nprocs;j++)
					{
						#ifndef USE_SSE2
							LIS_QUAD_ADD(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#else
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<np; i++)
			{
				#ifndef USE_SSE2
					LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#else
					LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#endif
			}
			for(i=0; i<n; i++)
			{
				js = A->L->ptr[i];
				je = A->L->ptr[i+1];
				for(j=js;j<je;j++)
				{
					jj  = A->L->index[j];
					#ifndef USE_SSE2
						LIS_QUAD_FMAD(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->L->value[j]);
					#else
						LIS_QUAD_FMAD_SSE2(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->L->value[j]);
					#endif
				}
				js = A->U->ptr[i];
				je = A->U->ptr[i+1];
				for(j=js;j<je;j++)
				{
					jj  = A->U->index[j];
					#ifndef USE_SSE2
						LIS_QUAD_FMAD(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->U->value[j]);
					#else
						LIS_QUAD_FMAD_SSE2(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->U->value[j]);
					#endif
				}
			}
		#endif
	}
	else
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR),"lis_matvect_csr_mp::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,jj,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for 
				for(i=0; i<n; i++)
				{
					js = A->ptr[i];
					je = A->ptr[i+1];
					for(j=js;j<je;j++)
					{
						jj  = k*np+A->index[j];
						#ifndef USE_SSE2
							LIS_QUAD_FMAD(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->value[j]);
						#else
							LIS_QUAD_FMAD_SSE2(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->value[j]);
						#endif
					}
				}
				#pragma omp for 
				for(i=0;i<np;i++)
				{
					y[i] = yl[i] = 0.0;
					for(j=0;j<nprocs;j++)
					{
						#ifndef USE_SSE2
							LIS_QUAD_ADD(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#else
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<np; i++)
			{
				y[i]  = 0.0;
				yl[i] = 0.0;
			}
			for(i=0; i<n; i++)
			{
				js = A->ptr[i];
				je = A->ptr[i+1];
				tt0.hi[0] = x[i];
				tt0.lo[0] = xl[i];
				for(j=js;j<je;j++)
				{
					jj  = A->index[j];
					#ifndef USE_SSE2
						LIS_QUAD_FMAD(y[jj],yl[jj],y[jj],yl[jj],tt0.hi[0],tt0.lo[0],A->value[j]);
					#else
						LIS_QUAD_FMAD_SSE2(y[jj],yl[jj],y[jj],yl[jj],tt0.hi[0],tt0.lo[0],A->value[j]);
					#endif
				}
			}
		#endif
	}
}

void lis_matvect_csr_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,js,je,j0,j1;
	#if defined(USE_AVX)
		LIS_INT			j2,j3;
	#endif
	LIS_INT				n,np;
	LIS_QUAD_PTR	tt0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#ifdef _OPENMP
		LIS_INT k,nprocs;
		LIS_SCALAR		*ww,*wwl;
	#endif
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvect_csr_mp2");
	#endif
	n    = A->n;
	np   = A->np;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	tt0.hi = &X->work[0];
	tt0.lo = &X->work[2];
	if( A->is_splited )
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR), "lis_matvect_csr_mp2::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,j0,j1,j2,j3,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,j0,j1,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,j0,j1,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(i=0; i<np; i++)
				{
					#ifndef USE_SSE2
						LIS_QUAD_MULD(ww[i],wwl[i],x[i],xl[i],A->D->value[i]);
					#else
						LIS_QUAD_MULD_SSE2(ww[i],wwl[i],x[i],xl[i],A->D->value[i]);
					#endif
				}
				#pragma omp for 
				for(i=0; i<n; i++)
				{
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					#if defined(USE_AVX)
						for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
						{
							j0 = k*np + A->L->index[j+0]; j1 = k*np + A->L->index[j+1];
							j2 = k*np + A->L->index[j+2]; j3 = k*np + A->L->index[j+3];
							#if LIS_AVX_SIZE == 4
								LIS_QUAD_FMAD4_AVX_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->L->value[j]);
							#else
								#error no implementation for this vector size
							#endif
						}
					#elif defined(USE_SSE2)
						for(j=js;j<je-1;j+=2)
						{
							j0  = k*np + A->L->index[j];
							j1  = k*np + A->L->index[j+1];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD2_SSE2_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j0],wwl[j0],ww[j1],wwl[j1],x[i],xl[i],x[i],xl[i],A->L->value[j]);
							#endif
						}
					#else
						#error no implementation
					#endif
					for(;j<je;j++)
					{
						j0  = A->L->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(ww[j0],wwl[j0],ww[j0],wwl[j0],x[i],xl[i],A->L->value[j]);
						#endif
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					#if defined(USE_AVX)
						for(j=js;j<je-1;j+=2)
						{
							j0 = k*np + A->U->index[j+0]; j1 = k*np + A->U->index[j+1];
							j2 = k*np + A->U->index[j+2]; j3 = k*np + A->U->index[j+3];
							#if LIS_AVX_SIZE == 4
								LIS_QUAD_FMAD4_AVX_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->U->value[j]);
							#else
								#error no implementation for this vector size
							#endif
						}
					#elif defined(USE_SSE2)
						for(j=js;j<je-1;j+=2)
						{
							j0  = k*np + A->U->index[j];
							j1  = k*np + A->U->index[j+1];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD2_SSE2_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j0],wwl[j0],ww[j1],wwl[j1],x[i],xl[i],x[i],xl[i],A->U->value[j]);
							#endif
						}
					#else
						#error no implementation
					#endif
					for(;j<je;j++)
					{
						j0  = A->U->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(ww[j0],wwl[j0],ww[j0],wwl[j0],x[i],xl[i],A->U->value[j]);
						#endif
					}
				}
				#pragma omp for 
				for(i=0;i<np;i++)
				{
					y[i] = yl[i] = 0.0;
					for(j=0;j<nprocs;j++)
					{
						#ifdef USE_SSE2
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<np; i++)
			{
				#ifndef USE_SSE2
					LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#else
					LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#endif
			}
			for(i=0; i<n; i++)
			{
				#if defined(USE_AVX)
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
					{
						j0 = A->L->index[j+0]; j1 = A->L->index[j+1];
						j2 = A->L->index[j+2]; j3 = A->L->index[j+3];
						#if LIS_AVX_SIZE == 4
							LIS_QUAD_FMAD4_AVX_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->L->value[j]);
						#else
							#error no implementation for this vector size
						#endif
						
					}
					for(;j<je;j++)
					{
						j0  = A->L->index[j];
						LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->L->value[j]);
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
					{
						j0 = A->U->index[j+0]; j1 = A->U->index[j+1];
						j2 = A->U->index[j+2]; j3 = A->U->index[j+3];
						#if LIS_AVX_SIZE == 4
							LIS_QUAD_FMAD4_AVX_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->U->value[j]);
						#else
							#error no implementation for this vector size
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->U->index[j];
						LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->U->value[j]);
					}
				#elif defined(USE_SSE2)
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					for(j=js;j<je-1;j+=2)
					{
						j0  = A->L->index[j];
						j1  = A->L->index[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j0],yl[j0],y[j1],yl[j1],x[i],xl[i],x[i],xl[i],A->L->value[j]);
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->L->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->L->value[j]);
						#endif
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					for(j=js;j<je-1;j+=2)
					{
						j0  = A->U->index[j];
						j1  = A->U->index[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j0],yl[j0],y[j1],yl[j1],x[i],xl[i],x[i],xl[i],A->U->value[j]);
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->U->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->U->value[j]);
						#endif
					}
				#else
					#error no implementation
				#endif
			}
		#endif
	}
	
	else
	{
	   //mp3_matvect
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR), "lis_matvect_csr_mp2::ww" );
			wwl = &ww[nprocs*np];
			double hpad1=0,hpad2=0,hpad3=0;
			double lpad1=0,lpad2=0,lpad3=0;
			
				LIS_INT sub=0;

			#if defined(USE_AVX)
				#pragma omp parallel private(sub,i,j,js,je,j0,j1,j2,j3,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,j0,j1,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,j0,j1,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for schedule (guided)
				for(i=0; i<n; i++)
				{
					#if defined(USE_AVX)
						js = A->ptr[i];
						je = A->ptr[i+1];
						for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
						{
					   	 	j0 = k*np + A->index[j+0]; j1 = k*np + A->index[j+1];
								j2 = k*np + A->index[j+2]; j3 = k*np + A->index[j+3];
							#if LIS_AVX_SIZE == 4
								LIS_QUAD_FMAD4_AVX_STSD
								(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
								ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
								x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],
								A->value[j]);
							#else
								#error no implementation for this vector size
							#endif
						}
						if(j==je-3)
						{
						   	j0 = k*np + A->index[j+0];
						       	j1 = k*np + A->index[j+1];
							j2 = k*np + A->index[j+2];
						
							LIS_QUAD_FMAD4_AVX_STSD
							(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],
							   hpad1,lpad1,
							 ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],
							   hpad1,lpad1,
							 x[i],xl[i],x[i],xl[i],x[i],xl[i],0,0,
							 A->value[j]);
						
							j+=3;
						}
						else if(j==je-2)
						{
							j0 = k*np + A->index[j+0];
						       	j1 = k*np + A->index[j+1];
						
							LIS_QUAD_FMAD4_AVX_STSD
							(ww[j0],wwl[j0],ww[j1],wwl[j1],
							   hpad1,lpad1,hpad2,lpad2,
							 ww[j0],wwl[j0],ww[j1],wwl[j1],
							   hpad1,lpad1,hpad2,lpad2,
							 x[i],xl[i],x[i],xl[i],0,0,0,0,
							 A->value[j]);
							
							j+=2;
						}
						else if(j==je-1)
						{
							j0 = k*np + A->index[j+0];
							
							LIS_QUAD_FMAD4_AVX_STSD
							(ww[j0],wwl[j0],
							   hpad1,lpad1,hpad2,lpad2,hpad3,lpad3,
							 ww[j0],wwl[j0],
							   hpad1,lpad1,hpad2,lpad2,hpad3,lpad3,
							 x[i],xl[i],0,0,0,0,0,0,
							 A->value[j]);
							
							j+=1;
						}

			#elif defined(USE_SSE2)
				js = A->ptr[i];
				je = A->ptr[i+1];
					for(j=js;j<je-1;j+=2)
						{
							j0  = k*np + A->index[j];
							j1  = k*np + A->index[j+1];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD2_SSE2_STSD
								(ww[j0],wwl[j0],ww[j1],wwl[j1],
								 ww[j0],wwl[j0],ww[j1],wwl[j1],
								 x[i],xl[i],x[i],xl[i],
								 A->value[j]);
							#endif
						}
					for(;j<je;j++)
						{
							j0  = A->index[j];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD_SSE2
								(ww[j0],wwl[j0],
								 ww[j0],wwl[j0],
								 x[i],xl[i],
								 A->value[j]);
							#endif
						}
					#else
						#error no implementation
					#endif
				}
			}
			for(i=0;i<np;i++)
			{
				y[i] = yl[i] = 0.0;
			}
			for(i=0;i<nprocs;i++)//nprocs=threads_num,np=N
			{
			   #pragma omp parallel for private(j,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			   for(j=0;j<np;j++)
			   {
				#ifdef USE_SSE2
					LIS_QUAD_ADD_SSE2
					(y[j],yl[j],
					 y[j],yl[j],
					 ww[np*i+j],wwl[np*i+j]);
				#endif
			   }
			}
			lis_free(ww);
		#else
			for(i=0; i<np; i++)
			{
				y[i]  = 0.0;
				yl[i] = 0.0;
			}
			for(i=0; i<n; i++)
			{
				#if defined(USE_AVX)
					js = A->ptr[i];
					je = A->ptr[i+1];
					for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
					{
						j0 = A->index[j+0]; j1 = A->index[j+1];
						j2 = A->index[j+2]; j3 = A->index[j+3];
						LIS_QUAD_FMAD4_AVX_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
												y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
												x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->value[j]);
					}
					for(;j<je;j++)
					{
						j0  = A->index[j];
						LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],x[i],xl[i],A->value[j]);
					}
				#elif defined(USE_SSE2)
					js = A->ptr[i];
					je = A->ptr[i+1];
					for(j=js;j<je-1;j+=2)
					{
						j0  = A->index[j];
						j1  = A->index[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j0],yl[j0],y[j1],yl[j1],x[i],xl[i],x[i],xl[i],A->value[j]);
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],x[i],xl[i],A->value[j]);
						#endif
					}
				#else
					#error no implementation
				#endif
			}
		#endif
	}
}
#endif

#ifdef USE_QUAD_PRECISION
void lis_matvec_csc_mp(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,js,je,jj;
	LIS_INT				n,np;
	LIS_QUAD_PTR	tt0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#ifdef _OPENMP
		LIS_INT k,nprocs;
		LIS_SCALAR		*ww,*wwl;
	#endif
	LIS_QUAD_DECLAR;

	n    = A->n;
	np   = A->np;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	tt0.hi = &X->work[0];
	tt0.lo = &X->work[1];
	if( A->is_splited )
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR),"lis_matvect_csr_mp::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,jj,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for 
				for(i=0; i<np; i++)
				{
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					for(j=js;j<je;j++)
					{
						jj  = k*np+A->L->index[j];
						#ifndef USE_SSE2
							LIS_QUAD_FMAD(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->L->value[j]);
						#else
							LIS_QUAD_FMAD_SSE2(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->L->value[j]);
						#endif
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					for(j=js;j<je;j++)
					{
						jj  = k*np+A->U->index[j];
						#ifndef USE_SSE2
							LIS_QUAD_FMAD(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->U->value[j]);
						#else
							LIS_QUAD_FMAD_SSE2(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->U->value[j]);
						#endif
					}
				}
				#pragma omp for 
				for(i=0;i<n;i++)
				{
					#ifndef USE_SSE2
						LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
					#else
						LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
					#endif
					for(j=0;j<nprocs;j++)
					{
						#ifndef USE_SSE2
							LIS_QUAD_ADD(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#else
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<n; i++)
			{
				#ifndef USE_SSE2
					LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#else
					LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#endif
			}
			for(i=0; i<np; i++)
			{
				js = A->L->ptr[i];
				je = A->L->ptr[i+1];
				for(j=js;j<je;j++)
				{
					jj  = A->L->index[j];
					#ifndef USE_SSE2
						LIS_QUAD_FMAD(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->L->value[j]);
					#else
						LIS_QUAD_FMAD_SSE2(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->L->value[j]);
					#endif
				}
				js = A->U->ptr[i];
				je = A->U->ptr[i+1];
				for(j=js;j<je;j++)
				{
					jj  = A->U->index[j];
					#ifndef USE_SSE2
						LIS_QUAD_FMAD(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->U->value[j]);
					#else
						LIS_QUAD_FMAD_SSE2(y[jj],yl[jj],y[jj],yl[jj],x[i],xl[i],A->U->value[j]);
					#endif
				}
			}
		#endif
	}
	else
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR),"lis_matvect_csr_mp::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,jj,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,jj,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for 
				for(i=0; i<np; i++)
				{
					js = A->ptr[i];
					je = A->ptr[i+1];
					for(j=js;j<je;j++)
					{
						jj  = k*np+A->L->index[j];
						#ifndef USE_SSE2
							LIS_QUAD_FMAD(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->L->value[j]);
						#else
							LIS_QUAD_FMAD_SSE2(ww[jj],wwl[jj],ww[jj],wwl[jj],x[i],xl[i],A->L->value[j]);
						#endif
					}
				}
				#pragma omp for 
				for(i=0;i<np;i++)
				{
					y[i] = yl[i] = 0.0;
					for(j=0;j<nprocs;j++)
					{
						#ifndef USE_SSE2
							LIS_QUAD_ADD(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#else
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<n; i++)
			{
				y[i]  = 0.0;
				yl[i] = 0.0;
			}
			for(i=0; i<np; i++)
			{
				js = A->ptr[i];
				je = A->ptr[i+1];
				tt0.hi[0] = x[i];
				tt0.lo[0] = xl[i];
				for(j=js;j<je;j++)
				{
					jj  = A->index[j];
					#ifndef USE_SSE2
						LIS_QUAD_FMAD(y[jj],yl[jj],y[jj],yl[jj],tt0.hi[0],tt0.lo[0],A->value[j]);
					#else
						LIS_QUAD_FMAD_SSE2(y[jj],yl[jj],y[jj],yl[jj],tt0.hi[0],tt0.lo[0],A->value[j]);
					#endif
				}
			}
		#endif
	}
}

void lis_matvec_csc_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,js,je,j0,j1;
	#if defined(USE_AVX)
		LIS_INT			j2,j3;
	#endif
	LIS_INT				n,np;
	LIS_QUAD_PTR	tt0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#ifdef _OPENMP
		LIS_INT k,nprocs;
		LIS_SCALAR		*ww,*wwl;
	#endif
	LIS_QUAD_DECLAR;

	n    = A->n;
	np   = A->np;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	tt0.hi = &X->work[0];
	tt0.lo = &X->work[2];
	if( A->is_splited )
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR), "lis_matvect_csr_mp2::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,j0,j1,j2,j3,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,j0,j1,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,j0,j1,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for 
				for(i=0; i<np; i++)
				{
					#if defined(USE_AVX)
						js = A->ptr[i];
						je = A->ptr[i+1];
						for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
						{
							j0 = k*np + A->index[j+0]; j1 = k*np + A->index[j+1];
							j2 = k*np + A->index[j+2]; j3 = k*np + A->index[j+3];
							#if LIS_AVX_SIZE == 4
								LIS_QUAD_FMAD4_AVX_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->value[j]);
							#else
								#error no implementation for this vector size
							#endif
						}
						for(;j<je;j++)
						{
							j0  = A->index[j];
							LIS_QUAD_FMAD_SSE2(ww[j0],wwl[j0],ww[j0],wwl[j0],x[i],xl[i],A->value[j]);
						}
					#elif defined(USE_SSE2)
						js = A->ptr[i];
						je = A->ptr[i+1];
						for(j=js;j<je-1;j+=2)
						{
							j0  = k*np + A->index[j];
							j1  = k*np + A->index[j+1];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD2_SSE2_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j0],wwl[j0],ww[j1],wwl[j1],x[i],xl[i],x[i],xl[i],A->value[j]);
							#endif
						}
						for(;j<je;j++)
						{
							j0  = A->index[j];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD_SSE2(ww[j0],wwl[j0],ww[j0],wwl[j0],x[i],xl[i],A->value[j]);
							#endif
						}
					#else
						#error no implementation
					#endif
				}
				#pragma omp for 
				for(i=0;i<n;i++)
				{
					y[i] = yl[i] = 0.0;
					for(j=0;j<nprocs;j++)
					{
						#ifdef USE_SSE2
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<n; i++)
			{
				#ifndef USE_SSE2
					LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#else
					LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
				#endif
			}
			for(i=0; i<np; i++)
			{
				#if defined(USE_AVX)
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
					{
						j0 = A->L->index[j+0]; j1 = A->L->index[j+1];
						j2 = A->L->index[j+2]; j3 = A->L->index[j+3];
						#if LIS_AVX_SIZE == 4
							LIS_QUAD_FMAD4_AVX_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->L->value[j]);
						#else
							#error no implementation for this vector size
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->L->index[j];
						LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->L->value[j]);
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
					{
						j0 = A->U->index[j+0]; j1 = A->U->index[j+1];
						j2 = A->U->index[j+2]; j3 = A->U->index[j+3];
						#if LIS_AVX_SIZE == 4
							LIS_QUAD_FMAD4_AVX_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->U->value[j]);
						#else
							#error no implementation for this vector size
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->U->index[j];
						LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->U->value[j]);
					}
				#elif defined(USE_SSE2)
					js = A->L->ptr[i];
					je = A->L->ptr[i+1];
					for(j=js;j<je-1;j+=2)
					{
						j0  = A->L->index[j];
						j1  = A->L->index[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j0],yl[j0],y[j1],yl[j1],x[i],xl[i],x[i],xl[i],A->L->value[j]);
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->L->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->L->value[j]);
						#endif
					}
					js = A->U->ptr[i];
					je = A->U->ptr[i+1];
					for(j=js;j<je-1;j+=2)
					{
						j0  = A->U->index[j];
						j1  = A->U->index[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j0],yl[j0],y[j1],yl[j1],x[i],xl[i],x[i],xl[i],A->U->value[j]);
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->U->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],tt0.hi[0],tt0.lo[0],A->U->value[j]);
						#endif
					}
				#else
					#error no implementation
				#endif
			}
		#endif
	}
	else
	{
		#ifdef _OPENMP
			nprocs = omp_get_max_threads();
			ww  = (LIS_SCALAR *)lis_malloc( 2*nprocs*np*sizeof(LIS_SCALAR), "lis_matvect_csr_mp2::ww" );
			wwl = &ww[nprocs*np];
			#if defined(USE_AVX)
				#pragma omp parallel private(i,j,js,je,j0,j1,j2,j3,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
			#elif defined(USE_SSE2)
				#pragma omp parallel private(i,j,js,je,j0,j1,k,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
			#else
				#pragma omp parallel private(i,j,js,je,j0,j1,k,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
			#endif
			{
				k = omp_get_thread_num();
				#pragma omp for
				for(j=0;j<nprocs;j++)
				{
					memset( &ww[j*np], 0, np*sizeof(LIS_SCALAR) );
					memset( &wwl[j*np], 0, np*sizeof(LIS_SCALAR) );
				}
				#pragma omp for 
				for(i=0; i<np; i++)
				{
					#if defined(USE_AVX)
						js = A->ptr[i];
						je = A->ptr[i+1];
						for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
						{
							j0 = k*np + A->index[j+0]; j1 = k*np + A->index[j+1];
							j2 = k*np + A->index[j+2]; j3 = k*np + A->index[j+3];
							#if LIS_AVX_SIZE == 4
								LIS_QUAD_FMAD4_AVX_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j2],wwl[j2],ww[j3],wwl[j3],
														x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->value[j]);
							#else
								#error no implementation for this vector size
							#endif
						}
						for(;j<je;j++)
						{
							j0  = A->index[j];
							LIS_QUAD_FMAD_SSE2(ww[j0],wwl[j0],ww[j0],wwl[j0],x[i],xl[i],A->value[j]);
						}
					#elif defined(USE_SSE2)
						js = A->ptr[i];
						je = A->ptr[i+1];
						for(j=js;j<je-1;j+=2)
						{
							j0  = k*np + A->index[j];
							j1  = k*np + A->index[j+1];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD2_SSE2_STSD(ww[j0],wwl[j0],ww[j1],wwl[j1],ww[j0],wwl[j0],ww[j1],wwl[j1],x[i],xl[i],x[i],xl[i],A->value[j]);
							#endif
						}
						for(;j<je;j++)
						{
							j0  = A->index[j];
							#ifdef USE_SSE2
								LIS_QUAD_FMAD_SSE2(ww[j0],wwl[j0],ww[j0],wwl[j0],x[i],xl[i],A->value[j]);
							#endif
						}
					#else
						#error no implementation
					#endif
				}
				#pragma omp for 
				for(i=0;i<n;i++)
				{
					y[i] = yl[i] = 0.0;
					for(j=0;j<nprocs;j++)
					{
						#ifdef USE_SSE2
							LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],ww[j*np+i],wwl[j*np+i]);
						#endif
					}
				}
			}
			lis_free(ww);
		#else
			for(i=0; i<n; i++)
			{
				y[i]  = 0.0;
				yl[i] = 0.0;
			}
			for(i=0; i<np; i++)
			{
				#if defined(USE_AVX)
					js = A->ptr[i];
					je = A->ptr[i+1];
					for(j=js;j<je-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
					{
						j0 = A->index[j+0]; j1 = A->index[j+1];
						j2 = A->index[j+2]; j3 = A->index[j+3];
						#if LIS_AVX_SIZE == 4
							LIS_QUAD_FMAD4_AVX_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													y[j0],yl[j0],y[j1],yl[j1],y[j2],yl[j2],y[j3],yl[j3],
													x[i],xl[i],x[i],xl[i],x[i],xl[i],x[i],xl[i],A->value[j]);
						#else
							#error no implementation for this vector size
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->index[j];
						LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],x[i],xl[i],A->value[j]);
					}
				#elif defined(USE_SSE2)
					js = A->ptr[i];
					je = A->ptr[i+1];
					for(j=js;j<je-1;j+=2)
					{
						j0  = A->index[j];
						j1  = A->index[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_STSD(y[j0],yl[j0],y[j1],yl[j1],y[j0],yl[j0],y[j1],yl[j1],x[i],xl[i],x[i],xl[i],A->value[j]);
						#endif
					}
					for(;j<je;j++)
					{
						j0  = A->index[j];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD_SSE2(y[j0],yl[j0],y[j0],yl[j0],x[i],xl[i],A->value[j]);
						#endif
					}
				#else
					#error no implementation
				#endif
			}
		#endif
	}
}

void lis_matvect_csc_mp(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,np;
	LIS_INT				is,ie,j0;
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	LIS_QUAD_DECLAR;


	np    = A->np;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	if( A->is_splited )
	{
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(i,j,is,ie,j0,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<np;i++)
		{
			#ifndef USE_SSE2
				LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#else
				LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#endif
			is = A->L->ptr[i];
			ie = A->L->ptr[i+1];
			for(j=is;j<ie-0;j+=1)
			{
				j0 = A->L->index[j+0];
				#ifndef USE_SSE2
					LIS_QUAD_FMAD(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
				#else
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
				#endif
			}
			is = A->U->ptr[i];
			ie = A->U->ptr[i+1];
			for(j=is;j<ie-0;j+=1)
			{
				j0 = A->U->index[j+0];
				#ifndef USE_SSE2
					LIS_QUAD_FMAD(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
				#else
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
				#endif
			}
		}
	}
	else
	{
		jj0 = A->index;
		vv0 = A->value;
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(i,j,is,ie,j0,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(i,j,is,ie,j0,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<np;i++)
		{
			y[i] = yl[i] = 0.0;

			is = A->ptr[i];
			ie = A->ptr[i+1];
			for(j=is;j<ie-0;j+=1)
			{
				j0 = jj0[j+0];
				#ifndef USE_SSE2
					LIS_QUAD_FMAD(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				#else
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				#endif
			}
		}
	}
}


void lis_matvect_csc_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,np;
	LIS_INT				is,ie;
	LIS_INT				j0,j1;
	#if defined(USE_AVX)
		LIS_INT			j2,j3;
	#endif
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#if defined(USE_AVX)
		LIS_AVX_TYPE	tt_hi,tt_lo;
		LIS_SCALAR		s_hi,s_lo;
	#else
		LIS_QUAD_PD		tt;
	#endif
	LIS_QUAD_DECLAR;

	np    = A->np;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	if( A->is_splited )
	{
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for private(i,j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(i,j,is,ie,j0,j1,tt,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(i,j,is,ie,j0,j1,tt,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif
		for(i=0;i<np;i++)
		{
			#ifndef USE_SSE2
				LIS_QUAD_MULD(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#else
				LIS_QUAD_MULD_SSE2(y[i],yl[i],x[i],xl[i],A->D->value[i]);
			#endif

			#if defined(USE_AVX)
				tt_hi = LIS_AVX_FUNC(setzero_pd)();
				tt_lo = LIS_AVX_FUNC(setzero_pd)();

				is = A->L->ptr[i];
				ie = A->L->ptr[i+1];
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					j0 = A->L->index[j+0]; j1 = A->L->index[j+1];
					j2 = A->L->index[j+2]; j3 = A->L->index[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],A->L->value[j]);
					#else
						#error no implementation for this vector size
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->L->index[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
				}
				is = A->U->ptr[i];
				ie = A->U->ptr[i+1];
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					j0 = A->U->index[j+0]; j1 = A->U->index[j+1];
					j2 = A->U->index[j+2]; j3 = A->U->index[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],A->U->value[i]);
					#else
						#error no implementation for this vector size
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->U->index[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
				}
				LIS_QUAD_HADDALL_AVX(s_hi,s_lo,tt_hi,tt_lo);
				LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],s_hi,s_lo);
			#elif defined(USE_SSE2)
				tt.hi[0] = tt.hi[1] = tt.lo[0] = tt.lo[1] = 0.0;
				is = A->L->ptr[i];
				ie = A->L->ptr[i+1];
				for(j=is;j<ie-1;j+=2)
				{
					j0 = A->L->index[j+0];
					j1 = A->L->index[j+1];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],A->L->value[j]);
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->L->index[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->L->value[j]);
					#endif
				}
				is = A->U->ptr[i];
				ie = A->U->ptr[i+1];
				for(j=is;j<ie-1;j+=2)
				{
					j0 = A->U->index[j+0];
					j1 = A->U->index[j+1];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],A->U->value[j]);
					#endif
				}
				for(;j<ie;j++)
				{
					j0 = A->U->index[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],A->U->value[j]);
					#endif
				}
				#ifdef USE_SSE2
					LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],tt.hi[0],tt.lo[0]);
					LIS_QUAD_ADD_SSE2(y[i],yl[i],y[i],yl[i],tt.hi[1],tt.lo[1]);
				#endif
			#else
				#error no implementation
			#endif
		}
	}
	else
	{
		jj0 = A->index;
		vv0 = A->value;
		#ifdef _OPENMP
		#if defined(USE_AVX)
			#pragma omp parallel for private(i,j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3)
		#elif defined(USE_SSE2)
			#pragma omp parallel for private(i,j,is,ie,j0,j1,tt,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh)
		#else
			#pragma omp parallel for private(i,j,is,ie,j0,j1,tt,p1,p2,tq,bhi,blo,chi,clo,sh,sl,th,tl,eh,el)
		#endif
		#endif

		#if defined(USE_AVX)
			for(i=0;i<np;i++)
			{
				tt_hi = LIS_AVX_FUNC(setzero_pd)();
				tt_lo = LIS_AVX_FUNC(setzero_pd)();

				is = A->ptr[i];
				ie = A->ptr[i+1];
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					j0 = jj0[j+0]; j1 = jj0[j+1];
					j2 = jj0[j+2]; j3 = jj0[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],vv0[j]);
					#else
						#error no implementation for this vector size
					#endif
				}
				LIS_QUAD_HADDALL_AVX(y[i],yl[i],tt_hi,tt_lo);
				for(;j<ie;j++)
				{
					j0 = jj0[j+0];
					LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
				}
			}
		#elif defined(USE_SSE2)
			for(i=0;i<np;i++)
			{
				tt.hi[0] = tt.hi[1] = tt.lo[0] = tt.lo[1] = 0.0;

				is = A->ptr[i];
				ie = A->ptr[i+1];
				for(j=is;j<ie-1;j+=2)
				{
					j0 = jj0[j+0];
					j1 = jj0[j+1];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],vv0[j]);
					#endif
				}
				#ifdef USE_SSE2
					LIS_QUAD_ADD_SSE2(y[i],yl[i],tt.hi[0],tt.lo[0],tt.hi[1],tt.lo[1]);
				#endif
				for(;j<ie;j++)
				{
					j0 = jj0[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
					#endif
				}
			}
		#else
			#error no implementation
		#endif
	}
}

#if USE_AVX
void lis_matvec_csr_balanced_mp2(LIS_MATRIX A, LIS_VECTOR X, LIS_VECTOR Y)
{
	LIS_INT				i,j,n;
	LIS_INT				is,ie;
	LIS_INT				j0,j1;
	#if defined(USE_AVX)
		LIS_INT j2,j3,ij;
	#endif
	LIS_INT				*jj0;
	LIS_SCALAR		*vv0;
	LIS_SCALAR		*x,*y,*xl,*yl;
	#if defined(USE_AVX)
		LIS_AVX_TYPE	tt_hi, tt_lo;
		LIS_SCALAR		s_hi, s_lo;
		LIS_AVX_TYPE	tmph,tmpl;
	#else
		LIS_QUAD_PD		tt;
	#endif
	LIS_QUAD_DECLAR;
	#ifdef LIS_DEBUG_IDENT
	printf("%s\n","lis_matvec_csr_mp2");
	#endif
	n     = A->n;
	x     = X->value;
	y     = Y->value;
	xl    = X->value_lo;
	yl    = Y->value_lo;
	if( A->is_splited )
	{
		printf("error NO IMPLEMENT\n");
	}
	else
	{
	   //mp3_matvec
		jj0 = A->index;
		vv0 = A->value;

		
		#ifdef _OPENMP
		#pragma omp parallel private(j,is,ie,j0,j1,j2,j3,bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,eh,t3,tt_hi,tt_lo,s_hi,s_lo)
		#endif
	{
		#ifdef _OPENMP
		LIS_INT ogtn1 = A->omp_sep[omp_get_thread_num()];
		LIS_INT ogtn2 = A->omp_sep[omp_get_thread_num() + 1];
		#else
		LIS_INT ogtn1 = A->omp_sep[1];
		LIS_INT ogtn2 = A->omp_sep[2];
		#endif
		for(i=ogtn1;i<ogtn2;i++)
		{
			#if defined(USE_AVX)
				LIS_AVX_TYPE tt_hi = LIS_AVX_FUNC(setzero_pd)();
				LIS_AVX_TYPE tt_lo = LIS_AVX_FUNC(setzero_pd)();
			
				is = A->ptr[i];
				ie = A->ptr[i+1];
				#if THREAD_DETAIL==1
					a[thnum] += ie - is;
				#endif
				for(j=is;j<ie-(LIS_AVX_SIZE-1);j+=LIS_AVX_SIZE)
				{
					#if 1
						j0 = jj0[j+0]; j1 = jj0[j+1];
						j2 = jj0[j+2]; j3 = jj0[j+3];
					#if LIS_AVX_SIZE == 4
						LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],x[j3],xl[j3],vv0[j]);
						#else
						#error no implementation for this vector size
						#endif
					#else
						j0 = jj0[j];
						LIS_QUAD_FMADN_AVX(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],vv0[j]);
					#endif
				}
				#define padding 1
				//AVX	
				#if padding == 1
				if(j==ie-3)
				{
				   j0 = jj0[j+0]; j1 = jj0[j+1];
					j2 = jj0[j+2]; 
					LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],x[j2],xl[j2],0,0,vv0[j]);
					j+=3;
				}
				else if(j==ie-2)
				{
					j0 = jj0[j+0]; j1 = jj0[j+1];
					LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],x[j1],xl[j1],0,0,0,0,vv0[j]);
					j+=2;
				}
				else if(j==ie-1)
				{
					j0 = jj0[j+0];
					LIS_QUAD_FMAD4_AVX_LDSD(tt_hi,tt_lo,tt_hi,tt_lo,x[j0],xl[j0],0,0,0,0,0,0,vv0[j]);
					j+=1;
				}

				LIS_QUAD_HADDALL_AVX(y[i],yl[i],tt_hi,tt_lo);
				#endif
				
			#elif defined(USE_SSE2)
				tt.hi[0] = tt.hi[1] = tt.lo[0] = tt.lo[1] = 0.0;

				is = A->ptr[i];
				ie = A->ptr[i+1];

				for(j=is;j<ie-1;j+=2)
				{
					#if 1
						j0 = jj0[j+0];
						j1 = jj0[j+1];
						#ifdef USE_SSE2
							LIS_QUAD_FMAD2_SSE2_LDSD(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],x[j1],xl[j1],vv0[j]);
						#endif
					#else
						j0 = jj0[j];
						LIS_QUAD_FMAD2_SSE2(tt.hi[0],tt.lo[0],tt.hi[0],tt.lo[0],x[j0],xl[j0],vv0[j]);
					#endif
				}
				#ifdef USE_SSE2
					LIS_QUAD_ADD_SSE2(y[i],yl[i],tt.hi[0],tt.lo[0],tt.hi[1],tt.lo[1]);
				#endif
				for(;j<ie;j++)
				{
					j0 = jj0[j+0];
					#ifdef USE_SSE2
						LIS_QUAD_FMAD_SSE2(y[i],yl[i],y[i],yl[i],x[j0],xl[j0],vv0[j]);
					#endif
				}
			#else
				#error no implementation
			#endif
		}
	}
}
}

void lis_omp_auto_select(LIS_MATRIX A)
{
	LIS_INT *sample;
	LIS_INT i=0, j=0, n;
	LIS_INT nnzrow;
	LIS_INT var=0;

	n = A->n;
	nnzrow = A->nnz / A->n;
	sample = malloc (sizeof(LIS_INT) * n);

	for(i=0; i<n; i++)
	{
		sample[i] = A->ptr[i+1] - A-> ptr[i];	
	}

	for(i=0; i<n; i++)
	{
		var += (nnzrow - sample[i]) * (nnzrow - sample[i]);
	}
	var = var / n;

	if(var > 50000)
	{
		lis_omp_balanced(A);
	}
	else
	{
		A->omp_sche_flag = 0;
	}
}

void lis_omp_balanced(LIS_MATRIX A)	
	{
#ifdef _OPENMP
		LIS_INT n = A->n;
		LIS_INT nnz_count = A->ptr[n] / omp_get_max_threads();
		LIS_INT nonzero = nnz_count;
		A->omp_sep = malloc(sizeof(LIS_INT) * omp_get_max_threads());
		LIS_INT i=0, j = 0;

		A->omp_sep[0] = 0;
		for(i=0;i<n;i++)
		{
			if(A->ptr[i] > nonzero)
			{
				A->omp_sep[j+1] = i;
				nonzero = nonzero + nnz_count;
				j++;
			}
		}
		A->omp_sep[omp_get_max_threads()] = n;
		A->omp_sche_flag = 1;
#endif
	}
#endif
#endif
//#endif
