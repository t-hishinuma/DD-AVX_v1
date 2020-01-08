/* Copyright (C) 2002-2007 The SSI Project. All rights reserved.

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

#define ONE 1
#ifndef __LIS_PRECISION_H__
#define __LIS_PRECISION_H__

#define SPLITTER 134217729.0

#if defined(USE_AVX)
	#include <immintrin.h>
	#define LIS_AVX_SIZE 4

	//AVX version needs SSE2 version
	#include <emmintrin.h>
	#define LIS_VEC_CAST(VAR) (*(__m128d*)&(VAR))
	#if !defined(USE_SSE2)
		#define USE_SSE2 1
	#endif
	#if !defined(USE_FMA2_SSE2)
		#define USE_FMA2_SSE2
	#endif

	extern const double lis_splitter_; ///used by broadcast_sd

	//per vector size configurations
	//have to implement additional code for new vector size
	#if LIS_AVX_SIZE == 4
		#define LIS_AVX_TYPE __m256d
		#define LIS_AVX_FUNC(NAME) _mm256_##NAME
	#else
		#error no implementation for this vector size
	#endif
#elif defined(USE_SSE2)
	#include <emmintrin.h>
	#define LIS_VEC_CAST(VAR) (VAR)
#endif

#if defined(USE_QUAD_PRECISION)
	#if defined(USE_AVX)
		#define LIS_QUAD_DECLAR \
			const LIS_AVX_TYPE sp = LIS_AVX_FUNC(broadcast_sd)(&lis_splitter_); \
			LIS_AVX_TYPE one,mi,bh,bl,ch,cl,sh,sl,wh,wl,th,tl,p1,p2,t0,t1,t2,eh,t3; \
	   	mi = LIS_AVX_FUNC(set_pd)(-1.0,-1.0,-1.0,-1.0)
	#elif defined(USE_SSE2)
		#define LIS_QUAD_DECLAR			__m128d bh,ch,sh,wh,th,bl,cl,sl,wl,tl,p1,p2,t0,t1,t2,t3,eh
	#else
		#define LIS_QUAD_DECLAR			double p1,p2,tq,bhi,blo,chi,clo,sh,eh,sl,el,th,tl
	#endif
#else
	#define LIS_QUAD_DECLAR
#endif

/**********************************************************
 *                                                        *
 *                      Scalar                            *
 *                                                        *
 **********************************************************/
/**********************************************************
 * LIS_QUAD_MINUS(a_hi,a_lo)                              *
 **********************************************************
  (a_hi,a_lo) <- (-a_hi,-a_lo)
 **********************************************************/
#define LIS_QUAD_MINUS(a_hi,a_lo)   \
				(a_hi) = -(a_hi); \
				(a_lo) = -(a_lo)
/**********************************************************
 * LIS_QUAD_ZERO(a_hi,a_lo)                               *
 **********************************************************
  (a_hi,a_lo) <- (0,0)
 **********************************************************/
#define LIS_QUAD_ZERO(a_hi,a_lo)   \
				(a_hi) = 0.0; \
				(a_lo) = 0.0
/**********************************************************
 * LIS_QUAD_ONE(a_hi,a_lo)                                *
 **********************************************************
  (a_hi,a_lo) <- (1,0)
 **********************************************************/
#define LIS_QUAD_ONE(a_hi,a_lo)   \
				(a_hi) = 1.0; \
				(a_lo) = 0.0
/**********************************************************
 * LIS_QUAD_FAST_TWO_SUM(a,b,r,e)                         *
 **********************************************************
  |a| > |b|
  a + b -> (r,e)
 **********************************************************/
#define LIS_QUAD_FAST_TWO_SUM(a,b,r,e)   \
				(r) = (a) + (b); \
				(e) = (b)  - ((r) - (a))
/**********************************************************
 * LIS_QUAD_TWO_SUM(a,b,r,e)                              *
 **********************************************************
  a + b -> (r,e)
 **********************************************************/
#define LIS_QUAD_TWO_SUM(a,b,r,e)   \
				(r) =  (a) + (b); \
				th  =  (r) - (a); \
				(e) =  ((a) - ((r) - th)) + ((b) - th)
/**********************************************************
 * LIS_QUAD_TWO_DIFF(a,b,r,e)                             *
 **********************************************************
  a - b -> (r,e)
 **********************************************************/
#define LIS_QUAD_TWO_DIFF(a,b,r,e)   \
				(r) =  (a) - (b); \
				th  =  (r) - (a); \
				(e) =  ((a) - ((r) - th)) - ((b) + th)
/**********************************************************
 * LIS_QUAD_SPLIT(b,b_hi,b_lo)                            *
 **********************************************************
  b -> (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SPLIT(b,b_hi,b_lo) \
				tq     = SPLITTER * (b); \
				(b_hi) = tq - (tq-(b));  \
				(b_lo) = (b) - (b_hi);   \
/**********************************************************
 * LIS_QUAD_TWO_PROD(a,b,r,e)                             *
 **********************************************************
  a x b -> (r,e)
 **********************************************************/
#ifndef HAS_FMA
#define LIS_QUAD_TWO_PROD(a,b,r,e)   \
				(r)  = (a) * (b); \
				LIS_QUAD_SPLIT((a),bhi,blo); \
				LIS_QUAD_SPLIT((b),chi,clo); \
				(e)  = ((bhi*chi-(r))+bhi*clo+blo*chi)+blo*clo
#else
#define LIS_QUAD_TWO_PROD(a,b,r,e)   \
				(r)  = (-(a)) * (b); \
				(e)  = (a) * (b) + (r)
#endif
/**********************************************************
 * LIS_QUAD_TWO_SQR(a,r,e)                                *
 **********************************************************
  a x a -> (r,e)
 **********************************************************/
#ifndef HAS_FMA
#define LIS_QUAD_TWO_SQR(a,r,e)   \
				(r)  = (a) * (a); \
				LIS_QUAD_SPLIT((a),bhi,blo); \
				(e)  = ((bhi*bhi-(r))+2.0*bhi*blo)+blo*blo
#else
#define LIS_QUAD_TWO_SQR(a,r,e)   \
				(r)  = (-(a)) * (a); \
				(e)  = (a) * (a) + (r)
#endif
/**********************************************************
 * LIS_QUAD_MUL(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)            *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_MUL(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_TWO_PROD((b_hi),(c_hi),p1,p2); \
				p2 += ((b_hi) * (c_lo)); \
				p2 += ((b_lo) * (c_hi)); \
				LIS_QUAD_FAST_TWO_SUM(p1,p2,(a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_MULD(a_hi,a_lo,b_hi,b_lo,c)                   *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x c
 **********************************************************/
#define LIS_QUAD_MULD(a_hi,a_lo,b_hi,b_lo,c) \
				LIS_QUAD_TWO_PROD((b_hi),(c),p1,p2); \
				p2 += ((b_lo) * (c)); \
				LIS_QUAD_FAST_TWO_SUM(p1,p2,(a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_SQR(a_hi,a_lo,b_hi,b_lo)                      *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SQR(a_hi,a_lo,b_hi,b_lo) \
				LIS_QUAD_TWO_SQR((b_hi),p1,p2); \
				p2 += (2.0*(b_hi) * (b_lo)); \
				p2 += ((b_lo) * (b_lo)); \
				LIS_QUAD_FAST_TWO_SUM(p1,p2,(a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_ADD(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)            *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) + (c_hi,c_lo)
 **********************************************************/
#ifndef USE_FAST_QUAD_ADD
#define LIS_QUAD_ADD(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_TWO_SUM((b_hi),(c_hi),sh,eh); \
				LIS_QUAD_TWO_SUM((b_lo),(c_lo),sl,el); \
				eh += sl; \
				LIS_QUAD_FAST_TWO_SUM(sh,eh,sh,eh); \
				eh += el; \
				LIS_QUAD_FAST_TWO_SUM(sh,eh,(a_hi),(a_lo))
#else
#define LIS_QUAD_ADD(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_TWO_SUM((b_hi),(c_hi),sh,eh); \
				eh += (b_lo); \
				eh += (c_lo); \
				LIS_QUAD_FAST_TWO_SUM(sh,eh,(a_hi),(a_lo))
#endif
/**********************************************************
 * LIS_QUAD_DIV(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)            *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) / (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_DIV(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				tl  = (b_hi) / (c_hi); \
				LIS_QUAD_MULD(eh,el,(c_hi),(c_lo),tl); \
				LIS_QUAD_TWO_DIFF((b_hi),eh,sh,sl); \
				sl -= el; \
				sl += (b_lo); \
				th  = (sh+sl) / (c_hi); \
				LIS_QUAD_FAST_TWO_SUM(tl,th,(a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_SQRT(a_hi,a_lo,b_hi,b_lo)                     *
 **********************************************************
  (a_hi,a_lo) <- SQRT( (b_hi,b_lo) )
 **********************************************************/
#define LIS_QUAD_SQRT(a_hi,a_lo,b_hi,b_lo) \
				if( (b_hi)==0 ) \
				{ \
					(a_hi) = (a_lo) = 0.0; \
					return LIS_FAILS; \
				} \
				if( (b_hi)<0 ) \
				{ \
					printf("ERROR bh=%e\n",(b_hi)); \
					(a_hi) = (a_lo) = 0.0; \
					return LIS_FAILS; \
				} \
				p1  = 1.0 / sqrt((b_hi)); \
				p2 = (b_hi) * p1; \
				p1  = p1 * 0.5; \
				LIS_QUAD_TWO_SQR(p2,chi,clo); \
				LIS_QUAD_ADD(th,eh,(b_hi),(b_lo),-chi,-clo); \
				p1  = p1 * th; \
				LIS_QUAD_FAST_TWO_SUM(p1,p2,(a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_FMA(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)  *
 **********************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_FMA(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_MUL(chi,p2,(b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_ADD((d_hi),(d_lo),(a_hi),(a_lo),chi,p2)
/**********************************************************
 * LIS_QUAD_FMAD(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c)         *
 **********************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * c
 **********************************************************/
#define LIS_QUAD_FMAD(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c) \
				LIS_QUAD_MULD(chi,p2,(b_hi),(b_lo),(c)); \
				LIS_QUAD_ADD((d_hi),(d_lo),(a_hi),(a_lo),chi,p2)
/**********************************************************
 * LIS_QUAD_FSA(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo)            *
 **********************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_FSA(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo) \
				LIS_QUAD_SQR(bhi,p2,(b_hi),(b_lo)); \
				LIS_QUAD_ADD((d_hi),(d_lo),(a_hi),(a_lo),bhi,p2)



/**********************************************************
 *                                                        *
 *                      SSE2(SD)                          *
 *                                                        *
 **********************************************************/
/**********************************************************
 * LIS_QUAD_MUL_SSE2_STORE(a_hi,a_lo)                     *
 **********************************************************
  (a_hi,a_lo) <-
 **********************************************************/
#define LIS_QUAD_MUL_SSE2_STORE(a_hi,a_lo) \
				_mm_store_sd(&(a_hi),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sh)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(sh)); \
				_mm_store_sd(&(a_lo),LIS_VEC_CAST(wh))
/**********************************************************
 * LIS_QUAD_ADD_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo)            *
 **********************************************************
  (b_hi,b_lo)  (c_hi,c_lo)
 **********************************************************/
#ifndef USE_FAST_QUAD_ADD
#define LIS_QUAD_ADD_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(sh) = _mm_set_pd((b_lo),(b_hi)); \
				LIS_VEC_CAST(bl) = _mm_set_pd((c_lo),(c_hi))
#else
#define LIS_QUAD_ADD_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(eh) = _mm_set_sd((b_hi)); \
				LIS_VEC_CAST(cl) = _mm_set_sd((b_lo)); \
				LIS_VEC_CAST(sh) = _mm_set_sd((c_hi)); \
				LIS_VEC_CAST(wh) = _mm_set_sd((c_lo))
#endif
/**********************************************************
 * LIS_QUAD_FMA_SSE2_LOAD(a_hi,a_lo)                      *
 **********************************************************
  (a_hi,a_lo)
 **********************************************************/
#ifndef USE_FAST_QUAD_ADD
#define LIS_QUAD_FMA_SSE2_LOAD(a_hi,a_lo) \
				LIS_VEC_CAST(th)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bl)  = _mm_set_pd((a_lo),(a_hi)); \
				LIS_VEC_CAST(sh)  = _mm_unpacklo_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(wh))
#else
#define LIS_QUAD_FMA_SSE2_LOAD(a_hi,a_lo) \
				LIS_VEC_CAST(th)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(eh) = _mm_set_sd((a_hi)); \
				LIS_VEC_CAST(cl) = _mm_set_sd((a_lo))
#endif
/**********************************************************
 * LIS_QUAD_MUL_SSE2_CORE(b_hi,b_lo,c_hi,c_lo)            *
 **********************************************************
  (b_hi,b_lo) x (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_MUL_SSE2_CORE(b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(sh)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(ch) = _mm_set_pd((c_hi),(b_hi)); \
				LIS_VEC_CAST(bh)  = _mm_unpackhi_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(tl) = _mm_set_pd((b_lo),(c_lo)); \
				LIS_VEC_CAST(bh)  = _mm_mul_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(th)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(th)  = _mm_mul_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(tl)); \
				LIS_VEC_CAST(ch) = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(eh)  = _mm_unpackhi_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(wh)  = _mm_unpacklo_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch) = _mm_unpackhi_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(wh));  \
				LIS_VEC_CAST(wh)  = _mm_mul_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch) = _mm_unpackhi_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(tl) = _mm_unpackhi_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(eh)  = _mm_unpackhi_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(tl)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(sh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bh))
/**********************************************************
 * LIS_QUAD_MULD_SSE2_CORE(b_hi,b_lo,c)                   *
 **********************************************************
  (b_hi,b_lo) x c
 **********************************************************/
#define LIS_QUAD_MULD_SSE2_CORE(b_hi,b_lo,c) \
				LIS_VEC_CAST(sh)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(ch) = _mm_set_pd((b_hi),(c)); \
				LIS_VEC_CAST(bh)  = _mm_unpackhi_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sl) = _mm_load_sd(&(b_lo)); \
				LIS_VEC_CAST(bh)  = _mm_mul_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sl) = _mm_mul_sd(LIS_VEC_CAST(sl),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(th)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(ch) = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(t1)  = _mm_unpackhi_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(wh)  = _mm_unpacklo_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch) = _mm_unpackhi_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(wh));  \
				LIS_VEC_CAST(wh)  = _mm_mul_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch) = _mm_unpackhi_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(th)  = _mm_unpackhi_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(sh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bh))
/**********************************************************
 * LIS_QUAD_SQR_SSE2_CORE(b_hi,b_lo)                      *
 **********************************************************
  (b_hi,b_lo) x (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SQR_SSE2_CORE(b_hi,b_lo) \
				LIS_VEC_CAST(sh)  = _mm_set_sd(SPLITTER); \
				LIS_VEC_CAST(th)  = _mm_load_sd(&(b_hi)); \
				LIS_VEC_CAST(bl)  = _mm_load_sd(&(b_lo)); \
				LIS_VEC_CAST(bh)  = _mm_mul_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(sh)  = _mm_mul_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(sh)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(cl)  = _mm_sub_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(wh)  = _mm_mul_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sh)  = _mm_add_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sh)  = _mm_mul_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(cl)  = _mm_mul_sd(LIS_VEC_CAST(cl),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(th)  = _mm_add_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(th)  = _mm_mul_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(bl)  = _mm_mul_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(sh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(bh))
/**********************************************************
 * LIS_QUAD_ADD_SSE2_CORE(a_hi,a_lo)                      *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) + (c_hi,c_lo)
 **********************************************************/
#ifndef USE_FAST_QUAD_ADD
#define LIS_QUAD_ADD_SSE2_CORE(a_hi,a_lo) \
				LIS_VEC_CAST(t0) = _mm_add_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(eh) = _mm_sub_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(th) = _mm_sub_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(sh) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(bl) = _mm_sub_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bl) = _mm_add_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(eh) = _mm_unpackhi_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(sh) = _mm_unpackhi_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(th) = LIS_VEC_CAST(t0); \
				LIS_VEC_CAST(bl) = _mm_add_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(th) = _mm_add_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(t0) = _mm_sub_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(bl) = _mm_sub_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(bl) = _mm_add_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(sh) = _mm_add_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bl)); \
				_mm_store_sd(&(a_hi),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sh) = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bl) = _mm_sub_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(sh)); \
				_mm_store_sd(&(a_lo),LIS_VEC_CAST(bl))
#else
#define LIS_QUAD_ADD_SSE2_CORE(a_hi,a_lo) \
				LIS_VEC_CAST(bl) = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(th) = _mm_sub_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(t0) = _mm_sub_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(sh) = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(eh) = _mm_sub_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(eh) = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(eh) = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(eh) = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(th) = _mm_add_sd(LIS_VEC_CAST(bl),LIS_VEC_CAST(eh)); \
				_mm_store_sd(&(a_hi),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(th) = _mm_sub_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(eh) = _mm_sub_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(th)); \
				_mm_store_sd(&(a_lo),LIS_VEC_CAST(eh))
#endif
/**********************************************************
 * LIS_QUAD_MUL_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)       *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_MUL_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_MUL_SSE2_CORE((b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_MUL_SSE2_STORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_MULD_SSE2(a_hi,a_lo,b_hi,b_lo,c)              *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x c
 **********************************************************/
#define LIS_QUAD_MULD_SSE2(a_hi,a_lo,b_hi,b_lo,c) \
				LIS_QUAD_MULD_SSE2_CORE((b_hi),(b_lo),(c)); \
				LIS_QUAD_MUL_SSE2_STORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_SQR_SSE2(a_hi,a_lo,b_hi,b_lo)                 *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SQR_SSE2(a_hi,a_lo,b_hi,b_lo) \
				LIS_QUAD_SQR_SSE2_CORE((b_hi),(b_lo)); \
				LIS_QUAD_MUL_SSE2_STORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_ADD_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)       *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) + (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_ADD_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_ADD_SSE2_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_ADD_SSE2_CORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_DIV_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)       *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) / (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_DIV_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(sh)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(bh)  = _mm_set_pd((b_lo),(b_hi)); \
				LIS_VEC_CAST(ch)  = _mm_set_pd((c_lo),(c_hi)); \
				LIS_VEC_CAST(p2)  = LIS_VEC_CAST(bh); \
				LIS_VEC_CAST(wh)  = LIS_VEC_CAST(ch); \
				LIS_VEC_CAST(p2)  = _mm_div_sd(LIS_VEC_CAST(p2),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(wh)  = _mm_unpacklo_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(ch)  = _mm_move_sd(LIS_VEC_CAST(ch),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(p2)  = LIS_VEC_CAST(wh); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(th)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(sh)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(ch)  = _mm_mul_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(wh)  = _mm_sub_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(th)  = LIS_VEC_CAST(sh); \
				LIS_VEC_CAST(p1)  = LIS_VEC_CAST(wh); \
				LIS_VEC_CAST(th)  = _mm_unpackhi_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(p1)  = _mm_unpackhi_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(eh)  = LIS_VEC_CAST(th); \
				LIS_VEC_CAST(th)  = _mm_mul_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(eh)  = _mm_mul_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(wh)  = _mm_mul_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(sh)  = _mm_mul_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p1)  = LIS_VEC_CAST(ch); \
				LIS_VEC_CAST(p1)  = _mm_unpackhi_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(eh)  = _mm_sub_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(eh)  = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(eh)  = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(eh)  = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(eh)  = _mm_add_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p1)  = LIS_VEC_CAST(ch); \
				LIS_VEC_CAST(p1)  = _mm_add_sd(LIS_VEC_CAST(p1),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(wh)  = LIS_VEC_CAST(p1); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = LIS_VEC_CAST(bh); \
				LIS_VEC_CAST(sh)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(eh)  = _mm_sub_sd(LIS_VEC_CAST(eh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(th)  = LIS_VEC_CAST(sh); \
				LIS_VEC_CAST(th)  = _mm_sub_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(p1)  = _mm_add_sd(LIS_VEC_CAST(p1),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(wh)  = LIS_VEC_CAST(sh); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(th)  = LIS_VEC_CAST(bh); \
				LIS_VEC_CAST(th)  = _mm_unpackhi_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bh)  = _mm_sub_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(bh)  = _mm_sub_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(bh)  = _mm_sub_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(bh)  = _mm_add_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(th)  = LIS_VEC_CAST(p2); \
				LIS_VEC_CAST(th)  = _mm_unpackhi_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(eh)  = LIS_VEC_CAST(th); \
				LIS_VEC_CAST(sh)  = _mm_add_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(sh)  = _mm_div_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(th)  = _mm_add_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(sh)); \
				_mm_store_sd(&(a_hi),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(th)  = _mm_sub_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(sh)  = _mm_sub_sd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				_mm_store_sd(&(a_lo),LIS_VEC_CAST(sh))
/**********************************************************
 * LIS_QUAD_SQRT_SSE2(a_hi,a_lo,b_hi,b_lo)                *
 **********************************************************
  (a_hi,a_lo) <- SQRT( (b_hi,b_lo) )
 **********************************************************/
#define LIS_QUAD_SQRT_SSE2(a_hi,a_lo,b_hi,b_lo) \
				if( (b_hi)==0 ) \
				{ \
					(a_hi) = (a_lo) = 0.0; \
					return LIS_FAILS; \
				} \
				if( (b_hi)<0 ) \
				{ \
					printf("ERROR LIS_VEC_CAST(bh)=%e\n",(b_hi)); \
					(a_hi) = (a_lo) = 0.0; \
					return LIS_FAILS; \
				} \
				LIS_VEC_CAST(wh)  = _mm_set_sd(SPLITTER); \
				LIS_VEC_CAST(bh) = _mm_load_pd(&(b_hi)); \
				LIS_VEC_CAST(bh) = LIS_VEC_CAST(bh); \
				LIS_VEC_CAST(t0) = _mm_set_sd(1.0); \
				LIS_VEC_CAST(t1) = _mm_set_sd(0.5); \
				LIS_VEC_CAST(t2) = _mm_sqrt_pd(LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(t0) = _mm_div_sd(LIS_VEC_CAST(t0),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(t2) = _mm_mul_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(t0) = _mm_mul_sd(LIS_VEC_CAST(t0),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(p1)  = _mm_mul_sd(LIS_VEC_CAST(t2),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(wh)  = _mm_mul_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(t1)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(wh)  = _mm_sub_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(wl)  = _mm_sub_sd(LIS_VEC_CAST(t2),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(t1)  = _mm_mul_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(wh)  = _mm_add_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(wh)  = _mm_mul_sd(LIS_VEC_CAST(wh),LIS_VEC_CAST(wl)); \
				LIS_VEC_CAST(wl)  = _mm_mul_sd(LIS_VEC_CAST(wl),LIS_VEC_CAST(wl)); \
				LIS_VEC_CAST(t1)  = _mm_sub_sd(LIS_VEC_CAST(t1),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(t1)  = _mm_add_sd(LIS_VEC_CAST(t1),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(t1)  = _mm_add_sd(LIS_VEC_CAST(t1),LIS_VEC_CAST(wl)); \
				LIS_VEC_CAST(p1) = _mm_unpacklo_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(sh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(eh) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(th) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(p1) = _mm_add_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(eh) = _mm_unpackhi_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(ch) = _mm_unpackhi_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(th) = LIS_VEC_CAST(sh); \
				LIS_VEC_CAST(bh) = _mm_add_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(th) = _mm_add_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(sh) = _mm_sub_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(bh) = _mm_sub_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(bh) = _mm_add_sd(LIS_VEC_CAST(bh),LIS_VEC_CAST(eh)); \
				LIS_VEC_CAST(th) = _mm_add_sd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(t0) = _mm_mul_sd(LIS_VEC_CAST(t0),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(p1) = _mm_add_sd(LIS_VEC_CAST(t2),LIS_VEC_CAST(t0)); \
				_mm_store_sd(&(a_hi),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p1) = _mm_sub_sd(LIS_VEC_CAST(p1),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(t0) = _mm_sub_sd(LIS_VEC_CAST(t0),LIS_VEC_CAST(p1)); \
				_mm_store_sd(&(a_lo),LIS_VEC_CAST(t0))
/***************************************************************
 * LIS_QUAD_FMA_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)  *
 ***************************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (c_hi,c_lo)
 ***************************************************************/
#define LIS_QUAD_FMA_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_MUL_SSE2_CORE((b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_FMA_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD_SSE2_CORE((d_hi),(d_lo))
/***************************************************************
 * LIS_QUAD_FMAD_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c)         *
 ***************************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * c
 ***************************************************************/
#define LIS_QUAD_FMAD_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c) \
				LIS_QUAD_MULD_SSE2_CORE((b_hi),(b_lo),(c)); \
				LIS_QUAD_FMA_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD_SSE2_CORE((d_hi),(d_lo))
/***************************************************************
 * LIS_QUAD_FSA_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo)            *
 ***************************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (b_hi,b_lo)
 ***************************************************************/
#define LIS_QUAD_FSA_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo) \
				LIS_QUAD_SQR_SSE2_CORE((b_hi),(b_lo)); \
				LIS_QUAD_FMA_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD_SSE2_CORE((d_hi),(d_lo))


/**********************************************************
 *                                                        *
 *                      SSE2(PD)                          *
 *                                                        *
 **********************************************************/
/**********************************************************
 * LIS_QUAD_MUL2_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo)           *
 **********************************************************
  (b_hi,b_lo)  (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_MUL2_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(t0)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(bh) = _mm_loadu_pd(&(b_hi)); \
				LIS_VEC_CAST(ch) = _mm_loadu_pd(&(c_hi)); \
				LIS_VEC_CAST(bl) = _mm_loadu_pd(&(b_lo)); \
				LIS_VEC_CAST(cl) = _mm_loadu_pd(&(c_lo))
/**********************************************************
 * LIS_QUAD_MULD2_SSE2_LOAD(b_hi,b_lo,c)                  *
 **********************************************************
  (b_hi,b_lo)  (c)
 **********************************************************/
#define LIS_QUAD_MULD2_SSE2_LOAD(b_hi,b_lo,c) \
				LIS_VEC_CAST(t0)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(bh)  = _mm_loadu_pd(&(b_hi)); \
				LIS_VEC_CAST(bl)  = _mm_loadu_pd(&(b_lo)); \
				LIS_VEC_CAST(ch)  = _mm_loadu_pd(&(c))
#define LIS_QUAD_MULD2_SSE2_LOAD_SD(b0_hi,b0_lo,b1_hi,b1_lo,c) \
				LIS_VEC_CAST(t0)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(bh)  = _mm_set_pd((b1_hi),(b0_hi)); \
				LIS_VEC_CAST(bl)  = _mm_set_pd((b1_lo),(b0_lo)); \
				LIS_VEC_CAST(ch)  = _mm_loadu_pd(&(c))
/**********************************************************
 * LIS_QUAD_SQR2_SSE2_LOAD(b_hi,b_lo)                     *
 **********************************************************
  (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SQR2_SSE2_LOAD(b_hi,b_lo) \
				LIS_VEC_CAST(ch)  = _mm_set_pd(SPLITTER,SPLITTER); \
				LIS_VEC_CAST(bh) = _mm_loadu_pd(&(b_hi)); \
				LIS_VEC_CAST(bl) = _mm_loadu_pd(&(b_lo))
/**********************************************************
 * LIS_QUAD_MUL2_SSE2_STORE(a_hi,a_lo)                    *
 **********************************************************
  (a_hi,a_lo) <-
 **********************************************************/
#define LIS_QUAD_MUL2_SSE2_STORE(a_hi,a_lo) \
				_mm_storeu_pd(&(a_hi),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(ch)); \
				_mm_storeu_pd(&(a_lo),LIS_VEC_CAST(p2))
#define LIS_QUAD_MUL2_SSE2_STOREU(a_hi,a_lo) \
				_mm_storeu_pd(&(a_hi),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(ch)); \
				_mm_storeu_pd(&(a_lo),LIS_VEC_CAST(p2))
/**********************************************************
 * LIS_QUAD_ADD2_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo)           *
 **********************************************************
  (b_hi,b_lo)  (c_hi,c_lo)
 **********************************************************/
#ifndef USE_FAST_QUAD_ADD
#define LIS_QUAD_ADD2_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(sh) = _mm_set_pd((b_lo),(b_hi)); \
				LIS_VEC_CAST(bl) = _mm_set_pd((c_lo),(c_hi))
#else
#define LIS_QUAD_ADD2_SSE2_LOAD(b_hi,b_lo,c_hi,c_lo) \
				LIS_VEC_CAST(eh) = _mm_set_sd((b_hi)); \
				LIS_VEC_CAST(cl) = _mm_set_sd((b_lo)); \
				LIS_VEC_CAST(sh) = _mm_set_sd((c_hi)); \
				LIS_VEC_CAST(wh) = _mm_set_sd((c_lo))
#endif
/**********************************************************
 * LIS_QUAD_ADD2_SSE2_STORE(a_hi,a_lo)                    *
 **********************************************************
  (a_hi,a_lo) <-
 **********************************************************/
#define LIS_QUAD_ADD2_SSE2_STORE(a_hi,a_lo) \
				_mm_storeu_pd(&(a_hi),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sh) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sh)); \
				_mm_storeu_pd(&(a_lo),LIS_VEC_CAST(bh))
#define LIS_QUAD_ADD2_SSE2_STORE_SD(a0_hi,a0_lo,a1_hi,a1_lo) \
				_mm_storel_pd(&(a0_hi),LIS_VEC_CAST(sh)); \
				_mm_storeh_pd(&(a1_hi),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sh) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sh)); \
				_mm_storel_pd(&(a0_lo),LIS_VEC_CAST(bh)); \
				_mm_storeh_pd(&(a1_lo),LIS_VEC_CAST(bh))
/**********************************************************
 * LIS_QUAD_FMA2_SSE2_LOAD(a_hi,a_lo)                     *
 **********************************************************
  (a_hi,a_lo)
 **********************************************************/
#define LIS_QUAD_FMA2_SSE2_LOAD(a_hi,a_lo) \
				LIS_VEC_CAST(t1)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(bh) = _mm_loadu_pd(&(a_hi)); \
				LIS_VEC_CAST(bl) = _mm_loadu_pd(&(a_lo))
#define LIS_QUAD_FMA2_SSE2_LOAD_SD(a0_hi,a0_lo,a1_hi,a1_lo) \
				LIS_VEC_CAST(t1)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(bh)  = _mm_set_pd((a1_hi),(a0_hi)); \
				LIS_VEC_CAST(bl)  = _mm_set_pd((a1_lo),(a0_lo))
/**********************************************************
 * LIS_QUAD_MUL2_SSE2_CORE                                *
 **********************************************************
  (b_hi,b_lo) x (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_MUL2_SSE2_CORE \
				LIS_VEC_CAST(p1)  = _mm_mul_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(sl)  = _mm_mul_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(th)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(tl)  = _mm_sub_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(sl)  = _mm_sub_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(tl)); \
				LIS_VEC_CAST(t1)  = _mm_mul_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(wh)  = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(t2)  = _mm_mul_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(wl)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(t0)  = _mm_mul_pd(LIS_VEC_CAST(wh),LIS_VEC_CAST(wl)); \
				LIS_VEC_CAST(p2)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(wl)); \
				LIS_VEC_CAST(sl)  = _mm_mul_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(ch)  = _mm_add_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(p2))
/**********************************************************
 * LIS_QUAD_MULD2_SSE2_CORE                               *
 **********************************************************
  (b_hi,b_lo) x c
 **********************************************************/
#define LIS_QUAD_MULD2_SSE2_CORE \
				LIS_VEC_CAST(p1)  = _mm_mul_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(bl)  = _mm_mul_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sh)  = _mm_mul_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(th)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(sh)  = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bh)  = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(sl)  = _mm_mul_pd(LIS_VEC_CAST(t0),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(tl)  = _mm_sub_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sl)  = _mm_sub_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(tl)); \
				LIS_VEC_CAST(ch)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(t2)  = _mm_mul_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(p2)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(t0)  = _mm_mul_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(t1)  = _mm_mul_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t1)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(t2)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(ch)  = _mm_add_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(p2))
/**********************************************************
 * LIS_QUAD_SQR2_SSE2_CORE                                *
 **********************************************************
  (b_hi,b_lo) x (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SQR2_SSE2_CORE \
				LIS_VEC_CAST(p1)  = _mm_mul_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(ch)  = _mm_mul_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(ch)  = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(cl)  = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(p2)  = _mm_mul_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch)  = _mm_add_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(ch)  = _mm_mul_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(cl)  = _mm_mul_pd(LIS_VEC_CAST(cl),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(p2)  = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(p1)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(cl)); \
				LIS_VEC_CAST(bh)  = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(bh)  = _mm_mul_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(bl)  = _mm_mul_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(p2)  = _mm_add_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(ch)  = _mm_add_pd(LIS_VEC_CAST(p1),LIS_VEC_CAST(p2))
/**********************************************************
 * LIS_QUAD_ADD2_SSE2_CORE                                *
 **********************************************************
  (b_hi,b_lo) + (c_hi,c_lo)
 **********************************************************/
#ifndef USE_FAST_QUAD_ADD
#define LIS_QUAD_ADD2_SSE2_CORE \
				LIS_VEC_CAST(sh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(th) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(t0) = _mm_sub_pd(LIS_VEC_CAST(sh),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(ch) = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(bh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(sl) = _mm_add_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(th) = _mm_sub_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(t0) = _mm_sub_pd(LIS_VEC_CAST(sl),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(p2) = _mm_sub_pd(LIS_VEC_CAST(p2),LIS_VEC_CAST(th)); \
				LIS_VEC_CAST(bl) = _mm_sub_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(bl) = _mm_add_pd(LIS_VEC_CAST(bl),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(bh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sl)); \
				LIS_VEC_CAST(th) = LIS_VEC_CAST(sh); \
				LIS_VEC_CAST(th) = _mm_add_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(sh) = _mm_sub_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(sh)); \
				LIS_VEC_CAST(bh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(sh) = _mm_add_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh))
#else
#define LIS_QUAD_ADD2_SSE2_CORE \
				LIS_VEC_CAST(th) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(wh) = _mm_sub_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh)); \
				LIS_VEC_CAST(t0) = _mm_sub_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(ch) = _mm_sub_pd(LIS_VEC_CAST(ch),LIS_VEC_CAST(wh)); \
				LIS_VEC_CAST(bh) = _mm_sub_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(t0)); \
				LIS_VEC_CAST(bh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(ch)); \
				LIS_VEC_CAST(bh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(bl)); \
				LIS_VEC_CAST(bh) = _mm_add_pd(LIS_VEC_CAST(bh),LIS_VEC_CAST(p2)); \
				LIS_VEC_CAST(sh) = _mm_add_pd(LIS_VEC_CAST(th),LIS_VEC_CAST(bh))
#endif
/**********************************************************
 * LIS_QUAD_MUL2_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)      *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_MUL2_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_MUL2_SSE2_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_MUL2_SSE2_CORE; \
				LIS_QUAD_MUL2_SSE2_STORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_MULD2_SSE2(a_hi,a_lo,b_hi,b_lo,c)             *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x c
 **********************************************************/
#define LIS_QUAD_MULD2_SSE2(a_hi,a_lo,b_hi,b_lo,c) \
				LIS_QUAD_MULD2_SSE2_LOAD((b_hi),(b_lo),(c)); \
				LIS_QUAD_MULD2_SSE2_CORE; \
				LIS_QUAD_MUL2_SSE2_STORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_SQR2_SSE2(a_hi,a_lo,b_hi,b_lo)                *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) x (b_hi,b_lo)
 **********************************************************/
#define LIS_QUAD_SQR2_SSE2(a_hi,a_lo,b_hi,b_lo) \
				LIS_QUAD_SQR2_SSE2_LOAD((b_hi),(b_lo)); \
				LIS_QUAD_SQR2_SSE2_CORE; \
				LIS_QUAD_MUL2_SSE2_STORE((a_hi),(a_lo))
/**********************************************************
 * LIS_QUAD_ADD2_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)      *
 **********************************************************
  (a_hi,a_lo) <- (b_hi,b_lo) + (c_hi,c_lo)
 **********************************************************/
#define LIS_QUAD_ADD2_SSE2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_ADD2_SSE2_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_ADD2_SSE2_CORE; \
				LIS_QUAD_ADD2_SSE2_STORE((a_hi),(a_lo))
/***************************************************************
 * LIS_QUAD_FMA2_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) *
 ***************************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (c_hi,c_lo)
 ***************************************************************/
#define LIS_QUAD_FMA2_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
				LIS_QUAD_MUL2_SSE2_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
				LIS_QUAD_MUL2_SSE2_CORE; \
				LIS_QUAD_FMA2_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD2_SSE2_CORE; \
				LIS_QUAD_ADD2_SSE2_STORE((d_hi),(d_lo))
/***************************************************************************
 * LIS_QUAD_FMAD2_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c)                    *
 * LIS_QUAD_FMAD2_SSE2_LDSD(d_hi,d_lo,a_hi,a_lo,b0_hi,b0_lo,b1_hi,b1_lo,c) *
 ***************************************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * c
 ***************************************************************/
#define LIS_QUAD_FMAD2_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c) \
				LIS_QUAD_MULD2_SSE2_LOAD((b_hi),(b_lo),(c)); \
				LIS_QUAD_MULD2_SSE2_CORE; \
				LIS_QUAD_FMA2_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD2_SSE2_CORE; \
				LIS_QUAD_ADD2_SSE2_STORE((d_hi),(d_lo))
/****************************************************************/
#define LIS_QUAD_FMAD2_SSE2_LDSD(d_hi,d_lo,a_hi,a_lo,b0_hi,b0_lo,b1_hi,b1_lo,c) \
				LIS_QUAD_MULD2_SSE2_LOAD_SD((b0_hi),(b0_lo),(b1_hi),(b1_lo),(c)); \
				LIS_QUAD_MULD2_SSE2_CORE; \
				LIS_QUAD_FMA2_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD2_SSE2_CORE; \
				LIS_QUAD_ADD2_SSE2_STORE((d_hi),(d_lo))
/****************************************************************/
#define LIS_QUAD_FMAD2_SSE2_STSD(d0_hi,d0_lo,d1_hi,d1_lo,a0_hi,a0_lo,a1_hi,a1_lo,b0_hi,b0_lo,b1_hi,b1_lo,c) \
				LIS_QUAD_MULD2_SSE2_LOAD_SD((b0_hi),(b0_lo),(b1_hi),(b1_lo),(c)); \
				LIS_QUAD_MULD2_SSE2_CORE; \
				LIS_QUAD_FMA2_SSE2_LOAD_SD((a0_hi),(a0_lo),(a1_hi),(a1_lo)); \
				LIS_QUAD_ADD2_SSE2_CORE; \
				LIS_QUAD_ADD2_SSE2_STORE_SD((d0_hi),(d0_lo),(d1_hi),(d1_lo))
/***************************************************************
 * LIS_QUAD_FSA2_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo)           *
 ***************************************************************
  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (b_hi,b_lo)
 ***************************************************************/
#define LIS_QUAD_FSA2_SSE2(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo) \
				LIS_QUAD_SQR2_SSE2_LOAD((b_hi),(b_lo)); \
				LIS_QUAD_SQR2_SSE2_CORE; \
				LIS_QUAD_FMA2_SSE2_LOAD((a_hi),(a_lo)); \
				LIS_QUAD_ADD2_SSE2_CORE; \
				LIS_QUAD_ADD2_SSE2_STORE((d_hi),(d_lo))

#if defined(USE_AVX)
	/**********************************************************
	 *                                                        *
	 *                AVX(Nbit; Quad x Quad)                  *
	 *                                                        *
	 **********************************************************/
	/**********************************************************
	 * lis_quad_addn_avx_load(b_hi,b_lo,c_hi,c_lo)            *
	 **********************************************************
	  (b_hi,b_lo)  (c_hi,c_lo)
	 **********************************************************/
	#ifndef use_fast_quad_add
	#define lis_quad_addn_avx_load(b_hi,b_lo,c_hi,c_lo) \
					sh = lis_avx_func(set_pd)((b_lo),(b_hi)); \
					bl = lis_avx_func(set_pd)((c_lo),(c_hi))
	#else
	#define lis_quad_addn_avx_load(b_hi,b_lo,c_hi,c_lo) \
	 				bh = lis_avx_func(loadu_pd)((double*)&(b_hi)); \
					bl = LIS_AVX_FUNC(loadu_pd)((double*)&(b_lo)); \
					ch = LIS_AVX_FUNC(loadu_pd)((double*)&(c_hi)); \
					p2 = LIS_AVX_FUNC(loadu_pd)((double*)&(c_lo))
	#endif

	/**********************************************************
	 * LIS_QUAD_MULN_AVX_LOAD(b_hi,b_lo,c_hi,c_lo)            *
	 **********************************************************
	  (b_hi,b_lo)  (c_hi,c_lo)
	 **********************************************************/
	#define LIS_QUAD_MULN_AVX_LOAD(b_hi,b_lo,c_hi,c_lo) \
					bh = LIS_AVX_FUNC(loadu_pd)((double*)&(b_hi)); \
					ch = LIS_AVX_FUNC(loadu_pd)((double*)&(c_hi)); \
					bl = LIS_AVX_FUNC(loadu_pd)((double*)&(b_lo)); \
					cl = LIS_AVX_FUNC(loadu_pd)((double*)&(c_lo))

	/**********************************************************
	 * LIS_QUAD_FMAN_AVX_LOAD(a_hi,a_lo)                      *
	 **********************************************************
	  (a_hi,a_lo)
	 **********************************************************/
	#define LIS_QUAD_FMAN_AVX_LOAD(a_hi,a_lo) \
				t1  = LIS_AVX_FUNC(sub_pd)(ch,p1); \
				p2  = LIS_AVX_FUNC(sub_pd)(p2,t1); \
				bh  = LIS_AVX_FUNC(loadu_pd)((double*)&(a_hi)); \
				bl  = LIS_AVX_FUNC(loadu_pd)((double*)&(a_lo))

	/**********************************************************
	 * LIS_QUAD_SQRN_AVX_LOAD(b_hi,b_lo)                      *
	 **********************************************************
	  (b_hi,b_lo)
	 **********************************************************/
	#define LIS_QUAD_SQRN_AVX_LOAD(b_hi,b_lo) \
					bh = LIS_AVX_FUNC(loadu_pd)((double*)&(b_hi)); \
					bl = LIS_AVX_FUNC(loadu_pd)((double*)&(b_lo))

	/**********************************************************
	 * LIS_QUAD_ADDN_AVX_CORE                                 *
	 **********************************************************
	  (b_hi,b_lo) + (c_hi,c_lo)
	 **********************************************************/
	#ifndef USE_FAST_QUAD_ADD
	#define LIS_QUAD_ADDN_AVX_CORE \
					sh = LIS_AVX_FUNC(add_pd)(bh,ch); \
					th = LIS_AVX_FUNC(sub_pd)(sh,bh); \
					t0 = LIS_AVX_FUNC(sub_pd)(sh,th); \
					ch = LIS_AVX_FUNC(sub_pd)(ch,th); \
					bh = LIS_AVX_FUNC(sub_pd)(bh,t0); \
					bh = LIS_AVX_FUNC(add_pd)(bh,ch); \
					sl = LIS_AVX_FUNC(add_pd)(bl,p2); \
					th = LIS_AVX_FUNC(sub_pd)(sl,bl); \
					t0 = LIS_AVX_FUNC(sub_pd)(sl,th); \
					p2 = LIS_AVX_FUNC(sub_pd)(p2,th); \
					bl = LIS_AVX_FUNC(sub_pd)(bl,t0); \
					bl = LIS_AVX_FUNC(add_pd)(bl,p2); \
					bh = LIS_AVX_FUNC(add_pd)(bh,sl); \
					th = sh; \
					th = LIS_AVX_FUNC(add_pd)(th,bh); \
					sh = LIS_AVX_FUNC(sub_pd)(th,sh); \
					bh = LIS_AVX_FUNC(sub_pd)(bh,sh); \
					bh = LIS_AVX_FUNC(add_pd)(bh,bl); \
					sh = LIS_AVX_FUNC(add_pd)(th,bh)
	#else
	#define LIS_QUAD_ADDN_AVX_CORE \
					th = LIS_AVX_FUNC(add_pd)(bh,ch); \
					wh = LIS_AVX_FUNC(sub_pd)(th,bh); \
					t0 = LIS_AVX_FUNC(sub_pd)(th,wh); \
					ch = LIS_AVX_FUNC(sub_pd)(ch,wh); \
					bh = LIS_AVX_FUNC(sub_pd)(bh,t0); \
					bh = LIS_AVX_FUNC(add_pd)(bh,ch); \
					bh = LIS_AVX_FUNC(add_pd)(bh,bl); \
					bh = LIS_AVX_FUNC(add_pd)(bh,p2); \
					sh = LIS_AVX_FUNC(add_pd)(th,bh)
	#endif

	/**********************************************************
	 * LIS_QUAD_MULN_AVX_CORE  ..mp3                             *
	 **********************************************************
	  (b_hi,b_lo) x (c_hi,c_lo)
	 **********************************************************/
	#ifdef USE_AVX2 
	#define LIS_QUAD_MULN_AVX_CORE \
		p1 = LIS_AVX_FUNC(mul_pd)(mi,bh); /*FMA*/ \
		p1 = LIS_AVX_FUNC(mul_pd)(p1,ch); /*FMA*/\
		p2 = _mm256_fmadd_pd(bh,ch,p1); /*FMA*/\
		p1 = LIS_AVX_FUNC(mul_pd)(mi,p1); /*FMA*/\
		p2 = _mm256_fmadd_pd(bh,cl,p2); \
		p2 = _mm256_fmadd_pd(bl,ch,p2); \
		ch = LIS_AVX_FUNC(add_pd)(p1,p2)
	#else
	#define LIS_QUAD_MULN_AVX_CORE \
		p1  = LIS_AVX_FUNC(mul_pd)(bh,ch); \
		sh  = LIS_AVX_FUNC(mul_pd)(sp,bh); \
		sl  = LIS_AVX_FUNC(mul_pd)(sp,ch); \
		th  = LIS_AVX_FUNC(sub_pd)(sh,bh); \
		tl  = LIS_AVX_FUNC(sub_pd)(sl,ch); \
		sh  = LIS_AVX_FUNC(sub_pd)(sh,th); \
		sl  = LIS_AVX_FUNC(sub_pd)(sl,tl); \
		t1  = LIS_AVX_FUNC(mul_pd)(bh,cl); \
		wh  = LIS_AVX_FUNC(sub_pd)(bh,sh); \
		t2  = LIS_AVX_FUNC(mul_pd)(ch,bl); \
		wl  = LIS_AVX_FUNC(sub_pd)(ch,sl); \
		t0  = LIS_AVX_FUNC(mul_pd)(wh,wl); \
		p2  = LIS_AVX_FUNC(mul_pd)(sh,sl); \
		sh  = LIS_AVX_FUNC(mul_pd)(sh,wl); \
		sl  = LIS_AVX_FUNC(mul_pd)(sl,wh); \
		p2  = LIS_AVX_FUNC(sub_pd)(p2,p1); \
		p2  = LIS_AVX_FUNC(add_pd)(p2,sh); \
		p2  = LIS_AVX_FUNC(add_pd)(p2,sl); \
		p2  = LIS_AVX_FUNC(add_pd)(p2,t0); \
		p2  = LIS_AVX_FUNC(add_pd)(p2,t1); \
		p2  = LIS_AVX_FUNC(add_pd)(p2,t2); \
		ch  = LIS_AVX_FUNC(add_pd)(p1,p2)
	#endif

	/**********************************************************
	 * LIS_QUAD_SQRN_AVX_CORE                                 *
	 **********************************************************
	  (b_hi,b_lo) x (b_hi,b_lo)
	 **********************************************************/
	#ifdef USE_AVX2 
	#define LIS_QUAD_SQRN_AVX_CORE \
		p1 = LIS_AVX_FUNC(mul_pd)(mi,bh); /*FMA*/ \
		p1 = LIS_AVX_FUNC(mul_pd)(p1,ch); /*FMA*/\
		p2 = _mm256_fmadd_pd(bh,ch,p1); /*FMA*/\
		p1 = LIS_AVX_FUNC(mul_pd)(mi,p1); /*FMA*/\
		p2 = _mm256_fmadd_pd(bh,cl,p2); \
		p2 = _mm256_fmadd_pd(bl,ch,p2); \
		ch = LIS_AVX_FUNC(add_pd)(p1,p2)
	#else
	#define LIS_QUAD_SQRN_AVX_CORE \
					p1  = LIS_AVX_FUNC(mul_pd)(bh,bh); \
					ch  = LIS_AVX_FUNC(mul_pd)(sp,bh); \
					p2  = LIS_AVX_FUNC(sub_pd)(ch,bh); \
					ch  = LIS_AVX_FUNC(sub_pd)(ch,p2); \
					cl  = LIS_AVX_FUNC(sub_pd)(bh,ch); \
					p2  = LIS_AVX_FUNC(mul_pd)(ch,ch); \
					ch  = LIS_AVX_FUNC(add_pd)(ch,ch); \
					ch  = LIS_AVX_FUNC(mul_pd)(ch,cl); \
					cl  = LIS_AVX_FUNC(mul_pd)(cl,cl); \
					p2  = LIS_AVX_FUNC(sub_pd)(p2,p1); \
					p2  = LIS_AVX_FUNC(add_pd)(p2,ch); \
					p2  = LIS_AVX_FUNC(add_pd)(p2,cl); \
					bh  = LIS_AVX_FUNC(add_pd)(bh,bh); \
					bh  = LIS_AVX_FUNC(mul_pd)(bh,bl); \
					bl  = LIS_AVX_FUNC(mul_pd)(bl,bl); \
					p2  = LIS_AVX_FUNC(add_pd)(p2,bh); \
					p2  = LIS_AVX_FUNC(add_pd)(p2,bl); \
					ch  = LIS_AVX_FUNC(add_pd)(p1,p2)
	#endif

	/**********************************************************
	 * LIS_QUAD_ADDN_AVX_STORE(a_hi,a_lo)                     *
	 **********************************************************
	  (a_hi,a_lo) <-
	 **********************************************************/
	#define LIS_QUAD_ADDN_AVX_STORE(a_hi,a_lo) \
					LIS_AVX_FUNC(storeu_pd)((double*)&(a_hi),sh); \
					sh = LIS_AVX_FUNC(sub_pd)(sh,th); \
					bh = LIS_AVX_FUNC(sub_pd)(bh,sh); \
					LIS_AVX_FUNC(storeu_pd)((double*)&(a_lo),bh)

	/**********************************************************
	 * LIS_QUAD_MULN_AVX_STORE(a_hi,a_lo)                     *
	 **********************************************************
	  (a_hi,a_lo) <-
	 **********************************************************/
	#define LIS_QUAD_MULN_AVX_STORE(a_hi,a_lo) \
					LIS_AVX_FUNC(storeu_pd)(&(a_hi),ch); \
					ch  = LIS_AVX_FUNC(sub_pd)(ch,p1); \
					p2  = LIS_AVX_FUNC(sub_pd)(p2,ch); \
					LIS_AVX_FUNC(storeu_pd)(&(a_lo),p2)
	#define LIS_QUAD_MULN_AVX_STOREU(a_hi,a_lo) \
					LIS_AVX_FUNC(storeu_pd)(&(a_hi),ch); \
					ch  = LIS_AVX_FUNC(sub_pd)(ch,p1); \
					p2  = LIS_AVX_FUNC(sub_pd)(p2,ch); \
					LIS_AVX_FUNC(storeu_pd)(&(a_lo),p2)



	/**********************************************************
	 * LIS_QUAD_ADDN_AVX(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)       *
	 **********************************************************
	  (a_hi,a_lo) <- (b_hi,b_lo) + (c_hi,c_lo)
	 **********************************************************/
	#define LIS_QUAD_ADDN_AVX(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
					LIS_QUAD_ADDN_AVX_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
					LIS_QUAD_ADDN_AVX_CORE; \
					LIS_QUAD_ADDN_AVX_STORE((a_hi),(a_lo))

	/**********************************************************
	 * LIS_QUAD_MULN_AVX(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)       *
	 **********************************************************
	  (a_hi,a_lo) <- (b_hi,b_lo) x (c_hi,c_lo)
	 **********************************************************/
	#define LIS_QUAD_MULN_AVX(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
					LIS_QUAD_MULN_AVX_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
					LIS_QUAD_MULN_AVX_CORE; \
					LIS_QUAD_MULN_AVX_STORE((a_hi),(a_lo))

	/**********************************************************
	 * LIS_QUAD_SQRN_AVX(a_hi,a_lo,b_hi,b_lo)                 *
	 **********************************************************
	  (a_hi,a_lo) <- (b_hi,b_lo) x (b_hi,b_lo)
	 **********************************************************/
	#define LIS_QUAD_SQRN_AVX(a_hi,a_lo,b_hi,b_lo) \
					LIS_QUAD_SQRN_AVX_LOAD((b_hi),(b_lo)); \
					LIS_QUAD_SQRN_AVX_CORE; \
					LIS_QUAD_MULN_AVX_STORE((a_hi),(a_lo))

	/***************************************************************
	 * LIS_QUAD_FMAN_AVX(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo)  *
	 ***************************************************************
	  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (c_hi,c_lo)
	 ***************************************************************/
	#define LIS_QUAD_FMAN_AVX(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) \
					LIS_QUAD_MULN_AVX_LOAD((b_hi),(b_lo),(c_hi),(c_lo)); \
					LIS_QUAD_MULN_AVX_CORE; \
					LIS_QUAD_FMAN_AVX_LOAD((a_hi),(a_lo)); \
					LIS_QUAD_ADDN_AVX_CORE; \
					LIS_QUAD_ADDN_AVX_STORE((d_hi),(d_lo))

	/***************************************************************
	 * LIS_QUAD_FSAN_AVX(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo)            *
	 ***************************************************************
	  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * (b_hi,b_lo)
	 ***************************************************************/
	#define LIS_QUAD_FSAN_AVX(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo) \
					LIS_QUAD_SQRN_AVX_LOAD((b_hi),(b_lo)); \
					LIS_QUAD_SQRN_AVX_CORE; \
					LIS_QUAD_FMAN_AVX_LOAD((a_hi),(a_lo)); \
					LIS_QUAD_ADDN_AVX_CORE; \
					LIS_QUAD_ADDN_AVX_STORE((d_hi),(d_lo))



	/**********************************************************
	 *                                                        *
	 *               AVX(Nbit; Double x Quad)                 *
	 *                                                        *
	 **********************************************************/

	/**********************************************************
	 * LIS_QUAD_MULDN_AVX_LOAD(b_hi,b_lo,c)                   *
	 **********************************************************
	  (b_hi,b_lo)  (c)
	 **********************************************************/
	#define LIS_QUAD_MULDN_AVX_LOAD(b_hi,b_lo,c) \
					bh  = LIS_AVX_FUNC(loadu_pd)((double*)&(b_hi)); \
					bl  = LIS_AVX_FUNC(loadu_pd)((double*)&(b_lo)); \
					ch  = LIS_AVX_FUNC(broadcast_sd)((double*)&(c))



	/**********************************************************
	 * LIS_QUAD_MULDN_AVX_CORE     .mp3                       *
	 **********************************************************
	  (b_hi,b_lo) x c (matvec)
	 **********************************************************/
	#ifdef USE_AVX2
	#define LIS_QUAD_MULDN_AVX_CORE \
			p1 = LIS_AVX_FUNC(mul_pd)(mi,bh); \
			p1 = LIS_AVX_FUNC(mul_pd)(p1,ch); \
			p2 = _mm256_fmadd_pd(bh,ch,p1); \
			p1 = LIS_AVX_FUNC(mul_pd)(mi,p1); \
			p2 = _mm256_fmadd_pd(bl,ch,p2); \
			ch  = LIS_AVX_FUNC(add_pd)(p1,p2)
	#else
	#define LIS_QUAD_MULDN_AVX_CORE \
			p1  = LIS_AVX_FUNC(mul_pd)(bh,ch); \
			bl  = LIS_AVX_FUNC(mul_pd)(bl,ch); \
			sh  = LIS_AVX_FUNC(mul_pd)(sp,bh); \
			th  = LIS_AVX_FUNC(sub_pd)(sh,bh); \
			sh  = LIS_AVX_FUNC(sub_pd)(sh,th); \
			bh  = LIS_AVX_FUNC(sub_pd)(bh,sh); \
			sl  = LIS_AVX_FUNC(mul_pd)(sp,ch); \
			tl  = LIS_AVX_FUNC(sub_pd)(sl,ch); \
			sl  = LIS_AVX_FUNC(sub_pd)(sl,tl); \
			ch  = LIS_AVX_FUNC(sub_pd)(ch,sl); \
			t2  = LIS_AVX_FUNC(mul_pd)(bh,ch); \
			p2  = LIS_AVX_FUNC(mul_pd)(sh,sl); \
			t0  = LIS_AVX_FUNC(mul_pd)(sh,ch); \
			t1  = LIS_AVX_FUNC(mul_pd)(sl,bh); \
			p2  = LIS_AVX_FUNC(sub_pd)(p2,p1); \
			p2  = LIS_AVX_FUNC(add_pd)(p2,t0); \
			p2  = LIS_AVX_FUNC(add_pd)(p2,t1); \
			p2  = LIS_AVX_FUNC(add_pd)(p2,t2); \
			p2  = LIS_AVX_FUNC(add_pd)(p2,bl); \
			ch  = LIS_AVX_FUNC(add_pd)(p1,p2)
	#endif

	/**********************************************************
	 * LIS_QUAD_MULDN_AVX(a_hi,a_lo,b_hi,b_lo,c)              *
	 **********************************************************
	  (a_hi,a_lo) <- (b_hi,b_lo) x c
	 **********************************************************/
	#define LIS_QUAD_MULDN_AVX(a_hi,a_lo,b_hi,b_lo,c) \
					LIS_QUAD_MULDN_AVX_LOAD((b_hi),(b_lo),(c)); \
					LIS_QUAD_MULDN_AVX_CORE; \
					LIS_QUAD_MULN_AVX_STORE((a_hi),(a_lo))

	/***************************************************************************
	 * LIS_QUAD_FMADN_AVX(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c)                     *
	 * LIS_QUAD_FMADN_AVX_LDSD(d_hi,d_lo,a_hi,a_lo,b0_hi,b0_lo,b1_hi,b1_lo,c)  *
	 ***************************************************************************
	  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * c
	 ***************************************************************/
	#define LIS_QUAD_FMADN_AVX(d_hi,d_lo,a_hi,a_lo,b_hi,b_lo,c) \
					LIS_QUAD_MULDN_AVX_LOAD((b_hi),(b_lo),(c)); \
					LIS_QUAD_MULDN_AVX_CORE; \
					LIS_QUAD_FMAN_AVX_LOAD((a_hi),(a_lo)); \
					LIS_QUAD_ADDN_AVX_CORE; \
					LIS_QUAD_ADDN_AVX_STORE((d_hi),(d_lo))



	//per vector size operations
	//have to implement additional code for new vector size
	#if LIS_AVX_SIZE == 4
	/**********************************************************
	 *                                                        *
	 *               AVX(256bit; Quad x Quad)                 *
	 *                                                        *
	 **********************************************************/

	/**********************************************************
	 * LIS_QUAD_FMA4_AVX_LOAD(a_hi,a_lo)                      *
	 **********************************************************
	  (a_hi,a_lo)
	 **********************************************************/
	#define LIS_QUAD_FMA4_AVX_LOAD_SD2(a0_hi,a0_lo) \
		t1  = _mm256_sub_pd(ch,p1); \
		p2  = _mm256_sub_pd(p2,t1); \
		bh  = _mm256_loadu_pd(&(a0_hi)); \
		bl  = _mm256_loadu_pd(&(a0_lo))
	
	////////////////////////////////////////////////////////////
	#define LIS_QUAD_FMA4_AVX_LOAD_SD(a0_hi,a0_lo,a1_hi,a1_lo,a2_hi,a2_lo,a3_hi,a3_lo) \
		t1  = _mm256_sub_pd(ch,p1); \
		p2  = _mm256_sub_pd(p2,t1); \
		bh  = _mm256_set_pd((a3_hi),(a2_hi),(a1_hi),(a0_hi)); \
		bl  = _mm256_set_pd((a3_lo),(a2_lo),(a1_lo),(a0_lo))

	/**********************************************************
	 * LIS_QUAD_ADD4_AVX_STORE(a_hi,a_lo)                     *
	 **********************************************************
	  (a_hi,a_lo) <-
	 **********************************************************/
	#define LIS_QUAD_ADD4_AVX_STORE_SD2(a0_hi,a0_lo) \
		_mm256_stream_pd(&(a0_hi), sh); \
		sh = _mm256_sub_pd(sh,th); \
		bh = _mm256_sub_pd(bh,sh); \
		_mm256_stream_pd(&(a0_lo), bh)
	
	////////////////////////////////////////////////////////////////
	#define LIS_QUAD_ADD4_AVX_STORE_SD(a0_hi,a0_lo,a1_hi,a1_lo,a2_hi,a2_lo,a3_hi,a3_lo) \
		*(__m128d*)&t3 = _mm256_extractf128_pd(sh, 1); \
		_mm_storel_pd(&(a0_hi), *(__m128d*)&sh); \
		_mm_storeh_pd(&(a1_hi), *(__m128d*)&sh); \
		_mm_storel_pd(&(a2_hi), *(__m128d*)&t3); \
		_mm_storeh_pd(&(a3_hi), *(__m128d*)&t3); \
		sh = _mm256_sub_pd(sh,th); \
		bh = _mm256_sub_pd(bh,sh); \
		*(__m128d*)&t3 = _mm256_extractf128_pd(bh, 1); \
		_mm_storel_pd(&(a0_lo), *(__m128d*)&bh); \
		_mm_storeh_pd(&(a1_lo), *(__m128d*)&bh); \
		_mm_storel_pd(&(a2_lo), *(__m128d*)&t3); \
		_mm_storeh_pd(&(a3_lo), *(__m128d*)&t3)



	/**********************************************************
	 * LIS_QUAD_HADDALL_AVX(a_hi,a_lo,b_hi,b_lo)              *
	 **********************************************************
	  (a_hi,a_lo) <- sum(i = 0 to 3, (b_hi, b_lo)[i])
	 **********************************************************/
	#define LIS_QUAD_HADDALL_AVX(a_hi, a_lo, b_hi, b_lo) \
					bh = LIS_AVX_FUNC(loadu_pd)((double*)&(b_hi)); \
					bl = LIS_AVX_FUNC(loadu_pd)((double*)&(b_lo)); \
					LIS_VEC_CAST(ch) = _mm256_extractf128_pd(bh, 1); \
					LIS_VEC_CAST(p2) = _mm256_extractf128_pd(bl, 1); \
					LIS_QUAD_ADD2_SSE2_CORE; \
					LIS_VEC_CAST(eh) = _mm_unpackhi_pd(LIS_VEC_CAST(sh), LIS_VEC_CAST(sh)); \
					LIS_VEC_CAST(t0) = _mm_sub_pd(LIS_VEC_CAST(sh), LIS_VEC_CAST(th)); \
					LIS_VEC_CAST(wh) = _mm_sub_pd(LIS_VEC_CAST(bh), LIS_VEC_CAST(t0)); \
					LIS_VEC_CAST(cl) = _mm_unpackhi_pd(LIS_VEC_CAST(wh), LIS_VEC_CAST(wh)); \
					LIS_QUAD_ADD_SSE2_CORE((a_hi), (a_lo));

	/**********************************************************
	 *                                                        *
	 *              AVX(256bit; Double x Quad)                *
	 *                                                        *
	 **********************************************************/
	/**********************************************************
	 * LIS_QUAD_MULD4_AVX_LOAD(b_hi,b_lo,c)                   *
	 **********************************************************
	  (b_hi,b_lo)  (c)
	 **********************************************************/
	#define LIS_QUAD_MULD4_AVX_LOAD_SD2(b0_hi,b0_lo,c) \
		bh  = _mm256_broadcast_sd((double*)&(b0_hi)); \
		bl  = _mm256_broadcast_sd((double*)&(b0_lo)); \
		ch  = _mm256_loadu_pd(&(c))
	
	////////////////////////////////////////////////////////////////////
	#define LIS_QUAD_MULD4_AVX_LOAD_SD(b0_hi,b0_lo,b1_hi,b1_lo,b2_hi,b2_lo,b3_hi,b3_lo,c) \
		bh  = _mm256_set_pd((b3_hi),(b2_hi),(b1_hi),(b0_hi)); \
		bl  = _mm256_set_pd((b3_lo),(b2_lo),(b1_lo),(b0_lo)); \
		ch  = _mm256_loadu_pd(&(c))

	/***************************************************************************
	 * LIS_QUAD_FMAD4_AVX_LDSD(d_hi,d_lo,a_hi,a_lo,b0_hi,b0_lo,b1_hi,b1_lo,c)  *
	 ***************************************************************************
	  (d_hi,d_lo) <- (a_hi,a_lo) + (b_hi,b_lo) * c
	 ***************************************************************/
	#define LIS_QUAD_FMAD4_AVX_LDSD(d_hi,d_lo,a_hi,a_lo,b0_hi,b0_lo,b1_hi,b1_lo,b2_hi,b2_lo,b3_hi,b3_lo,c) \
		LIS_QUAD_MULD4_AVX_LOAD_SD((b0_hi),(b0_lo),(b1_hi),(b1_lo),(b2_hi),(b2_lo),(b3_hi),(b3_lo),(c)); \
		LIS_QUAD_MULDN_AVX_CORE; \
		LIS_QUAD_FMAN_AVX_LOAD((a_hi),(a_lo)); \
		LIS_QUAD_ADDN_AVX_CORE; \
		LIS_QUAD_ADDN_AVX_STORE((d_hi),(d_lo))

	#ifdef USE_AVX2 
	#define LIS_DOUBLE_FMA4_AVX_LDSD(d_hi,a_hi,b0_hi,b1_hi,b2_hi,b3_hi,c) \
	bl = _mm256_load_pd(&(c)); \
	ch = _mm256_set_pd((b0_hi),(b1_hi),(b2_hi),(b3_hi)); \
	d_hi = _mm256_fmadd_pd(bl,ch,d_hi);
	#else
	#define LIS_DOUBLE_FMA4_AVX_LDSD(d_hi,a_hi,b0_hi,b1_hi,b2_hi,b3_hi,c) \
	bl = _mm256_load_pd(&(c)); \
	ch = _mm256_set_pd((b0_hi),(b1_hi),(b2_hi),(b3_hi)); \
	bl = _mm256_mul_pd(bl,ch); \
	d_hi = _mm256_add_pd(d_hi,bl); 
	#endif

	/****************************************************************/
	#define LIS_QUAD_FMAD4_AVX_STSD(d0_hi,d0_lo,d1_hi,d1_lo,d2_hi,d2_lo,d3_hi,d3_lo,a0_hi,a0_lo,a1_hi,a1_lo,a2_hi,a2_lo,a3_hi,a3_lo,b0_hi,b0_lo,b1_hi,b1_lo,b2_hi,b2_lo,b3_hi,b3_lo,c) \
		LIS_QUAD_MULD4_AVX_LOAD_SD2((b0_hi),(b0_lo),(c)); \
		LIS_QUAD_MULDN_AVX_CORE; \
		LIS_QUAD_FMA4_AVX_LOAD_SD((a0_hi),(a0_lo),(a1_hi),(a1_lo),(a2_hi),(a2_lo),(a3_hi),(a3_lo)); \
		LIS_QUAD_ADDN_AVX_CORE; \
		LIS_QUAD_ADD4_AVX_STORE_SD((d0_hi),(d0_lo),(d1_hi),(d1_lo),(d2_hi),(d2_lo),(d3_hi),(d3_lo))
	
	////////////////////////////////////////////////////////////////////
	#define LIS_QUAD_FMAD4_AVX_STSD2(d0_hi,d0_lo,a0_hi,a0_lo,b0_hi,b0_lo,c) \
		LIS_QUAD_MULD4_AVX_LOAD_SD2((b0_hi),(b0_lo),(c)); \
		LIS_QUAD_MULDN_AVX_CORE; \
		LIS_QUAD_FMA4_AVX_LOAD_SD2((a0_hi),(a0_lo)); \
		LIS_QUAD_ADDN_AVX_CORE; \
		LIS_QUAD_ADD4_AVX_STORE_SD2((d0_hi),(d0_lo))
	
	/////////////////////////////////////////////////////////////////////
	#define LIS_QUAD_FMAD4_AVX_STSD3(d0_hi,d0_lo,d1_hi,d1_lo,d2_hi,d2_lo,d3_hi,d3_lo,a0_hi,a0_lo,a1_hi,a1_lo,a2_hi,a2_lo,a3_hi,a3_lo,b0_hi,b0_lo,c) \
		LIS_QUAD_MULD4_AVX_LOAD_SD2((b0_hi),(b0_lo),(c)); \
		LIS_QUAD_MULDN_AVX_CORE; \
		LIS_QUAD_FMA4_AVX_LOAD_SD((a0_hi),(a0_lo),(a1_hi),(a1_lo),(a2_hi),(a2_lo),(a3_hi),(a3_lo)); \
		LIS_QUAD_ADDN_AVX_CORE; \
		LIS_QUAD_ADD4_AVX_STORE_SD((d0_hi),(d0_lo),(d1_hi),(d1_lo),(d2_hi),(d2_lo),(d3_hi),(d3_lo))
	/**********************************************************
	 *                                                        *
	 *                 AVX(256bit; helper)                    *
	 *                                                        *
	 **********************************************************/

	/* LIS_TRANSPOSE_YMM(v1, v2, v3, v4)
	 * before                 after
	 * v1 |1,1|1,2|1,3|1,4|   v1 |1,1|2,1|3,1|4,1|
	 * v2 |2,1|2,2|2,3|2,4|   v2 |1,2|2,2|3,2|4,2|
	 * v3 |3,1|3,2|3,3|3,4|   v3 |1,3|2,3|3,3|4,3|
	 * v4 |4,1|4,2|4,3|4,4|   v4 |1,4|2,4|3,4|4,4|
	 */
	#define LIS_TRANSPOSE_AVX(v1, v2, v3, v4) \
	{ \
		__m256d u1, u3; \
		__m128d t0; \
		t0 = _mm256_extractf128_pd(v1, 1); \
		v1 = _mm256_insertf128_pd(v1, *(__m128d*)&v3, 1); \
		v3 = _mm256_insertf128_pd(v3, t0, 0); \
		t0 = _mm256_extractf128_pd(v2, 1); \
		v2 = _mm256_insertf128_pd(v2, *(__m128d*)&v4, 1); \
		v4 = _mm256_insertf128_pd(v4, t0, 0); \
		u1 = _mm256_shuffle_pd(v1, v2, 0x0); \
		v2 = _mm256_shuffle_pd(v1, v2, 0xf); \
		u3 = _mm256_shuffle_pd(v3, v4, 0x0); \
		v4 = _mm256_shuffle_pd(v3, v4, 0xf); \
		v1 = u1; \
		v3 = u3; \
	}

	#define LIS_MASK_GEN_AVX(MASK,SIZE) \
	{ \
		const __m256d mask_base_ = {0x1.0000000000000p0,0x1.0000000000001p0,0x1.0000000000002p0,0x1.0000000000003p0}; \
		uint64_t r_ = 0x3ff0000000000000; \
		__m256d rv_; \
		r_ |= SIZE; \
		rv_ = _mm256_broadcast_sd((double*)&r_); \
		MASK = _mm256_cmp_pd(mask_base_, rv_, _CMP_LT_OQ); \
	}

	#define LIS_MASK_LOAD_AVX(DST,SRC_ADR,MASK) \
			DST = _mm256_maskload_pd(SRC_ADR, *(__m256i*)&MASK);

	#define LIS_MASK_STORE_AVX(DST_ADR,SRC,MASK) \
			_mm256_maskstore_pd(DST_ADR,*(__m256i*)&MASK,SRC);

	#else
		#error no implementation for this vector size
	#endif /* LIS_AVX_SIZE */

#endif /* USE_AVX */


extern double *lis_quad_scalar_tmp;

#define LIS_QUAD_SCALAR_MALLOC(s,pos,num) \
				(s).hi = &lis_quad_scalar_tmp[2*(pos)]; \
				(s).lo = &lis_quad_scalar_tmp[2*(pos)+(num)]

#ifdef __cplusplus
extern "C"
{
#endif

extern void lis_quad_x87_fpu_init(unsigned int *cw_old);
extern void lis_quad_x87_fpu_finalize(unsigned int cw);

extern void lis_quad_minus(LIS_QUAD *a);
extern void lis_quad_zero(LIS_QUAD *a);
extern void lis_quad_one(LIS_QUAD *a);
extern void lis_quad_min(LIS_QUAD *a, LIS_QUAD *b, LIS_QUAD *c);
extern void lis_quad_max(LIS_QUAD *a, LIS_QUAD *b, LIS_QUAD *c);

extern void lis_quad_add(LIS_QUAD *a, const LIS_QUAD *b, const LIS_QUAD *c);
extern void lis_quad_sub(LIS_QUAD *a, const LIS_QUAD *b, const LIS_QUAD *c);
extern void lis_quad_mul(LIS_QUAD *a, const LIS_QUAD *b, const LIS_QUAD *c);
extern void lis_quad_mul_dd_d(LIS_QUAD *a, const LIS_QUAD *b, const double c);
extern void lis_quad_sqr(LIS_QUAD *a, const LIS_QUAD *b);
extern void lis_quad_div(LIS_QUAD *a, const LIS_QUAD *b, const LIS_QUAD *c);
extern LIS_INT  lis_quad_sqrt(LIS_QUAD *a, const LIS_QUAD *b);

#ifdef __cplusplus
}
#endif

#endif /* __LIS_PRECISION_H__ */
