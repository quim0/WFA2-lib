#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "ksw2.h"

#ifdef __SSE2__
#include <emmintrin.h>

#ifdef KSW_SSE2_ONLY
#undef __SSE4_1__
#endif

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#if defined(__AVX2__) || defined (__AVX512BW__)
#include <immintrin.h>
#endif

#ifdef KSW_CPU_DISPATCH
#ifdef __SSE4_1__
void ksw_extd2_sse41(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat,
				   int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t *ez)
#else
void ksw_extd2_sse2(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat,
				   int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t *ez)
#endif
#else
void ksw_extd2_sse(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat,
				   int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t *ez)
#endif // ~KSW_CPU_DISPATCH
{
#define __dp_code_block1 \
	z = _mm_load_si128(&s[t]); \
	xt1 = _mm_load_si128(&x[t]);                     /* xt1 <- x[r-1][t..t+15] */ \
	tmp = _mm_srli_si128(xt1, 15);                   /* tmp <- x[r-1][t+15] */ \
	xt1 = _mm_or_si128(_mm_slli_si128(xt1, 1), x1_); /* xt1 <- x[r-1][t-1..t+14] */ \
	x1_ = tmp; \
	vt1 = _mm_load_si128(&v[t]);                     /* vt1 <- v[r-1][t..t+15] */ \
	tmp = _mm_srli_si128(vt1, 15);                   /* tmp <- v[r-1][t+15] */ \
	vt1 = _mm_or_si128(_mm_slli_si128(vt1, 1), v1_); /* vt1 <- v[r-1][t-1..t+14] */ \
	v1_ = tmp; \
	a = _mm_add_epi8(xt1, vt1);                      /* a <- x[r-1][t-1..t+14] + v[r-1][t-1..t+14] */ \
	ut = _mm_load_si128(&u[t]);                      /* ut <- u[t..t+15] */ \
	b = _mm_add_epi8(_mm_load_si128(&y[t]), ut);     /* b <- y[r-1][t..t+15] + u[r-1][t..t+15] */ \
	x2t1= _mm_load_si128(&x2[t]); \
	tmp = _mm_srli_si128(x2t1, 15); \
	x2t1= _mm_or_si128(_mm_slli_si128(x2t1, 1), x21_); \
	x21_= tmp; \
	a2= _mm_add_epi8(x2t1, vt1); \
	b2= _mm_add_epi8(_mm_load_si128(&y2[t]), ut);

#define __dp_code_block2 \
	_mm_store_si128(&u[t], _mm_sub_epi8(z, vt1));    /* u[r][t..t+15] <- z - v[r-1][t-1..t+14] */ \
	_mm_store_si128(&v[t], _mm_sub_epi8(z, ut));     /* v[r][t..t+15] <- z - u[r-1][t..t+15] */ \
	tmp = _mm_sub_epi8(z, q_); \
	a = _mm_sub_epi8(a, tmp); \
	b = _mm_sub_epi8(b, tmp); \
	tmp = _mm_sub_epi8(z, q2_); \
	a2= _mm_sub_epi8(a2, tmp); \
	b2= _mm_sub_epi8(b2, tmp);

	int r, t, qe = q + e, n_col_, *off = 0, *off_end = 0, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc, long_thres, long_diff;
	int with_cigar = !(flag&KSW_EZ_SCORE_ONLY), approx_max = !!(flag&KSW_EZ_APPROX_MAX);
	int32_t *H = 0, H0 = 0, last_H0_t = 0;
	uint8_t *qr, *sf, *mem, *mem2 = 0;
	__m128i q_, q2_, qe_, qe2_, zero_, sc_mch_, sc_mis_, m1_, sc_N_;
	__m128i *u, *v, *x, *y, *x2, *y2, *s, *p = 0;

	ksw_reset_extz(ez);
	if (m <= 1 || qlen <= 0 || tlen <= 0) return;

	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	zero_   = _mm_set1_epi8(0);
	q_      = _mm_set1_epi8(q);
	q2_     = _mm_set1_epi8(q2);
	qe_     = _mm_set1_epi8(q + e);
	qe2_    = _mm_set1_epi8(q2 + e2);
	sc_mch_ = _mm_set1_epi8(mat[0]);
	sc_mis_ = _mm_set1_epi8(mat[1]);
	sc_N_   = mat[m*m-1] == 0? _mm_set1_epi8(-e2) : _mm_set1_epi8(mat[m*m-1]);
	m1_     = _mm_set1_epi8(m - 1); // wildcard

	if (w < 0) w = tlen > qlen? tlen : qlen;
	wl = wr = w;
	tlen_ = (tlen + 15) / 16;
	n_col_ = qlen < tlen? qlen : tlen;
	n_col_ = ((n_col_ < w + 1? n_col_ : w + 1) + 15) / 16 + 1;
	qlen_ = (qlen + 15) / 16;
	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
		max_sc = max_sc > mat[t]? max_sc : mat[t];
		min_sc = min_sc < mat[t]? min_sc : mat[t];
	}
	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	long_thres = e != e2? (q2 - q) / (e - e2) - 1 : 0;
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	mem = (uint8_t*)kcalloc(km, tlen_ * 8 + qlen_ + 1, 16);
	u = (__m128i*)(((size_t)mem + 15) >> 4 << 4); // 16-byte aligned
	v = u + tlen_, x = v + tlen_, y = x + tlen_, x2 = y + tlen_, y2 = x2 + tlen_;
	s = y2 + tlen_, sf = (uint8_t*)(s + tlen_), qr = sf + tlen_ * 16;
	memset(u,  -q  - e,  tlen_ * 16);
	memset(v,  -q  - e,  tlen_ * 16);
	memset(x,  -q  - e,  tlen_ * 16);
	memset(y,  -q  - e,  tlen_ * 16);
	memset(x2, -q2 - e2, tlen_ * 16);
	memset(y2, -q2 - e2, tlen_ * 16);
	if (!approx_max) {
		H = (int32_t*)kmalloc(km, tlen_ * 16 * 4);
		for (t = 0; t < tlen_ * 16; ++t) H[t] = KSW_NEG_INF;
	}
	if (with_cigar) {
		mem2 = (uint8_t*)kmalloc(km, ((size_t)(qlen + tlen - 1) * n_col_ + 1) * 16);
		p = (__m128i*)(((size_t)mem2 + 15) >> 4 << 4);
		off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int) * 2);
		off_end = off + qlen + tlen - 1;
	}

	for (t = 0; t < qlen; ++t) qr[t] = query[qlen - 1 - t];
	memcpy(sf, target, tlen);

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		uint8_t *qrr = qr + (qlen - 1 - r);
		int8_t *u8 = (int8_t*)u, *v8 = (int8_t*)v, *x8 = (int8_t*)x, *x28 = (int8_t*)x2;
		__m128i x1_, x21_, v1_;
		// find the boundaries
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r-wr+1)>>1) st = (r-wr+1)>>1; // take the ceil
		if (en > (r+wl)>>1) en = (r+wl)>>1; // take the floor
		if (st > en) {
			ez->zdropped = 1;
			break;
		}
		st0 = st, en0 = en;
		st = st / 16 * 16, en = (en + 16) / 16 * 16 - 1;
		// set boundary conditions
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x8[st - 1], x21 = x28[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
			} else {
				x1 = -q - e, x21 = -q2 - e2;
				v1 = -q - e;
			}
		} else {
			x1 = -q - e, x21 = -q2 - e2;
			v1 = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		if (en >= r) {
			((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u8[r] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {
			for (t = st0; t <= en0; t += 16) {
				__m128i sq, st, tmp, mask;
				sq = _mm_loadu_si128((__m128i*)&sf[t]);
				st = _mm_loadu_si128((__m128i*)&qrr[t]);
				mask = _mm_or_si128(_mm_cmpeq_epi8(sq, m1_), _mm_cmpeq_epi8(st, m1_));
				tmp = _mm_cmpeq_epi8(sq, st);
#ifdef __SSE4_1__
				tmp = _mm_blendv_epi8(sc_mis_, sc_mch_, tmp);
				tmp = _mm_blendv_epi8(tmp,     sc_N_,   mask);
#else
				tmp = _mm_or_si128(_mm_andnot_si128(tmp,  sc_mis_), _mm_and_si128(tmp,  sc_mch_));
				tmp = _mm_or_si128(_mm_andnot_si128(mask, tmp),     _mm_and_si128(mask, sc_N_));
#endif
				_mm_storeu_si128((__m128i*)((int8_t*)s + t), tmp);
			}
		} else {
			for (t = st0; t <= en0; ++t)
				((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
		}
		// core loop
		x1_  = _mm_cvtsi32_si128((uint8_t)x1);
		x21_ = _mm_cvtsi32_si128((uint8_t)x21);
		v1_  = _mm_cvtsi32_si128((uint8_t)v1);
		st_ = st / 16, en_ = en / 16;
		assert(en_ - st_ + 1 <= n_col_);
		if (!with_cigar) { // score only
			for (t = st_; t <= en_; ++t) {
				__m128i z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
#ifdef __SSE4_1__
				z = _mm_max_epi8(z, a);
				z = _mm_max_epi8(z, b);
				z = _mm_max_epi8(z, a2);
				z = _mm_max_epi8(z, b2);
				z = _mm_min_epi8(z, sc_mch_);
				__dp_code_block2; // save u[] and v[]; update a, b, a2 and b2
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_max_epi8(a,  zero_), qe_));
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_max_epi8(b,  zero_), qe_));
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_max_epi8(a2, zero_), qe2_));
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_max_epi8(b2, zero_), qe2_));
#else
				tmp = _mm_cmpgt_epi8(a,  z);
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
				tmp = _mm_cmpgt_epi8(b,  z);
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
				tmp = _mm_cmpgt_epi8(a2, z);
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2));
				tmp = _mm_cmpgt_epi8(b2, z);
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b2));
				tmp = _mm_cmplt_epi8(sc_mch_, z);
				z = _mm_or_si128(_mm_and_si128(tmp, sc_mch_), _mm_andnot_si128(tmp, z));
				__dp_code_block2;
				tmp = _mm_cmpgt_epi8(a, zero_);
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_and_si128(tmp, a),  qe_));
				tmp = _mm_cmpgt_epi8(b, zero_);
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_and_si128(tmp, b),  qe_));
				tmp = _mm_cmpgt_epi8(a2, zero_);
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_));
				tmp = _mm_cmpgt_epi8(b2, zero_);
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_));
#endif
			}
		} else if (!(flag&KSW_EZ_RIGHT)) { // gap left-alignment
			__m128i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m128i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
#ifdef __SSE4_1__
				d = _mm_and_si128(_mm_cmpgt_epi8(a, z), _mm_set1_epi8(1));       // d = a  > z? 1 : 0
				z = _mm_max_epi8(z, a);
				d = _mm_blendv_epi8(d, _mm_set1_epi8(2), _mm_cmpgt_epi8(b,  z)); // d = b  > z? 2 : d
				z = _mm_max_epi8(z, b);
				d = _mm_blendv_epi8(d, _mm_set1_epi8(3), _mm_cmpgt_epi8(a2, z)); // d = a2 > z? 3 : d
				z = _mm_max_epi8(z, a2);
				d = _mm_blendv_epi8(d, _mm_set1_epi8(4), _mm_cmpgt_epi8(b2, z)); // d = a2 > z? 3 : d
				z = _mm_max_epi8(z, b2);
				z = _mm_min_epi8(z, sc_mch_);
#else // we need to emulate SSE4.1 intrinsics _mm_max_epi8() and _mm_blendv_epi8()
				tmp = _mm_cmpgt_epi8(a,  z);
				d = _mm_and_si128(tmp, _mm_set1_epi8(1));
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a));
				tmp = _mm_cmpgt_epi8(b,  z);
				d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(2)));
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b));
				tmp = _mm_cmpgt_epi8(a2, z);
				d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(3)));
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, a2));
				tmp = _mm_cmpgt_epi8(b2, z);
				d = _mm_or_si128(_mm_andnot_si128(tmp, d), _mm_and_si128(tmp, _mm_set1_epi8(4)));
				z = _mm_or_si128(_mm_andnot_si128(tmp, z), _mm_and_si128(tmp, b2));
				tmp = _mm_cmplt_epi8(sc_mch_, z);
				z = _mm_or_si128(_mm_and_si128(tmp, sc_mch_), _mm_andnot_si128(tmp, z));
#endif
				__dp_code_block2;
				tmp = _mm_cmpgt_epi8(a, zero_);
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_and_si128(tmp, a),  qe_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x08))); // d = a > 0? 1<<3 : 0
				tmp = _mm_cmpgt_epi8(b, zero_);
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_and_si128(tmp, b),  qe_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x10))); // d = b > 0? 1<<4 : 0
				tmp = _mm_cmpgt_epi8(a2, zero_);
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_and_si128(tmp, a2), qe2_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x20))); // d = a > 0? 1<<5 : 0
				tmp = _mm_cmpgt_epi8(b2, zero_);
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_and_si128(tmp, b2), qe2_));
				d = _mm_or_si128(d, _mm_and_si128(tmp, _mm_set1_epi8(0x40))); // d = b > 0? 1<<6 : 0
				_mm_store_si128(&pr[t], d);
			}
		} else { // gap right-alignment
			__m128i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m128i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
#ifdef __SSE4_1__
				d = _mm_andnot_si128(_mm_cmpgt_epi8(z, a), _mm_set1_epi8(1));    // d = z > a?  0 : 1
				z = _mm_max_epi8(z, a);
				d = _mm_blendv_epi8(_mm_set1_epi8(2), d, _mm_cmpgt_epi8(z, b));  // d = z > b?  d : 2
				z = _mm_max_epi8(z, b);
				d = _mm_blendv_epi8(_mm_set1_epi8(3), d, _mm_cmpgt_epi8(z, a2)); // d = z > a2? d : 3
				z = _mm_max_epi8(z, a2);
				d = _mm_blendv_epi8(_mm_set1_epi8(4), d, _mm_cmpgt_epi8(z, b2)); // d = z > b2? d : 4
				z = _mm_max_epi8(z, b2);
				z = _mm_min_epi8(z, sc_mch_);
#else // we need to emulate SSE4.1 intrinsics _mm_max_epi8() and _mm_blendv_epi8()
				tmp = _mm_cmpgt_epi8(z, a);
				d = _mm_andnot_si128(tmp, _mm_set1_epi8(1));
				z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a));
				tmp = _mm_cmpgt_epi8(z, b);
				d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(2)));
				z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b));
				tmp = _mm_cmpgt_epi8(z, a2);
				d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(3)));
				z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, a2));
				tmp = _mm_cmpgt_epi8(z, b2);
				d = _mm_or_si128(_mm_and_si128(tmp, d), _mm_andnot_si128(tmp, _mm_set1_epi8(4)));
				z = _mm_or_si128(_mm_and_si128(tmp, z), _mm_andnot_si128(tmp, b2));
				tmp = _mm_cmplt_epi8(sc_mch_, z);
				z = _mm_or_si128(_mm_and_si128(tmp, sc_mch_), _mm_andnot_si128(tmp, z));
#endif
				__dp_code_block2;
				tmp = _mm_cmpgt_epi8(zero_, a);
				_mm_store_si128(&x[t],  _mm_sub_epi8(_mm_andnot_si128(tmp, a),  qe_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x08))); // d = a > 0? 1<<3 : 0
				tmp = _mm_cmpgt_epi8(zero_, b);
				_mm_store_si128(&y[t],  _mm_sub_epi8(_mm_andnot_si128(tmp, b),  qe_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x10))); // d = b > 0? 1<<4 : 0
				tmp = _mm_cmpgt_epi8(zero_, a2);
				_mm_store_si128(&x2[t], _mm_sub_epi8(_mm_andnot_si128(tmp, a2), qe2_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x20))); // d = a > 0? 1<<5 : 0
				tmp = _mm_cmpgt_epi8(zero_, b2);
				_mm_store_si128(&y2[t], _mm_sub_epi8(_mm_andnot_si128(tmp, b2), qe2_));
				d = _mm_or_si128(d, _mm_andnot_si128(tmp, _mm_set1_epi8(0x40))); // d = b > 0? 1<<6 : 0
				_mm_store_si128(&pr[t], d);
			}
		}
		if (!approx_max) { // find the exact max with a 32-bit score array
			int32_t max_H, max_t;
			// compute H[], max_H and max_t
			if (r > 0) {
				int32_t HH[4], tt[4], en1 = st0 + (en0 - st0) / 4 * 4, i;
				__m128i max_H_, max_t_;
				max_H = H[en0] = en0 > 0? H[en0-1] + u8[en0] : H[en0] + v8[en0]; // special casing the last element
				max_t = en0;
				max_H_ = _mm_set1_epi32(max_H);
				max_t_ = _mm_set1_epi32(max_t);
				for (t = st0; t < en1; t += 4) { // this implements: H[t]+=v8[t]-qe; if(H[t]>max_H) max_H=H[t],max_t=t;
					__m128i H1, tmp, t_;
					H1 = _mm_loadu_si128((__m128i*)&H[t]);
					t_ = _mm_setr_epi32(v8[t], v8[t+1], v8[t+2], v8[t+3]);
					H1 = _mm_add_epi32(H1, t_);
					_mm_storeu_si128((__m128i*)&H[t], H1);
					t_ = _mm_set1_epi32(t);
					tmp = _mm_cmpgt_epi32(H1, max_H_);
#ifdef __SSE4_1__
					max_H_ = _mm_blendv_epi8(max_H_, H1, tmp);
					max_t_ = _mm_blendv_epi8(max_t_, t_, tmp);
#else
					max_H_ = _mm_or_si128(_mm_and_si128(tmp, H1), _mm_andnot_si128(tmp, max_H_));
					max_t_ = _mm_or_si128(_mm_and_si128(tmp, t_), _mm_andnot_si128(tmp, max_t_));
#endif
				}
				_mm_storeu_si128((__m128i*)HH, max_H_);
				_mm_storeu_si128((__m128i*)tt, max_t_);
				for (i = 0; i < 4; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
				for (; t < en0; ++t) { // for the rest of values that haven't been computed with SSE
					H[t] += (int32_t)v8[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}
			} else H[0] = v8[0] - qe, max_H = H[0], max_t = 0; // special casing r==0
			// update ez
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop(ez, 1, max_H, r, max_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
		} else { // find approximate max; Z-drop might be inaccurate, too.
			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v8[last_H0_t];
					int32_t d1 = u8[last_H0_t + 1];
					if (d0 > d1) H0 += d0;
					else H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v8[last_H0_t];
				} else {
					++last_H0_t, H0 += u8[last_H0_t];
				}
			} else H0 = v8[0] - qe, last_H0_t = 0;
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop(ez, 1, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}
	kfree(km, mem);
	if (!approx_max) kfree(km, H);
	if (with_cigar) { // backtrack
		int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
		if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
			ez->reach_end = 1;
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (ez->max_t >= 0 && ez->max_q >= 0) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*16, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}
		kfree(km, mem2); kfree(km, off);
	}
	#undef __dp_code_block1
	#undef __dp_code_block2
}
#endif // __SSE2__

#ifdef __AVX2__
__attribute__((optimize("O3")))
void ksw_extd2_avx2(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat,
				   int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t *ez)
{
#define _mm256_cvtsi32_si256(a)  _mm256_zextsi128_si256(_mm_cvtsi32_si128(a))
#define _mm256_cvtsi128_si256(a) _mm256_zextsi128_si256(_mm256_castsi256_si128(a))
#define __dp_code_block1												\
	z 	= _mm256_load_si256(&s[t]);										\
	xt1 = _mm256_load_si256(&x[t]);                     				\
	__m256i shifted = _mm256_slli_si256(xt1, 1);						\
	__m256i extract = _mm256_srli_si256(xt1, 15); 						\
	__m256i swapped = _mm256_permute2f128_si256(extract, extract, 1);	\
	tmp 	= _mm256_cvtsi128_si256(swapped);							\
	swapped = _mm256_and_si256(swapped, mask); 							\
	swapped = _mm256_or_si256(swapped, x1_);							\
	xt1 	= _mm256_or_si256(swapped, shifted);						\
	x1_ 	= tmp; 														\
	vt1 	= _mm256_load_si256(&v[t]);                    				\
	shifted = _mm256_slli_si256(vt1, 1);								\
	extract = _mm256_srli_si256(vt1, 15); 								\
	swapped = _mm256_permute2f128_si256(extract, extract, 1);			\
	tmp 	= _mm256_cvtsi128_si256(swapped);							\
	swapped = _mm256_and_si256(swapped, mask); 							\
	swapped = _mm256_or_si256(swapped, v1_);							\
	vt1 	= _mm256_or_si256(swapped, shifted);						\
	v1_ 	= tmp; 														\
	a 		= _mm256_add_epi8(xt1, vt1);           			        	\
	ut 		= _mm256_load_si256(&u[t]);              			        \
	b 		= _mm256_add_epi8(_mm256_load_si256(&y[t]), ut);     		\
	x2t1 	= _mm256_load_si256(&x2[t]);								\
	shifted = _mm256_slli_si256(x2t1, 1);								\
	extract = _mm256_srli_si256(x2t1, 15); 								\
	swapped = _mm256_permute2f128_si256(extract, extract, 1);			\
	tmp 	= _mm256_cvtsi128_si256(swapped);							\
	swapped = _mm256_and_si256(swapped, mask); 							\
	swapped = _mm256_or_si256(swapped, x21_);							\
	x2t1 	= _mm256_or_si256(swapped, shifted);						\
	x21_ 	= tmp;														\
	a2		= _mm256_add_epi8(x2t1, vt1); 								\
	b2		= _mm256_add_epi8(_mm256_load_si256(&y2[t]), ut);

#define __dp_code_block2 																			\
	_mm256_store_si256(&u[t], _mm256_sub_epi8(z, vt1));/* u[r][t..t+15] <- z - v[r-1][t-1..t+14] */ \
	_mm256_store_si256(&v[t], _mm256_sub_epi8(z, ut)); /* v[r][t..t+15] <- z - u[r-1][t..t+15] */ 	\
	tmp = _mm256_sub_epi8(z, q_); 																	\
	a = _mm256_sub_epi8(a, tmp); 																	\
	b = _mm256_sub_epi8(b, tmp); 																	\
	tmp = _mm256_sub_epi8(z, q2_); 																	\
	a2= _mm256_sub_epi8(a2, tmp); 																	\
	b2= _mm256_sub_epi8(b2, tmp);

	int r, t, qe = q + e, n_col_, *off = 0, *off_end = 0, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc, long_thres, long_diff;
	int with_cigar = !(flag&KSW_EZ_SCORE_ONLY), approx_max = !!(flag&KSW_EZ_APPROX_MAX);
	int32_t *H = 0, H0 = 0, last_H0_t = 0;
	uint8_t *qr, *sf, *mem, *mem2 = 0;
	__m256i q_, q2_, qe_, qe2_, zero_, sc_mch_, sc_mis_, m1_, sc_N_;
	__m256i flag1_, flag2_, flag3_, flag4_, flag8_, flag16_, flag32_, flag64_; 
	__m256i *u, *v, *x, *y, *x2, *y2, *s, *p = 0;

	ksw_reset_extz(ez);
	if (m <= 1 || qlen <= 0 || tlen <= 0) return;


	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	zero_   = _mm256_set1_epi8(0);
	q_      = _mm256_set1_epi8(q);
	q2_     = _mm256_set1_epi8(q2);
	qe_     = _mm256_set1_epi8(q + e);
	qe2_    = _mm256_set1_epi8(q2 + e2);
	sc_mch_ = _mm256_set1_epi8(mat[0]);
	sc_mis_ = _mm256_set1_epi8(mat[1]);
	sc_N_   = mat[m*m-1] == 0? _mm256_set1_epi8(-e2) : _mm256_set1_epi8(mat[m*m-1]);
	m1_     = _mm256_set1_epi8(m - 1); // wildcard

	flag1_  = _mm256_set1_epi8(1);
	flag2_  = _mm256_set1_epi8(2);
	flag3_  = _mm256_set1_epi8(3);
	flag4_  = _mm256_set1_epi8(4);
	flag8_  = _mm256_set1_epi8(8);
	flag16_ = _mm256_set1_epi8(16);
	flag32_ = _mm256_set1_epi8(32);
	flag64_ = _mm256_set1_epi8(64);

	__m256i mask = _mm256_set_epi64x(-1, -1, -1, -256);

	if (w < 0) w = tlen > qlen? tlen : qlen;
	wl = wr = w;
	tlen_ = (tlen + 32 - 1) / 32;
	n_col_ = qlen < tlen? qlen : tlen;
	n_col_ = ((n_col_ < w + 1? n_col_ : w + 1) + 32 - 1) / 32 + 1;
	qlen_ = (qlen + 32 - 1) / 32;
	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
		max_sc = max_sc > mat[t]? max_sc : mat[t];
		min_sc = min_sc < mat[t]? min_sc : mat[t];
	}
	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	long_thres = e != e2? (q2 - q) / (e - e2) - 1 : 0;
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	mem = (uint8_t*)kcalloc(km, tlen_ * 8 + qlen_ + 1, 32);
	u = (__m256i*)(((size_t)mem + 32 - 1) >> 5 << 5); // 16-byte aligned
	v = u + tlen_, x = v + tlen_, y = x + tlen_, x2 = y + tlen_, y2 = x2 + tlen_;
	s = y2 + tlen_, sf = (uint8_t*)(s + tlen_), qr = sf + tlen_ * 32;
	memset(u,  -q  - e,  tlen_ * 32);
	memset(v,  -q  - e,  tlen_ * 32);
	memset(x,  -q  - e,  tlen_ * 32);
	memset(y,  -q  - e,  tlen_ * 32);
	memset(x2, -q2 - e2, tlen_ * 32);
	memset(y2, -q2 - e2, tlen_ * 32);
	if (!approx_max) {
		H = (int32_t*)kmalloc(km, tlen_ * 32 * 4);
		for (t = 0; t < tlen_ * 32; ++t) H[t] = KSW_NEG_INF;
	}
	if (with_cigar) {
		mem2 = (uint8_t*)kmalloc(km, ((size_t)(qlen + tlen - 1) * n_col_ + 1) * 32);
		p = (__m256i*)(((size_t)mem2 + 32 - 1) >> 5 << 5);
		off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int) * 2);
		off_end = off + qlen + tlen - 1;
	}

	for (t = 0; t < qlen; ++t) qr[t] = query[qlen - 1 - t];
	memcpy(sf, target, tlen);

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		uint8_t *qrr = qr + (qlen - 1 - r);
		int8_t *u8 = (int8_t*)u, *v8 = (int8_t*)v, *x8 = (int8_t*)x, *x28 = (int8_t*)x2;
		__m256i x1_, x21_, v1_;
		// find the boundaries
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r-wr+1)>>1) st = (r-wr+1)>>1; // take the ceil
		if (en > (r+wl)>>1) en = (r+wl)>>1; // take the floor
		if (st > en) {
			ez->zdropped = 1;
			break;
		}
		st0 = st, en0 = en;
		st = st / 32 * 32, en = (en + 32) / 32 * 32 - 1;
		// set boundary conditions
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x8[st - 1], x21 = x28[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
			} else {
				x1 = -q - e, x21 = -q2 - e2;
				v1 = -q - e;
			}
		} else {
			x1 = -q - e, x21 = -q2 - e2;
			v1 = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		if (en >= r) {
			((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u8[r] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {
			for (t = st0; t <= en0; t += 32) {
				__m256i sq, st, tmp;
				//better than loadu if data in L1
				sq = _mm256_lddqu_si256((__m256i*)&sf[t]); 
				st = _mm256_lddqu_si256((__m256i*)&qrr[t]);

				__m256i msk = _mm256_or_si256(_mm256_cmpeq_epi8(sq, m1_), _mm256_cmpeq_epi8(st, m1_));
				tmp = _mm256_cmpeq_epi8(sq, st);
				tmp = _mm256_blendv_epi8(sc_mis_, sc_mch_, tmp);
				tmp = _mm256_blendv_epi8(tmp,     sc_N_,   msk);
				_mm256_storeu_si256((__m256i*)((int8_t*)s + t), tmp);
			}
		} else {
			for (t = st0; t < (en0 % 8)+1; ++t)
			{
				((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
			}
			
			__m256i m256v 		 = _mm256_set1_epi32(m);
			__m256i shuffle_mask = _mm256_setr_epi64x(0xFFFFFFFF0C080400, -1, 0xFFFFFFFF0C080400, -1);
			__m256i perm_mask    = _mm256_setr_epi32(0,4,1,1,1,1,1,1);
			uint8_t* s_aux = ((uint8_t*)s);
			for (; t <= en0; t+=8)
			{
				__m256i sf_  = _mm256_cvtepi8_epi32(_mm_lddqu_si128((__m128i*)&sf[t]));
				__m256i qrr_ = _mm256_cvtepi8_epi32(_mm_lddqu_si128((__m128i*)&qrr[t]));
				__m256i val  = _mm256_add_epi32(_mm256_mullo_epi16(sf_, m256v), qrr_);
				__m256i sti  = _mm256_i32gather_epi32((const int*)&mat[0], val, 1);
				sti  = _mm256_shuffle_epi8(sti, shuffle_mask);
				sti  = _mm256_permutevar8x32_epi32(sti, perm_mask);
				_mm_storeu_si64(&s_aux[t], _mm256_castsi256_si128(sti));				
			}
		}

		// core loop
		x1_ = _mm256_cvtsi32_si256((uint8_t)x1);
		x21_= _mm256_cvtsi32_si256((uint8_t)x21);
		v1_ = _mm256_cvtsi32_si256((uint8_t)v1);
		st_ = st / 32, en_ = en / 32;
		assert(en_ - st_ + 1 <= n_col_);
		if (!with_cigar) { // score only
			for (t = st_; t <= en_; ++t) {
				__m256i z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				z = _mm256_max_epi8(z, a);
				z = _mm256_max_epi8(z, b);
				z = _mm256_max_epi8(z, a2);
				z = _mm256_max_epi8(z, b2);
				z = _mm256_min_epi8(z, sc_mch_);
				__dp_code_block2; // save u[] and v[]; update a, b, a2 and b2
				_mm256_store_si256(&x[t],  _mm256_sub_epi8(_mm256_max_epi8(a,  zero_), qe_));
				_mm256_store_si256(&y[t],  _mm256_sub_epi8(_mm256_max_epi8(b,  zero_), qe_));
				_mm256_store_si256(&x2[t], _mm256_sub_epi8(_mm256_max_epi8(a2, zero_), qe2_));
				_mm256_store_si256(&y2[t], _mm256_sub_epi8(_mm256_max_epi8(b2, zero_), qe2_));
			}
		} else if (!(flag&KSW_EZ_RIGHT)) { // gap left-alignment
			__m256i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m256i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				d = _mm256_and_si256(_mm256_cmpgt_epi8(a, z), flag1_);       // d = a  > z? 1 : 0
				z = _mm256_max_epi8(z, a);
				d = _mm256_blendv_epi8(d, flag2_, _mm256_cmpgt_epi8(b,  z)); // d = b  > z? 2 : d
				z = _mm256_max_epi8(z, b);
				d = _mm256_blendv_epi8(d, flag3_, _mm256_cmpgt_epi8(a2, z)); // d = a2 > z? 3 : d
				z = _mm256_max_epi8(z, a2);
				d = _mm256_blendv_epi8(d, flag4_, _mm256_cmpgt_epi8(b2, z)); // d = a2 > z? 3 : d
				z = _mm256_max_epi8(z, b2);
				z = _mm256_min_epi8(z, sc_mch_);
				__dp_code_block2;
				tmp = _mm256_cmpgt_epi8(a, zero_);
				_mm256_store_si256(&x[t],  _mm256_sub_epi8(_mm256_and_si256(tmp, a),  qe_));
				d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag8_)); // d = a > 0? 1<<3 : 0
				tmp = _mm256_cmpgt_epi8(b, zero_);
				_mm256_store_si256(&y[t],  _mm256_sub_epi8(_mm256_and_si256(tmp, b),  qe_));
				d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag16_)); // d = b > 0? 1<<4 : 0
				tmp = _mm256_cmpgt_epi8(a2, zero_);
				_mm256_store_si256(&x2[t], _mm256_sub_epi8(_mm256_and_si256(tmp, a2), qe2_));
				d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag32_)); // d = a > 0? 1<<5 : 0
				tmp = _mm256_cmpgt_epi8(b2, zero_);
				_mm256_store_si256(&y2[t], _mm256_sub_epi8(_mm256_and_si256(tmp, b2), qe2_));
				d = _mm256_or_si256(d, _mm256_and_si256(tmp, flag64_)); // d = b > 0? 1<<6 : 0
				_mm256_store_si256(&pr[t], d);
			}
		} else { // gap right-alignment
			__m256i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m256i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				d = _mm256_andnot_si256(_mm256_cmpgt_epi8(z, a), flag1_);    // d = z > a?  0 : 1
				z = _mm256_max_epi8(z, a);
				d = _mm256_blendv_epi8(flag2_, d, _mm256_cmpgt_epi8(z, b));  // d = z > b?  d : 2
				z = _mm256_max_epi8(z, b);
				d = _mm256_blendv_epi8(flag3_, d, _mm256_cmpgt_epi8(z, a2)); // d = z > a2? d : 3
				z = _mm256_max_epi8(z, a2);
				d = _mm256_blendv_epi8(flag4_, d, _mm256_cmpgt_epi8(z, b2)); // d = z > b2? d : 4
				z = _mm256_max_epi8(z, b2);
				z = _mm256_min_epi8(z, sc_mch_);
				__dp_code_block2;
				tmp = _mm256_cmpgt_epi8(zero_, a);
				_mm256_store_si256(&x[t],  _mm256_sub_epi8(_mm256_andnot_si256(tmp, a),  qe_));
				d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag8_)); // d = a > 0? 1<<3 : 0
				tmp = _mm256_cmpgt_epi8(zero_, b);
				_mm256_store_si256(&y[t],  _mm256_sub_epi8(_mm256_andnot_si256(tmp, b),  qe_));
				d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag16_)); // d = b > 0? 1<<4 : 0
				tmp = _mm256_cmpgt_epi8(zero_, a2);
				_mm256_store_si256(&x2[t], _mm256_sub_epi8(_mm256_andnot_si256(tmp, a2), qe2_));
				d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag32_)); // d = a > 0? 1<<5 : 0
				tmp = _mm256_cmpgt_epi8(zero_, b2);
				_mm256_store_si256(&y2[t], _mm256_sub_epi8(_mm256_andnot_si256(tmp, b2), qe2_));
				d = _mm256_or_si256(d, _mm256_andnot_si256(tmp, flag64_)); // d = b > 0? 1<<6 : 0
				_mm256_store_si256(&pr[t], d);
			}
		}
		if (!approx_max) { // find the exact max with a 32-bit score array
			int32_t max_H, max_t;
			// compute H[], max_H and max_t
			if (r > 0) {
				int32_t HH[32/4], tt[32/4], en1 = st0 + (en0 - st0) / (32/4) * (32/4), i;
				__m256i max_H_, max_t_;
				max_H = H[en0] = en0 > 0? H[en0-1] + u8[en0] : H[en0] + v8[en0]; // special casing the last element
				max_t = en0;
				max_H_ = _mm256_set1_epi32(max_H);
				max_t_ = _mm256_set1_epi32(max_t);
				for (t = st0; t < en1; t += 32/4) { // this implements: H[t]+=v8[t]; if(H[t]>max_H) max_H=H[t],max_t=t;
					__m256i H1, t_;
					H1 = _mm256_lddqu_si256((__m256i*)&H[t]);
					t_ = _mm256_cvtepi8_epi32(_mm_lddqu_si128((__m128i*)&v8[t]));
					H1 = _mm256_add_epi32(H1, t_);
					_mm256_storeu_si256((__m256i*)&H[t], H1);
					t_ = _mm256_set1_epi32(t);
					__m256i tmp = _mm256_cmpgt_epi32(H1, max_H_);
					max_H_ = _mm256_blendv_epi8(max_H_, H1, tmp);
					max_t_ = _mm256_blendv_epi8(max_t_, t_, tmp);
				}
				_mm256_storeu_si256((__m256i*)HH, max_H_);
				_mm256_storeu_si256((__m256i*)tt, max_t_);
				for (i = 0; i < 32/4; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
				for (; t < en0; ++t) { // for the rest of values that haven't been computed with SSE
					H[t] += (int32_t)v8[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}
			} else H[0] = v8[0] - qe, max_H = H[0], max_t = 0; // special casing r==0
			// update ez
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop(ez, 1, max_H, r, max_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
		} else { // find approximate max; Z-drop might be inaccurate, too.
			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v8[last_H0_t];
					int32_t d1 = u8[last_H0_t + 1];
					if (d0 > d1) H0 += d0;
					else H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v8[last_H0_t];
				} else {
					++last_H0_t, H0 += u8[last_H0_t];
				}
			} else H0 = v8[0] - qe, last_H0_t = 0;
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop(ez, 1, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}
	kfree(km, mem);
	if (!approx_max) kfree(km, H);
	if (with_cigar) { // backtrack
		int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
		if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*32, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
			ez->reach_end = 1;
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*32, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (ez->max_t >= 0 && ez->max_q >= 0) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*32, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}
		/*
		//This is not in minimap2
		if (flag & KSW_EZ_EQX) {
			int32_t nc0 = ez->n_cigar;
			uint32_t *ci0;
			ci0 = (uint32_t*)kmalloc(km, nc0 * sizeof(uint32_t));
			memcpy(ci0, ez->cigar, nc0 * sizeof(uint32_t));
			ksw_cigar2eqx(km, query, target, nc0, ci0, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
			kfree(km, ci0);
		}
		*/
		kfree(km, mem2); kfree(km, off);
	}
	#undef __dp_code_block1
	#undef __dp_code_block2
}
#endif //__AVX2__


#ifdef __AVX512BW__
__attribute__((optimize("O3")))
void ksw_extd2_avx512(void *km, int qlen, const uint8_t *query, int tlen, const uint8_t *target, int8_t m, const int8_t *mat,
				   int8_t q, int8_t e, int8_t q2, int8_t e2, int w, int zdrop, int end_bonus, int flag, ksw_extz_t *ez)
{
#define _mm512_cvtsi32_si512(a)  _mm512_zextsi128_si512(_mm_cvtsi32_si128(a))
#define _mm512_cvtsi128_si512(a) _mm512_zextsi128_si512(_mm512_castsi512_si128(a))
#define __dp_code_block1										\
	z   	= _mm512_load_si512(&s[t]);							\
	xt1 	= _mm512_load_si512(&x[t]);   						\
	__m512i shifted = _mm512_bslli_epi128(xt1, 1);				\
	__m512i extract = _mm512_bsrli_epi128(xt1, 15);				\
	extract = _mm512_shuffle_i32x4(extract, extract, 0x93);		\
	xt1 	= _mm512_or_si512(shifted, extract);				\
	tmp 	= _mm512_cvtsi128_si512(xt1);						\
	xt1 	= _mm512_mask_blend_epi8(1, xt1, x1_);				\
	x1_ 	= tmp;												\
	vt1 	= _mm512_load_si512(&v[t]);							\
	shifted = _mm512_bslli_epi128(vt1, 1);						\
	extract = _mm512_bsrli_epi128(vt1, 15);						\
	extract = _mm512_shuffle_i32x4(extract, extract, 0x93);		\
	vt1 	= _mm512_or_si512(shifted, extract);				\
	tmp 	= _mm512_cvtsi128_si512(vt1);						\
	vt1 	= _mm512_mask_blend_epi8(1, vt1, v1_);				\
	v1_ 	= tmp;												\
	a   	= _mm512_add_epi8(xt1, vt1);						\
	ut  	= _mm512_load_si512(&u[t]);							\
	b   	= _mm512_add_epi8(_mm512_load_si512(&y[t]), ut);	\
	x2t1	= _mm512_load_si512(&x2[t]);						\
	shifted = _mm512_bslli_epi128(x2t1, 1);						\
	extract = _mm512_bsrli_epi128(x2t1, 15);					\
	extract = _mm512_shuffle_i32x4(extract, extract, 0x93);		\
	x2t1	= _mm512_or_si512(shifted, extract);				\
	tmp 	= _mm512_cvtsi128_si512(x2t1);						\
	x2t1	= _mm512_mask_blend_epi8(1, x2t1, x21_);			\
	x21_	= tmp;												\
	a2 		= _mm512_add_epi8(x2t1, vt1);						\
	b2		= _mm512_add_epi8(_mm512_load_si512(&y2[t]), ut);

#define __dp_code_block2                                                                       			\
	_mm512_store_si512(&u[t], _mm512_sub_epi8(z, vt1));	/* u[r][t..t+15] <- z - v[r-1][t-1..t+14] */    \
	_mm512_store_si512(&v[t], _mm512_sub_epi8(z, ut));  /* v[r][t..t+15] <- z - u[r-1][t..t+15]   */    \
	tmp = _mm512_sub_epi8(z, q_);                                                                       \
	a = _mm512_sub_epi8(a, tmp);                                                                        \
	b = _mm512_sub_epi8(b, tmp);                                                                        \
	tmp = _mm512_sub_epi8(z, q2_);                                                                      \
	a2= _mm512_sub_epi8(a2, tmp);                                                                       \
	b2= _mm512_sub_epi8(b2, tmp);

	int r, t, qe = q + e, n_col_, *off = 0, *off_end = 0, tlen_, qlen_, last_st, last_en, wl, wr, max_sc, min_sc, long_thres, long_diff;
	int with_cigar = !(flag&KSW_EZ_SCORE_ONLY), approx_max = !!(flag&KSW_EZ_APPROX_MAX);
	int32_t *H = 0, H0 = 0, last_H0_t = 0;
	uint8_t *qr, *sf, *mem, *mem2 = 0;
	__m512i q_, q2_, qe_, qe2_, zero_, sc_mch_, sc_mis_, m1_, sc_N_;
	__m512i flag1_, flag2_, flag3_, flag4_, flag8_, flag16_, flag32_, flag64_; 
	__m512i *u, *v, *x, *y, *x2, *y2, *s, *p = 0;

	ksw_reset_extz(ez);
	if (m <= 1 || qlen <= 0 || tlen <= 0) return;


	if (q2 + e2 < q + e) t = q, q = q2, q2 = t, t = e, e = e2, e2 = t; // make sure q+e no larger than q2+e2

	zero_   = _mm512_set1_epi8(0);
	q_      = _mm512_set1_epi8(q);
	q2_     = _mm512_set1_epi8(q2);
	qe_     = _mm512_set1_epi8(q + e);
	qe2_    = _mm512_set1_epi8(q2 + e2);
	sc_mch_ = _mm512_set1_epi8(mat[0]);
	sc_mis_ = _mm512_set1_epi8(mat[1]);
	sc_N_   = mat[m*m-1] == 0? _mm512_set1_epi8(-e2) : _mm512_set1_epi8(mat[m*m-1]);
	m1_     = _mm512_set1_epi8(m - 1); // wildcard

	flag1_  = _mm512_set1_epi8(1);
	flag2_  = _mm512_set1_epi8(2);
	flag3_  = _mm512_set1_epi8(3);
	flag4_  = _mm512_set1_epi8(4);
	flag8_  = _mm512_set1_epi8(8);
	flag16_ = _mm512_set1_epi8(16);
	flag32_ = _mm512_set1_epi8(32);
	flag64_ = _mm512_set1_epi8(64);

	if (w < 0) w = tlen > qlen? tlen : qlen;
	wl = wr = w;
	tlen_ = (tlen + 64 - 1) / 64;
	n_col_ = qlen < tlen? qlen : tlen;
	n_col_ = ((n_col_ < w + 1? n_col_ : w + 1) + 64 - 1) / 64 + 1;
	qlen_ = (qlen + 64 - 1) / 64;
	for (t = 1, max_sc = mat[0], min_sc = mat[1]; t < m * m; ++t) {
		max_sc = max_sc > mat[t]? max_sc : mat[t];
		min_sc = min_sc < mat[t]? min_sc : mat[t];
	}
	if (-min_sc > 2 * (q + e)) return; // otherwise, we won't see any mismatches

	long_thres = e != e2? (q2 - q) / (e - e2) - 1 : 0;
	if (q2 + e2 + long_thres * e2 > q + e + long_thres * e)
		++long_thres;
	long_diff = long_thres * (e - e2) - (q2 - q) - e2;

	mem = (uint8_t*)kcalloc(km, tlen_ * 8 + qlen_ + 1, 64);
	u = (__m512i*)(((size_t)mem + 64 - 1) >> 6 << 6); // 16-byte aligned
	v = u + tlen_, x = v + tlen_, y = x + tlen_, x2 = y + tlen_, y2 = x2 + tlen_;
	s = y2 + tlen_, sf = (uint8_t*)(s + tlen_), qr = sf + tlen_ * 64;
	memset(u,  -q  - e,  tlen_ * 64);
	memset(v,  -q  - e,  tlen_ * 64);
	memset(x,  -q  - e,  tlen_ * 64);
	memset(y,  -q  - e,  tlen_ * 64);
	memset(x2, -q2 - e2, tlen_ * 64);
	memset(y2, -q2 - e2, tlen_ * 64);
	if (!approx_max) {
		H = (int32_t*)kmalloc(km, tlen_ * 64 * 4);
		for (t = 0; t < tlen_ * 64; ++t) H[t] = KSW_NEG_INF;
	}
	if (with_cigar) {
		mem2 = (uint8_t*)kmalloc(km, ((size_t)(qlen + tlen - 1) * n_col_ + 1) * 64);
		p = (__m512i*)(((size_t)mem2 + 64 - 1) >> 6 << 6);
		off = (int*)kmalloc(km, (qlen + tlen - 1) * sizeof(int) * 2);
		off_end = off + qlen + tlen - 1;
	}

	for (t = 0; t < qlen; ++t) qr[t] = query[qlen - 1 - t];
	memcpy(sf, target, tlen);

	for (r = 0, last_st = last_en = -1; r < qlen + tlen - 1; ++r) {
		int st = 0, en = tlen - 1, st0, en0, st_, en_;
		int8_t x1, x21, v1;
		uint8_t *qrr = qr + (qlen - 1 - r);
		int8_t *u8 = (int8_t*)u, *v8 = (int8_t*)v, *x8 = (int8_t*)x, *x28 = (int8_t*)x2;
		__m512i x1_, x21_, v1_;
		// find the boundaries
		if (st < r - qlen + 1) st = r - qlen + 1;
		if (en > r) en = r;
		if (st < (r-wr+1)>>1) st = (r-wr+1)>>1; // take the ceil
		if (en > (r+wl)>>1) en = (r+wl)>>1; // take the floor
		if (st > en) {
			ez->zdropped = 1;
			break;
		}
		st0 = st, en0 = en;
		st = st / 64 * 64, en = (en + 64) / 64 * 64 - 1;
		// set boundary conditions
		if (st > 0) {
			if (st - 1 >= last_st && st - 1 <= last_en) {
				x1 = x8[st - 1], x21 = x28[st - 1], v1 = v8[st - 1]; // (r-1,s-1) calculated in the last round
			} else {
				x1 = -q - e, x21 = -q2 - e2;
				v1 = -q - e;
			}
		} else {
			x1 = -q - e, x21 = -q2 - e2;
			v1 = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		if (en >= r) {
			((int8_t*)y)[r] = -q - e, ((int8_t*)y2)[r] = -q2 - e2;
			u8[r] = r == 0? -q - e : r < long_thres? -e : r == long_thres? long_diff : -e2;
		}
		// loop fission: set scores first
		if (!(flag & KSW_EZ_GENERIC_SC)) {
			for (t = st0; t <= en0; t += 64) {
				__m512i sq, st, tmp;
				sq = _mm512_loadu_si512((__m512i*)&sf[t]);
				st = _mm512_loadu_si512((__m512i*)&qrr[t]);
				__mmask64 msk = _mm512_cmpeq_epi8_mask(sq, m1_) | _mm512_cmpeq_epi8_mask(st, m1_);
				tmp = _mm512_mask_blend_epi8(_mm512_cmpeq_epi8_mask(sq, st), sc_mis_, sc_mch_);
				tmp = _mm512_mask_blend_epi8(msk, tmp, sc_N_);
				_mm512_storeu_si512((__m512i*)((int8_t*)s + t), tmp);
			}
		} else {
			for (t = st0; t < (en0 % 8)+1; ++t)
			{
				((uint8_t*)s)[t] = mat[sf[t] * m + qrr[t]];
			}
			
			__m256i m256v 		 = _mm256_set1_epi32(m);
			__m256i shuffle_mask = _mm256_setr_epi64x(0xFFFFFFFF0C080400, -1, 0xFFFFFFFF0C080400, -1);
			__m256i perm_mask    = _mm256_setr_epi32(0,4,1,1,1,1,1,1);
			uint8_t* s_aux = ((uint8_t*)s);
			for (; t <= en0; t+=8)
			{
				__m256i sf_  = _mm256_cvtepi8_epi32(_mm_lddqu_si128((__m128i*)&sf[t]));
				__m256i qrr_ = _mm256_cvtepi8_epi32(_mm_lddqu_si128((__m128i*)&qrr[t]));
				__m256i val  = _mm256_add_epi32(_mm256_mullo_epi16(sf_, m256v), qrr_);
				__m256i sti  = _mm256_i32gather_epi32((const int*)&mat[0], val, 1);
				sti  = _mm256_shuffle_epi8(sti, shuffle_mask);
				sti  = _mm256_permutevar8x32_epi32(sti, perm_mask);
				_mm_storeu_si64(&s_aux[t], _mm256_castsi256_si128(sti));				
			}
		}

		// core loop
		x1_ = _mm512_cvtsi32_si512((uint8_t)x1);
		x21_= _mm512_cvtsi32_si512((uint8_t)x21);
		v1_ = _mm512_cvtsi32_si512((uint8_t)v1);
		st_ = st / 64, en_ = en / 64;
		assert(en_ - st_ + 1 <= n_col_);
		if (!with_cigar) { // score only
			for (t = st_; t <= en_; ++t) {
				__m512i z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				z = _mm512_max_epi8(z, a);
				z = _mm512_max_epi8(z, b);
				z = _mm512_max_epi8(z, a2);
				z = _mm512_max_epi8(z, b2);
				z = _mm512_min_epi8(z, sc_mch_);
				__dp_code_block2; // save u[] and v[]; update a, b, a2 and b2
				_mm512_store_si512(&x[t],  _mm512_sub_epi8(_mm512_max_epi8(a,  zero_), qe_));
				_mm512_store_si512(&y[t],  _mm512_sub_epi8(_mm512_max_epi8(b,  zero_), qe_));
				_mm512_store_si512(&x2[t], _mm512_sub_epi8(_mm512_max_epi8(a2, zero_), qe2_));
				_mm512_store_si512(&y2[t], _mm512_sub_epi8(_mm512_max_epi8(b2, zero_), qe2_));
			}
		} else if (!(flag&KSW_EZ_RIGHT)) { // gap left-alignment
			__m512i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m512i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				d = _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(a,  z), zero_, flag1_);
				z = _mm512_max_epi8(z, a);
				d = _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(b,  z), d, flag2_);
				z = _mm512_max_epi8(z, b);
				d = _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(a2, z), d, flag3_);
				z = _mm512_max_epi8(z, a2);
				d = _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(b2, z), d, flag4_);
				z = _mm512_max_epi8(z, b2);
				z = _mm512_min_epi8(z, sc_mch_);
				__dp_code_block2;
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(a,  zero_),  zero_, flag8_));  // d = a  > 0? 1<<3 : 0
				_mm512_store_si512(&x[t],  _mm512_sub_epi8(_mm512_max_epi8(a,  zero_), qe_));
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(b,  zero_),  zero_, flag16_)); // d = b  > 0? 1<<4 : 0
				_mm512_store_si512(&y[t],  _mm512_sub_epi8(_mm512_max_epi8(b,  zero_), qe_));
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(a2,  zero_), zero_, flag32_)); // d = a2 > 0? 1<<5 : 0
				_mm512_store_si512(&x2[t], _mm512_sub_epi8(_mm512_max_epi8(a2, zero_), qe2_));
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpgt_epi8_mask(b2,  zero_), zero_, flag64_)); // d = b2 > 0? 1<<6 : 0
				_mm512_store_si512(&y2[t], _mm512_sub_epi8(_mm512_max_epi8(b2, zero_), qe2_));
				_mm512_store_si512(&pr[t], d);
			}
		} else { // gap right-alignment
			__m512i *pr = p + (size_t)r * n_col_ - st_;
			off[r] = st, off_end[r] = en;
			for (t = st_; t <= en_; ++t) {
				__m512i d, z, a, b, a2, b2, xt1, x2t1, vt1, ut, tmp;
				__dp_code_block1;
				d = _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(a,  z), zero_, flag1_);
				z = _mm512_max_epi8(z, a);
				d = _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(b,  z), d,     flag2_);
				z = _mm512_max_epi8(z, b);
				d = _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(a2, z), d,     flag3_);
				z = _mm512_max_epi8(z, a2);
				d = _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(b2, z), d,     flag4_);
				z = _mm512_max_epi8(z, b2);
				z = _mm512_min_epi8(z, sc_mch_);
				__dp_code_block2;
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(a,  zero_),  zero_, flag8_));  // d = a  >= 0? 1<<3 : 0
				_mm512_store_si512(&x[t],  _mm512_sub_epi8(_mm512_max_epi8(a,  zero_), qe_));
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(b,  zero_),  zero_, flag16_)); // d = b  >= 0? 1<<4 : 0
				_mm512_store_si512(&y[t],  _mm512_sub_epi8(_mm512_max_epi8(b,  zero_), qe_));
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(a2,  zero_), zero_, flag32_)); // d = a2 >= 0? 1<<5 : 0
				_mm512_store_si512(&x2[t], _mm512_sub_epi8(_mm512_max_epi8(a2, zero_), qe2_));
				d = _mm512_or_si512(d, _mm512_mask_blend_epi8(_mm512_cmpge_epi8_mask(b2,  zero_), zero_, flag64_)); // d = b2 >= 0? 1<<6 : 0
				_mm512_store_si512(&y2[t], _mm512_sub_epi8(_mm512_max_epi8(b2, zero_), qe2_));
				_mm512_store_si512(&pr[t], d);
			}
		}
		if (!approx_max) { // find the exact max with a 32-bit score array
			int32_t max_H, max_t;
			// compute H[], max_H and max_t
			if (r > 0) {
				int32_t HH[64/4], tt[64/4], en1 = st0 + (en0 - st0) / (64/4) * (64/4), i;
				__m512i max_H_, max_t_;
				max_H = H[en0] = en0 > 0? H[en0-1] + u8[en0] : H[en0] + v8[en0]; // special casing the last element
				max_t = en0;
				max_H_ = _mm512_set1_epi32(max_H);
				max_t_ = _mm512_set1_epi32(max_t);
				for (t = st0; t < en1; t += 64/4) { // this implements: H[t]+=v8[t]; if(H[t]>max_H) max_H=H[t],max_t=t;
					__m512i H1, t_;
					H1 = _mm512_loadu_si512((__m512i*)&H[t]);
					t_ = _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i*)&v8[t]));
					H1 = _mm512_add_epi32(H1, t_);
					_mm512_storeu_si512((__m512i*)&H[t], H1);
					t_ = _mm512_set1_epi32(t);
					__mmask64 tmp = _mm512_cmpgt_epi32_mask(H1, max_H_);
					max_H_ = _mm512_mask_blend_epi32(tmp, max_H_, H1);
					max_t_ = _mm512_mask_blend_epi32(tmp, max_t_, t_);
				}
				_mm512_storeu_si512((__m512i*)HH, max_H_);
				_mm512_storeu_si512((__m512i*)tt, max_t_);
				for (i = 0; i < 64/4; ++i)
					if (max_H < HH[i]) max_H = HH[i], max_t = tt[i] + i;
				for (; t < en0; ++t) { // for the rest of values that haven't been computed with SSE
					H[t] += (int32_t)v8[t];
					if (H[t] > max_H)
						max_H = H[t], max_t = t;
				}
			} else H[0] = v8[0] - qe, max_H = H[0], max_t = 0; // special casing r==0
			// update ez
			if (en0 == tlen - 1 && H[en0] > ez->mte)
				ez->mte = H[en0], ez->mte_q = r - en;
			if (r - st0 == qlen - 1 && H[st0] > ez->mqe)
				ez->mqe = H[st0], ez->mqe_t = st0;
			if (ksw_apply_zdrop(ez, 1, max_H, r, max_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H[tlen - 1];
		} else { // find approximate max; Z-drop might be inaccurate, too.
			if (r > 0) {
				if (last_H0_t >= st0 && last_H0_t <= en0 && last_H0_t + 1 >= st0 && last_H0_t + 1 <= en0) {
					int32_t d0 = v8[last_H0_t];
					int32_t d1 = u8[last_H0_t + 1];
					if (d0 > d1) H0 += d0;
					else H0 += d1, ++last_H0_t;
				} else if (last_H0_t >= st0 && last_H0_t <= en0) {
					H0 += v8[last_H0_t];
				} else {
					++last_H0_t, H0 += u8[last_H0_t];
				}
			} else H0 = v8[0] - qe, last_H0_t = 0;
			if ((flag & KSW_EZ_APPROX_DROP) && ksw_apply_zdrop(ez, 1, H0, r, last_H0_t, zdrop, e2)) break;
			if (r == qlen + tlen - 2 && en0 == tlen - 1)
				ez->score = H0;
		}
		last_st = st, last_en = en;
		//for (t = st0; t <= en0; ++t) printf("(%d,%d)\t(%d,%d,%d,%d)\t%d\n", r, t, ((int8_t*)u)[t], ((int8_t*)v)[t], ((int8_t*)x)[t], ((int8_t*)y)[t], H[t]); // for debugging
	}
	kfree(km, mem);
	if (!approx_max) kfree(km, H);
	if (with_cigar) { // backtrack
		int rev_cigar = !!(flag & KSW_EZ_REV_CIGAR);
		if (!ez->zdropped && !(flag&KSW_EZ_EXTZ_ONLY)) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*64, tlen-1, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (!ez->zdropped && (flag&KSW_EZ_EXTZ_ONLY) && ez->mqe + end_bonus > (int)ez->max) {
			ez->reach_end = 1;
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*64, ez->mqe_t, qlen-1, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		} else if (ez->max_t >= 0 && ez->max_q >= 0) {
			ksw_backtrack(km, 1, rev_cigar, 0, (uint8_t*)p, off, off_end, n_col_*64, ez->max_t, ez->max_q, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
		}
		/*
		//This is not in minimap2
		if (flag & KSW_EZ_EQX) {
			int32_t nc0 = ez->n_cigar;
			uint32_t *ci0;
			ci0 = (uint32_t*)kmalloc(km, nc0 * sizeof(uint32_t));
			memcpy(ci0, ez->cigar, nc0 * sizeof(uint32_t));
			ksw_cigar2eqx(km, query, target, nc0, ci0, &ez->m_cigar, &ez->n_cigar, &ez->cigar);
			kfree(km, ci0);
		}
		*/
		kfree(km, mem2); kfree(km, off);
	}
	#undef __dp_code_block1
	#undef __dp_code_block2
}
#endif //__AVX512BW__
