/*
 *                             The MIT License
 *
 * Wavefront Alignment Algorithms
 * Copyright (c) 2017 by Santiago Marco-Sola  <santiagomsola@gmail.com>
 *
 * This file is part of Wavefront Alignment Algorithms.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * PROJECT: Wavefront Alignment Algorithms
 * AUTHOR(S): Santiago Marco-Sola <santiagomsola@gmail.com>
 * DESCRIPTION: WaveFront-Alignment module for the "extension" of exact matches
 */

#include "wavefront_extend.h"
#include "wavefront_align.h"
#include "wavefront_compute.h"
#include "wavefront_heuristic.h"
#include "wavefront_extend_kernels.h"
#include "wavefront_extend_kernels_sve.h"
#include "wavefront_termination.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>

/*
 * Wavefront-Extend Inner Kernel (Scalar)
 */
FORCE_INLINE wf_offset_t wavefront_extend_matches_packed_kernel(
    wavefront_aligner_t* const wf_aligner,
    const int k,
    wf_offset_t offset) {
  // Fetch pattern/text blocks
  uint64_t* pattern_blocks = (uint64_t*)(wf_aligner->sequences.pattern+WAVEFRONT_V(k,offset));
  uint64_t* text_blocks = (uint64_t*)(wf_aligner->sequences.text+WAVEFRONT_H(k,offset));
  // Compare 64-bits blocks
  uint64_t cmp = *pattern_blocks ^ *text_blocks;
  while (__builtin_expect(cmp==0,0)) {
    // Increment offset (full block)
    offset += 8;
    // Next blocks
    ++pattern_blocks;
    ++text_blocks;
    // Compare
    cmp = *pattern_blocks ^ *text_blocks;
  }
  // Count equal characters
  const int equal_right_bits = __builtin_ctzl(cmp);
  const int equal_chars = DIV_FLOOR(equal_right_bits,8);
  offset += equal_chars;
  // Return extended offset
  return offset;
}

/*
 * Wavefront-Extend Inner Kernel (SIMD SVE)
 */
FORCE_NO_INLINE void wavefront_extend_matches_packed_end2end_sve(
    wavefront_aligner_t* const wf_aligner,
    wavefront_t* const mwavefront,
    const int lo,
    const int hi) {
  // Parameters
  wf_offset_t* const offsets = mwavefront->offsets;
  int k_min = lo;
  int k_max = hi;
  const char* pattern = wf_aligner->sequences.pattern;
  const char* text = wf_aligner->sequences.text;

  uint64_t num_elems = svcntw();

  int k;
  int num_diagonals = k_max-k_min+1;
  int loop_peeling_iters = num_diagonals % num_elems;

  for (k=k_min; k<k_min + loop_peeling_iters; k++) {
    wf_offset_t offset = offsets[k];

    if (offset < 0) continue;

    offsets[k] = wavefront_extend_matches_packed_kernel(wf_aligner,k,offset);
  }

  if (num_diagonals < num_elems) return;

  // Perform one extend step in a SIMD manner, if some diagonal needs more steps
  // to finish the extension, do it in scalar

  for (k=k_min+loop_peeling_iters; k<=k_max;k+=num_elems) {
    // Get number of elements that will be computed in this iteration
    //int active_elements = wf_length - k;
    svint32_t ks = svindex_s32(k, 1);
    //mask = svcmple_s32(svptrue_b32(), ks, svdup_s32(k_max));

    svint32_t sv_offsets = svld1(svptrue_b32(), &offsets[k]);

    // Deactivate null offsets (as they will produce a segfault in the gathers)
    // mask <- cmp less than 0
    svbool_t mask = svcmpge_n_s32(svptrue_b32(), sv_offsets, 0);

    svbool_t original_mask = mask;
    // h = o
    svint32_t h = sv_offsets;
    // v = o - k
    svint32_t v = svsub_z(mask, sv_offsets, ks);

    svuint32_t bases_pattern = svld1_gather_s32offset_u32(mask, (uint32_t*)pattern, v);
    svuint32_t bases_text = svld1_gather_s32offset_u32(mask, (uint32_t*)text, h);

    svuint32_t xor_result = sveor_u32_z(mask, bases_pattern, bases_text);
    xor_result = svrevb_u32_z(mask, xor_result);
    svuint32_t clz_res = svclz_u32_z(mask, xor_result);
    svint32_t Eq = svreinterpret_s32(svlsr_u32_z(mask, clz_res, svdup_u32(3U)));

    // Make sure we don't count beyond the sequence
    // This is not needed as we have sentinels at the end of the sequence.
    // Leaving it as a comment just in case we need it in the future
    // svint32_t remaining_v = svsub_s32_z(mask, svdup_s32(pattern_length), v);
    // svint32_t remaining_h = svsub_s32_z(mask, svdup_s32(text_length), h);
    // Eq = svmin_s32_z(mask, Eq, remaining_v);
    // Eq = svmin_s32_z(mask, Eq, remaining_h);

    sv_offsets = svadd_s32_m(mask, sv_offsets, Eq);

    // Only diagonals that have 4 elements equal (so they have not finished) will continue
    mask = svcmpgt_n_s32(mask, Eq, 3U);

    // This part is not needed if the sequences has sentinels at the end
    // v < pattern_length
    // svbool_t mask_v = svcmplt_n_s32(mask, v, pattern_length);
    // mask = svand_b_z(mask, mask, mask_v);
    // h < text_length
    // svbool_t mask_h = svcmplt_n_s32(mask, h, text_length);
    // mask = svand_b_z(mask, mask, mask_h);

    // Store the partial results
    svst1_s32(original_mask, &offsets[k], sv_offsets);

    // If all diagonals have found a mismatch, we have finished the current
    // iteration, continue with the loop. If not, finish the remaining diagonal
    // in a scalar way
    bool svtest;
    while( (svtest = svptest_any(svptrue_b32(), mask)) ) {
      // Extract the K corresponding to the last (i.e. most significant bit)
      // active element in the predicate
      int32_t curr_k = svlastb_s32(mask, ks);
      int32_t curr_offset = svlastb_s32(mask, sv_offsets);

      // Remove the currently extracted element from the predicate:
      // 1. Create a new predicate with all elements to 1 except the selected
      //    one (using a cmp). Latency: 4 cycles. Throughput: 1 per cycle
      // 2. Perform an AND operation between the two predicates. Latency: 2.
      //    Throughput: 2

      // In cmpne with an immediate, the immediate is limited to -16 to 15, use
      // a full register as we need more range than that. Latency of dup:
      // (latency=3, throughout=1/cycle)
      svbool_t mask_to_remove_sel = svcmpne_s32(mask, ks, svdup_s32(curr_k));
      mask = svand_b_z(mask, mask, mask_to_remove_sel);

      // Scalar extend
      if (curr_offset >= 0) {
        offsets[curr_k] = wavefront_extend_matches_packed_kernel(wf_aligner,curr_k,curr_offset);
      } else {
        offsets[curr_k] = WAVEFRONT_OFFSET_NULL;
      }
    } // while (svtest)
  } // for kmin...kmax
}


FORCE_NO_INLINE wf_offset_t wavefront_extend_matches_packed_end2end_max_sve(
    wavefront_aligner_t* const wf_aligner,
    wavefront_t* const mwavefront,
    const int lo,
    const int hi) {
  // Parameters
  const char* pattern = wf_aligner->sequences.pattern;
  const char* text    = wf_aligner->sequences.text;
  
  wf_offset_t* const offsets = mwavefront->offsets;
  int k_min = lo;
  int k_max = hi;
  wf_offset_t max_antidiag   = 0;
  uint64_t num_elems = svcntw();

  int k;
  int num_diagonals = k_max-k_min+1;
  int loop_peeling_iters = num_diagonals % num_elems;

  for (k=k_min; k<k_min + loop_peeling_iters; k++) {
    wf_offset_t offset = offsets[k];

    if (offset < 0) continue;
    offset = wavefront_extend_matches_packed_kernel(wf_aligner,k,offset);
    offsets[k] = offset;
    const wf_offset_t antidiag = WAVEFRONT_ANTIDIAGONAL(k,offset);
    if (max_antidiag < antidiag) max_antidiag = antidiag;
  }

  if (num_diagonals < num_elems) return max_antidiag;

  // Perform one extend step in a SIMD manner, if some diagonal needs more steps
  // to finish the extension, do it in scalar

  for (k=k_min+loop_peeling_iters; k<=k_max;k+=num_elems) {
    svint32_t ks = svindex_s32(k, 1);
    svint32_t sv_offsets = svld1(svptrue_b32(), &offsets[k]);

    // Deactivate null offsets (as they will produce a segfault in the gathers)
    // mask <- cmp less than 0
    svbool_t mask = svcmpge_n_s32(svptrue_b32(), sv_offsets, 0);
    svbool_t original_mask = mask;

    // h = o
    svint32_t h = sv_offsets;
    // v = o - k
    svint32_t v = svsub_z(mask, sv_offsets, ks);

    svuint32_t bases_pattern = svld1_gather_s32offset_u32(mask, (uint32_t*)pattern, v);
    svuint32_t bases_text = svld1_gather_s32offset_u32(mask, (uint32_t*)text, h);

    svuint32_t xor_result = sveor_u32_z(mask, bases_pattern, bases_text);
    xor_result = svrevb_u32_z(mask, xor_result);
    svuint32_t clz_res = svclz_u32_z(mask, xor_result);
    svint32_t Eq = svreinterpret_s32(svlsr_u32_z(mask, clz_res, svdup_u32(3U)));

    // Make sure we don't count beyond the sequence
    // This is not needed as we have sentinels at the end of the sequence.
    // Leaving it as a comment just in case we need it in the future
    // svint32_t remaining_v = svsub_s32_z(mask, svdup_s32(pattern_length), v);
    // svint32_t remaining_h = svsub_s32_z(mask, svdup_s32(text_length), h);
    // Eq = svmin_s32_z(mask, Eq, remaining_v);
    // Eq = svmin_s32_z(mask, Eq, remaining_h);

    sv_offsets = svadd_s32_m(mask, sv_offsets, Eq);

    // Only diagonals that have 4 elements equal (so they have not finished) will continue
    mask = svcmpgt_n_s32(mask, Eq, 3U);

    // This part is not needed if the sequences has sentinels at the end
    // v < pattern_length
    // svbool_t mask_v = svcmplt_n_s32(mask, v, pattern_length);
    // mask = svand_b_z(mask, mask, mask_v);
    // h < text_length
    // svbool_t mask_h = svcmplt_n_s32(mask, h, text_length);
    // mask = svand_b_z(mask, mask, mask_h);

    // Store the partial results
    svst1_s32(original_mask, &offsets[k], sv_offsets);

    // Calculate maximum antiadiagonal. Antiadiagonal = 2*offset - k
    svint32_t curr_antidiags = svlsl_n_s32_m(original_mask, sv_offsets, 1U);
    curr_antidiags = svsub_s32_m(original_mask, curr_antidiags, ks);
    // MAX() reduction
    wf_offset_t curr_max_antidig = (wf_offset_t)svmaxv_s32(original_mask, curr_antidiags);
    if (max_antidiag < curr_max_antidig) max_antidiag = curr_max_antidig;

    // If all diagonals have found a mismatch, we have finished the current
    // iteration, continue with the loop. If not, finish the remaining diagonal
    // in a scalar way
    bool svtest;
    while( (svtest = svptest_any(svptrue_b32(), mask)) ) {
      // Extract the K corresponding to the last (i.e. most significant bit)
      // active element in the predicate
      int32_t curr_k = svlastb_s32(mask, ks);
      int32_t curr_offset = svlastb_s32(mask, sv_offsets);

      // Remove the currently extracted element from the predicate:
      // 1. Create a new predicate with all elements to 1 except the selected
      //    one (using a cmp). Latency: 4 cycles. Throughput: 1 per cycle
      // 2. Perform an AND operation between the two predicates. Latency: 2.
      //    Throughput: 2

      // In cmpne with an immediate, the immediate is limited to -16 to 15, use
      // a full register as we need more range than that. Latency of dup:
      // (latency=3, throughout=1/cycle)
      svbool_t mask_to_remove_sel = svcmpne_s32(mask, ks, svdup_s32(curr_k));
      mask = svand_b_z(mask, mask, mask_to_remove_sel);

      // Scalar extend
      if (curr_offset >= 0) {
        curr_offset = wavefront_extend_matches_packed_kernel(wf_aligner,curr_k,curr_offset);
        offsets[curr_k] = curr_offset;
        const wf_offset_t antidiag = WAVEFRONT_ANTIDIAGONAL(curr_k, curr_offset);
        if (max_antidiag < antidiag) max_antidiag = antidiag;
      } else {
        offsets[curr_k] = WAVEFRONT_OFFSET_NULL;
      }
    } // while (svtest)
  } // for kmin...kmax
  return max_antidiag;
}


FORCE_NO_INLINE bool wavefront_extend_matches_packed_endsfree_sve(
    wavefront_aligner_t* const wf_aligner,
    wavefront_t* const mwavefront,
    const int score,
    const int lo,
    const int hi) {
  // Parameters
  const char* pattern = wf_aligner->sequences.pattern;
  const char* text    = wf_aligner->sequences.text;
  
  wf_offset_t* const offsets = mwavefront->offsets;
  int k_min = lo;
  int k_max = hi;
  uint64_t num_elems = svcntw();

  int k;
  int num_diagonals = k_max-k_min+1;
  int loop_peeling_iters = num_diagonals % num_elems;

  for (k=k_min; k<k_min + loop_peeling_iters; k++) {
    wf_offset_t offset = offsets[k];

    if (offset < 0) continue;
    offset = wavefront_extend_matches_packed_kernel(wf_aligner,k,offset);
    offsets[k] = offset;
    // Check ends-free reaching boundaries
    if (wavefront_termination_endsfree(wf_aligner,mwavefront,score,k,offset)) {
      return true; // Quit (we are done)
    }
  }

  if (num_diagonals < num_elems) return false;

  // Perform one extend step in a SIMD manner, if some diagonal needs more steps
  // to finish the extension, do it in scalar

  for (k=k_min+loop_peeling_iters; k<=k_max;k+=num_elems) {
    svint32_t ks = svindex_s32(k, 1);
    svint32_t sv_offsets = svld1(svptrue_b32(), &offsets[k]);

    // Deactivate null offsets (as they will produce a segfault in the gathers)
    // mask <- cmp less than 0
    svbool_t mask = svcmpge_n_s32(svptrue_b32(), sv_offsets, 0);
    svbool_t original_mask = mask;

    // h = o
    svint32_t h = sv_offsets;
    // v = o - k
    svint32_t v = svsub_z(mask, sv_offsets, ks);

    svuint32_t bases_pattern = svld1_gather_s32offset_u32(mask, (uint32_t*)pattern, v);
    svuint32_t bases_text = svld1_gather_s32offset_u32(mask, (uint32_t*)text, h);

    svuint32_t xor_result = sveor_u32_z(mask, bases_pattern, bases_text);
    xor_result = svrevb_u32_z(mask, xor_result);
    svuint32_t clz_res = svclz_u32_z(mask, xor_result);
    svint32_t Eq = svreinterpret_s32(svlsr_u32_z(mask, clz_res, svdup_u32(3U)));

    // Make sure we don't count beyond the sequence
    // This is not needed as we have sentinels at the end of the sequence.
    // Leaving it as a comment just in case we need it in the future
    // svint32_t remaining_v = svsub_s32_z(mask, svdup_s32(pattern_length), v);
    // svint32_t remaining_h = svsub_s32_z(mask, svdup_s32(text_length), h);
    // Eq = svmin_s32_z(mask, Eq, remaining_v);
    // Eq = svmin_s32_z(mask, Eq, remaining_h);

    sv_offsets = svadd_s32_m(mask, sv_offsets, Eq);

    // Only diagonals that have 4 elements equal (so they have not finished) will continue
    mask = svcmpgt_n_s32(mask, Eq, 3U);

    // This part is not needed if the sequences has sentinels at the end
    // v < pattern_length
    // svbool_t mask_v = svcmplt_n_s32(mask, v, pattern_length);
    // mask = svand_b_z(mask, mask, mask_v);
    // h < text_length
    // svbool_t mask_h = svcmplt_n_s32(mask, h, text_length);
    // mask = svand_b_z(mask, mask, mask_h);

    // Store the partial results
    svst1_s32(original_mask, &offsets[k], sv_offsets);

    // Check if ends-free has reached a boundary, if so, return
    for (int kk=k; k<num_elems; kk++) {
      if (wavefront_termination_endsfree(wf_aligner,mwavefront,score,kk,offsets[kk])) {
        return true; // Quit (we are done)
      }
    }

    // If all diagonals have found a mismatch, we have finished the current
    // iteration, continue with the loop. If not, finish the remaining diagonal
    // in a scalar way
    bool svtest;
    while( (svtest = svptest_any(svptrue_b32(), mask)) ) {
      // Extract the K corresponding to the last (i.e. most significant bit)
      // active element in the predicate
      int32_t curr_k = svlastb_s32(mask, ks);
      int32_t curr_offset = svlastb_s32(mask, sv_offsets);

      // Remove the currently extracted element from the predicate:
      // 1. Create a new predicate with all elements to 1 except the selected
      //    one (using a cmp). Latency: 4 cycles. Throughput: 1 per cycle
      // 2. Perform an AND operation between the two predicates. Latency: 2.
      //    Throughput: 2

      // In cmpne with an immediate, the immediate is limited to -16 to 15, use
      // a full register as we need more range than that. Latency of dup:
      // (latency=3, throughout=1/cycle)
      svbool_t mask_to_remove_sel = svcmpne_s32(mask, ks, svdup_s32(curr_k));
      mask = svand_b_z(mask, mask, mask_to_remove_sel);

      // Scalar extend
      if (curr_offset >= 0) {
        curr_offset = wavefront_extend_matches_packed_kernel(wf_aligner,curr_k,curr_offset);
        offsets[curr_k] = curr_offset;
      // Check ends-free reaching boundaries
      if (wavefront_termination_endsfree(wf_aligner,mwavefront,score,curr_k,curr_offset)) {
        return true; // Quit (we are done)
      }

      } else {
        offsets[curr_k] = WAVEFRONT_OFFSET_NULL;
      }
    } // while (svtest)
  } // for kmin...kmax
  return false; 
}

#endif // __ARM_FEATURE_SVE