// SIMD optimizations for slice operations using AVX2
// Optimization #3 from MULTITHREADING_OPTIMIZATION_STUDY.md

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-optimized version of sub_slice
/// Performs: lhs[i] -= rhs[i] for all i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn sub_slice_avx2(lhs: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());

    let len = lhs.len();
    let mut i = 0;

    // Process 8 elements at a time with AVX2
    while i + 8 <= len {
        let l_vec = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let r_vec = _mm256_loadu_ps(rhs.as_ptr().add(i));
        let res = _mm256_sub_ps(l_vec, r_vec);
        _mm256_storeu_ps(lhs.as_mut_ptr().add(i), res);
        i += 8;
    }

    // Handle remaining elements
    for j in i..len {
        *lhs.get_unchecked_mut(j) -= *rhs.get_unchecked(j);
    }
}

/// AVX2-optimized version of mul_slice
/// Performs: lhs[i] *= rhs[i] for all i
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn mul_slice_avx2(lhs: &mut [f32], rhs: &[f32]) {
    debug_assert_eq!(lhs.len(), rhs.len());

    let len = lhs.len();
    let mut i = 0;

    // Process 8 elements at a time with AVX2
    while i + 8 <= len {
        let l_vec = _mm256_loadu_ps(lhs.as_ptr().add(i));
        let r_vec = _mm256_loadu_ps(rhs.as_ptr().add(i));
        let res = _mm256_mul_ps(l_vec, r_vec);
        _mm256_storeu_ps(lhs.as_mut_ptr().add(i), res);
        i += 8;
    }

    // Handle remaining elements
    for j in i..len {
        *lhs.get_unchecked_mut(j) *= *rhs.get_unchecked(j);
    }
}

/// AVX2-optimized version of sum_slices_uninit
/// Performs: dst = sum of all chunks of src
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn sum_slices_uninit_avx2<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();

    // Initialize dst with first chunk
    let mut i = 0;
    while i + 8 <= len {
        let s_vec = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i) as *mut f32, s_vec);
        i += 8;
    }
    for j in i..len {
        dst.get_unchecked_mut(j).write(*src.get_unchecked(j));
    }

    let dst = &mut *(dst as *mut _ as *mut [f32]);

    // Sum remaining chunks
    src[len..].chunks_exact(len).for_each(|chunk| {
        let mut i = 0;
        while i + 8 <= len {
            let d_vec = _mm256_loadu_ps(dst.as_ptr().add(i));
            let s_vec = _mm256_loadu_ps(chunk.as_ptr().add(i));
            let res = _mm256_add_ps(d_vec, s_vec);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), res);
            i += 8;
        }
        for j in i..len {
            *dst.get_unchecked_mut(j) += *chunk.get_unchecked(j);
        }
    });

    dst
}

/// AVX2-optimized version of fma_slices_uninit (THE MOST CRITICAL OPERATION!)
/// Performs: dst = sum of (src1[i] * src2[i]) across all chunks
/// This is the heart of CFR calculation and benefits massively from FMA instructions
///
/// Uses 4x loop unrolling to process 32 elements per iteration, hiding FMA latency
/// by keeping 4 independent operations in flight.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn fma_slices_uninit_avx2<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src1: &[f32],
    src2: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();

    // Initialize dst with first chunk (s1[i] * s2[i]) - 4x unrolled
    let mut i = 0;
    while i + 32 <= len {
        let s1_0 = _mm256_loadu_ps(src1.as_ptr().add(i));
        let s2_0 = _mm256_loadu_ps(src2.as_ptr().add(i));
        let s1_1 = _mm256_loadu_ps(src1.as_ptr().add(i + 8));
        let s2_1 = _mm256_loadu_ps(src2.as_ptr().add(i + 8));
        let s1_2 = _mm256_loadu_ps(src1.as_ptr().add(i + 16));
        let s2_2 = _mm256_loadu_ps(src2.as_ptr().add(i + 16));
        let s1_3 = _mm256_loadu_ps(src1.as_ptr().add(i + 24));
        let s2_3 = _mm256_loadu_ps(src2.as_ptr().add(i + 24));

        let res0 = _mm256_mul_ps(s1_0, s2_0);
        let res1 = _mm256_mul_ps(s1_1, s2_1);
        let res2 = _mm256_mul_ps(s1_2, s2_2);
        let res3 = _mm256_mul_ps(s1_3, s2_3);

        _mm256_storeu_ps(dst.as_mut_ptr().add(i) as *mut f32, res0);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8) as *mut f32, res1);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16) as *mut f32, res2);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24) as *mut f32, res3);
        i += 32;
    }
    // Handle remaining 8-element blocks
    while i + 8 <= len {
        let s1_vec = _mm256_loadu_ps(src1.as_ptr().add(i));
        let s2_vec = _mm256_loadu_ps(src2.as_ptr().add(i));
        let res = _mm256_mul_ps(s1_vec, s2_vec);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i) as *mut f32, res);
        i += 8;
    }
    // Handle remaining scalar elements
    for j in i..len {
        dst.get_unchecked_mut(j)
            .write(*src1.get_unchecked(j) * *src2.get_unchecked(j));
    }

    let dst = &mut *(dst as *mut _ as *mut [f32]);

    // Accumulate remaining chunks with FMA: dst += s1 * s2 - 4x unrolled
    let num_chunks = (src1.len() - len) / len;
    let s1_rest = &src1[len..];
    let s2_rest = &src2[len..];

    for chunk_idx in 0..num_chunks {
        let s1_chunk = &s1_rest[chunk_idx * len..];
        let s2_chunk = &s2_rest[chunk_idx * len..];

        let mut i = 0;
        // 4x unrolled FMA loop for better instruction-level parallelism
        while i + 32 <= len {
            // Load dst values
            let d0 = _mm256_loadu_ps(dst.as_ptr().add(i));
            let d1 = _mm256_loadu_ps(dst.as_ptr().add(i + 8));
            let d2 = _mm256_loadu_ps(dst.as_ptr().add(i + 16));
            let d3 = _mm256_loadu_ps(dst.as_ptr().add(i + 24));

            // Load src1 values
            let s1_0 = _mm256_loadu_ps(s1_chunk.as_ptr().add(i));
            let s1_1 = _mm256_loadu_ps(s1_chunk.as_ptr().add(i + 8));
            let s1_2 = _mm256_loadu_ps(s1_chunk.as_ptr().add(i + 16));
            let s1_3 = _mm256_loadu_ps(s1_chunk.as_ptr().add(i + 24));

            // Load src2 values
            let s2_0 = _mm256_loadu_ps(s2_chunk.as_ptr().add(i));
            let s2_1 = _mm256_loadu_ps(s2_chunk.as_ptr().add(i + 8));
            let s2_2 = _mm256_loadu_ps(s2_chunk.as_ptr().add(i + 16));
            let s2_3 = _mm256_loadu_ps(s2_chunk.as_ptr().add(i + 24));

            // FMA: d = d + (s1 * s2) - 4 independent operations
            let res0 = _mm256_fmadd_ps(s1_0, s2_0, d0);
            let res1 = _mm256_fmadd_ps(s1_1, s2_1, d1);
            let res2 = _mm256_fmadd_ps(s1_2, s2_2, d2);
            let res3 = _mm256_fmadd_ps(s1_3, s2_3, d3);

            // Store results
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), res0);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), res1);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 16), res2);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i + 24), res3);

            i += 32;
        }
        // Handle remaining 8-element blocks
        while i + 8 <= len {
            let d_vec = _mm256_loadu_ps(dst.as_ptr().add(i));
            let s1_vec = _mm256_loadu_ps(s1_chunk.as_ptr().add(i));
            let s2_vec = _mm256_loadu_ps(s2_chunk.as_ptr().add(i));
            let res = _mm256_fmadd_ps(s1_vec, s2_vec, d_vec);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), res);
            i += 8;
        }
        // Handle remaining scalar elements
        for j in i..len {
            *dst.get_unchecked_mut(j) +=
                *s1_chunk.get_unchecked(j) * *s2_chunk.get_unchecked(j);
        }
    }

    dst
}
