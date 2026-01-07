// SIMD optimizations for slice operations using AVX2
// Optimization #3 from MULTITHREADING_OPTIMIZATION_STUDY.md

use std::mem::MaybeUninit;

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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(crate) unsafe fn fma_slices_uninit_avx2<'a>(
    dst: &'a mut [MaybeUninit<f32>],
    src1: &[f32],
    src2: &[f32],
) -> &'a mut [f32] {
    let len = dst.len();

    // Initialize dst with first chunk (s1[i] * s2[i])
    let mut i = 0;
    while i + 8 <= len {
        let s1_vec = _mm256_loadu_ps(src1.as_ptr().add(i));
        let s2_vec = _mm256_loadu_ps(src2.as_ptr().add(i));
        let res = _mm256_mul_ps(s1_vec, s2_vec);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i) as *mut f32, res);
        i += 8;
    }
    for j in i..len {
        dst.get_unchecked_mut(j)
            .write(*src1.get_unchecked(j) * *src2.get_unchecked(j));
    }

    let dst = &mut *(dst as *mut _ as *mut [f32]);

    // Accumulate remaining chunks with FMA: dst += s1 * s2
    src1[len..]
        .chunks_exact(len)
        .zip(src2[len..].chunks_exact(len))
        .for_each(|(s1_chunk, s2_chunk)| {
            let mut i = 0;
            while i + 8 <= len {
                let d_vec = _mm256_loadu_ps(dst.as_ptr().add(i));
                let s1_vec = _mm256_loadu_ps(s1_chunk.as_ptr().add(i));
                let s2_vec = _mm256_loadu_ps(s2_chunk.as_ptr().add(i));
                // FMA: d = d + (s1 * s2)
                let res = _mm256_fmadd_ps(s1_vec, s2_vec, d_vec);
                _mm256_storeu_ps(dst.as_mut_ptr().add(i), res);
                i += 8;
            }
            for j in i..len {
                *dst.get_unchecked_mut(j) +=
                    *s1_chunk.get_unchecked(j) * *s2_chunk.get_unchecked(j);
            }
        });

    dst
}

// Runtime detection of AVX2 support (cached for performance)
use std::sync::atomic::{AtomicU8, Ordering};

static AVX2_AVAILABLE: AtomicU8 = AtomicU8::new(2); // 0=no, 1=yes, 2=unknown

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn has_avx2() -> bool {
    let cached = AVX2_AVAILABLE.load(Ordering::Relaxed);
    if cached != 2 {
        return cached == 1;
    }

    #[cfg(target_feature = "avx2")]
    {
        AVX2_AVAILABLE.store(1, Ordering::Relaxed);
        true
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        let available = is_x86_feature_detected!("avx2");
        AVX2_AVAILABLE.store(if available { 1 } else { 0 }, Ordering::Relaxed);
        available
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn has_avx2() -> bool {
    false
}
