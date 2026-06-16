//! Small numerical helpers shared between the backend and the sampler.

/// In-place, numerically-stable softmax over `x`.
///
/// We subtract the maximum before exponentiating so that large logits don't
/// overflow to `inf`. A no-op on an empty slice.
pub(crate) fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SiLU / "swish" activation: `x * sigmoid(x)`.
#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}
