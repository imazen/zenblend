//! `f32` math that works with and without `std`.
//!
//! `core` has no `sqrt`/`floor`/`ceil` on `f32`; without `std` these route
//! through `libm`. All three are exact (sqrt is correctly rounded in both),
//! so the two builds produce identical bits.

#[cfg(feature = "std")]
#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn floor(x: f32) -> f32 {
    x.floor()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn floor(x: f32) -> f32 {
    libm::floorf(x)
}

#[cfg(feature = "std")]
#[inline]
pub(crate) fn ceil(x: f32) -> f32 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn ceil(x: f32) -> f32 {
    libm::ceilf(x)
}
