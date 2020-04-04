/// A trait implmented for types which have some sensible notion
/// of arithmetic, a [`Debug`][1] implementation, and are [`Copy`].
/// This trait is sealed, and can not be implemented outside of
/// `const_linear`. It is implemented for every primitive
/// numeric type.
///
/// # Examples
///
/// ## One and Zero values
/// ```
/// use const_linear::traits::Scalar;
///
/// let x = i32::ZERO;
/// let y = i32::ONE;
///
/// assert_eq!(x - y, -1);
/// ```
///
/// ## Converting to floating point
///
/// ```
/// use const_linear::traits::Scalar;
///
/// assert_eq!(1.to_real::<f64>(), 1 as f64);
/// assert_eq!(1.to_real::<f32>(), 1 as f32);
/// ```
///
/// [1]: std::fmt::Debug
pub trait Scalar: sealed::Ops<Self> {
    /// The "zero" value for this Scalar type. Acts as the
    /// additive identity.
    const ZERO: Self;

    /// The "one" value for this Scalar type. Acts as the
    /// multiplicative identity.
    const ONE: Self;

    fn into_real<R: Real>(self) -> R;
}

/// A trait
pub trait Signed: Scalar + sealed::Ops<Self> {
    fn negate(self) -> Self;

    /// Returns true if self is positive and false if the
    /// number is zero or negative.
    fn is_positive(self) -> bool;

    /// Returns true if self is negative and false if the
    /// number is zero or positive.
    fn is_negative(self) -> bool;
}

/// A trait that is implemented for types that can represent
/// real numbers. This trait is sealed, and can not be
/// implemented outside of `const_linear`. It is implemented
/// for [`f64`] and [`f32`].
pub trait Real: Signed + sealed::Ops<Self> {
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: Self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn ln(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn cbrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn recip(self) -> Self;
    fn exp_m1(self) -> Self;
    fn ln_1p(self) -> Self;

    /// Euler's number, [`e`](https://en.wikipedia.org/wiki/E_(mathematical_constant))
    fn e() -> Self;

    /// The square root of `2`.
    fn sqrt_2() -> Self;

    /// Archimedes' constant, [`π`](https://en.wikipedia.org/wiki/Pi)
    fn pi() -> Self;

    /// `1 / π`
    fn one_over_pi() -> Self;

    /// `2 / π`
    fn two_over_pi() -> Self;

    /// `2 / √π`
    fn two_over_sqrt_pi() -> Self;

    /// `1 / √2`
    fn one_over_sqrt_two() -> Self;

    /// `π / 2`
    fn pi_over_2() -> Self;

    /// `π / 3`
    fn pi_over_3() -> Self;

    /// `π / 4`
    fn pi_over_4() -> Self;

    /// `π / 6`
    fn pi_over_6() -> Self;

    /// `π / 8`
    fn pi_over_8() -> Self;

    /// `ln(2)`
    fn ln_2() -> Self;

    /// `ln(10)`
    fn ln_10() -> Self;

    /// `log2(e)`
    fn log2_e() -> Self;

    /// `log(e)`
    fn log10_e() -> Self;

    fn from_scalar<T: Scalar>(t: T) -> Self;

    /// Tests for approximate equality between two real numbers. This is a
    /// workaround for the limited precision of floating point numbers.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::traits::Real;
    ///
    /// // 0.1 + 0.2 != 0.3 (In floating point arithmetic).
    /// assert_ne!(0.1 + 0.2, 0.3);
    ///
    /// // ... But it is *approximately* equal.
    /// assert!((0.1 + 0.2).approx_eq(0.3, std::f64::EPSILON, 1.0E-15))
    /// ```
    fn approx_eq(self, other: Self, epsilon: Self, max_relative: Self) -> bool;
}

macro_rules! scalar_impls {
    ($($t:ident, $zero:literal, $one:literal),+) => {
        $(
            impl $crate::traits::Scalar for $t {
                const ZERO: Self = $zero;
                const ONE: Self = $one;

                fn into_real<R: Real>(self) -> R {
                    R::from_scalar(self)
                }
            }
        )*
    }
}

macro_rules! signed_impls {
    ($($t:ty),+) => {
        $(
            impl $crate::traits::Signed for $t {
                fn negate(self) -> Self {
                    -self
                }

                fn is_positive(self) -> bool {
                    self > Self::ZERO
                }

                fn is_negative(self) -> bool {
                    self < Self::ZERO
                }
            }
        )*
    };
}

macro_rules! real_impls {
    ($($t:tt, $func:tt),+) => {
        $(
            impl Real for $t {
                fn floor(self) -> Self {
                    <$t>::floor(self)
                }

                fn ceil(self) -> Self {
                    <$t>::ceil(self)
                }

                fn round(self) -> Self {
                    <$t>::round(self)
                }

                fn trunc(self) -> Self {
                    <$t>::trunc(self)
                }

                fn fract(self) -> Self {
                    <$t>::fract(self)
                }

                fn powi(self, n: i32) -> Self {
                    <$t>::powi(self, n)
                }

                fn powf(self, n: Self) -> Self {
                    <$t>::powf(self, n)
                }

                fn sqrt(self) -> Self {
                    <$t>::sqrt(self)
                }

                fn exp(self) -> Self {
                    <$t>::exp(self)
                }

                fn exp2(self) -> Self {
                    <$t>::exp2(self)
                }

                fn ln(self) -> Self {
                    <$t>::ln(self)
                }

                fn log(self, base: Self) -> Self {
                    <$t>::log(self, base)
                }

                fn log2(self) -> Self {
                    <$t>::log2(self)
                }

                fn log10(self) -> Self {
                    <$t>::log10(self)
                }

                fn cbrt(self) -> Self {
                    <$t>::cbrt(self)
                }

                fn sin(self) -> Self {
                    <$t>::sin(self)
                }

                fn cos(self) -> Self {
                    <$t>::cos(self)
                }

                fn tan(self) -> Self {
                    <$t>::tan(self)
                }

                fn asin(self) -> Self {
                    <$t>::asin(self)
                }

                fn acos(self) -> Self {
                    <$t>::acos(self)
                }

                fn atan(self) -> Self {
                    <$t>::atan(self)
                }

                fn atan2(self, other: Self) -> Self {
                    <$t>::atan2(self, other)
                }

                fn sinh(self) -> Self {
                    <$t>::sinh(self)
                }

                fn cosh(self) -> Self {
                    <$t>::cosh(self)
                }

                fn tanh(self) -> Self {
                    <$t>::tanh(self)
                }

                fn asinh(self) -> Self {
                    <$t>::asinh(self)
                }

                fn acosh(self) -> Self {
                    <$t>::acosh(self)
                }

                fn atanh(self) -> Self {
                    <$t>::atanh(self)
                }

                fn recip(self) -> Self {
                    <$t>::recip(self)
                }

                fn exp_m1(self) -> Self {
                    <$t>::exp_m1(self)
                }

                fn ln_1p(self) -> Self {
                    <$t>::ln_1p(self)
                }

                fn e() -> Self {
                    std::$t::consts::E
                }

                fn sqrt_2() -> Self {
                    std::$t::consts::SQRT_2
                }

                fn pi() -> Self {
                    std::$t::consts::PI
                }

                fn one_over_pi() -> Self {
                    std::$t::consts::FRAC_1_PI
                }

                fn two_over_pi() -> Self {
                    std::$t::consts::FRAC_2_PI
                }

                fn two_over_sqrt_pi() -> Self {
                    std::$t::consts::FRAC_2_SQRT_PI
                }

                fn one_over_sqrt_two() -> Self {
                    std::$t::consts::FRAC_1_SQRT_2
                }

                fn pi_over_2() -> Self {
                    std::$t::consts::FRAC_PI_2
                }

                fn pi_over_3() -> Self {
                    std::$t::consts::FRAC_PI_3
                }

                fn pi_over_4() -> Self {
                    std::$t::consts::FRAC_PI_4
                }

                fn pi_over_6() -> Self {
                    std::$t::consts::FRAC_PI_6
                }

                fn pi_over_8() -> Self {
                    std::$t::consts::FRAC_PI_8
                }

                fn ln_2() -> Self {
                    std::$t::consts::LN_2
                }

                fn ln_10() -> Self {
                    std::$t::consts::LN_10
                }

                fn log2_e() -> Self {
                    std::$t::consts::LOG2_E
                }

                fn log10_e() -> Self {
                    std::$t::consts::LOG10_E
                }

                fn from_scalar<T: Scalar>(t: T) -> Self {
                    t.$func()
                }

                // Adapted from the relative_eq method in the approx crate.
                // https://docs.rs/approx/0.3.2/src/approx/relative_eq.rs.html#54-83
                fn approx_eq(self, other: Self, epsilon: Self, max_relative: Self) -> bool {
                    // The special case in which both are *exactly* equal.
                    if self == other {
                        return true;
                    } else if self.is_infinite() || other.is_infinite() || self.is_nan() || other.is_nan() {
                        return false;
                    }

                    let diff = (self - other).abs();

                    if diff <= epsilon {
                        true
                    } else {
                        let abs_self = self.abs();
                        let abs_other = other.abs();

                        let largest = Self::max(abs_self, abs_other);

                        diff < largest * max_relative
                    }
                }
            }
        )*
    }
}

scalar_impls! {
    i8, 0, 1,
    u8, 0, 1,
    i16, 0, 1,
    u16, 0, 1,
    i32, 0, 1,
    u32, 0, 1,
    i64, 0, 1,
    u64, 0, 1,
    i128, 0, 1,
    u128, 0, 1,
    isize, 0, 1,
    usize, 0, 1,
    f32, 0.0, 1.0,
    f64, 0.0, 1.0
}

signed_impls! {
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64
}

real_impls! {
    f64, to_f64,
    f32, to_f32
}

mod sealed {
    use std::cmp::PartialEq;
    use std::fmt::Debug;
    use std::ops::*;

    pub trait Ops<T>:
        Copy
        + Clone
        + Debug
        + Add<Output = T>
        + AddAssign<T>
        + Div<Output = T>
        + DivAssign<T>
        + Mul<Output = T>
        + MulAssign<T>
        + Rem<Output = T>
        + RemAssign<T>
        + Sub<Output = T>
        + SubAssign<T>
        + PartialEq<T>
        + ToFloat
    {
    }

    pub trait ToFloat {
        fn to_f64(self) -> f64;

        fn to_f32(self) -> f32;
    }

    macro_rules! ops_impls {
        ($($t:ty),+) => {
            $(
                impl self::ToFloat for $t {
                    fn to_f64(self) -> f64 {
                        self as f64
                    }

                    fn to_f32(self) -> f32 {
                        self as f32
                    }
                }

                impl self::Ops<$t> for $t {}
            )*
        }
    }

    ops_impls! {
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        i128,
        u128,
        isize,
        usize,
        f32,
        f64
    }
}
