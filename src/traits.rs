/// A value which has some sensible notion of arithmetic,
/// a [`Debug`][1] implementation, and is [`Copy`]. This
/// trait is sealed, and can not be implemented outside of
/// `const_linear`.
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
/// assert_eq!(1.to_f64(), 1 as f64);
/// assert_eq!(1.to_f32(), 1 as f32);
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

    /// Converts `T` to an [`f64`]. Equivalent to `T as f64`.
    fn to_f64(self) -> f64;

    /// Converts `T` to an [`f32`]. Equivalent to `T as f32`.
    fn to_f32(self) -> f32;
}

/// A value which is a scalar value and is also
/// capable of representing real numbers. This trait
/// is sealed, and can not be implemented outside of
/// `const_linear`.
pub trait Real: Scalar + sealed::Ops<Self> {
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
    ($($t:ty, $zero:literal, $one:literal),+) => {
        $(
            impl $crate::traits::Scalar for $t {
                const ZERO: Self = $zero;
                const ONE: Self = $one;

                fn to_f64(self) -> f64 {
                    self as f64
                }

                fn to_f32(self) -> f32 {
                    self as f32
                }
            }

            // Unfortunately, it's not possible to do a blanket impl, so this is a workaround
            // for all types which impl Scalar. Inverse of impl Mul<T> for Matrix<T>
            impl<const M: usize, const N: usize> ::std::ops::Mul<$crate::Matrix<$t, { M }, { N }>> for $t {
                type Output = $crate::core::Matrix<$t, { M }, { N }>;

                fn mul(self, mut rhs: Self::Output) -> Self::Output {
                    for elem in rhs.column_iter_mut() {
                        *elem *= self;
                    }

                    rhs
                }
            }
        )*
    }
}

macro_rules! real_impls {
    ($($t:ident),+) => {
        $(
            impl ::std::ops::Mul<$crate::geometry::angle::Angle<$t>> for $t {
                type Output = $crate::geometry::angle::Angle<$t>;

                fn mul(
                    self,
                    rhs: $crate::geometry::angle::Angle<$t>
                ) -> $crate::geometry::angle::Angle<$t> {
                    match rhs {
                        $crate::geometry::angle::Angle::Deg(_) =>
                            $crate::geometry::angle::Angle::Deg(rhs.val() * self),
                        $crate::geometry::angle::Angle::Rad(_) =>
                            $crate::geometry::angle::Angle::Rad(rhs.val() * self),
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

impl Real for f64 {
    fn e() -> Self {
        std::f64::consts::E
    }

    fn sqrt_2() -> Self {
        std::f64::consts::SQRT_2
    }

    fn pi() -> Self {
        std::f64::consts::PI
    }

    fn one_over_pi() -> Self {
        std::f64::consts::FRAC_1_PI
    }

    fn two_over_pi() -> Self {
        std::f64::consts::FRAC_2_PI
    }

    fn two_over_sqrt_pi() -> Self {
        std::f64::consts::FRAC_2_SQRT_PI
    }

    fn one_over_sqrt_two() -> Self {
        std::f64::consts::FRAC_1_SQRT_2
    }

    fn pi_over_2() -> Self {
        std::f64::consts::FRAC_PI_2
    }

    fn pi_over_3() -> Self {
        std::f64::consts::FRAC_PI_3
    }

    fn pi_over_4() -> Self {
        std::f64::consts::FRAC_PI_4
    }

    fn pi_over_6() -> Self {
        std::f64::consts::FRAC_PI_6
    }

    fn pi_over_8() -> Self {
        std::f64::consts::FRAC_PI_8
    }

    fn ln_2() -> Self {
        std::f64::consts::LN_2
    }

    fn ln_10() -> Self {
        std::f64::consts::LN_10
    }

    fn log2_e() -> Self {
        std::f64::consts::LOG2_E
    }

    fn log10_e() -> Self {
        std::f64::consts::LOG10_E
    }

    fn from_scalar<T: Scalar>(t: T) -> Self {
        t.to_f64()
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

impl Real for f32 {
    fn e() -> Self {
        std::f32::consts::E
    }

    fn sqrt_2() -> Self {
        std::f32::consts::SQRT_2
    }

    fn pi() -> Self {
        std::f32::consts::PI
    }

    fn one_over_pi() -> Self {
        std::f32::consts::FRAC_1_PI
    }

    fn two_over_pi() -> Self {
        std::f32::consts::FRAC_2_PI
    }

    fn two_over_sqrt_pi() -> Self {
        std::f32::consts::FRAC_2_SQRT_PI
    }

    fn one_over_sqrt_two() -> Self {
        std::f32::consts::FRAC_1_SQRT_2
    }

    fn pi_over_2() -> Self {
        std::f32::consts::FRAC_PI_2
    }

    fn pi_over_3() -> Self {
        std::f32::consts::FRAC_PI_3
    }

    fn pi_over_4() -> Self {
        std::f32::consts::FRAC_PI_4
    }

    fn pi_over_6() -> Self {
        std::f32::consts::FRAC_PI_6
    }

    fn pi_over_8() -> Self {
        std::f32::consts::FRAC_PI_8
    }

    fn ln_2() -> Self {
        std::f32::consts::LN_2
    }

    fn ln_10() -> Self {
        std::f32::consts::LN_10
    }

    fn log2_e() -> Self {
        std::f32::consts::LOG2_E
    }

    fn log10_e() -> Self {
        std::f32::consts::LOG10_E
    }

    fn from_scalar<T: Scalar>(t: T) -> Self {
        t.to_f32()
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

real_impls! {
    f64,
    f32
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
    {
    }

    macro_rules! ops_impls {
        ($($t:ty),+) => {
            $(
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
