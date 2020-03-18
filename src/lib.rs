#![allow(incomplete_features)]
#![feature(const_generics)]
#![feature(const_fn)]
#![feature(const_loop)]
#![feature(const_if_match)]
#![feature(const_fn_union)]
#![feature(const_mut_refs)]
#![feature(const_raw_ptr_deref)]
#![feature(untagged_unions)]
#![feature(const_raw_ptr_to_usize_cast)]
#![feature(const_slice_from_raw_parts)]

/// Constructs a matrix from a type, a value, or
/// a series of values.
///
/// # Examples
///
/// ## Identity matrix
/// An invocation in the form `matrix![T; N]`
/// creates an N x N identity matrix of type `T`.
///
/// ```
/// use const_linear::matrix;
///
/// let expected = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/// let m = matrix![i32; 3];
///
/// for (x, y) in m.column_iter().zip(expected.iter()) {
///     assert_eq!(x, y);
/// }
/// ```
///
/// ## Matrix from value
///
/// An invocation in the form `matrix![x; M, N]` creates
/// an M x N matrix filled with the value `x`.
///
/// ```
/// use const_linear::matrix;
///
/// let expected = [1; 16];
///
/// let m = matrix![1; 4, 4];
///
/// for (x, y) in m.column_iter().zip(expected.iter()) {
///     assert_eq!(x, y);
/// }
/// ```
///
/// ## Matrix from columns
/// An invocation in the form
/// ```text
/// matrix![
///     a, b, c, d, ...;
///     e, f, g, h, ...;
///     ...
/// ]
/// ```
/// creates a matrix which has columns `[a, b, c, d, ...], [e, f, g, h...], ...`
/// A semicolon indicates the end of a column.
///
/// ```
/// use const_linear::matrix;
///
/// let m = matrix![
///     1, 2, 3;
///     4, 5, 6;
///     7, 8, 9;
/// ];
///
/// assert_eq!(m.det(), 0.0);
/// ```
#[macro_export]
macro_rules! matrix {
    ($t:ty; $n:expr) => {
        $crate::matrix::Matrix::<$t, { $n }, { $n }>::id()
    };
    ($val:expr; $rows:expr, $cols:expr) => {
        $crate::matrix::Matrix::<_,  { $rows },  { $cols }>::from_val($val)
    };
    ($($($elem:expr),+);+$(;)?) => {
        $crate::matrix::Matrix::from_array([$([$($elem),*]),*])
    };
}

/// Constructs a vector from either a value or a series of
/// values.
///
/// # Examples
///
/// ## Vector from value
///
/// An invocation in the form `matrix![x; N]` creates
/// an N dimensional vector filled with the value `x`.
///
/// ```
/// use const_linear::vector;
///
/// let expected = [1; 3];
///
/// let v = vector![1; 3];
///
/// for (x, y) in v.column_iter().zip(expected.iter()) {
///     assert_eq!(x, y);
/// }
/// ```
///
/// ## Vector from a series of values
///
/// An invocation in the form `vector![x, y, z, w, ...]`
/// creates a vector with the elements `[x, y, z, w, ...]`.
///
/// ```
/// use const_linear::vector;
///
/// let v = vector![1, 2, 3, 4];
///
/// assert_eq!(v.length(), f64::sqrt(30.0));
///
/// ```
#[macro_export]
macro_rules! vector {
    ($val:expr; $dim:expr) => {
        $crate::Vector::<_, { $dim }>::from_val($val)
    };
    ($($elem:expr),+$(,)?) => {
        $crate::Vector::from_array([[$($elem),*]])
    };
}

pub mod matrix;
pub mod traits;
pub(crate) mod utils;

pub use crate::{
    matrix::*,
    traits::{Real as __Real, Scalar as __Scalar},
};
