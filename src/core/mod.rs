pub(crate) mod boxed;
pub(crate) mod imp;
pub(crate) mod iter;
pub(crate) mod matrix;

pub use boxed::*;
pub use iter::*;
pub use matrix::*;

use crate::traits::Scalar;

pub(crate) fn mm_dot<'a, T: Scalar, const M: usize, const N: usize, const P: usize>(
    v: Row<'a, T, { M }, { N }>,
    u: Column<'a, T, { N }, { P }>,
) -> T {
    let mut out = T::ZERO;

    v.zip(u).for_each(|(&x, &y)| out += x * y);

    out
}
