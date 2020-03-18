use crate::{traits::Scalar, utils::MaybeArray};
use std::mem::MaybeUninit;

#[derive(Clone)]
pub(crate) struct MatrixImpl<T: Scalar, const M: usize, const N: usize> {
    data: [[T; M]; N],
}

impl<T: Scalar, const R: usize, const C: usize> MatrixImpl<T, { R }, { C }> {
    // Returns a matrix whose elements are all zero.
    pub(crate) const fn zero() -> Self {
        Self::from_val(T::ZERO)
    }

    /// Returns a matrix whose elements are all equal to the given value.
    pub(crate) const fn from_val(t: T) -> Self {
        unsafe {
            let mut outer = MaybeArray::uninit();
            let mut outer_idx = 0;

            while outer_idx < C {
                let inner = MaybeArray::from_elem(t);

                *outer.get_mut(outer_idx) = MaybeUninit::new(inner.assume_init());
                outer_idx += 1;
            }

            Self {
                data: outer.assume_init(),
            }
        }
    }

    pub(crate) const fn from_array(array: [[T; R]; C]) -> Self {
        unsafe {
            Self {
                data: MaybeArray::from_array(array).assume_init(),
            }
        }
    }

    pub(crate) const fn as_array(&self) -> &[[T; R]; C] {
        &self.data
    }

    pub(crate) const fn as_mut_array(&mut self) -> &mut [[T; R]; C] {
        &mut self.data
    }
}

// The identity matrix only exists when the number of rows and
// columns are equal to each other.
impl<T: Scalar, const N: usize> MatrixImpl<T, { N }, { N }> {
    /// Creates an identtity matrix whose dimensions are N * N
    pub(crate) const fn id() -> Self {
        unsafe {
            let mut outer = MaybeArray::uninit();
            let mut outer_idx = 0;

            while outer_idx < N {
                let mut inner = MaybeArray::uninit();
                let mut inner_idx = 0;

                while inner_idx < N {
                    let val = if inner_idx == outer_idx {
                        T::ONE
                    } else {
                        T::ZERO
                    };

                    *inner.get_mut(inner_idx) = MaybeUninit::new(val);
                    inner_idx += 1;
                }

                *outer.get_mut(outer_idx) = MaybeUninit::new(inner.assume_init());
                outer_idx += 1;
            }

            Self {
                data: outer.assume_init(),
            }
        }
    }
}
