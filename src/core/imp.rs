use crate::{traits::Scalar, utils::MaybeArray};
use std::fmt;
use std::mem::MaybeUninit;

#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct MatrixImpl<T: Scalar, const M: usize, const N: usize> {
    data: [[T; M]; N],
}

impl<T: Scalar, const M: usize, const N: usize> MatrixImpl<T, { M }, { N }> {
    /// Returns a matrix whose elements are all zero.
    pub(crate) const fn zero() -> Self {
        Self::from_val(T::ZERO)
    }

    /// Returns a matrix whose elements are all equal to the given value.
    pub(crate) const fn from_val(t: T) -> Self {
        unsafe {
            let mut outer = MaybeArray::uninit();
            let mut outer_idx = 0;

            while outer_idx < N {
                let inner = MaybeArray::from_elem(t);

                *outer.get_mut(outer_idx) = MaybeUninit::new(inner.assume_init());
                outer_idx += 1;
            }

            Self {
                data: outer.assume_init(),
            }
        }
    }

    pub(crate) const fn from_array(array: [[T; M]; N]) -> Self {
        unsafe {
            Self {
                data: MaybeArray::from_array(array).assume_init(),
            }
        }
    }

    pub(crate) const fn as_array(&self) -> &[[T; M]; N] {
        &self.data
    }

    pub(crate) const fn as_mut_array(&mut self) -> &mut [[T; M]; N] {
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

impl<T: Scalar, const M: usize, const N: usize> fmt::Debug for MatrixImpl<T, { M }, { N }> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list()
            .entries(self.data.iter().map(|arr| &arr[..]))
            .finish()
    }
}
