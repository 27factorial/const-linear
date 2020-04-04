use crate::{
    core::{
        iter::*,
        matrix::{Matrix, SquareMatrix},
    },
    traits::{Real, Scalar, Signed},
};
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoxedMatrix<T: Scalar, const M: usize, const N: usize> {
    data: Box<Matrix<T, { M }, { N }>>,
}

impl<T: Scalar, const M: usize, const N: usize> BoxedMatrix<T, { M }, { N }> {
    pub fn zero() -> Self {
        Self {
            data: box Matrix::zero(),
        }
    }

    pub fn from_val(t: T) -> Self {
        Self {
            data: box Matrix::from_val(t),
        }
    }

    pub fn from_array(array: [[T; M]; N]) -> Self {
        Self {
            data: box Matrix::from_array(array),
        }
    }

    pub const fn as_array(&self) -> &[[T; M]; N] {
        self.data.as_array()
    }

    pub const fn as_mut_array(&mut self) -> &mut [[T; M]; N] {
        self.data.as_mut_array()
    }

    pub const fn as_ptr(&self) -> *const T {
        self.as_array() as *const _ as _
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_array() as *mut _ as _
    }

    pub const fn dimensions(&self) -> (usize, usize) {
        (M, N)
    }

    pub const fn column_iter(&self) -> ColumnIter<'_, T, { M }, { N }> {
        ColumnIter {
            matrix: &self.data,
            row: 0,
            col: 0,
        }
    }

    pub const fn column_iter_mut(&mut self) -> ColumnIterMut<'_, T, { M }, { N }> {
        ColumnIterMut {
            matrix: &mut self.data,
            row: 0,
            col: 0,
        }
    }

    pub const fn row_iter(&self) -> RowIter<'_, T, { M }, { N }> {
        RowIter {
            matrix: &self.data,
            row: 0,
            col: 0,
        }
    }

    pub const fn row_iter_mut(&mut self) -> RowIterMut<'_, T, { M }, { N }> {
        RowIterMut {
            matrix: &mut self.data,
            row: 0,
            col: 0,
        }
    }

    pub const fn column(&self, column: usize, n: usize) -> Column<'_, T, { M }, { N }> {
        Column {
            matrix: &self.data,
            column,
            idx: n,
        }
    }

    pub const fn column_mut(&mut self, column: usize, n: usize) -> ColumnMut<'_, T, { M }, { N }> {
        ColumnMut {
            matrix: &mut self.data,
            column,
            idx: n,
        }
    }

    pub const fn row(&self, row: usize, n: usize) -> Row<'_, T, { M }, { N }> {
        Row {
            matrix: &self.data,
            row,
            idx: n,
        }
    }

    pub const fn row_mut(&mut self, row: usize, n: usize) -> RowMut<'_, T, { M }, { N }> {
        RowMut {
            matrix: &mut *self.data,
            row,
            idx: n,
        }
    }

    pub fn scale(mut self, s: T) -> Self {
        self.scale_mut(s);
        self
    }

    pub fn scale_mut(&mut self, s: T) {
        self.data.scale_mut(s);
    }

    pub fn into_real<R: Real>(self) -> BoxedMatrix<R, { M }, { N }> {
        BoxedMatrix {
            data: box self.data.into_real(),
        }
    }

    pub fn transpose(&self) -> BoxedMatrix<T, { N }, { M }> {
        BoxedMatrix {
            data: box self.data.transpose(),
        }
    }

    pub fn into_inner(self) -> Box<Matrix<T, { M }, { N }>> {
        self.data
    }
}

impl<T: Signed, const M: usize, const N: usize> BoxedMatrix<T, { M }, { N }> {
    pub fn negate(mut self) -> Self {
        self.negate_mut();
        self
    }

    pub fn negate_mut(&mut self) {
        self.scale_mut(T::ONE.negate());
    }
}

impl<T: Real, const M: usize, const N: usize> BoxedMatrix<T, { M }, { N }> {
    pub fn gauss(mut self) -> Self {
        self.gauss_mut();
        self
    }

    pub fn gauss_mut(&mut self) {
        self.data.gauss_mut();
    }
}

pub type BoxedSquareMatrix<T, const N: usize> = BoxedMatrix<T, { N }, { N }>;
pub type BoxedHomogeneousMatrix<T, const N: usize> = BoxedSquareMatrix<T, { N + 1 }>;

impl<T: Scalar, const N: usize> BoxedSquareMatrix<T, { N }> {
    pub fn id() -> Self {
        Self {
            data: box SquareMatrix::id(),
        }
    }

    pub fn into_homogeneous(self) -> BoxedHomogeneousMatrix<T, { N }> {
        let mut out = BoxedHomogeneousMatrix::id();

        for i in 0..N {
            out.column_mut(i, 0)
                .zip(self.column(i, 0))
                .for_each(|(val, new)| *val = *new);
        }

        out
    }
}

impl<T: Real, const N: usize> BoxedSquareMatrix<T, { N }> {
    pub fn det(&self) -> T {
        match N {
            // We need to do this part separately from the
            // normal Matrix impl because it would be
            // possible to cause a stack overflow if we
            // used the normal version, since the matrix
            // is cloned onto the stack in that case.
            n if n >= 5 => {
                let matrix = self.clone().gauss();

                let mut prod = T::ONE;

                for i in 0..N {
                    prod *= matrix.data[(i, i)];
                }

                prod
            }
            _ => self.data.det(),
        }
    }
}

// ========================= BoxedMatrix Trait Impls

// Indexing (in the order (row, col))
impl<T: Scalar, const M: usize, const N: usize> Index<(usize, usize)>
    for BoxedMatrix<T, { M }, { N }>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        self.data.index(index)
    }
}

impl<T: Scalar, const M: usize, const N: usize> IndexMut<(usize, usize)>
    for BoxedMatrix<T, { M }, { N }>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

// Addition
impl<T: Scalar, const M: usize, const N: usize> Add for BoxedMatrix<T, { M }, { N }> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data + rhs.data,
        }
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add for &'_ BoxedMatrix<T, { M }, { N }> {
    type Output = BoxedMatrix<T, { M }, { N }>;

    fn add(self, rhs: Self) -> Self::Output {
        BoxedMatrix {
            data: &self.data + &rhs.data,
        }
    }
}

// Subtraction
impl<T: Scalar, const M: usize, const N: usize> Sub for BoxedMatrix<T, { M }, { N }> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            data: self.data - rhs.data,
        }
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub for &'_ BoxedMatrix<T, { M }, { N }> {
    type Output = BoxedMatrix<T, { M }, { N }>;

    fn sub(self, rhs: Self) -> Self::Output {
        BoxedMatrix {
            data: &self.data - &rhs.data,
        }
    }
}

// Multiplication
impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<BoxedMatrix<T, { N }, { P }>>
    for BoxedMatrix<T, { M }, { N }>
{
    type Output = BoxedMatrix<T, { M }, { P }>;

    fn mul(self, rhs: BoxedMatrix<T, { N }, { P }>) -> Self::Output {
        BoxedMatrix {
            data: self.data * rhs.data,
        }
    }
}

impl<T: Scalar, const M: usize, const N: usize, const P: usize>
    Mul<&'_ BoxedMatrix<T, { N }, { P }>> for &'_ BoxedMatrix<T, { M }, { N }>
{
    type Output = BoxedMatrix<T, { M }, { P }>;

    fn mul(self, rhs: &'_ BoxedMatrix<T, { N }, { P }>) -> Self::Output {
        BoxedMatrix {
            data: &self.data * &rhs.data,
        }
    }
}

// Negation
impl<T: Signed, const M: usize, const N: usize> Neg for BoxedMatrix<T, { M }, { N }> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            data: self.data.neg(),
        }
    }
}

impl<T: Signed, const M: usize, const N: usize> Neg for &'_ BoxedMatrix<T, { M }, { N }> {
    type Output = BoxedMatrix<T, { M }, { N }>;

    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}
