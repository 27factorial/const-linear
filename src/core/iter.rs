use crate::{core::Matrix, traits::Scalar};
use std::ops::{Index, IndexMut};

/// A column-wise iterator over the elements of a matrix.
pub struct ColumnIter<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a Matrix<T, { M }, { N }>,
    pub(super) row: usize,
    pub(super) col: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator
    for ColumnIter<'a, T, { M }, { N }>
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.col == N {
            None
        } else {
            let ret = &self.matrix.index((self.row, self.col));

            if self.row == M - 1 {
                self.row = 0;
                self.col += 1;
            } else {
                self.row += 1;
            }

            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (M * N, Some(M * N))
    }
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> ExactSizeIterator
    for ColumnIter<'a, T, { M }, { N }>
{
}

/// A column-wise iterator over mutable references to the elements of a matrix.
pub struct ColumnIterMut<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a mut Matrix<T, { M }, { N }>,
    pub(super) row: usize,
    pub(super) col: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator
    for ColumnIterMut<'a, T, { M }, { N }>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.col == N {
            None
        } else {
            let ret = self.matrix.index_mut((self.row, self.col)) as *mut T;

            if self.row == M - 1 {
                self.row = 0;
                self.col += 1;
            } else {
                self.row += 1;
            }

            // SAFETY:
            // 1. The pointer can never be invalid, since the Matrix can not
            // be dropped due to it being mutably borrowed.
            // 2. The code above only ever gets one mutable borrow to each
            // element, so there is no pointer aliasing.
            unsafe { Some(&mut *ret) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (M * N, Some(M * N))
    }
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> ExactSizeIterator
    for ColumnIterMut<'a, T, { M }, { N }>
{
}

/// A row-wise iterator over the elements of a matrix.
pub struct RowIter<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a Matrix<T, { M }, { N }>,
    pub(super) row: usize,
    pub(super) col: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator for RowIter<'a, T, { M }, { N }> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.row == M {
            None
        } else {
            let ret = &self.matrix.index((self.row, self.col));

            if self.col == N - 1 {
                self.col = 0;
                self.row += 1;
            } else {
                self.col += 1;
            }

            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (M * N, Some(M * N))
    }
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> ExactSizeIterator
    for RowIter<'a, T, { M }, { N }>
{
}

/// A row-wise iterator over mutable references to the elements of a matrix.
pub struct RowIterMut<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a mut Matrix<T, { M }, { N }>,
    pub(super) row: usize,
    pub(super) col: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator
    for RowIterMut<'a, T, { M }, { N }>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.row == M {
            None
        } else {
            let ret = self.matrix.index_mut((self.row, self.col)) as *mut T;

            if self.col == N - 1 {
                self.col = 0;
                self.row += 1;
            } else {
                self.col += 1;
            }

            // SAFETY:
            // 1. The pointer can never be invalid, since the Matrix can not
            // be dropped due to it being mutably borrowed.
            // 2. The code above only ever gets one mutable borrow to each
            // element, so there is no pointer aliasing.
            unsafe { Some(&mut *ret) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (M * N, Some(M * N))
    }
}

impl<'a, T: Scalar, const M: usize, const N: usize> ExactSizeIterator
    for RowIterMut<'a, T, { M }, { N }>
{
}

/// An iterator over elements in one column of a matrix.
pub struct Column<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a Matrix<T, { M }, { N }>,
    pub(super) column: usize,
    pub(super) idx: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator for Column<'a, T, { M }, { N }> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.idx == M {
            None
        } else {
            let ret = &self.matrix.index((self.idx, self.column));

            self.idx += 1;

            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (M, Some(M))
    }
}

impl<'a, T: Scalar, const M: usize, const N: usize> ExactSizeIterator
    for Column<'a, T, { M }, { N }>
{
}

/// An iterator over mutable references to elements in one column of a matrix.
pub struct ColumnMut<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a mut Matrix<T, { M }, { N }>,
    pub(super) column: usize,
    pub(super) idx: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator
    for ColumnMut<'a, T, { M }, { N }>
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.idx == M {
            None
        } else {
            let ret = self.matrix.index_mut((self.idx, self.column)) as *mut T;

            self.idx += 1;

            unsafe { Some(&mut *ret) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (M, Some(M))
    }
}

impl<'a, T: Scalar, const M: usize, const N: usize> ExactSizeIterator
    for ColumnMut<'a, T, { M }, { N }>
{
}

/// An iterator over elements in one row of a matrix.
pub struct Row<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a Matrix<T, { M }, { N }>,
    pub(super) row: usize,
    pub(super) idx: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator for Row<'a, T, { M }, { N }> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        if self.idx == N {
            None
        } else {
            let ret = &self.matrix.index((self.row, self.idx));

            self.idx += 1;

            Some(ret)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N, Some(N))
    }
}

impl<'a, T: Scalar, const M: usize, const N: usize> ExactSizeIterator for Row<'a, T, { M }, { N }> {}

/// An iterator over mutable references to elements in one row of a matrix.
pub struct RowMut<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    pub(super) matrix: &'a mut Matrix<T, { M }, { N }>,
    pub(super) row: usize,
    pub(super) idx: usize,
}

impl<'a, T: Scalar + 'a, const M: usize, const N: usize> Iterator for RowMut<'a, T, { M }, { N }> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        if self.idx == N {
            None
        } else {
            let ret = self.matrix.index_mut((self.row, self.idx)) as *mut T;

            self.idx += 1;

            unsafe { Some(&mut *ret) }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (N, Some(N))
    }
}

impl<'a, T: Scalar, const M: usize, const N: usize> ExactSizeIterator
    for RowMut<'a, T, { M }, { N }>
{
}
