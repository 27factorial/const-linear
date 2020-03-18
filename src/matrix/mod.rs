mod imp;

use crate::{
    traits::{Real, Scalar},
    utils::ConstVec,
};
use imp::MatrixImpl;
use std::{
    cmp::PartialEq,
    fmt,
    ops::{Add, Div, Index, IndexMut, Mul, Sub},
};

fn mm_dot<'a, T: Scalar, const M: usize, const N: usize, const P: usize>(
    v: Row<'a, T, { M }, { N }>,
    u: Column<'a, T, { N }, { P }>,
) -> T {
    let mut out = T::ZERO;

    v.zip(u).for_each(|(&x, &y)| out += x * y);

    out
}

// ========================= Matrix

/// An M * N column-major Matrix of scalar values backed by an array. Scalar
/// refers to anything that implements the [`Scalar`] trait, which is any value
/// that is [`Copy`] and has a notion of arithmetic. However, many operations,
/// such as the determinant and Gaussian elimination may not be possible
/// with integer values.
#[derive(Clone)]
pub struct Matrix<T: Scalar, const M: usize, const N: usize> {
    data: MatrixImpl<T, { M }, { N }>,
}

impl<T: Scalar, const M: usize, const N: usize> Matrix<T, { M }, { N }> {
    /// Constructs a matrix filled with all zero values.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[0.0, 0.0], [0.0, 0.0]];
    /// let a = Matrix::<f64, 2, 2>::zero();
    /// let b = Matrix::from_array(array);
    ///
    /// assert_eq!(a, b);
    /// ```
    pub const fn zero() -> Self {
        Self {
            data: MatrixImpl::zero(),
        }
    }

    /// Constructs a matrix filled with all values equal to `t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 1.0], [1.0, 1.0]];
    /// let a = Matrix::<_, 2, 2>::from_val(1.0);
    /// let b = Matrix::from_array(array);
    ///
    /// assert_eq!(a, b);
    /// ```
    pub const fn from_val(t: T) -> Self {
        Self {
            data: MatrixImpl::from_val(t),
        }
    }

    /// Constructs a matrix from a given 2 dimensional array.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::from_array(array);
    ///
    /// for (x, y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn from_array(array: [[T; M]; N]) -> Self {
        Self {
            data: MatrixImpl::from_array(array),
        }
    }

    /// Returns a reference to the backing array of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let m = Matrix::<_, 2, 2>::from_val(1.0);
    /// let slice = m.as_array();
    ///
    /// for a in slice {
    ///     println!("{:?}", &a[..]);
    /// }
    /// ```
    pub const fn as_array(&self) -> &[[T; M]; N] {
        self.data.as_array()
    }

    /// Returns a mutable reference to the backing array of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let mut m = Matrix::<_, 2, 2>::from_val(1.0);
    /// let slice = m.as_mut_array();
    ///
    /// for arr in slice {
    ///     for elem in arr {
    ///         *elem = 2.0;
    ///     }
    /// }
    ///
    /// assert_eq!(m, Matrix::from_val(2.0));
    /// ```
    pub const fn as_mut_array(&mut self) -> &mut [[T; M]; N] {
        self.data.as_mut_array()
    }

    /// Returns a raw pointer to the first element in the array. Keep in mind that
    /// since the matrix is represented in column-major order, offsetting this
    /// pointer by `mem::size_of::<T>` bytes will cause it to point to the next
    /// element in the column.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let m = Matrix::<_, 2, 2>::from_val(1.0);
    /// let ptr = m.as_ptr();
    ///
    /// unsafe {
    ///     assert_eq!(*ptr, 1.0);
    /// }
    /// ```
    pub const fn as_ptr(&self) -> *const T {
        self.as_array() as *const _ as _
    }

    /// Returns a mutable raw pointer to the first element in the array. Keep in mind
    /// that since the matrix is represented in column-major order, offsetting this
    /// pointer by `mem::size_of::<T>` bytes will cause it to point to the next
    /// element in the column.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let mut m = Matrix::<_, 2, 2>::from_val(1.0);
    /// let ptr = m.as_mut_ptr();
    ///
    /// unsafe {
    ///     *ptr = 2.0;
    ///     assert_ne!(*ptr, 1.0);
    /// }
    /// ```
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_array() as *mut _ as _
    }

    /// Returns the dimensions of the matrix, in the order (M, N).
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let m = Matrix::<_, 2, 3>::from_val(1.0);
    ///
    /// assert_eq!(m.dimensions(), (2, 3))
    /// ```
    pub const fn dimensions(&self) -> (usize, usize) {
        (M, N)
    }

    /// Returns a column-wise iterator of all elements in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::from_array(array);
    ///
    /// for (x, y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn column_iter(&self) -> ColumnIter<'_, T, { M }, { N }> {
        ColumnIter {
            matrix: self,
            row: 0,
            col: 0,
        }
    }

    /// Returns a column-wise iterator of mutable references to
    /// all elements in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [2.0, 4.0, 6.0, 8.0];
    /// let mut m = Matrix::from_array(array);
    ///
    /// for elem in m.column_iter_mut() {
    ///     *elem *= 2.0;
    /// }
    ///
    /// for (x, y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn column_iter_mut(&mut self) -> ColumnIterMut<'_, T, { M }, { N }> {
        ColumnIterMut {
            matrix: self,
            row: 0,
            col: 0,
        }
    }

    /// Returns a row-wise iterator of all elements in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [1.0, 3.0, 2.0, 4.0];
    /// let m = Matrix::from_array(array);
    ///
    /// for (x, y) in m.row_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn row_iter(&self) -> RowIter<'_, T, { M }, { N }> {
        RowIter {
            matrix: self,
            row: 0,
            col: 0,
        }
    }

    /// Returns a row-wise iterator of mutable references to
    /// all elements in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [2.0, 6.0, 4.0, 8.0];
    /// let mut m = Matrix::from_array(array);
    ///
    /// for elem in m.row_iter_mut() {
    ///     *elem *= 2.0;
    /// }
    ///
    /// for (x, y) in m.row_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn row_iter_mut(&mut self) -> RowIterMut<'_, T, { M }, { N }> {
        RowIterMut {
            matrix: self,
            row: 0,
            col: 0,
        }
    }

    /// Returns an iterator of elements over the specified column in
    /// the matrix, starting from the `n`th element.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [1.0, 2.0];
    /// let m = Matrix::from_array(array);
    ///
    /// for (x, y) in m.column(0, 0).zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn column(&self, column: usize, n: usize) -> Column<'_, T, { M }, { N }> {
        Column {
            matrix: self,
            column,
            idx: n,
        }
    }

    /// Returns an iterator of mutable references over the specified column
    /// in the matrix, starting from the `n`th element.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [2.0, 4.0, 3.0, 4.0];
    /// let mut m = Matrix::from_array(array);
    ///
    /// for elem in m.column_mut(0, 0) {
    ///     *elem *= 2.0;
    /// }
    ///
    /// for (x, y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn column_mut(&mut self, column: usize, n: usize) -> ColumnMut<'_, T, { M }, { N }> {
        ColumnMut {
            matrix: self,
            column,
            idx: n,
        }
    }

    /// Returns an iterator of elements over the specified row in
    /// the matrix, starting from the `n`th element.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [1.0, 3.0];
    /// let m = Matrix::from_array(array);
    ///
    /// for (x, y) in m.row(0, 0).zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn row(&self, row: usize, n: usize) -> Row<'_, T, { M }, { N }> {
        Row {
            matrix: self,
            row,
            idx: n,
        }
    }

    /// Returns an iterator of mutable references over the specified row
    /// in the matrix, starting from the `n`th element.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1.0, 2.0], [3.0, 4.0]];
    /// let expected = [2.0, 6.0, 2.0, 4.0];
    /// let mut m = Matrix::from_array(array);
    ///
    /// for elem in m.row_mut(0, 0) {
    ///     *elem *= 2.0;
    /// }
    ///
    /// for (x, y) in m.row_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y)
    /// }
    /// ```
    pub const fn row_mut(&mut self, row: usize, n: usize) -> RowMut<'_, T, { M }, { N }> {
        RowMut {
            matrix: self,
            row,
            idx: n,
        }
    }

    /// Converts the provided matrix into one which holds [`f64`]
    /// values. This is required for some operations, such as
    /// computing the determinant of a matrix or putting the matrix
    /// into [row echelon form][1].
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;     
    ///      
    /// let array = [[1, 2], [3, 4]];
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::from_array(array).into_f64();
    ///
    /// for (&x, &y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y as f64);
    /// }
    /// ```
    ///
    /// [1]: https://en.wikipedia.org/wiki/Row_echelon_form
    pub fn into_f64(self) -> Matrix<f64, { M }, { N }> {
        let mut out = Matrix::zero();

        out.column_iter_mut()
            .zip(self.column_iter().map(|elem| elem.to_f64()))
            .for_each(|(x, y)| *x = y);

        out
    }

    /// Converts the provided matrix into one which holds [`f32`]
    /// values. This is required for some operations, such as
    /// computing the determinant of a matrix or putting the matrix
    /// into [row echelon form][1].
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;     
    ///      
    /// let array = [[1, 2], [3, 4]];
    /// let expected = [1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::from_array(array).into_f32();
    ///
    /// for (&x, &y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y as f32);
    /// }
    /// ```
    ///
    /// [1]: https://en.wikipedia.org/wiki/Row_echelon_form
    pub fn into_f32(self) -> Matrix<f32, { M }, { N }> {
        let mut out = Matrix::zero();

        out.column_iter_mut()
            .zip(self.column_iter().map(|elem| elem.to_f32()))
            .for_each(|(x, y)| *x = y);

        out
    }

    /// Returns the transpose of a matrix. This is a
    /// non-consuming operation, since the matrix needs
    /// to be copied anyways.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{matrix, Matrix};
    ///
    /// let m = matrix![
    ///     1.0, 2.0;
    ///     3.0, 4.0;
    ///     5.0, 6.0;
    /// ];
    ///
    /// let m_t = m.transpose();
    ///
    /// for (x, y) in m.column_iter().zip(m_t.row_iter()) {
    ///     assert_eq!(x, y);
    /// }
    /// ```
    pub fn transpose(&self) -> Matrix<T, { N }, { M }> {
        let mut out = Matrix::<T, { N }, { M }>::zero();

        for row in 0..M {
            for col in 0..N {
                out[(col, row)] = self[(row, col)];
            }
        }

        out
    }
}

// Methods that are only applicable when the matrix is square.
impl<T: Scalar, const N: usize> Matrix<T, { N }, { N }> {
    /// Returns the N x N identity matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::Matrix;
    ///
    /// let array = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    ///
    /// let m = Matrix::<usize, 3, 3>::id();
    ///
    /// assert_eq!(m, Matrix::from_array(array));
    /// ```
    pub const fn id() -> Self {
        Self {
            data: MatrixImpl::id(),
        }
    }

    /// Returns the determinant of a given matrix. For N < 5,
    /// this is done manually, as matrices with dimensions of
    /// less than 5x5 are the most common ones encountered. For
    /// N >= 5, the matrix is first row reduced, and then the
    /// product of the main diagonal is returned. Note that
    /// this operation may not be 100% correct, as the matrix
    /// must first be converted into floating point
    /// representation, which may result in a loss of precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{matrix, Matrix};
    ///
    /// let m = matrix![
    ///     3, 3, 3;
    ///     -2, -1, -2;
    ///     1, -2, -3;
    /// ];
    ///
    /// assert_eq!(m.det(), -12.0);
    /// ```
    pub fn det(&self) -> f64 {
        // For matrix dimensions 0 to 4, we do the determinant
        // manually in order to speed up computations.
        // For dimensions above 4, use gaussian elimination
        // and find the product of the diagonals.
        match N {
            0 => 1.0,
            1 => self[(0, 0)].to_f64(),
            2 => {
                // | a b |
                // | c d |

                (self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]).to_f64()
            }
            3 => {
                // This is horrible.
                // | a b c |
                // | d e f |
                // | g h i |

                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(0, 2)];

                let d = self[(1, 0)];
                let e = self[(1, 1)];
                let f = self[(1, 2)];

                let g = self[(2, 0)];
                let h = self[(2, 1)];
                let i = self[(2, 2)];

                (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)).to_f64()
            }
            4 => {
                // And this is even worse.
                // | a b c d |
                // | e f g h |
                // | i j k l |
                // | m n o p |

                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(0, 2)];
                let d = self[(0, 3)];

                let e = self[(1, 0)];
                let f = self[(1, 1)];
                let g = self[(1, 2)];
                let h = self[(1, 3)];

                let i = self[(2, 0)];
                let j = self[(2, 1)];
                let k = self[(2, 2)];
                let l = self[(2, 3)];

                let m = self[(3, 0)];
                let n = self[(3, 1)];
                let o = self[(3, 2)];
                let p = self[(3, 3)];

                let x = a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n));
                let y = b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m));
                let z = c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m));
                let w = d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m));

                (x - y + z - w).to_f64()
            }
            _ => {
                let matrix = self.clone().into_f64().gauss();

                let mut prod = 1.0;

                for i in 0..N {
                    prod *= matrix[(i, i)];
                }

                prod
            }
        }
    }

    /// Returns the determinant of a given matrix. For N < 5,
    /// this is done manually, as matrices with dimensions of
    /// less than 5x5 are the most common ones encountered. For
    /// N >= 5, the matrix is first row reduced, and then the
    /// product of the main diagonal is returned. Note that
    /// this operation may not be 100% correct, as the matrix
    /// must first be converted into floating point
    /// representation, which may result in a loss of precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{matrix, Matrix};
    ///
    /// let m = matrix![
    ///     3, 3, 3;
    ///     -2, -1, -2;
    ///     1, -2, -3;
    /// ];
    ///
    /// assert_eq!(m.det_f32(), -12.0);
    /// ```
    pub fn det_f32(&self) -> f32 {
        match N {
            0 => 1.0,
            1 => self[(0, 0)].to_f32(),
            2 => (self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]).to_f32(),
            3 => {
                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(0, 2)];

                let d = self[(1, 0)];
                let e = self[(1, 1)];
                let f = self[(1, 2)];

                let g = self[(2, 0)];
                let h = self[(2, 1)];
                let i = self[(2, 2)];

                (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)).to_f32()
            }
            4 => {
                let a = self[(0, 0)];
                let b = self[(0, 1)];
                let c = self[(0, 2)];
                let d = self[(0, 3)];

                let e = self[(1, 0)];
                let f = self[(1, 1)];
                let g = self[(1, 2)];
                let h = self[(1, 3)];

                let i = self[(2, 0)];
                let j = self[(2, 1)];
                let k = self[(2, 2)];
                let l = self[(2, 3)];

                let m = self[(3, 0)];
                let n = self[(3, 1)];
                let o = self[(3, 2)];
                let p = self[(3, 3)];

                let x = a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n));
                let y = b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m));
                let z = c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m));
                let w = d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m));

                (x - y + z - w).to_f32()
            }
            _ => {
                let matrix = self.clone().into_f32().gauss();

                let mut prod = 1.0;

                for i in 0..N {
                    prod *= matrix[(i, i)];
                }

                prod
            }
        }
    }
}

// Methods that are only applicable when the matrix contains floating point
// values.
impl<T: Real, const M: usize, const N: usize> Matrix<T, { M }, { N }> {
    /// Reduces the provided matrix to [row echelon form][1],
    /// consuming the original matrix in the process.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{matrix, Matrix};
    ///
    /// let m = matrix![
    ///     3, 3, 3;
    ///     -2, -1, -2;
    ///     1, -2, -3;
    /// ].into_f64().gauss();
    ///
    /// let rows = m.dimensions().0;
    ///
    /// for i in 0..rows {
    ///     println!("{:?}", m[(i, i)]);
    /// }
    ///
    /// ```
    ///
    /// [1]: https://en.wikipedia.org/wiki/Row_echelon_form
    pub fn gauss(mut self) -> Self {
        self.gauss_in_place();
        self
    }

    /// Reduces the provided matrix to [row echelon form][1] without
    /// consuming the original matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{matrix, Matrix};
    ///
    /// let mut m = matrix![
    ///     3, 3, 3;
    ///     -2, -1, -2;
    ///     1, -2, -3;
    /// ].into_f64();
    ///
    /// m.gauss_in_place();
    /// let rows = m.dimensions().0;
    ///
    /// for i in 0..rows {
    ///     println!("{:?}", m[(i, i)]);
    /// }
    ///
    /// ```
    ///
    /// [1]: https://en.wikipedia.org/wiki/Row_echelon_form
    pub fn gauss_in_place(&mut self) {
        for pivot_idx in 0..M - 1 {
            if self[(pivot_idx, pivot_idx)] != T::ZERO {
                let mut pivot = ConstVec::<_, { N }>::new();

                self.row(pivot_idx, pivot_idx)
                    .for_each(|&val| pivot.push(val).ok().unwrap());

                for row_idx in pivot_idx + 1..M {
                    // Find the ratio needed to remove the elements below the
                    // pivot.
                    let ratio = self[(row_idx, pivot_idx)] / pivot[0];

                    // subtract a multiple of the pivot row from this row.
                    self.row_mut(row_idx, pivot_idx)
                        .zip(pivot.iter())
                        .for_each(|(val, &sub)| *val -= ratio * sub);
                }
            }
        }
    }
}

// ========================= Iterators

/// A column-wise iterator over the elements of a matrix.
pub struct ColumnIter<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    matrix: &'a Matrix<T, { M }, { N }>,
    row: usize,
    col: usize,
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
    matrix: &'a mut Matrix<T, { M }, { N }>,
    row: usize,
    col: usize,
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
    matrix: &'a Matrix<T, { M }, { N }>,
    row: usize,
    col: usize,
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
    matrix: &'a mut Matrix<T, { M }, { N }>,
    row: usize,
    col: usize,
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
    matrix: &'a Matrix<T, { M }, { N }>,
    column: usize,
    idx: usize,
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

/// An iterator over mutable references to elements in one column of a matrix.
pub struct ColumnMut<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    matrix: &'a mut Matrix<T, { M }, { N }>,
    column: usize,
    idx: usize,
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

/// An iterator over elements in one row of a matrix.
pub struct Row<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    matrix: &'a Matrix<T, { M }, { N }>,
    row: usize,
    idx: usize,
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

/// An iterator over mutable references to elements in one row of a matrix.
pub struct RowMut<'a, T: Scalar + 'a, const M: usize, const N: usize> {
    matrix: &'a mut Matrix<T, { M }, { N }>,
    row: usize,
    idx: usize,
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

// ========================= Matrix Trait Impls

impl<T: Scalar, const M: usize, const N: usize> fmt::Debug for Matrix<T, { M }, { N }> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Matrix {{")?;
        writeln!(f, "    [")?;

        for col in self.as_array().iter() {
            writeln!(f, "        {:?},", &col[..])?;
        }

        writeln!(f, "    ],")?;
        writeln!(f, "}}")
    }
}

impl<T: Scalar, const M: usize, const N: usize> PartialEq for Matrix<T, { M }, { N }> {
    fn eq(&self, other: &Self) -> bool {
        self.column_iter()
            .zip(other.column_iter())
            .all(|(x, y)| x == y)
    }
}

impl<T: Scalar + Eq, const M: usize, const N: usize> Eq for Matrix<T, { M }, { N }> {}

impl<T: Scalar, const M: usize, const N: usize> Index<(usize, usize)> for Matrix<T, { M }, { N }> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        &self.as_array()[index.1][index.0]
    }
}

impl<T: Scalar, const M: usize, const N: usize> IndexMut<(usize, usize)>
    for Matrix<T, { M }, { N }>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.as_mut_array()[index.1][index.0]
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add for Matrix<T, { M }, { N }> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.column_iter_mut()
            .zip(rhs.column_iter())
            .for_each(|(self_val, &rhs_val)| *self_val += rhs_val);
        self
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub for Matrix<T, { M }, { N }> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.column_iter_mut()
            .zip(rhs.column_iter())
            .for_each(|(self_val, &rhs_val)| *self_val -= rhs_val);
        self
    }
}

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, { N }, { P }>>
    for Matrix<T, { M }, { N }>
{
    type Output = Matrix<T, { M }, { P }>;

    fn mul(self, rhs: Matrix<T, { N }, { P }>) -> Self::Output {
        let mut out = Self::Output::zero();

        for i in 0..M {
            for j in 0..P {
                let row = self.row(i, 0);
                let column = rhs.column(j, 0);

                out[(i, j)] = mm_dot(row, column);
            }
        }

        out
    }
}

impl<T: Scalar, const R: usize, const C: usize> Mul<T> for Matrix<T, { R }, { C }> {
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self {
        for elem in self.column_iter_mut() {
            *elem *= rhs;
        }

        self
    }
}

// ========================= Vectors (Special case of Matrix)

/// An N dimensional vector, represented as a matrix
/// with dimensions N x 1.
pub type Vector<T, const N: usize> = Matrix<T, { N }, 1>;

/// An N dimensional vector of f64 values.
pub type VectorF64<const N: usize> = Vector<f64, { N }>;

/// An N dimensional vector of f32 values.
pub type VectorF32<const N: usize> = Vector<f32, { N }>;

impl<T: Scalar, const N: usize> Vector<T, { N }> {
    /// Returns the length of the vector as an [`f64`].
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{vector, Vector};
    ///
    /// let v = vector![1, 1, 1];
    ///
    /// assert_eq!(v.length(), f64::sqrt(3.0));
    /// ```
    pub fn length(&self) -> f64 {
        f64::sqrt(self.squared_sum().to_f64())
    }

    /// Returns the length of the vector as an [`f32`].
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{vector, Vector};
    ///
    /// let v = vector![1, 1, 1];
    ///
    /// assert_eq!(v.length(), f64::sqrt(3.0));
    /// ```
    ///
    pub fn length_f32(&self) -> f32 {
        f32::sqrt(self.squared_sum().to_f32())
    }

    /// Consumes and normalizes the vector, returning the
    /// unit vector of [`f64`] values in the same direction
    /// as the original vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{vector, Vector};
    ///
    /// let v = vector![1, 1, 1].normalize();
    ///
    /// assert_eq!(v.length(), 1.0);
    /// ```
    pub fn normalize(self) -> VectorF64<{ N }> {
        let len = self.length();

        // A vector's length is zero iff it is the zero vector.
        if len == f64::ZERO {
            VectorF64::<{ N }>::zero()
        } else {
            self.into_f64() / len
        }
    }

    /// Consumes and normalizes the vector, returning the
    /// unit vector of [`f32`] values in the same direction
    /// as the original vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     traits::Real,
    ///     vector, Vector,
    /// };
    ///
    /// let v = vector![1, 1, 1].normalize_f32();
    ///
    /// // Relative equality is needed here, since
    /// // the value doesn't come out as *exactly*
    /// // 1.0f32.
    /// let epsilon = std::f32::EPSILON;
    /// let relative = 1.0E-7f32;
    ///
    /// assert!(v.length_f32().approx_eq(1.0f32, epsilon, relative));
    /// ```
    pub fn normalize_f32(self) -> VectorF32<{ N }> {
        let len = self.length_f32();

        if len == f32::ZERO {
            VectorF32::<{ N }>::zero()
        } else {
            self.into_f32() / len
        }
    }

    pub fn dot(&self, rhs: &Self) -> T {
        self.column_iter()
            .zip(rhs.column_iter())
            .fold(T::ZERO, |acc, (&x, &y)| acc + (x * y))
    }

    /// Returns the sum of every elements squared.
    fn squared_sum(&self) -> T {
        let mut sum = T::ZERO;
        for &x in self.column_iter() {
            sum += x * x;
        }
        sum
    }
}

impl<T: Scalar> Vector<T, { 3 }> {
    pub fn cross(&self, rhs: &Self) -> Self {
        vector![
            self[(1, 0)] * rhs[(2, 0)] - self[(2, 0)] * rhs[(1, 0)],
            self[(2, 0)] * rhs[(0, 0)] - self[(0, 0)] * rhs[(2, 0)],
            self[(0, 0)] * rhs[(1, 0)] - self[(1, 0)] * rhs[(0, 0)],
        ]
    }
}

// ========================= Vector Trait Impls

impl<T: Scalar, const N: usize> Add<T> for Vector<T, { N }> {
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        for elem in self.column_iter_mut() {
            *elem += rhs;
        }

        self
    }
}

impl<T: Scalar, const N: usize> Sub<T> for Vector<T, { N }> {
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self::Output {
        for elem in self.column_iter_mut() {
            *elem -= rhs;
        }

        self
    }
}

/// Note that this impl will do *integer* division if `T`
/// happens to be an integer. If you want a floating point
/// representation, call [`Matrix::into_f64`] or
/// [`Matrix::into_f32`] before dividing.
impl<T: Scalar, const N: usize> Div<T> for Vector<T, { N }> {
    type Output = Vector<T, { N }>;

    fn div(mut self, rhs: T) -> Self::Output {
        for elem in self.column_iter_mut() {
            *elem /= rhs
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn index() {
        const ARRAY: [[f64; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let matrix = Matrix::from_array(ARRAY);

        assert_eq!(matrix[(1, 2)], 6.0);
    }

    #[test]
    fn iter() {
        const ARRAY: [[f64; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        const COLUMN_FLATTENED: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        const ROW_FLATTENED: [f64; 6] = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];

        let matrix = Matrix::from_array(ARRAY);

        // Check that column_iter() is indeed a column-wise iterator.
        for (i, &matrix_elem) in matrix.column_iter().enumerate() {
            assert_eq!(COLUMN_FLATTENED[i], matrix_elem);
        }

        // And likewise for row_iter()
        for (i, &matrix_elem) in matrix.row_iter().enumerate() {
            assert_eq!(ROW_FLATTENED[i], matrix_elem);
        }

        // The below code shows that `matrix` can not be dropped
        // as it is mutably borrowed. This is just a test two ensure
        // that the unsafe code works as intended.
        //        let mut mut_col_iter = matrix.column_iter_mut();
        //
        //        let mut_borrowed = mut_col_iter.next().unwrap();
        //
        //        drop(mut_col_iter);
        //        drop(matrix);
        //
        //        *mut_borrowed = 0.0;
    }

    #[test]
    fn transpose() {
        const ARRAY: [[f64; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        const ITER_TRANSPOSED: [f64; 6] = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];

        let matrix = Matrix::from_array(ARRAY).transpose();

        // Ensure we get the expected results.
        for (i, &matrix_elem) in matrix.column_iter().enumerate() {
            // eprintln!("{}", matrix_elem);
            assert_eq!(ITER_TRANSPOSED[i], matrix_elem);
        }

        // Check that (Matrix^T)^T is the same as the original
        // matrix.
        assert!(matrix
            .column_iter()
            .zip(matrix.transpose().transpose().column_iter())
            .all(|(x, y)| x == y));

        // Four times?
        assert!(matrix
            .column_iter()
            .zip(
                matrix
                    .transpose()
                    .transpose()
                    .transpose()
                    .transpose()
                    .column_iter()
            )
            .all(|(x, y)| x == y));
    }

    #[test]
    fn matrix_matrix_mul() {
        let first = [[1usize, 2], [3, 4]];
        let second = [[1, 2], [3, 4], [5, 6]];

        let expected = [7usize, 10, 15, 22, 23, 34];

        let multiplied = Matrix::from_array(first) * Matrix::from_array(second);

        // Check each element is correct.
        for (i, &elem) in multiplied.column_iter().enumerate() {
            assert_eq!(expected[i], elem);
        }

        // Sanity check for square matrices, as Matrix::<N, N> * Matrix<N, N> -> Matrix<N, N>
        let m = Matrix::<usize, 5, 5>::from_val(1);
        let n = Matrix::<usize, 5, 5>::from_val(1);

        for arr in m.as_array() {
            for elem in arr {
                eprint!("{:?}, ", elem)
            }
        }

        let square_matrix = m * n;
        let expected = [5; 25];

        for (i, &elem) in square_matrix.column_iter().enumerate() {
            assert_eq!(expected[i], elem);
        }
    }

    #[test]
    fn gauss_elim() {
        let array = [[2, -3, -2], [1, -1, 1], [-1, 2, 2]];
        let expected = [[2.0, 0.0, 0.0], [1.0, 0.5, 0.0], [-1.0, 0.5, -1.0]];

        let matrix = Matrix::from_array(array).into_f64().gauss();
        let eliminated = Matrix::from_array(expected).into_f64();

        // For this matrix, all of these turn out to be completely equal,
        // but approx_eq is used as a safety measure, since this may not
        // always be the case.
        dbg!(matrix.as_array());
        dbg!(eliminated.as_array());
        for (x, y) in matrix.column_iter().zip(eliminated.column_iter()) {
            assert!(x.approx_eq(*y, std::f64::EPSILON, 1.0E-10));
        }
    }

    #[test]
    fn matrix_macro() {
        let first = [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let second = [1usize; 16];
        let third = [1usize, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

        let mat_one = matrix![
             1usize, 2, 3, 4;
             5, 6, 7, 8;
             9, 10, 11, 12;
             13, 14, 15, 16;
        ];

        let mat_two = matrix![1usize; 4, 4];

        let mat_three = matrix![usize; 4];

        for (i, &elem) in mat_one.column_iter().enumerate() {
            assert_eq!(first[i], elem);
        }

        for (i, &elem) in mat_two.column_iter().enumerate() {
            assert_eq!(second[i], elem);
        }

        for (i, &elem) in mat_three.column_iter().enumerate() {
            assert_eq!(third[i], elem);
        }
    }

    #[test]
    fn vector_macro() {
        let first = [1, 2, 3];
        let second = [1, 1, 1];

        let vec_one = vector![1, 2, 3];
        let vec_two = vector![1; 3];

        for (i, &elem) in vec_one.column_iter().enumerate() {
            assert_eq!(first[i], elem);
        }

        for (i, &elem) in vec_two.column_iter().enumerate() {
            assert_eq!(second[i], elem);
        }
    }

    #[test]
    fn vector_length() {
        use crate::traits::Real as _;

        let expected_f64 = std::f64::consts::SQRT_2;
        let expected_f32 = std::f32::consts::SQRT_2;

        let vec_one = vector![1.0f64; 2];
        let vec_two = vector![1.0f32; 2];

        // eprintln!("{}, {}", vec_one.length(), expected_f64);
        // eprintln!("{}, {}", vec_two.length_f32(), expected_f32);

        assert!(vec_one
            .length()
            .approx_eq(expected_f64, std::f64::EPSILON, 1.0e-6));
        assert!(vec_two
            .length_f32()
            .approx_eq(expected_f32, std::f32::EPSILON, 1.0e-6));
    }

    #[test]
    fn normalize() {
        use crate::traits::Real as _;

        let vector = vector![1, 2, 3];
        let normalized = vector.normalize();

        // eprintln!("{}", normalized.length());
        // eprintln!("{:?}", normalized.as_array());

        assert!(normalized
            .length()
            .approx_eq(1.0f64, std::f64::EPSILON, 1.0e-15));
    }

    #[test]
    fn cross() {
        let v = vector![0, 4, -2];
        let w = vector![3, -1, 5];

        let expected = vector![18, -6, -12];

        assert!(v.cross(&w) == expected);
    }

    #[test]
    #[ignore]
    fn large_matrix_time() {
        // let m1 = matrix![1.0; 16, 16];
        // let n1 = matrix![1.0; 16, 16];
        // let m2 = matrix![1; 32, 32];
        // let n2 = matrix![1; 32, 32];
        // let m3 = matrix![1; 64, 64];
        // let n3 = matrix![1; 64, 64];
        // let m4 = matrix![1; 128, 128];
        // let n4 = matrix![1; 128, 128];
        // let m5 = matrix![1; 4, 4];
        // let n5 = matrix![1; 4, 4];
        // // let n2 = n.clone();
        // // let n3 = n.clone();
        // // let n4 = n.clone();
        // // let n5 = n.clone();
        //
        // let start = Instant::now();
        // let _ = m5 * n5;
        // eprintln!("4: {:?}", start.elapsed());
        //
        // let start = Instant::now();
        // let _ = m1 * n1;
        // eprintln!("16: {:?}", start.elapsed());
        //
        // let start = Instant::now();
        // let _ = m2 * n2;
        // eprintln!("32: {:?}", start.elapsed());
        //
        // let start = Instant::now();
        // let _ = m3 * n3;
        // eprintln!("64: {:?}", start.elapsed());
        //
        // let start = Instant::now();
        // let _ = m4 * n4;
        // eprintln!("128: {:?}", start.elapsed());

        let mut a = matrix![
            1.0, -2.0, 5.0, 2.0;
            5.0, 3.0, 1.0, 3.0;
            4.0, 6.0, 0.0, -4.0;
            2.0, 4.0, -1.0, 0.0;
        ];

        let b = a.clone();

        let c = a.clone();

        let start_gauss = Instant::now();
        a.gauss_in_place();
        eprintln!("gauss: {:?}", start_gauss.elapsed());

        let start_mm = Instant::now();
        let _ = b * c;
        eprintln!("mm: {:?}", start_mm.elapsed());

        let start_det = Instant::now();
        let _ = a.det();
        eprintln!("det: {:?}", start_det.elapsed());
    }

    #[test]
    #[ignore]
    fn debug() {
        let m = matrix![usize; 16];

        dbg!(m);
    }
}
