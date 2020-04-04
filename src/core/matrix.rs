use super::mm_dot;
use crate::{
    algebra::unit::UnitVector,
    core::{imp::MatrixImpl, iter::*},
    traits::{Real, Scalar, Signed},
    utils::ConstVec,
};
use std::{
    alloc,
    cmp::PartialEq,
    fmt, mem,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
};

// ========================= Matrix

/// An M * N column-major Matrix of scalar values backed by an array. Scalar
/// refers to anything that implements the [`Scalar`] trait, which is any value
/// that is [`Copy`] and has a notion of arithmetic. However, many operations,
/// such as the determinant and Gaussian elimination may not be possible
/// with integer values.
#[repr(transparent)]
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

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn scale(mut self, s: T) -> Self {
        self.scale_mut(s);
        self
    }

    pub fn scale_mut(&mut self, s: T) {
        for elem in self.column_iter_mut() {
            *elem *= s
        }
    }

    /// Converts the provided matrix into one which holds [`Real`]
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
    /// let m = Matrix::from_array(array).into_real::<f64>();
    ///
    /// for (&x, &y) in m.column_iter().zip(expected.iter()) {
    ///     assert_eq!(x, y as f64);
    /// }
    /// ```
    ///
    /// [1]: https://en.wikipedia.org/wiki/Row_echelon_form
    pub fn into_real<R: Real>(self) -> Matrix<R, { M }, { N }> {
        let mut out = Matrix::zero();

        out.column_iter_mut()
            .zip(self.column_iter().map(|elem| elem.into_real()))
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

impl<T: Signed, const M: usize, const N: usize> Matrix<T, { M }, { N }> {
    pub fn negate(mut self) -> Self {
        self.negate_mut();
        self
    }

    pub fn negate_mut(&mut self) {
        self.scale_mut(T::ONE.negate());
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
    /// ].into_real::<f64>().gauss();
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
        self.gauss_mut();
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
    /// ].into_real::<f64>();
    ///
    /// m.gauss_mut();
    /// let rows = m.dimensions().0;
    ///
    /// for i in 0..rows {
    ///     println!("{:?}", m[(i, i)]);
    /// }
    ///
    /// ```
    ///
    /// [1]: https://en.wikipedia.org/wiki/Row_echelon_form
    pub fn gauss_mut(&mut self) {
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

/// A matrix that is square.
pub type SquareMatrix<T, const N: usize> = Matrix<T, { N }, { N }>;

/// A matrix which represents `N`-dimensional space with `N + 1` dimensions.
pub type HomogeneousMatrix<T, const N: usize> = SquareMatrix<T, { N + 1 }>;

// Methods that are only applicable when the matrix is square.
impl<T: Scalar, const N: usize> SquareMatrix<T, { N }> {
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

    pub fn into_homogeneous(self) -> HomogeneousMatrix<T, { N }> {
        let mut out = HomogeneousMatrix::id();

        for i in 0..N {
            out.column_mut(i, 0)
                .zip(self.column(i, 0))
                .for_each(|(val, new)| *val = *new);
        }

        out
    }
}

impl<T: Real, const N: usize> SquareMatrix<T, { N }> {
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
    /// ].into_real::<f64>();
    ///
    /// assert_eq!(m.det(), -12.0);
    /// ```
    pub fn det(&self) -> T {
        // For matrix dimensions 0 to 4, we do the determinant
        // manually in order to speed up computations.
        // For dimensions above 4, use Gaussian elimination
        // and find the product of the diagonals.
        match N {
            0 => T::ONE,
            1 => self[(0, 0)],
            2 => {
                // | a b |
                // | c d |

                self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
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

                a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
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

                x - y + z - w
            }
            _ => {
                let matrix = self.clone().gauss();

                let mut prod = T::ONE;

                for i in 0..N {
                    prod *= matrix[(i, i)];
                }

                prod
            }
        }
    }
}

// ========================= Matrix Trait Impls

// Debug
impl<T: Scalar, const M: usize, const N: usize> fmt::Debug for Matrix<T, { M }, { N }> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Matrix").field("data", &self.data).finish()
    }
}

// Equality
impl<T: Scalar, const M: usize, const N: usize> PartialEq for Matrix<T, { M }, { N }> {
    fn eq(&self, other: &Self) -> bool {
        self.column_iter()
            .zip(other.column_iter())
            .all(|(x, y)| x == y)
    }
}

impl<T: Scalar + Eq, const M: usize, const N: usize> Eq for Matrix<T, { M }, { N }> {}

// Indexing (in the order (row, col))
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

// Addition
impl<T: Scalar, const M: usize, const N: usize> Add for Matrix<T, { M }, { N }> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.column_iter_mut()
            .zip(rhs.column_iter())
            .for_each(|(self_val, &rhs_val)| *self_val += rhs_val);
        self
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add for &'_ Matrix<T, { M }, { N }> {
    type Output = Matrix<T, { M }, { N }>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = Self::Output::zero();

        self.column_iter()
            .zip(rhs.column_iter())
            .map(|(&x, &y)| x + y)
            .zip(out.column_iter_mut())
            .for_each(|(val, elem)| *elem = val);

        out
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add for Box<Matrix<T, { M }, { N }>> {
    type Output = Box<Matrix<T, { M }, { N }>>;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.column_iter_mut()
            .zip(rhs.column_iter())
            .for_each(|(self_val, &rhs_val)| *self_val += rhs_val);
        self
    }
}

impl<T: Scalar, const M: usize, const N: usize> Add for &'_ Box<Matrix<T, { M }, { N }>> {
    type Output = Box<Matrix<T, { M }, { N }>>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut out = box Matrix::<T, { M }, { N }>::zero();

        self.column_iter()
            .zip(rhs.column_iter())
            .map(|(&x, &y)| x + y)
            .zip(out.column_iter_mut())
            .for_each(|(val, elem)| *elem = val);

        out
    }
}

// Subtraction
impl<T: Scalar, const M: usize, const N: usize> Sub for Matrix<T, { M }, { N }> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.column_iter_mut()
            .zip(rhs.column_iter())
            .for_each(|(self_val, &rhs_val)| *self_val -= rhs_val);
        self
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub for &'_ Matrix<T, { M }, { N }> {
    type Output = Matrix<T, { M }, { N }>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = Self::Output::zero();

        self.column_iter()
            .zip(rhs.column_iter())
            .map(|(&x, &y)| x - y)
            .zip(out.column_iter_mut())
            .for_each(|(val, elem)| *elem = val);

        out
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub for Box<Matrix<T, { M }, { N }>> {
    type Output = Box<Matrix<T, { M }, { N }>>;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.column_iter_mut()
            .zip(rhs.column_iter())
            .for_each(|(self_val, &rhs_val)| *self_val -= rhs_val);
        self
    }
}

impl<T: Scalar, const M: usize, const N: usize> Sub for &'_ Box<Matrix<T, { M }, { N }>> {
    type Output = Box<Matrix<T, { M }, { N }>>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = box Matrix::<T, { M }, { N }>::zero();

        self.column_iter()
            .zip(rhs.column_iter())
            .map(|(&x, &y)| x - y)
            .zip(out.column_iter_mut())
            .for_each(|(val, elem)| *elem = val);

        out
    }
}

// Multiplication
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

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<&'_ Matrix<T, { N }, { P }>>
    for &'_ Matrix<T, { M }, { N }>
{
    type Output = Matrix<T, { M }, { P }>;

    fn mul(self, rhs: &'_ Matrix<T, { N }, { P }>) -> Self::Output {
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

impl<T: Scalar, const M: usize, const N: usize, const P: usize> Mul<Box<Matrix<T, { N }, { P }>>>
    for Box<Matrix<T, { M }, { N }>>
{
    type Output = Box<Matrix<T, { M }, { P }>>;

    fn mul(self, rhs: Box<Matrix<T, { N }, { P }>>) -> Self::Output {
        let mut out = box Matrix::<T, { M }, { P }>::zero();

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

impl<T: Scalar, const M: usize, const N: usize, const P: usize>
    Mul<&'_ Box<Matrix<T, { N }, { P }>>> for &'_ Box<Matrix<T, { M }, { N }>>
{
    type Output = Box<Matrix<T, { M }, { P }>>;

    fn mul(self, rhs: &Box<Matrix<T, { N }, { P }>>) -> Self::Output {
        let mut out = box Matrix::<T, { M }, { P }>::zero();

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

// Negation
impl<T: Signed, const M: usize, const N: usize> Neg for Matrix<T, { M }, { N }> {
    type Output = Self;

    fn neg(mut self) -> Self {
        self.scale_mut(T::ONE.negate());

        self
    }
}

impl<T: Signed, const M: usize, const N: usize> Neg for &'_ Matrix<T, { M }, { N }> {
    type Output = Matrix<T, { M }, { N }>;

    fn neg(self) -> Self::Output {
        self.clone().neg()
    }
}

impl<T: Signed, const M: usize, const N: usize> Neg for Box<Matrix<T, { M }, { N }>> {
    type Output = Self;

    fn neg(mut self) -> Self {
        self.negate_mut();
        self
    }
}

impl<T: Signed, const M: usize, const N: usize> Neg for &'_ Box<Matrix<T, { M }, { N }>> {
    type Output = Box<Matrix<T, { M }, { N }>>;

    fn neg(self) -> Self::Output {
        Box::clone(self).neg()
    }
}

// ========================= Vectors (Special case of Matrix)

/// An N dimensional vector, represented as a matrix
/// with dimensions N x 1.
pub type Vector<T, const N: usize> = Matrix<T, { N }, 1>;

impl<T: Scalar, const N: usize> Vector<T, { N }> {
    pub const fn basis(dim: usize) -> Option<Self> {
        if dim >= N {
            None
        } else {
            let mut v = Self::zero();
            v.as_mut_array()[0][dim] = T::ONE;

            Some(v)
        }
    }

    pub const fn comp(&self, dim: usize) -> T {
        if dim >= N {
            T::ZERO
        } else {
            self.as_array()[0][dim]
        }
    }

    pub fn dot(&self, rhs: &Self) -> T {
        self.column_iter()
            .zip(rhs.column_iter())
            .fold(T::ZERO, |acc, (&x, &y)| acc + (x * y))
    }

    /// Returns the sum of every element squared.
    pub(crate) fn squared_sum(&self) -> T {
        let mut sum = T::ZERO;
        for &x in self.column_iter() {
            sum += x * x;
        }
        sum
    }
}

impl<T: Real, const N: usize> Vector<T, { N }> {
    /// Returns the length of the vector as an [`f64`].
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{vector, Vector};
    ///
    /// let v = vector![1, 1, 1].into_real::<f64>();
    ///
    /// assert_eq!(v.length(), f64::sqrt(3.0));
    /// ```
    pub fn length(&self) -> T {
        T::sqrt(self.squared_sum())
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
    /// let v = vector![1, 1, 1].into_real::<f64>().normalize();
    ///
    /// assert_eq!(v.length(), 1.0);
    /// ```
    pub fn normalize(self) -> Self {
        let len = self.length();

        // A vector's length is zero iff it is the zero vector.
        if len == T::ZERO {
            Self::zero()
        } else {
            self / len
        }
    }
}

impl<T: Signed> Vector<T, { 3 }> {
    pub fn cross(&self, rhs: &Self) -> Self {
        let x = self.comp(1) * rhs.comp(2) - self.comp(2) * rhs.comp(1);
        let y = self.comp(2) * rhs.comp(0) - self.comp(0) * rhs.comp(2);
        let z = self.comp(0) * rhs.comp(1) - self.comp(1) * rhs.comp(0);

        vector![x, y, z]
    }
}

/// Note that this impl will do *integer* division if `T`
/// happens to be an integer. If you want a floating point
/// representation, call [`Matrix::into_f64`] or
/// [`Matrix::into_f32`] before dividing.
impl<T: Scalar, const N: usize> Div<T> for Vector<T, { N }> {
    type Output = Self;

    fn div(mut self, rhs: T) -> Self {
        for elem in self.column_iter_mut() {
            *elem /= rhs
        }

        self
    }
}

impl<T: Scalar, const N: usize> Div<T> for &'_ Vector<T, { N }> {
    type Output = Vector<T, { N }>;

    fn div(self, rhs: T) -> Self::Output {
        let mut out = self.clone();

        for elem in out.column_iter_mut() {
            *elem /= rhs
        }

        out
    }
}

impl<T: Scalar, const N: usize> Div<T> for Box<Vector<T, { N }>> {
    type Output = Self;

    fn div(mut self, rhs: T) -> Self {
        for elem in self.column_iter_mut() {
            *elem /= rhs
        }

        self
    }
}

impl<T: Scalar, const N: usize> Div<T> for &'_ Box<Vector<T, { N }>> {
    type Output = Box<Vector<T, { N }>>;

    fn div(self, rhs: T) -> Self::Output {
        let mut out = Box::clone(&self);

        for elem in out.column_iter_mut() {
            *elem /= rhs
        }

        out
    }
}

impl<T: Real, const N: usize> From<UnitVector<T, { N }>> for Vector<T, { N }> {
    fn from(unit: UnitVector<T, { N }>) -> Self {
        unit.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{matrix, vector};
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

        let matrix = Matrix::from_array(array).into_real::<f64>().gauss();
        let eliminated = Matrix::from_array(expected).into_real::<f64>();

        // For this matrix, all of these turn out to be completely equal,
        // but approx_eq is used as a safety measure, since this may not
        // always be the case.
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
            .length()
            .approx_eq(expected_f32, std::f32::EPSILON, 1.0e-6));
    }

    #[test]
    fn normalize() {
        use crate::traits::Real as _;

        let vector = vector![1, 2, 3].into_real::<f64>();
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

        assert_eq!(v.cross(&w), expected);
    }

    #[test]
    fn vector_basis() {
        assert_eq!(Vector::<usize, 1>::basis(1), None);
        assert_eq!(Vector::<usize, 3>::basis(1), Some(vector![0, 1, 0]));
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
        a.gauss_mut();
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
        let m = matrix![usize; 3];

        eprintln!("{:?}", m);
    }
}
