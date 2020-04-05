use crate::{
    algebra::unit::{UnitQuaternion, UnitVector},
    core::{SquareMatrix, Vector},
    geometry::angle::Angle,
    traits::{Real, Scalar},
};

use std::ops::{Add, Div, Mul, Sub};

/// A quaternion, represented as a scalar part and
/// a vector part.
#[derive(Clone, Debug, PartialEq)]
pub struct Quaternion<T: Real> {
    scalar: T,
    vector: Vector<T, 3>,
}

impl<T: Real> Quaternion<T> {
    /// Constructs a quaternion from a real component and a
    /// vector component.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion
    /// };
    ///
    /// let v = vector![2.0, 3.0, 4.0];
    /// let _ = Quaternion::new(1.0, v);
    /// ```
    pub fn new(scalar: T, vector: Vector<T, 3>) -> Self {
        Self { scalar, vector }
    }

    /// Constructs a unit quaternion from a given axis and
    /// an angle.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    ///     geometry::Angle,
    /// };
    ///
    /// let axis = vector![1.0, 0.0, 0.0];
    /// let angle = Angle::from_deg(60.0).to_rad();
    ///
    /// let q = Quaternion::from_axis_angle(axis.clone(), angle);
    ///
    /// let half_angle = angle.scale(0.5);
    ///
    /// let scalar = half_angle.cos();
    /// let vector = axis.normalize().scale(half_angle.sin());
    ///
    /// assert_eq!(q, Quaternion::new(scalar, vector))
    /// ```
    pub fn from_axis_angle(axis: Vector<T, 3>, angle: Angle<T>) -> Self {
        let angle = angle.to_rad().scale(2.into_real::<T>().recip());

        let scalar = angle.cos();
        let vector = axis.normalize().scale(angle.sin());

        Self { scalar, vector }
    }

    pub fn scalar(&self) -> T {
        self.scalar
    }

    pub fn vector(&self) -> &Vector<T, 3> {
        &self.vector
    }

    /// Returns the conjugate of the quaternion, consuming
    /// the original quaternion in the process.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0];
    /// let v = vector![-2.0, -3.0, -4.0];
    ///
    /// let q = Quaternion::new(1.0, u);
    /// let p = Quaternion::new(1.0, v);
    ///
    /// assert_eq!(q.conjugate(), p);
    /// ```
    pub fn conjugate(mut self) -> Self {
        self.conjugate_mut();
        self
    }

    /// Sets the original quaternion to its conjugate without
    /// consuming the original quaternion.
    ///
    /// # Examples
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0];
    /// let v = vector![-2.0, -3.0, -4.0];
    ///
    /// let mut q = Quaternion::new(1.0, u);
    /// let p = Quaternion::new(1.0, v);
    /// q.conjugate_mut();
    ///
    /// assert_eq!(q, p);
    /// ```
    pub fn conjugate_mut(&mut self) {
        self.vector.negate_mut();
    }

    /// Returns the norm of the quaternion.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion
    /// };
    ///
    /// let v = vector![2.0, 3.0, 4.0];
    /// let q = Quaternion::new(1.0, v);
    ///
    /// assert_eq!(q.norm(), f64::sqrt(30.0));
    /// ```
    pub fn norm(&self) -> T {
        let scalar_squared = self.scalar * self.scalar;
        let sum = scalar_squared + self.vector.squared_sum();
        sum.sqrt()
    }

    /// Returns the squared norm of the quaternion.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion
    /// };
    ///
    /// let v = vector![2.0, 3.0, 4.0];
    /// let q = Quaternion::new(1.0, v);
    /// let norm_squared = q.norm_squared();
    ///
    /// assert_eq!(q.clone().inverse(), q.conjugate() / norm_squared);
    /// ```
    pub fn norm_squared(&self) -> T {
        let norm = self.norm();
        norm * norm
    }

    pub fn normalize(mut self) -> Self {
        self.normalize_mut();
        self
    }

    pub fn normalize_mut(&mut self) {
        let norm = self.norm();

        self.scalar /= norm;
        self.vector.scale_mut(norm.recip())
    }

    /// Returns a quaternion which is the original quaternion
    /// scaled by a factor of `s`. This is the same as multiplying
    /// the quaternion by the real value `s`.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0];
    /// let v = vector![4.0, 6.0, 8.0];
    ///
    /// let q = Quaternion::new(1.0, u);
    /// let p = Quaternion::new(2.0, v);
    ///
    /// assert_eq!(q.scale(2.0), p);
    /// ```
    pub fn scale(mut self, s: T) -> Self {
        self.scale_mut(s);
        self
    }

    /// Scales the quaternion  by a factor of `s` without consuming it.
    /// This is the same as multiplying the quaternion by the real
    /// value `s`.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0];
    /// let v = vector![4.0, 6.0, 8.0];
    ///
    /// let mut q = Quaternion::new(1.0, u);
    /// let p = Quaternion::new(2.0, v);
    ///
    /// q.scale_mut(2.0);
    /// assert_eq!(q, p);
    /// ```
    pub fn scale_mut(&mut self, s: T) {
        self.scalar *= s;
        self.vector.scale_mut(s);
    }

    /// Returns the inverse of the quaternion, consuming the original
    /// quaternion in the process.
    ///
    /// Equivalent to `self.conjugate / self.norm_squared()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let v = vector![2.0, 3.0, 4.0];
    /// let q = Quaternion::new(1.0, v);
    ///
    /// let conjugate = q.clone().conjugate();
    /// let norm_squared = q.norm_squared();
    ///
    /// assert_eq!(q.inverse(), conjugate / norm_squared);
    /// ```
    pub fn inverse(self) -> Self {
        let norm_squared = self.norm_squared();
        self.conjugate() / norm_squared
    }

    /// Returns a vector representing the quaternion, consuming the original
    /// quaternion in the process. The vector is represented as `[x, y, z, w]`,
    /// where `x`, `y`, and `z` are the imaginary components of the quaternion,
    /// and `w` is the scalar component.
    ///
    /// # Example
    ///
    /// ```
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0, 1.0];
    /// let v = vector![2.0, 3.0, 4.0];
    ///
    /// let q = Quaternion::new(1.0, v);
    ///
    /// assert_eq!(q.into_vector(), u);
    ///
    /// ```
    pub fn into_vector(self) -> Vector<T, 4> {
        vector![
            self.vector[(0, 0)],
            self.vector[(1, 0)],
            self.vector[(2, 0)],
            self.scalar,
        ]
    }

    /// Divides `self` by `other` from the right.
    /// Equivalent to `other.inverse() * self`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Note that this example is not tested here
    /// // as it causes the compiler to overflow its
    /// // stack when it runs as a doctest as of
    /// // 2020-03-19. It is, however, tested in the
    /// // regular tests for this crate.
    ///
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0];
    /// let v = vector![6.0, 7.0, 8.0];
    /// let q = Quaternion::new(1.0, u);
    /// let p = Quaternion::new(5.0, v);
    ///
    /// let q_cloned = q.clone();
    /// let p_cloned = p.clone();
    ///
    /// assert_eq!(q.div_right(p), p_cloned.inverse() * q_cloned);
    /// ```
    pub fn div_right(self, other: Self) -> Self {
        let inverse = other.inverse();
        inverse * self
    }

    /// Divides `self` by `other` from the left.
    /// Equivalent to `self * other.inverse()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Note that this example is not tested here
    /// // as it causes the compiler to overflow its
    /// // stack when it runs as a doctest as of
    /// // 2020-03-19. It is, however, tested in the
    /// // regular tests for this crate.
    ///
    /// use const_linear::{
    ///     vector,
    ///     algebra::Quaternion,
    /// };
    ///
    /// let u = vector![2.0, 3.0, 4.0];
    /// let v = vector![6.0, 7.0, 8.0];
    /// let q = Quaternion::new(1.0, u);
    /// let p = Quaternion::new(5.0, v);
    ///
    /// let q_cloned = q.clone();
    /// let p_cloned = p.clone();
    ///
    /// assert_eq!(q.div_left(p), q_cloned * p_cloned.inverse());
    /// ```
    pub fn div_left(self, other: Self) -> Self {
        let inverse = other.inverse();
        self * inverse
    }

    pub fn as_matrix(&self) -> SquareMatrix<T, 4> {
        let a = self.scalar;
        let b = self.vector[(0, 0)];
        let c = self.vector[(0, 1)];
        let d = self.vector[(0, 2)];

        let nb = b.negate();
        let nc = c.negate();
        let nd = d.negate();

        matrix![
            a,  b,  c,  d;
            nb, a,  d,  nc;
            nc, nd, a,  b;
            nd, c,  nb, a;
        ]
    }
}

impl<T: Real> Add for Quaternion<T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self.scalar += rhs.scalar;
        self.vector
            .column_iter_mut()
            .zip(rhs.vector.column_iter())
            .for_each(|(x, &y)| *x += y);

        self
    }
}

impl<T: Real> Sub for Quaternion<T> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self.scalar -= rhs.scalar;
        self.vector
            .column_iter_mut()
            .zip(rhs.vector.column_iter())
            .for_each(|(x, &y)| *x -= y);

        self
    }
}

impl<T: Real> Mul for Quaternion<T> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self {
        // (q0 * p0 - q . p, q0 * p + p0 * q + q X p)
        let dot = self.vector.dot(&rhs.vector);
        let cross = self.vector.cross(&rhs.vector);

        self.vector = self.vector.scale(rhs.scalar) + rhs.vector.scale(self.scalar) + cross;
        self.scalar = self.scalar * rhs.scalar - dot;

        self
    }
}

impl<T: Real> Div<T> for Quaternion<T> {
    type Output = Self;

    fn div(mut self, rhs: T) -> Self {
        self.scalar /= rhs;
        self.vector.scale_mut(rhs.recip());

        self
    }
}

impl<T: Real> From<UnitQuaternion<T>> for Quaternion<T> {
    fn from(unit: UnitQuaternion<T>) -> Self {
        unit.into_inner()
    }
}

/// Converts a 3-dimensional vector into a pure quaternion.
impl<T: Real> From<Vector<T, 3>> for Quaternion<T> {
    fn from(vector: Vector<T, 3>) -> Self {
        Quaternion::new(T::ZERO, vector)
    }
}

/// Converts a 4-dimensional vector of the form `[x, y, z, w]`
/// into a quaternion, where `[x, y, z]` is the vector
/// component and `w` is the scalar component.
impl<T: Real> From<Vector<T, 4>> for Quaternion<T> {
    fn from(vector: Vector<T, 4>) -> Self {
        let scalar = vector.comp(3);
        let vector_3 = vector![vector.comp(0), vector.comp(1), vector.comp(2)];

        Quaternion::new(scalar, vector_3)
    }
}

#[cfg(test)]
mod quaternion_tests {
    use super::*;

    #[test]
    fn new() {
        let q = Quaternion {
            scalar: 1.0,
            vector: vector![2.0, 3.0, 4.0],
        };
        let q_new = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);

        assert_eq!(q_new, q);
    }

    #[test]
    fn conjugate() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let q_conj = Quaternion::new(1.0, vector![-2.0, -3.0, -4.0]);

        assert_eq!(q.conjugate(), q_conj);
    }

    #[test]
    fn norm() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let expected_norm = f64::sqrt(30.0);

        assert_eq!(q.norm(), expected_norm);
    }

    #[test]
    fn norm_squared() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let expected_norm_squared = 30.0;

        assert_eq!(q.norm_squared(), expected_norm_squared);
    }

    // Covers scale_mut as well.
    #[test]
    fn scale() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let scaled = Quaternion::new(2.0, vector![4.0, 6.0, 8.0]);

        assert_eq!(q.scale(2.0), scaled);
    }

    #[test]
    fn inverse() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let inverse = Quaternion::new(
            0.03333333333333333,
            vector![-0.06666666666666667, -0.1, -0.13333333333333333],
        );

        assert_eq!(q.inverse(), inverse);
    }

    #[test]
    fn into_vector() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let vector = vector![2.0, 3.0, 4.0, 1.0];

        assert_eq!(q.into_vector(), vector);
    }

    #[test]
    fn arithmetic() {
        let q = Quaternion::new(1.0, vector![2.0, 3.0, 4.0]);
        let p = Quaternion::new(5.0, vector![6.0, 7.0, 8.0]);

        let sum = Quaternion::new(6.0, vector![8.0, 10.0, 12.0]);
        let diff = Quaternion::new(-4.0, vector![-4.0, -4.0, -4.0]);
        let prod_qp = Quaternion::new(-60.0, vector![12.0, 30.0, 24.0]);
        let prod_pq = Quaternion::new(-60.0, vector![20.0, 14.0, 32.0]);
        let div_q_2 = Quaternion::new(0.5, vector![1.0, 1.5, 2.0]);
        let div_right_q_p = Quaternion::new(
            0.40229885057471265,
            vector![0.0, 0.09195402298850575, 0.04597701149425287],
        );
        let div_left_q_p = Quaternion::new(
            0.40229885057471265,
            vector![
                0.04597701149425287,
                0.000000000000000006938893903907228,
                0.09195402298850575
            ],
        );

        assert_eq!(q.clone() + p.clone(), sum);
        assert_eq!(q.clone() - p.clone(), diff);
        assert_eq!(q.clone() * p.clone(), prod_qp);
        assert_eq!(p.clone() * q.clone(), prod_pq);
        assert_eq!(q.clone() / 2.0, div_q_2);
        assert_eq!(q.clone().div_right(p.clone()), div_right_q_p);
        assert_eq!(q.clone().div_left(p.clone()), div_left_q_p);
    }
}
