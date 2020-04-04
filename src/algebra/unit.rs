use crate::{algebra::quaternion::Quaternion, core::Vector, traits::Real};

pub struct Unit<T>(T);

impl<T> Unit<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> AsRef<T> for Unit<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

pub type UnitVector<T, const N: usize> = Unit<Vector<T, { N }>>;

impl<T: Real, const N: usize> From<Vector<T, { N }>> for UnitVector<T, { N }> {
    fn from(vector: Vector<T, { N }>) -> Self {
        Unit(vector.normalize())
    }
}

pub type UnitQuaternion<T> = Unit<Quaternion<T>>;

impl<T: Real> From<Quaternion<T>> for UnitQuaternion<T> {
    fn from(quat: Quaternion<T>) -> Self {
        Unit(quat.normalize())
    }
}
