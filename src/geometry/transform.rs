use crate::{
    core::{HomogeneousMatrix, Vector},
    traits::Real,
};

pub trait Rotation<T: Real, const N: usize> {
    fn id() -> Self;
    fn rotate(&self, _: Vector<T, { N }>) -> Vector<T, { N }>;
}

pub struct Isometry<T: Real, R, const N: usize>
where
    R: Rotation<T, { N }>,
{
    translation: Vector<T, { N }>,
    rotation: R,
}

impl<T: Real, R, const N: usize> Isometry<T, R, { N }>
where
    R: Rotation<T, { N }>,
{
    pub fn new() -> Self {
        Self {
            translation: Vector::zero(),
            rotation: R::id(),
        }
    }

    // pub fn translate(mut self, v: &Vector<T, { N }>) -> Self {
    //     self.translate_mut(v);
    //     self
    // }
    //
    // pub fn translate_mut(&mut self, v: &Vector<T, { N }>) {
    //     self.matrix
    //         .column_mut(N - 1, 0)
    //         .zip(v.column_iter())
    //         .for_each(|(mat_val, &v_val)| *mat_val += v_val);
    // }
    //
    // pub fn into_inner(self) -> HomogeneousMatrix<T, { N }> {
    //     self.matrix
    // }
}
