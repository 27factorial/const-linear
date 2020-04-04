use crate::{core::Vector, geometry::angle::Angle, traits::Real};

use std::ops::{Add, Div, Mul, Sub};

pub struct Complex<T: Real> {
    real: T,
    imaginary: T,
}

impl<T: Real> Complex<T> {
    pub fn new(real: T, imaginary: T) -> Self {
        Self { real, imaginary }
    }

    pub fn re(&self) -> T {
        self.real
    }

    pub fn im(&self) -> T {
        self.imaginary
    }

    pub fn modulus(&self) -> T {
        (self.real * self.real + self.imaginary * self.imaginary).sqrt()
    }

    pub fn modulus_squared(&self) -> T {
        let modulus = self.modulus();
        modulus * modulus
    }

    pub fn scale(mut self, s: T) -> Self {
        self.scale_mut(s);
        self
    }

    pub fn scale_mut(&mut self, s: T) {
        self.real *= s;
        self.imaginary *= s;
    }

    pub fn conjugate(mut self) -> Self {
        self.conjugate_mut();
        self
    }

    pub fn conjugate_mut(&mut self) {
        self.imaginary = self.imaginary.negate();
    }

    pub fn arg(&self) -> Angle<T> {
        Angle::from_rad(self.imaginary.atan2(self.real))
    }

    pub fn arg_deg(&self) -> Angle<T> {
        self.arg().to_deg()
    }

    pub fn into_vector(self) -> Vector<T, 2> {
        vector![self.real, self.imaginary]
    }
}
