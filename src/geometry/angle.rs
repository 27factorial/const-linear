use crate::traits::Real;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Copy, Clone, Debug)]
pub enum Angle<T: Real> {
    Deg(T),
    Rad(T),
}

impl<T: Real> Angle<T> {
    pub const fn from_deg(val: T) -> Self {
        Self::Deg(val)
    }

    pub const fn from_rad(val: T) -> Self {
        Self::Rad(val)
    }

    pub fn to_rad(self) -> Self {
        match self {
            Self::Deg(val) => Self::Rad(val * T::pi() / T::from_scalar(180)),
            Self::Rad(val) => Self::Rad(val),
        }
    }

    pub fn to_deg(self) -> Self {
        match self {
            Self::Deg(val) => Self::Deg(val),
            Self::Rad(val) => Self::Deg(val * T::from_scalar(180) / T::pi()),
        }
    }

    pub fn is_deg(&self) -> bool {
        matches!(self, Self::Deg(_))
    }

    pub fn is_rad(&self) -> bool {
        matches!(self, Self::Rad(_))
    }

    pub fn val(&self) -> T {
        match self {
            Self::Deg(val) => *val,
            Self::Rad(val) => *val,
        }
    }
}

impl<T: Real> Add for Angle<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match self {
            Self::Deg(_) => {
                let rhs_val = rhs.to_deg().val();
                Self::Deg(self.val() + rhs_val)
            }
            Self::Rad(_) => {
                let rhs_val = rhs.to_rad().val();
                Self::Rad(self.val() + rhs_val)
            }
        }
    }
}

impl<T: Real> Sub for Angle<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        match self {
            Self::Deg(_) => {
                let rhs_val = rhs.to_deg().val();
                Self::Deg(self.val() - rhs_val)
            }
            Self::Rad(_) => {
                let rhs_val = rhs.to_rad().val();
                Self::Rad(self.val() - rhs_val)
            }
        }
    }
}

impl<T: Real> Mul<Self> for Angle<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        match self {
            Self::Deg(_) => {
                let rhs_val = rhs.to_deg().val();
                Self::Deg(self.val() * rhs_val)
            }
            Self::Rad(_) => {
                let rhs_val = rhs.to_rad().val();
                Self::Rad(self.val() * rhs_val)
            }
        }
    }
}

impl<T: Real> Mul<T> for Angle<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        match self {
            Self::Deg(_) => Self::Deg(self.val() * rhs),
            Self::Rad(_) => Self::Rad(self.val() * rhs),
        }
    }
}

impl<T: Real> Div<T> for Angle<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self {
        match self {
            Self::Deg(_) => Self::Deg(self.val() / rhs),
            Self::Rad(_) => Self::Rad(self.val() / rhs),
        }
    }
}
