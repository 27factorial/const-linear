// Suppress the unused warnings from rustc, as eventually
// everything in this module will be used.
#![allow(unused)]

use std::ptr;
use std::{
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::{Index, IndexMut},
};

// This is akin to a transmute from MaybeUninit<T> -> T, as
// neither mem::transmute nor MaybeUninit::assume_init() are
// const fns.
const unsafe fn const_assume_init<T>(this: MaybeUninit<T>) -> T {
    union Init<Ty> {
        maybe_uninit: MaybeUninit<Ty>,
        init: ManuallyDrop<Ty>,
    }

    let transmute = Init { maybe_uninit: this };

    ManuallyDrop::into_inner(transmute.init)
}

// This is a workaround for use until lazy normalization is implemented.
// It allows arrays to use a generic parameter as a size. This works under
// the assumption that both ManuallyDrop<T> and MaybeUninit<T> are
// repr(transparent), and casting between them should be fine
// (so long as T is initialized). Since reads and writes of zero-sized types
// are no-ops anyways, the uninit field is ().
pub(crate) union MaybeArray<T, const N: usize> {
    uninit: (),
    maybe_uninit: ManuallyDrop<[MaybeUninit<T>; N]>,
    init: ManuallyDrop<[T; N]>,
}

impl<T, const N: usize> MaybeArray<T, { N }> {
    /// Creates a new `MaybeArray<T, N>` in an uninitialized state.
    pub(crate) const fn uninit() -> Self {
        MaybeArray { uninit: () }
    }

    /// Creates a new `MaybeArray<T, N>` from an array. It is always
    /// safe to call `init()` on this value.
    pub(crate) const fn from_array(array: [T; N]) -> Self {
        Self {
            init: ManuallyDrop::new(array),
        }
    }

    pub(crate) const fn get(&self, idx: usize) -> &MaybeUninit<T> {
        let maybe_uninit =
            unsafe { &*(&self.maybe_uninit as *const _ as *const [MaybeUninit<T>; N]) };
        &maybe_uninit[idx]
    }

    pub(crate) const fn get_mut(&mut self, idx: usize) -> &mut MaybeUninit<T> {
        let maybe_uninit =
            unsafe { &mut *(&mut self.maybe_uninit as *mut _ as *mut [MaybeUninit<T>; N]) };
        &mut maybe_uninit[idx]
    }

    /// Returns an array that is assumed to be in an initialized state. Calling
    /// this method on a non-initialized `MaybeArray<T>` is undefined behavior.
    pub(crate) const unsafe fn assume_init(self) -> [T; N] {
        ManuallyDrop::into_inner(self.init)
    }
}

impl<T: Copy + Clone, const N: usize> MaybeArray<T, { N }> {
    /// Creates a new `MaybeArray<T, N>` from a specified element. It
    /// is always safe to call `init()` this value.
    pub(crate) const fn from_elem(t: T) -> Self {
        let mut uninit = Self::uninit();
        let mut idx = 0;

        while idx < N {
            *uninit.get_mut(idx) = MaybeUninit::new(t);
            idx += 1;
        }

        uninit
    }
}

pub struct PushError<T>(pub(crate) T);
pub struct PopError;

pub(crate) struct ConstVec<T, const N: usize> {
    inner: MaybeArray<T, { N }>,
    length: usize,
    capacity: usize,
}

impl<T, const N: usize> ConstVec<T, { N }> {
    /// Constructs a new ConstVec<T> with length 0 and
    /// capacity N.
    pub const fn new() -> Self {
        Self {
            inner: MaybeArray::uninit(),
            length: 0,
            capacity: N,
        }
    }

    /// Constructs a ConstVec<T> from an array of length N.
    pub const fn from_array(array: [T; N]) -> Self {
        Self {
            inner: MaybeArray::from_array(array),
            length: N,
            capacity: N,
        }
    }

    /// Returns the length of this vector.
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns the capacity of this vector.
    pub const fn capacity(&self) -> usize {
        N
    }

    pub const fn get(&self, idx: usize) -> Option<&T> {
        if idx > self.length - 1 {
            None
        } else {
            unsafe {
                let ret = &*(&self.inner.get(idx) as *const _ as *const T);
                Some(ret)
            }
        }
    }

    pub const fn as_slice(&self) -> &[T] {
        unsafe {
            let first_ptr = &self.inner.maybe_uninit as *const _ as *const T;
            &*ptr::slice_from_raw_parts(first_ptr, self.length)
        }
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            let first_ptr = &mut self.inner.maybe_uninit as *mut _ as *mut T;
            &mut *ptr::slice_from_raw_parts_mut(first_ptr, self.length)
        }
    }

    /// Pushes an element to the vector, returning it if the vector is
    /// full.
    pub const fn push(&mut self, t: T) -> Result<(), PushError<T>> {
        if self.length == self.capacity {
            return Err(PushError(t));
        } else {
            *self.inner.get_mut(self.length) = MaybeUninit::new(t);
            self.length += 1;
            Ok(())
        }
    }

    /// Pops an element from the vector, returning an error if
    /// there are zero elements. Note that if the type you pop
    /// happens to be Copy and you need a const fn version, you
    /// should call pop_copy instead.
    pub fn pop(&mut self) -> Result<T, PopError> {
        if self.length == 0 {
            return Err(PopError);
        } else {
            unsafe {
                let ret = mem::replace(
                    &mut self.inner.maybe_uninit[self.length - 1],
                    MaybeUninit::uninit(),
                )
                .assume_init();
                self.length -= 1;
                Ok(ret)
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.as_slice().iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.as_mut_slice().iter_mut()
    }
}

impl<T: Copy + Clone, const N: usize> ConstVec<T, { N }> {
    /// A const fn version of pop which only works for Copy types.
    pub const fn pop_copy(&mut self) -> Result<T, PopError> {
        if self.length == 0 {
            return Err(PopError);
        } else {
            unsafe {
                let ret = const_assume_init(*self.inner.get(self.length - 1));
                self.length -= 1;
                Ok(ret)
            }
        }
    }
}

impl<T, const N: usize> Index<usize> for ConstVec<T, { N }> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        &self.as_slice()[idx]
    }
}

impl<T, const N: usize> IndexMut<usize> for ConstVec<T, { N }> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.as_mut_slice()[idx]
    }
}

#[cfg(test)]
mod maybe_array_tests {
    use super::*;

    #[test]
    fn from_elem() {
        let array = unsafe { MaybeArray::<_, 32>::from_elem(1).assume_init() };

        assert_eq!(array.len(), 32);

        for elem in &array[..] {
            assert_eq!(*elem, 1);
        }
    }
}
