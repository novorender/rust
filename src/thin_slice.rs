use core::{ptr::NonNull, marker::PhantomData, slice};
use crate::range::RangeInstance;

#[derive(Clone, Copy)]
pub struct ThinSlice<'a, T> {
    start: NonNull<T>,
    #[cfg(debug_assertions)]
    len: usize,
    marker: PhantomData<&'a ()>,
}

impl<'a, T> ThinSlice<'a, T> {
    pub fn from_data_and_offset(data: &'a [u8], offset: usize, _len: u32) -> ThinSlice<'a, T> {
        let ptr = unsafe{ NonNull::new_unchecked(data[offset..].as_ptr() as *const T as *mut T) };
        ThinSlice {
            start: ptr,
            #[cfg(debug_assertions)]
            len: _len as usize,
            marker: PhantomData,
        }
    }

    pub fn empty() -> ThinSlice<'a, T>
    where T: 'a
    {
        Self::from(&[])
    }

    pub unsafe fn as_slice(self, len: u32) -> &'a [T] {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.len, len as usize);

        NonNull::slice_from_raw_parts(self.start, len as usize).as_ref()
    }

    pub unsafe fn slice_range(self, range: RangeInstance<u32>) -> &'a [T] {
        #[cfg(debug_assertions)]
        debug_assert!(self.len >= (range.start + range.count) as usize, "len: {} range: {:?}", self.len, range);

        slice::from_raw_parts(self.start.as_ptr().add(range.start as usize), range.count as usize).as_ref()
    }

    pub unsafe fn range(self, range: RangeInstance<u32>) -> ThinSlice<'a, T>
    where T: 'a
    {
        #[cfg(debug_assertions)]
        debug_assert!(self.len >= (range.start + range.count) as usize);

        self.slice_range(range).into()
    }

    pub unsafe fn get_unchecked(self, index: usize) -> &'a T {
        #[cfg(debug_assertions)]
        debug_assert!(index < self.len);

        &*self.start.as_ptr().add(index)
    }

    pub fn iter(self) -> ThinSliceIter<'a, T> {
        ThinSliceIter {
            start: self.start,
            #[cfg(debug_assertions)]
            len: self.len,
            marker: PhantomData
        }
    }

    #[cfg(debug_assertions)]
    pub fn len(&self) -> usize {
        self.len
    }
}

pub struct ThinSliceIter<'a, T> {
    start: NonNull<T>,
    #[cfg(debug_assertions)]
    len: usize,
    marker: PhantomData<&'a ()>,
}

impl<'a, T: 'a> ThinSliceIterator for ThinSliceIter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        unsafe{
            #[cfg(debug_assertions)]
            {
                debug_assert_ne!(self.len, 0);
                self.len -= 1;
            }
            let ret = self.start.as_ref();
            self.start = NonNull::new_unchecked(self.start.as_ptr().add(1));
            ret
        }
    }
}

impl<'a, T> From<&'a [T]> for ThinSlice<'a, T> {
    fn from(value: &'a [T]) -> Self {
        let ptr = value.as_ptr() as *mut T;
        ThinSlice {
            // SAFETY: We are referencing into a slice so it can't be null
            start: unsafe{ NonNull::new_unchecked(ptr) },
            #[cfg(debug_assertions)]
            len: value.len(),
            marker: PhantomData,
        }
    }
}

impl<'a, T> From<&'a [T;0]> for ThinSlice<'a, T> {
    fn from(value: &'a [T;0]) -> Self {
        let ptr = value.as_ptr() as *mut T;
        ThinSlice {
            // SAFETY: We are referencing into a slice so it can't be null
            start: unsafe{ NonNull::new_unchecked(ptr) },
            #[cfg(debug_assertions)]
            len: value.len(),
            marker: PhantomData,
        }
    }
}

pub trait ThinSliceIterator {
    type Item;

    unsafe fn next(&mut self) -> Self::Item;
}
