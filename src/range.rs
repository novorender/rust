use crate::thin_slice::{ThinSlice, ThinSliceIter, ThinSliceIterator};

#[derive(Clone, Copy)]
pub struct Range<'a, T> {
    pub start: ThinSlice<'a, T>,
    pub count: ThinSlice<'a, T>,
}

#[derive(Clone, Copy, Debug)]
pub struct RangeInstance<T> {
    pub start: T,
    pub count: T,
}

impl<'a, T: Copy> Range<'a, T> {
    pub unsafe fn get_unchecked(&self, index: usize) -> RangeInstance<T> {
        RangeInstance{
            start: *self.start.get_unchecked(index),
            count: *self.count.get_unchecked(index)
        }
    }

    pub fn iter(&self) -> RangeIter<'a, T> {
        RangeIter { start: self.start.iter(), count: self.count.iter() }
    }
}

impl<T: std::ops::Add<Output = T> + Copy> Into<std::ops::Range<T>> for RangeInstance<T> {
    fn into(self) -> std::ops::Range<T> {
        self.start .. self.start + self.count
    }
}
pub struct RangeIter<'a, T> {
    pub(crate) start: ThinSliceIter<'a, T>,
    pub(crate) count: ThinSliceIter<'a, T>,
}

impl<'a, T: Copy + 'a> ThinSliceIterator for RangeIter<'a, T> {
    type Item = RangeInstance<T>;

    #[inline(always)]
    unsafe fn next(&mut self) -> RangeInstance<T> {
        RangeInstance {
            start: unsafe{ *self.start.next() },
            count: unsafe{ *self.count.next() },
        }
    }
}

#[macro_export]
macro_rules! impl_range_iter {
    ($name: ident, $ty: ty) => {
        paste::paste!{
            pub struct [<$name Instance>] (pub $crate::range::RangeInstance<$ty>);

            pub struct [<$name Iter>]<'a> ($crate::range::RangeIter<'a, $ty>);

            impl<'a> $name<'a> {
                pub fn iter(&self) -> [<$name Iter>]<'a> {
                    [<$name Iter>] (self.0.iter())
                }

                pub unsafe fn get_unchecked(&self, index: usize) -> [<$name Instance>] {
                    [<$name Instance>](self.0.get_unchecked(index))
                }
            }

            impl<'a> $crate::thin_slice::ThinSliceIterator for [<$name Iter>]<'a> {
                type Item = [<$name Instance>];

                #[inline(always)]
                unsafe fn next(&mut self) -> [<$name Instance>] {
                    [<$name Instance>](unsafe{ self.0.next() })
                }
            }
        }
    };
}