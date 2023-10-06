use crate::thin_slice::{ThinSlice, ThinSliceIter, ThinSliceIterator};

#[derive(Clone, Copy)]
pub struct RangeSlice<'a, T> {
    pub start: ThinSlice<'a, T>,
    pub count: ThinSlice<'a, T>,
}

#[derive(Clone, Copy, Debug)]
pub struct Range<T> {
    pub start: T,
    pub count: T,
}

impl<'a, T: Copy> RangeSlice<'a, T> {
    pub unsafe fn get_unchecked(&self, index: usize) -> Range<T> {
        Range{
            start: *self.start.get_unchecked(index),
            count: *self.count.get_unchecked(index)
        }
    }

    pub fn iter(&self) -> RangeIter<'a, T> {
        RangeIter { start: self.start.iter(), count: self.count.iter() }
    }
}

impl<T: std::ops::Add<Output = T> + Copy> Into<std::ops::Range<T>> for Range<T> {
    fn into(self) -> std::ops::Range<T> {
        self.start .. self.start + self.count
    }
}
pub struct RangeIter<'a, T> {
    pub(crate) start: ThinSliceIter<'a, T>,
    pub(crate) count: ThinSliceIter<'a, T>,
}

impl<'a, T: Copy + 'a> ThinSliceIterator for RangeIter<'a, T> {
    type Item = Range<T>;

    #[inline(always)]
    unsafe fn next(&mut self) -> Range<T> {
        Range {
            start: unsafe{ *self.start.next() },
            count: unsafe{ *self.count.next() },
        }
    }
}

#[macro_export(local_inner_macros)]
macro_rules! impl_range {
    (
        $(#[$outer:meta])*
        $name: ident: $ty: ty
    ) => {
        paste::paste!{
            $(#[$outer])*
            #[derive(Clone, Copy)]
            pub struct [<$name Slice>]<'a> (pub $crate::range::RangeSlice<'a, $ty>);

            $(#[$outer])*
            pub struct $name (pub $crate::range::Range<$ty>);

            pub struct [<$name Iter>]<'a> ($crate::range::RangeIter<'a, $ty>);

            impl<'a> [<$name Slice>]<'a> {
                pub fn iter(&self) -> [<$name Iter>]<'a> {
                    [<$name Iter>] (self.0.iter())
                }

                pub unsafe fn get_unchecked(&self, index: usize) -> $name {
                    $name(self.0.get_unchecked(index))
                }
            }

            impl<'a> $crate::thin_slice::ThinSliceIterator for [<$name Iter>]<'a> {
                type Item = $name;

                #[inline(always)]
                unsafe fn next(&mut self) -> $name {
                    $name(unsafe{ self.0.next() })
                }
            }

            impl core::ops::Deref for $name {
                type Target = $crate::range::Range<$ty>;

                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }
        }
    };
}