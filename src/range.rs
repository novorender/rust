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

    pub fn thin_iter(&self) -> RangeIter<'a, T> {
        RangeIter { start: self.start.iter(), count: self.count.iter() }
    }

    pub unsafe fn range(&self, range: &Range<u32>) -> RangeSlice<'a, T>
    where T: 'a
    {
        RangeSlice {
            start: unsafe{ self.start.range(range) },
            count: unsafe{ self.count.range(range) },
        }
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
            pub struct $name (pub $crate::range::Range<u32>);

            pub struct [<$name ThinIter>]<'a> ($crate::range::RangeIter<'a, $ty>);

            impl<'a> [<$name Slice>]<'a> {
                pub fn thin_iter(&self) -> [<$name ThinIter>]<'a> {
                    [<$name ThinIter>] (self.0.thin_iter())
                }

                pub unsafe fn get_unchecked(&self, index: usize) -> $name {
                    let range = unsafe{ self.0.get_unchecked(index) };
                    $name($crate::range::Range{
                        start: range.start as u32,
                        count: range.count as u32,
                    })
                }

                pub unsafe fn range(&self, range: &$crate::range::Range<u32>) -> [<$name Slice>]<'a> {
                    [<$name Slice>](self.0.range(range))
                }
            }

            impl<'a> $crate::thin_slice::ThinSliceIterator for [<$name ThinIter>]<'a> {
                type Item = $name;

                #[inline(always)]
                unsafe fn next(&mut self) -> $name {
                    let range = unsafe{ self.0.next() };
                    $name($crate::range::Range{
                        start: range.start as u32,
                        count: range.count as u32,
                    })
                }
            }

            impl core::ops::Deref for $name {
                type Target = $crate::range::Range<u32>;

                #[inline(always)]
                fn deref(&self) -> &Self::Target {
                    &self.0
                }
            }
        }
    };
}