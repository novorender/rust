use std::mem::size_of;

pub fn copy_to_interleaved_array<T: Copy + Send + Sync>(dst: &mut [T], src: &[T], byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    for (dst, src) in dst[offset..].iter_mut().step_by(stride).zip(&src[begin..end]) {
        *dst = *src
    }
}


pub fn fill_to_interleaved_array<T: Copy + Send + Sync>(dst: &mut [T], src: T, byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    let end = (offset + stride * (end - begin)).min(dst.len());

    for dst in dst[offset..end].iter_mut().step_by(stride) {
        *dst = src;
    }
}

#[cfg(test)]
mod test {
    use std::mem::size_of;

    #[test]
    fn test_copy_interleaved() {
        let src = [0f32, 1., 2., 3., 4., 5., 6.];
        let mut dst = [0.; 14];
        super::copy_to_interleaved_array(&mut dst, &src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
        assert_eq!(dst, [0., 0., 1., 0., 2., 0., 3., 0., 4., 0., 5., 0., 6., 0.]);
    }

    #[test]
    fn test_fill_interleaved() {
        let src = 1.;
        let mut dst = [0f32; 14];
        super::fill_to_interleaved_array(&mut dst, src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
        assert_eq!(dst, [0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]);
    }
}