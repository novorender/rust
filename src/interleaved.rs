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

    if stride == 1 {
        dst[offset..end].fill(src);
    }else{
        for dst in dst[offset..end].iter_mut().step_by(stride) {
            *dst = src;
        }
    }
}

pub fn interleave_one<T: Copy + 'static>(dst: &mut[T], src: &[T], byte_offset: usize, byte_stride: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    if stride == 1 {
        dst[offset..offset + src.len()].copy_from_slice(src)
    }else{
        for (dst, src) in dst[offset..].iter_mut().step_by(stride).zip(src) {
            *dst = *src;
        }
    }
}

pub fn interleave_two<T: Copy + bytemuck::Pod + 'static>(dst: &mut[T], src0: &[T], src1: &[T], byte_offset: usize, byte_stride: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    if stride == 2 {
        for ((dst, src0), src1) in bytemuck::cast_slice_mut::<T, [T;2]>(&mut dst[offset..])
            .iter_mut()
            .zip(src0)
            .zip(src1)
        {
            dst[0] = *src0;
            dst[1] = *src1;
        }
    }else{
        for ((dst, src0), src1) in dst[offset..]
            .chunks_mut(stride)
            .zip(src0)
            .zip(src1)
        {
            dst[0] = *src0;
            dst[1] = *src1;
        }
    }
}

pub fn interleave_three<T: Copy + bytemuck::Pod + 'static>(dst: &mut[T], src0: &[T], src1: &[T], src2: &[T], byte_offset: usize, byte_stride: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    if stride == 3 {
        for (((dst, src0), src1), src2) in bytemuck::cast_slice_mut::<T, [T;3]>(&mut dst[offset..])
            .iter_mut()
            .zip(src0)
            .zip(src1)
            .zip(src2)
        {
            dst[0] = *src0;
            dst[1] = *src1;
            dst[2] = *src2;
        }
    }else{
        for (((dst, src0), src1), src2) in dst[offset..]
            .chunks_mut(stride)
            .zip(src0)
            .zip(src1)
            .zip(src2)
        {
            dst[0] = *src0;
            dst[1] = *src1;
            dst[2] = *src2;
        }
    }
}

pub fn interleave_four<T: Copy + bytemuck::Pod + 'static>(dst: &mut[T], src0: &[T], src1: &[T], src2: &[T], src3: &[T], byte_offset: usize, byte_stride: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    if stride == 4 {
        for ((((dst, src0), src1), src2), src3) in bytemuck::cast_slice_mut::<T, [T;4]>(&mut dst[offset..])
            .iter_mut()
            .zip(src0)
            .zip(src1)
            .zip(src2)
            .zip(src3)
        {
            dst[0] = *src0;
            dst[1] = *src1;
            dst[2] = *src2;
            dst[3] = *src3;
        }
    }else{
        for ((((dst, src0), src1), src2), src3) in dst[offset..]
            .chunks_mut(stride)
            .zip(src0)
            .zip(src1)
            .zip(src2)
            .zip(src3)
        {
            dst[0] = *src0;
            dst[1] = *src1;
            dst[2] = *src2;
            dst[3] = *src3;
        }
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
    fn test_copy_interleaved_three() {
        let src0 = [0f32;7];
        let src1 = [0f32;7];
        let src2 = [0f32, 1., 2., 3., 4., 5., 6.];
        let mut dst = [0.; 21];
        super::interleave_three(&mut dst, &src0, &src1, &src2, 0, size_of::<f32>() * 3);
        assert_eq!(dst, [0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 3., 0., 0., 4., 0., 0., 5., 0., 0., 6.]);
    }

    #[test]
    fn test_fill_interleaved() {
        let src = 1.;
        let mut dst = [0f32; 14];
        super::fill_to_interleaved_array(&mut dst, src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
        assert_eq!(dst, [0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]);
    }
}