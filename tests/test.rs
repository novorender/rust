use core::mem::size_of;

#[test]
fn test_copy_interleaved() {
    let src = [0f32, 1., 2., 3., 4., 5., 6.];
    let mut dst = [0.; 14];
    wasm_parser::copy_to_interleaved_array_f32(&mut dst, &src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
    assert_eq!(dst, [0., 0., 1., 0., 2., 0., 3., 0., 4., 0., 5., 0., 6., 0.]);
}

#[test]
fn test_fill_interleaved() {
    let src = 1.;
    let mut dst = [0f32; 14];
    wasm_parser::fill_to_interleaved_array_f32(&mut dst, src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
    assert_eq!(dst, [0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]);
}