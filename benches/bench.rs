#![cfg_attr(feature="unstable", feature(test))]

#[cfg(feature = "unstable")]
mod benches {
    use std::mem::size_of;

    extern crate test;
    extern crate wasm_parser;

    #[bench]
    fn copy(b: &mut test::Bencher) {
        let src = vec![0.;1_000_000];
        let mut dst = vec![0.;1_000_000];
        b.iter(|| {
            wasm_parser::copy_to_interleaved_array_f32(&mut dst, &src, 0, size_of::<f32>(), 0, src.len());
        });
    }

    #[bench]
    fn fill(b: &mut test::Bencher) {
        let src = 1f32;
        let mut dst = vec![0.;1_000_000];
        let end = dst.len();
        b.iter(|| {
            wasm_parser::fill_to_interleaved_array_f32(&mut dst, src, 0, size_of::<f32>(), 0, end);
        });
    }
}