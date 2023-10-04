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

    #[bench]
    fn parse(b: &mut test::Bencher) {
        use std::io::Read;
        let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        let mut schema = wasm_parser::types::Schema::read(&data).unwrap();
        b.iter(|| {
            schema = wasm_parser::types::Schema::read(&data).unwrap();
        });

        for p in schema.sub_mesh_projection.primitive_type() {
            assert!((*p as u8) < 7);
        }
        for p in schema.sub_mesh_projection.attributes() {
            let mut p = *p;
            p.remove(wasm_parser::types::OptionalVertexAttribute::NORMAL);
            p.remove(wasm_parser::types::OptionalVertexAttribute::COLOR);
            p.remove(wasm_parser::types::OptionalVertexAttribute::TEX_COORD);
            p.remove(wasm_parser::types::OptionalVertexAttribute::PROJECTED_POS);
            assert!(p.is_empty());
        }
    }



    #[bench]
    fn children(b: &mut test::Bencher) {
        use std::io::Read;
        let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        let schema = wasm_parser::types::Schema::read(&data).unwrap();
        b.iter(|| {
            let _ = schema.children(true, |_| true).collect::<Vec<_>>();
        });
    }
}