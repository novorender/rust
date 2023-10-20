#![cfg_attr(feature="unstable", feature(test))]

#[cfg(feature = "unstable")]
mod benches {
    use std::mem::size_of;

    extern crate test;
    extern crate wasm_parser;

    #[bench]
    fn copy(b: &mut test::Bencher) {
        let src = vec![0f32;1_000_000];
        let mut dst = vec![0f32;1_000_000];
        b.iter(|| {
            wasm_parser::interleaved::copy_to_interleaved_array(&mut dst, &src, 0, size_of::<f32>(), 0, src.len());
        });
    }

    #[bench]
    fn fill(b: &mut test::Bencher) {
        let src = 1f32;
        let mut dst = vec![0f32;1_000_000];
        let end = dst.len();
        b.iter(|| {
            wasm_parser::interleaved::fill_to_interleaved_array(&mut dst, src, 0, size_of::<f32>(), 0, end);
        });
    }

    #[bench]
    fn parse(b: &mut test::Bencher) {
        use std::io::Read;
        let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        let mut schema = wasm_parser::types_2_1::Schema::parse(&data);
        b.iter(|| {
            schema = wasm_parser::types_2_1::Schema::parse(&data);
        });

        for p in unsafe{ schema.sub_mesh_projection.primitive_type.as_slice(schema.sub_mesh_projection.len) } {
            assert!((*p as u8) < 7);
        }
        for p in unsafe{ schema.sub_mesh_projection.attributes.as_slice(schema.sub_mesh_projection.len) } {
            let mut p = *p;
            p.remove(wasm_parser::types_2_1::OptionalVertexAttribute::NORMAL);
            p.remove(wasm_parser::types_2_1::OptionalVertexAttribute::COLOR);
            p.remove(wasm_parser::types_2_1::OptionalVertexAttribute::TEX_COORD);
            p.remove(wasm_parser::types_2_1::OptionalVertexAttribute::PROJECTED_POS);
            assert!(p.is_empty());
        }
    }



    #[bench]
    fn children(b: &mut test::Bencher) {
        use std::io::Read;
        let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        let schema = wasm_parser::types_2_1::Schema::parse(&data);
        b.iter(|| {
            let _ = schema.children(|_| true).collect::<Vec<_>>();
        });
    }


    #[bench]
    fn geometry(b: &mut test::Bencher) {
        use std::io::Read;
        let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
        let mut data = Vec::new();
        file.read_to_end(&mut data).unwrap();
        let schema = wasm_parser::types_2_1::Schema::parse(&data);
        b.iter(|| {
            let _ = schema.geometry(
                false,
                &wasm_parser::parser::Highlights::default(),
                |_| true
            );
        });
    }
}