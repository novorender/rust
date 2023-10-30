#![cfg_attr(feature="unstable", feature(test))]

#[cfg(feature = "unstable")]
mod benches {
    use std::mem::size_of;

    use wasm_parser::thin_slice::ThinSlice;

    extern crate test;
    extern crate wasm_parser;

    #[bench]
    fn copy(b: &mut test::Bencher) {
        let src0 = vec![0f32; 333_333];
        let src1 = vec![0f32; 333_333];
        let src2 = vec![0f32; 333_333];
        let mut dst = vec![0f32; 999_999];
        b.iter(|| {
            wasm_parser::interleaved::copy_to_interleaved_array(&mut dst, &src0, size_of::<f32>() * 0, size_of::<f32>() * 3, 0, src0.len());
            wasm_parser::interleaved::copy_to_interleaved_array(&mut dst, &src1, size_of::<f32>() * 1, size_of::<f32>() * 3, 0, src1.len());
            wasm_parser::interleaved::copy_to_interleaved_array(&mut dst, &src2, size_of::<f32>() * 2, size_of::<f32>() * 3, 0, src2.len());
        });
    }

    #[bench]
    fn copy_three(b: &mut test::Bencher) {
        let data0 = vec![0f32; 333_333];
        let data1 = vec![0f32; 333_333];
        let data2 = vec![0f32; 333_333];
        let src0 = ThinSlice::from_data_and_offset(bytemuck::cast_slice(&data0), 0, (data0.len() * size_of::<f32>()) as u32);
        let src1 = ThinSlice::from_data_and_offset(bytemuck::cast_slice(&data1), 0, (data1.len() * size_of::<f32>()) as u32);
        let src2 = ThinSlice::from_data_and_offset(bytemuck::cast_slice(&data2), 0, (data2.len() * size_of::<f32>()) as u32);
        let mut dst = vec![0f32; 999_999];
        b.iter(|| {
            wasm_parser::interleaved::interleave_three(
                &mut dst,
                unsafe{ src0.as_slice(data0.len() as u32) },
                unsafe{ src1.as_slice(data1.len() as u32) },
                unsafe{ src2.as_slice(data2.len() as u32) },
                size_of::<f32>() * 0,
                size_of::<f32>() * 3,
            );
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
        let bump = bumpalo::Bump::new();
        let arena = unsafe{ &*(&bump as *const bumpalo::Bump) };
        let schema = wasm_parser::types_2_1::Schema::parse(&data);
        b.iter(|| {
            let _ = schema.geometry(
                arena,
                false,
                &wasm_parser::parser::Highlights::new(arena),
                |_| true
            );
        });
    }
}