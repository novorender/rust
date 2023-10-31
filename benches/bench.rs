#![cfg_attr(feature="unstable", feature(test))]

#[cfg(feature = "unstable")]
mod benches {
    use std::{mem::size_of, hint::black_box};

    use na::*;
    use rand::random;
    use wasm_parser::{thin_slice::ThinSlice, Arena};

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

    #[bench]
    fn intersections(b: &mut test::Bencher) {
        fn model_local_matrix(local_space_translation: Vec3, offset: Vec3, scale: f32) -> Mat4 {
            let (ox, oy, oz) = (offset.x, offset.y, offset.z);
            let (tx, ty, tz) = (local_space_translation.x, local_space_translation.y, local_space_translation.z);
            Mat4::new(
                scale, 0., 0., 0.,
                0., scale, 0., 0.,
                0., 0., scale, 0.,
                ox - tx, oy - ty, oz - tz, 1.,
            )
        }

        let pos = (0..1_000_000).flat_map(|_| {
            let p0 = (random::<i16>(), random::<i16>(), random::<i16>());
            let p1 = (random::<i16>(), random::<i16>(), random());
            let p2 = (random::<i16>(), random::<i16>(), random::<i16>());
            [p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]
        }).collect::<Vec<_>>();

        let idx = (0u32 .. 1_000_000).flat_map(|i|
            [i * 3 + 0, i * 3 + 1, i * 3 + 2]
        ).collect::<Vec<_>>();

        let mut arena = Arena::new();

        let mut output = vec![0.; 10_000_000];

        let local_space_translation = vec3(100., 10., 1000.);
        let offset = vec3(10., 1000., 10000.);
        let scale = 3.;
        let plane_normal = vec3(1., 3., 5.).normalize();
        let plane_offset = 2000.;
        let plane = vec4!(plane_normal, plane_offset);
        let model_local_matrix = model_local_matrix(local_space_translation, offset, scale);
        let (plane_local_matrix, local_plane_matrix) = wasm_parser::outlines::plane_matrices(plane, local_space_translation);
        let model_to_plane_mat = local_plane_matrix * model_local_matrix * wasm_parser::outlines::DENORM_MATRIX;

        b.iter(|| {
            let idx = test::black_box(idx.as_slice());
            let pos = test::black_box(pos.as_slice());
            let model_to_plane_mat = test::black_box(model_to_plane_mat);
            let output = test::black_box(output.as_mut_slice());
            wasm_parser::outlines::intersect_triangles(idx, pos, model_to_plane_mat, output);
            black_box(output);
            arena.reset();
        });
    }

    #[bench]
    fn intersections_random_idx(b: &mut test::Bencher) {
        fn model_local_matrix(local_space_translation: Vec3, offset: Vec3, scale: f32) -> Mat4 {
            let (ox, oy, oz) = (offset.x, offset.y, offset.z);
            let (tx, ty, tz) = (local_space_translation.x, local_space_translation.y, local_space_translation.z);
            Mat4::new(
                scale, 0., 0., 0.,
                0., scale, 0., 0.,
                0., 0., scale, 0.,
                ox - tx, oy - ty, oz - tz, 1.,
            )
        }

        let pos = (0..1_000_000).flat_map(|_| {
            let p0 = (random::<i16>(), random::<i16>(), random::<i16>());
            let p1 = (random::<i16>(), random::<i16>(), random());
            let p2 = (random::<i16>(), random::<i16>(), random::<i16>());
            [p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]
        }).collect::<Vec<_>>();

        let idx = (0u32 .. 1_000_000).flat_map(|i| {
            let idx_offset = ((random::<f32>() * 2. - 1.) * 100.) as u32;
            let i = (i + idx_offset).min(999_999);
            [i * 3 + 0, i * 3 + 1, i * 3 + 2]
        }).collect::<Vec<_>>();

        let mut arena = Arena::new();

        let mut output = vec![0.; 10_000_000];

        let local_space_translation = vec3(100., 10., 1000.);
        let offset = vec3(10., 1000., 10000.);
        let scale = 3.;
        let plane_normal = vec3(1., 3., 5.).normalize();
        let plane_offset = 2000.;
        let plane = vec4!(plane_normal, plane_offset);
        let model_local_matrix = model_local_matrix(local_space_translation, offset, scale);
        let (plane_local_matrix, local_plane_matrix) = wasm_parser::outlines::plane_matrices(plane, local_space_translation);
        let model_to_plane_mat = local_plane_matrix * model_local_matrix * wasm_parser::outlines::DENORM_MATRIX;

        b.iter(|| {
            let idx = test::black_box(idx.as_slice());
            let pos = test::black_box(pos.as_slice());
            let model_to_plane_mat = test::black_box(model_to_plane_mat);
            let output = test::black_box(output.as_mut_slice());
            wasm_parser::outlines::intersect_triangles(idx, pos, model_to_plane_mat, output);
            black_box(output);
            arena.reset();
        });
    }
}