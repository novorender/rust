#[cfg(feature="cap")]
use std::alloc;
#[cfg(feature="cap")]
use cap::Cap;
use js_sys::Float32Array;
use js_sys::Int16Array;
use js_sys::Uint16Array;
use js_sys::Uint32Array;
use js_sys::Uint8Array;

#[cfg(feature="cap")]
#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::MAX);

use core::{mem, ffi::c_void};
use js_sys::Array;
use parser::Highlights;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;
use allocator_api2::vec::Vec;
use bumpalo::Bump;

pub mod reader_2_0;
pub mod reader_2_1;
pub mod types_2_1;
pub mod types_2_0;
pub mod thin_slice;
pub mod range;
pub mod parser;
pub mod ktx;
pub mod interleaved;
pub mod outlines;
mod gl_bindings;
mod js_types;

#[cfg(all(feature = "console", target_family = "wasm"))]
#[macro_export]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
    // ( $t:expr ) => {
    //     web_sys::console::log_1(&JsValue::from_str($t));
    // }
}

#[cfg(all(feature = "console", not(target_family = "wasm")))]
use log::debug as log;

#[cfg(not(feature = "console"))]
#[macro_export]
macro_rules! log {
    ($($tt:tt)*) => {

    };
}

#[wasm_bindgen]
pub fn init_console() {
    #[cfg(all(target_family="wasm", feature = "console_error_panic_hook"))]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Arena(*mut Bump);

#[wasm_bindgen]
impl Arena {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Arena {
        let bump = Bump::with_capacity(32 * 1024);
        bump.set_allocation_limit(Some(32 * 1024));
        Arena(Box::into_raw(Box::new(bump)))
    }

    pub fn with_capacity(size: usize) -> Arena {
        let bump = Bump::with_capacity(size);
        bump.set_allocation_limit(Some(size));
        Arena(Box::into_raw(Box::new(bump)))

    }

    pub fn clone(&self) -> Arena {
        Arena(self.0)
    }

    pub fn allocate_u16(&self, size: usize) -> Uint16Array {
        let bump = unsafe {&*self.0};
        let memory = bump.alloc_slice_fill_default(size);
        unsafe{ Uint16Array::view(memory) }
    }

    pub fn allocate_u32(&self, size: usize) -> Uint32Array {
        let bump = unsafe {&*self.0};
        let memory = bump.alloc_slice_fill_default(size);
        unsafe{ Uint32Array::view(memory) }
    }

    pub fn allocate_i16(&self, size: usize) -> Int16Array {
        let bump = unsafe {&*self.0};
        let memory = bump.alloc_slice_fill_default(size);
        unsafe{ Int16Array::view(memory) }
    }

    pub fn allocate_f32(&self, size: usize) -> Float32Array {
        let bump = unsafe {&*self.0};
        let memory = bump.alloc_slice_fill_default(size);
        unsafe{ Float32Array::view(memory) }
    }

    pub fn reset(&mut self) {
        let bump = unsafe {&mut *self.0};
        bump.reset()
    }
}

#[wasm_bindgen]
pub struct Schema {
    _data: Vec<u8>,
    version: &'static str,
    schema: *mut c_void,
    arena: Arena,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "{subMeshes: Array<ReturnSubMesh_2_0 | ReturnSubMesh_2_1>, textures: Array<Texture_2_0 | Texture_2_1>}")]
    pub type NodeGeometry;
    #[wasm_bindgen(typescript_type = "Array<Child_2_0> | Array<Child_2_1>")]
    pub type ArrayChild;
}

impl Schema {
    pub fn schema_2_0(&self) -> Option<&types_2_0::Schema> {
        if self.version == "2.0" {
            // SAFETY: schema is put in a `Box` when created in `Schema::parse` and we check
            // it's the correct type through the version check
            Some(unsafe{ &*(self.schema as *mut types_2_0::Schema) })
        }else{
            None
        }
    }

    pub fn schema_2_1(&self) -> Option<&types_2_1::Schema> {
        if self.version == "2.1" {
            // SAFETY: schema is put in a `Box` when created in `Schema::parse` and we check
            // it's the correct type through the version check
            Some(unsafe{ &*(self.schema as *mut types_2_1::Schema) })
        }else{
            None
        }
    }
}

#[wasm_bindgen]
impl Schema {
    pub fn parse_2_0(data_js: &Uint8Array, arena: Arena) -> Schema {
        let layout = std::alloc::Layout::from_size_align(data_js.length() as usize, 4)
            .unwrap();
        let ptr = unsafe{ &*arena.0 }.alloc_layout(layout);
        let mut data = unsafe{ Vec::from_raw_parts(
            ptr.as_ptr(),
            data_js.length() as usize,
            data_js.length() as usize
        ) };
        // let mut data = Vec::with_capacity_in(data_js.length() as usize, unsafe{ &*arena.0 });
        // unsafe{ data.set_len(data_js.length() as usize) };
        data_js.copy_to(&mut data);
        let schema = types_2_0::Schema::parse(&data);
        Schema {
            version: "2.0",
            schema: Box::into_raw(Box::new(schema)) as *mut c_void,
            _data: data,
            arena
        }
    }

    pub fn parse_2_1(data_js: &Uint8Array, arena: Arena) -> Schema {
        let layout = std::alloc::Layout::from_size_align(data_js.length() as usize, 4)
            .unwrap();
        let ptr = unsafe{ &*arena.0 }.alloc_layout(layout);
        let mut data = unsafe{ Vec::from_raw_parts(
            ptr.as_ptr(),
            data_js.length() as usize,
            data_js.length() as usize
        ) };
        // let mut data = Vec::with_capacity_in(data_js.length() as usize, unsafe{ &*arena.0 });
        // unsafe{ data.set_len(data_js.length() as usize) };
        data_js.copy_to(&mut data);
        let schema = types_2_1::Schema::parse(&data);
        Schema {
            version: "2.1",
            schema: Box::into_raw(Box::new(schema)) as *mut c_void,
            _data: data,
            arena
        }
    }

    pub fn children(&self) -> ArrayChild {
        let array: Array = match self.version {
            "2.0" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_0::Schema) };
                schema.children(|_| true)
                    .map(JsValue::from)
                    .collect()
            }
            "2.1" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_1::Schema) };
                schema.children(|_| true)
                    .map(JsValue::from)
                    .collect()
            }
            _ => todo!("version {} not implemented yet", self.version)
        };
        let js_value: JsValue = array.into();
        js_value.into()
    }

    pub fn create_highlights(&self, indices_js: &Uint8Array) -> Highlights {
        let mut indices = Vec::with_capacity_in(indices_js.length() as usize, unsafe{ &*self.arena.0 });
        unsafe{ indices.set_len(indices_js.length() as usize) };
        indices_js.copy_to(&mut indices);
        Highlights { indices }
    }

    pub fn geometry(&self, enable_outlines: bool, apply_filter: bool, highlights: Highlights) -> NodeGeometry {
        #[cfg(feature="memory-monitor")]
        log!("Currently allocated: {}MB", ALLOCATOR.allocated() as f32 / 1024. / 1024.);

        #[wasm_bindgen]
        pub struct InnerNodeGeometry {
            sub_meshes: Array,
            textures: Array,
        }

        #[wasm_bindgen]
        impl InnerNodeGeometry {
            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "subMeshes")]
            pub fn sub_meshes(&self) -> Array {
                self.sub_meshes.clone()
            }

            #[wasm_bindgen(getter)]
            pub fn textures(&self) -> Array {
                self.textures.clone()
            }
        }

        let js_value: JsValue = match self.version {
            "2.0" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_0::Schema) };
                let (sub_meshes, textures) = schema.geometry(
                    unsafe{ &*self.arena.0 },
                    enable_outlines,
                    &highlights,
                    |object_id| !apply_filter || highlights.indices[object_id as usize] != u8::MAX
                );
                let sub_meshes: Array = sub_meshes
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                let textures: Array = textures
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                InnerNodeGeometry{
                    sub_meshes,
                    textures
                }.into()
            }
            "2.1" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_1::Schema) };
                let (sub_meshes, index) = schema.geometry(
                    unsafe{ &*self.arena.0 },
                    enable_outlines,
                    &highlights,
                    |object_id| !apply_filter || highlights.indices[object_id as usize] != u8::MAX
                );
                let sub_meshes: Array = sub_meshes
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                let textures: Array = index
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                InnerNodeGeometry{
                    sub_meshes,
                    textures
                }.into()
            }
            _ => todo!("version {} not implemented yet", self.version)
        };
        js_value.into()
    }
}

impl Drop for Schema {
    fn drop(&mut self) {
        // SAFETY: schema is put in a `Box` when created in `Schema::parse`
        match self.version {
            "2.0" => mem::drop(unsafe{ Box::from_raw(self.schema as *mut types_2_0::Schema) }),
            "2.1" => mem::drop(unsafe{ Box::from_raw(self.schema as *mut types_2_1::Schema) }),
            _ => todo!("version {} not implemented yet", self.version)
        }

        #[cfg(feature="memory-monitor")]
        log!("Currently allocated after dropping schema: {}MB", ALLOCATOR.allocated() as f32 / 1024. / 1024.);
    }
}

#[cfg(feature="wasm-bindgen-bench")]
#[wasm_bindgen]
pub fn bench_intersections() {
    use glam::*;
    use rand::random;
    use std::cell::UnsafeCell;
    use std::hint::black_box;

    const INTERSECTIONS_LEN: usize = 1_000_000;

    fn model_local_matrix(local_space_translation: Vec3, offset: Vec3, scale: f32) -> Mat4 {
        let (ox, oy, oz) = (offset.x, offset.y, offset.z);
        let (tx, ty, tz) = (local_space_translation.x, local_space_translation.y, local_space_translation.z);
        mat4(
            vec4(scale, 0., 0., 0.),
            vec4(0., scale, 0., 0.),
            vec4(0., 0., scale, 0.),
            vec4(ox - tx, oy - ty, oz - tz, 1.),
        )
    }

    let pos = (0..INTERSECTIONS_LEN).flat_map(|_| {
        let p0 = (random::<i16>(), random::<i16>(), random::<i16>());
        let p1 = (random::<i16>(), random::<i16>(), random());
        let p2 = (random::<i16>(), random::<i16>(), random::<i16>());
        [p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2]
    }).collect::<Vec<_>>();

    let sequential_idx = (0 .. INTERSECTIONS_LEN as u32).flat_map(|i|
        [i * 3 + 0, i * 3 + 1, i * 3 + 2]
    ).collect::<Vec<_>>();

    let random_offset_idx = (0 .. INTERSECTIONS_LEN as u32).flat_map(|i| {
        let idx_offset = ((random::<f32>() * 2. - 1.) * 100.) as u32;
        let i = (i + idx_offset).min(INTERSECTIONS_LEN as u32 - 1);
        [i * 3 + 0, i * 3 + 1, i * 3 + 2]
    }).collect::<Vec<_>>();

    let random_idx = (0 .. INTERSECTIONS_LEN).flat_map(|_| {
        let i = (random::<f32>() * INTERSECTIONS_LEN as f32) as u32;
        [i * 3 + 0, i * 3 + 1, i * 3 + 2]
    }).collect::<Vec<_>>();

    let output = vec![0f32; INTERSECTIONS_LEN * 10];
    let output = UnsafeCell::new(output);

    let local_space_translation = vec3(100., 10., 1000.);
    let offset = vec3(10., 1000., 10000.);
    let scale = 3.;
    let plane_normal = vec3(1., 3., 5.).normalize();
    let plane_offset = 2000.;
    let plane = vec4(plane_normal.x, plane_normal.y, plane_normal.z, plane_offset);
    let model_local_matrix = model_local_matrix(local_space_translation, offset, scale);
    let (plane_local_matrix, local_plane_matrix) = crate::outlines::plane_matrices(plane, local_space_translation);
    let model_to_plane_mat = local_plane_matrix * model_local_matrix * crate::outlines::DENORM_MATRIX;

    crate::log!("sequential idx: {}", easybench_wasm::bench(|| {
        let idx = black_box(sequential_idx.as_slice());
        let pos = black_box(pos.as_slice());
        let model_to_plane_mat = black_box(model_to_plane_mat);
        let output = black_box(unsafe{ &mut *output.get() });
        crate::outlines::intersect_triangles(idx, pos, model_to_plane_mat, output);
        output
    }));

    crate::log!("random offset idx: {}", easybench_wasm::bench(|| {
        let idx = black_box(random_offset_idx.as_slice());
        let pos = black_box(pos.as_slice());
        let model_to_plane_mat = black_box(model_to_plane_mat);
        let output = black_box(unsafe{ &mut *output.get() });
        crate::outlines::intersect_triangles(idx, pos, model_to_plane_mat, output);
        output
    }));

    crate::log!("random idx: {}", easybench_wasm::bench(|| {
        let idx = black_box(random_idx.as_slice());
        let pos = black_box(pos.as_slice());
        let model_to_plane_mat = black_box(model_to_plane_mat);
        let output = black_box(unsafe{ &mut *output.get() });
        crate::outlines::intersect_triangles(idx, pos, model_to_plane_mat, output);
        output
    }));
}