// #![cfg_attr(target_family="wasm", no_std)]

#[cfg(feature="cap")]
use std::alloc;
#[cfg(feature="cap")]
use cap::Cap;

#[cfg(feature="cap")]
#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::MAX);

use core::{mem, ffi::c_void};
use js_sys::Array;
use parser::Highlights;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;

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
pub struct Schema {
    _data: Vec<u8>,
    version: &'static str,
    schema: *mut c_void,
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
    pub fn parse_2_0(data: Vec<u8>) -> Schema {
        let schema = types_2_0::Schema::parse(&data);
        Schema {
            version: "2.0",
            schema: Box::into_raw(Box::new(schema)) as *mut c_void,
            _data: data,
        }
    }

    pub fn parse_2_1(data: Vec<u8>) -> Schema {
        let schema = types_2_1::Schema::parse(&data);
        Schema {
            version: "2.1",
            schema: Box::into_raw(Box::new(schema)) as *mut c_void,
            _data: data,
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

    pub fn create_highlights(&self, indices: Vec<u8>) -> Highlights {
        Highlights { indices }
    }

    // Array<Array> contains 2 Arrays, first is the vertex buffer, the other the textures
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

// TODO: Check this is valid in wasm-bindgen
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
