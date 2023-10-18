// #![cfg_attr(target_family="wasm", no_std)]

use core::{mem, ffi::c_void};
use js_sys::Array;
use parser::Highlights;
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
pub struct Schema{
    _data: Vec<u8>,
    version: &'static str,
    schema: *mut c_void,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "{subMeshes: Array<ReturnSubMesh_2_0 | ReturnSubMesh_2_1>, textures: Array<Texture_2_0 | Texture_2_1>}")]
    pub type NodeGeometry;

    #[wasm_bindgen(typescript_type = "Array<Child_2_0 | Child_2_1>")]
    pub type ArrayChild;
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
                    .map(|child| serde_wasm_bindgen::to_value(&child).unwrap())
                    .collect()
            }
            "2.1" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_1::Schema) };
                schema.children(|_| true)
                    .map(|child| serde_wasm_bindgen::to_value(&child).unwrap())
                    .collect()
            }
            _ => todo!()
        };

        let js_value: JsValue = array.into();
        js_value.into()
    }

    // Array<Array> contains 2 Arrays, first is the vertex buffer, the other the textures
    pub fn geometry(&self, enable_outlines: bool) -> NodeGeometry {
        #[wasm_bindgen]
        pub struct InnerNodeGeometry {
            sub_meshes: JsValue,
            textures: JsValue,
        }

        #[wasm_bindgen]
        impl InnerNodeGeometry {
            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "subMeshes")]
            pub fn sub_meshes(&self) -> JsValue {
                self.sub_meshes.clone()
            }

            #[wasm_bindgen(getter)]
            pub fn textures(&self) -> JsValue {
                self.textures.clone()
            }
        }

        let js_value: JsValue = match self.version {
            "2.0" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_0::Schema) };
                let (sub_meshes, textures) = schema.geometry(
                    enable_outlines,
                    Highlights{indices: &[]},
                    |_| true
                );
                let sub_meshes = serde_wasm_bindgen::to_value(&sub_meshes).unwrap();
                let textures = serde_wasm_bindgen::to_value(&textures).unwrap();
                InnerNodeGeometry{
                    sub_meshes,
                    textures
                }.into()
            }
            "2.1" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_1::Schema) };
                let (sub_meshes, textures) = schema.geometry(
                    enable_outlines,
                    Highlights{indices: &[]},
                    |_| true
                );
                let sub_meshes = serde_wasm_bindgen::to_value(&sub_meshes).unwrap();
                let textures = serde_wasm_bindgen::to_value(&textures).unwrap();
                InnerNodeGeometry{
                    sub_meshes,
                    textures
                }.into()
            }
            _ => todo!()
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
            _ => todo!()
        }
    }
}
