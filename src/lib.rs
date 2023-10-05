// #![cfg_attr(target_family="wasm", no_std)]

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use core::{mem::size_of, slice, mem, ffi::c_void};
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

pub mod parser_2_1;
pub mod parser_2_0;
pub mod utils_2_1;
pub mod utils_2_0;
pub mod types_2_1;
pub mod types_2_0;
pub mod thin_slice;
pub mod range;


#[cfg(feature = "console")]
#[macro_export]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
    // ( $t:expr ) => {
    //     web_sys::console::log_1(&JsValue::from_str($t));
    // }
}

#[cfg(not(any(feature = "console", target_family = "wasm")))]
use log::debug as log;

#[cfg(all(not(feature = "console"), target_family = "wasm"))]
#[macro_export]
macro_rules! log {
    ($($tt:tt)*) => {

    };
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub fn init_console() {
    #[cfg(all(target_family="wasm", feature = "console_error_panic_hook"))]
    console_error_panic_hook::set_once();
}

macro_rules! primitive_function_impl {
    ($name: ident, $dst_ty: ty, $src_ty: ty) => {
        paste::paste!{
            #[cfg_attr(target_family = "wasm", wasm_bindgen)]
            pub fn [<$name _ $dst_ty>](dst: &mut [$dst_ty], src: $src_ty, byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
                $name(dst, src, byte_offset, byte_stride, begin, end)
            }
        }
    };
}

fn copy_to_interleaved_array<T: Copy + Send + Sync>(dst: &mut [T], src: &[T], byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    for (dst, src) in dst[offset..].iter_mut().step_by(stride).zip(&src[begin..end]) {
        *dst = *src
    }
}

primitive_function_impl!(copy_to_interleaved_array, u8, &[u8]);
primitive_function_impl!(copy_to_interleaved_array, u16, &[u16]);
primitive_function_impl!(copy_to_interleaved_array, u32, &[u32]);
primitive_function_impl!(copy_to_interleaved_array, i8, &[i8]);
primitive_function_impl!(copy_to_interleaved_array, i16, &[i16]);
primitive_function_impl!(copy_to_interleaved_array, i32, &[i32]);
primitive_function_impl!(copy_to_interleaved_array, f32, &[f32]);
primitive_function_impl!(copy_to_interleaved_array, f64, &[f64]);

fn fill_to_interleaved_array<T: Copy + Send + Sync>(dst: &mut [T], src: T, byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
    debug_assert_eq!(byte_offset % size_of::<T>(), 0);
    debug_assert_eq!(byte_stride % size_of::<T>(), 0);

    let offset = byte_offset / size_of::<T>();
    let stride = byte_stride / size_of::<T>();

    let end = (offset + stride * (end - begin)).min(dst.len());

    for dst in dst[offset..end].iter_mut().step_by(stride) {
        *dst = src;
    }
}

primitive_function_impl!(fill_to_interleaved_array, u8, u8);
primitive_function_impl!(fill_to_interleaved_array, u16, u16);
primitive_function_impl!(fill_to_interleaved_array, u32, u32);
primitive_function_impl!(fill_to_interleaved_array, i8, i8);
primitive_function_impl!(fill_to_interleaved_array, i16, i16);
primitive_function_impl!(fill_to_interleaved_array, i32, i32);
primitive_function_impl!(fill_to_interleaved_array, f32, f32);
primitive_function_impl!(fill_to_interleaved_array, f64, f64);

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Schema{
    _data: Vec<u8>,
    version: &'static str,
    schema: *mut c_void,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct ChildVec(usize, usize);

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Schema {
    pub fn parse_2_0(data: Vec<u8>) -> Schema {
        let schema = types_2_0::Schema::parse(&data).expect("Error parsing schema");
        Schema {
            version: "2.0",
            schema: Box::into_raw(Box::new(schema)) as *mut c_void,
            _data: data,
        }
    }

    pub fn parse_2_1(data: Vec<u8>) -> Schema {
        let schema = types_2_1::Schema::parse(&data).expect("Error parsing schema");
        Schema {
            version: "2.1",
            schema: Box::into_raw(Box::new(schema)) as *mut c_void,
            _data: data,
        }
    }

    pub fn children(&self, separate_positions_buffer: bool) -> ChildVec {
        match self.version {
            "2.0" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_0::Schema) };
                let children = schema.children(separate_positions_buffer, |_| true).collect::<Vec<_>>();
                ChildVec(children.as_ptr() as usize, children.len())
            }
            "2.1" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_1::Schema) };
                let children = schema.children(separate_positions_buffer, |_| true).collect::<Vec<_>>();
                ChildVec(children.as_ptr() as usize, children.len())
            }
            _ => todo!()
        }
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
