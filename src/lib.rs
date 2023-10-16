// #![cfg_attr(target_family="wasm", no_std)]

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use core::{mem::size_of, mem, ffi::c_void};
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

macro_rules! primitive_function_impl {
    ($name: ident, $dst_ty: ty, $src_ty: ty) => {
        paste::paste!{
            #[wasm_bindgen]
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

// #[wasm_bindgen]
// #[repr(transparent)]
// #[derive(Copy, Clone)]
// pub struct HandleU8(usize);

// #[wasm_bindgen]
// pub fn allocate_u8(size: usize) -> HandleU8 {
//     HandleU8(DATA_U8.with(|data| data.borrow_mut().insert_key_gen(vec![0; size])))
// }

// #[wasm_bindgen]
// pub fn array_u8(handle: &HandleU8) -> js_sys::Uint8Array {
//     unsafe{
//         DATA_U8.with(|data| {
//             let mut data = data.borrow_mut();
//             let slice = &mut data[handle.0];
//             js_sys::Uint8Array::view_mut_raw(slice.as_mut_ptr(), slice.len())
//         })
//     }
// }

// #[wasm_bindgen]
// #[repr(transparent)]
// #[derive(Copy, Clone)]
// pub struct HandleI16(usize);

// #[wasm_bindgen]
// pub fn allocate_i16(size: usize) -> HandleI16 {
//     HandleI16(DATA_I16.with(|data| data.borrow_mut().insert_key_gen(vec![0; size])))
// }

// #[wasm_bindgen]
// pub fn array_i16(handle: &HandleI16) -> js_sys::Int16Array {
//     unsafe{
//         DATA_I16.with(|data| {
//             let mut data = data.borrow_mut();
//             let slice = &mut data[handle.0];
//             js_sys::Int16Array::view_mut_raw(slice.as_mut_ptr(), slice.len())
//         })
//     }
// }
// #[wasm_bindgen]
// /// Copies an array into another interleaved taking into account offset and stride
// ///
// /// # Arguments
// ///
// /// * `dst` - The destination array as a [`HandleI16`]
// /// * `src` - The source array as an `i16` slice
// pub fn copy_to_interleaved_array_i16(dst: &HandleI16, src: &[i16], byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
//     debug_assert_eq!(byte_offset % size_of::<i16>(), 0);
//     debug_assert_eq!(byte_stride % size_of::<i16>(), 0);

//     let offset = byte_offset / size_of::<i16>();
//     let stride = byte_stride / size_of::<i16>();

//     DATA_I16.with(|data| {
//         let dst_data = &mut data.borrow_mut()[dst.0];
//         for (dst, src) in dst_data[offset..].iter_mut().step_by(stride).zip(&src[begin..end]) {
//             *dst = *src
//         }
//     })
// }


// #[wasm_bindgen]
// #[derive(Copy, Clone)]
// pub struct HandleI16(*mut i16, usize);

// #[wasm_bindgen]
// pub fn allocate_i16(size: usize) -> HandleI16 {
//     // let layout = unsafe{
//     //     Layout::from_size_align(size, Layout::new::<i16>().align()).unwrap_unchecked()
//     // };
//     // HandleI16(unsafe{ WeeAlloc::INIT.alloc(layout) } as *mut i16, size)

//     BUMP.with(|bump| HandleI16(bump.alloc_slice_fill_default::<i16>(size).as_mut_ptr() as *mut i16, size))
// }

// #[wasm_bindgen]
// pub fn array_i16(handle: &HandleI16) -> js_sys::Int16Array {
//     unsafe{
//         let HandleI16(ptr, size) = *handle;
//         js_sys::Int16Array::view_mut_raw(ptr, size)
//     }
// }

// #[wasm_bindgen]
// /// Copies an array into another interleaved taking into account offset and stride
// ///
// /// # Arguments
// ///
// /// * `dst` - The destination array as a [`HandleI16`]
// /// * `src` - The source array as an `i16` slice
// pub fn copy_to_interleaved_array_i16(dst: &HandleI16, src: &[i16], byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
//     debug_assert_eq!(byte_offset % size_of::<i16>(), 0);
//     debug_assert_eq!(byte_stride % size_of::<i16>(), 0);

//     let offset = byte_offset / size_of::<i16>();
//     let stride = byte_stride / size_of::<i16>();

//     let HandleI16(ptr, size) = *dst;
//     let dst_data = unsafe{ slice::from_raw_parts_mut(ptr, size) };
//     for (dst, src) in dst_data[offset..].iter_mut().step_by(stride).zip(&src[begin..end]) {
//         *dst = *src
//     }
// }



#[wasm_bindgen]
#[derive(Copy, Clone)]
pub struct HandleI16(*mut i16, usize);

#[wasm_bindgen]
pub fn allocate_i16(size: usize) -> HandleI16 {
    // let layout = unsafe{
    //     Layout::from_size_align(size, Layout::new::<i16>().align()).unwrap_unchecked()
    // };
    // HandleI16(unsafe{ WeeAlloc::INIT.alloc(layout) } as *mut i16, size)

    let mut mem = vec![0; size];
    let handle = HandleI16(mem.as_mut_ptr() as *mut i16, size);
    mem::forget(mem);
    handle
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen]
pub fn array_i16(handle: &HandleI16) -> js_sys::Int16Array {
    unsafe{
        let HandleI16(ptr, size) = *handle;
        js_sys::Int16Array::view_mut_raw(ptr, size)
    }
}

#[wasm_bindgen]
/// Copies an array into another interleaved taking into account offset and stride
///
/// # Arguments
///
/// * `dst` - The destination array as a [`HandleI16`]
/// * `src` - The source array as an `i16` slice
pub fn copy_to_interleaved_array_i16(dst: &HandleI16, src: &[i16], byte_offset: usize, byte_stride: usize, begin: usize, end: usize) {
    debug_assert_eq!(byte_offset % size_of::<i16>(), 0);
    debug_assert_eq!(byte_stride % size_of::<i16>(), 0);

    let offset = byte_offset / size_of::<i16>();
    let stride = byte_stride / size_of::<i16>();

    let HandleI16(ptr, size) = *dst;
    let dst_data = unsafe{ core::slice::from_raw_parts_mut(ptr, size) };
    for (dst, src) in dst_data[offset..].iter_mut().step_by(stride).zip(&src[begin..end]) {
        *dst = *src
    }
}

primitive_function_impl!(copy_to_interleaved_array, u8, &[u8]);
primitive_function_impl!(copy_to_interleaved_array, u16, &[u16]);
primitive_function_impl!(copy_to_interleaved_array, u32, &[u32]);
primitive_function_impl!(copy_to_interleaved_array, i8, &[i8]);
// primitive_function_impl!(copy_to_interleaved_array, i16, &[i16]);
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

#[wasm_bindgen]
pub struct Schema{
    _data: Vec<u8>,
    version: &'static str,
    schema: *mut c_void,
}

#[wasm_bindgen]
pub struct ChildVec(usize, usize);

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

    pub fn children(&self) -> Array {
        match self.version {
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
            _ => todo!()
        }
    }

    // Array<Array> contains 2 Arrays, first is the vertex buffer, the other the textures
    pub fn geometry(&self, enable_outlines: bool) -> Array {
        match self.version {
            "2.0" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_0::Schema) };
                let (sub_meshes, textures) = schema.geometry(
                    enable_outlines,
                    Highlights{indices: &[]},
                    |_| true
                );
                let sub_meshes: Array = sub_meshes
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                let textures: Array = textures
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                [sub_meshes, textures].into_iter().collect()
            }
            "2.1" => {
                // SAFETY: schema is put in a `Box` when created in `Schema::parse`
                let schema = unsafe{ &*(self.schema as *mut types_2_1::Schema) };
                let (sub_meshes, index) = schema.geometry(
                    enable_outlines,
                    Highlights{indices: &[]},
                    |_| true
                );
                let sub_meshes: Array = sub_meshes
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                let textures: Array = index
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                [sub_meshes, textures].into_iter().collect()
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
