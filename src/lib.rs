#![cfg_attr(target_family="wasm", no_std)]

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use core::mem::size_of;
use wasm_bindgen::prelude::*;
pub mod parser;
pub mod utils;
pub mod types;


#[cfg(feature = "console")]
macro_rules! log {
    // ( $( $t:tt )* ) => {
    //     web_sys::console::log_1(&format!( $( $t )* ).into());
    // }
    ( $t:expr ) => {
        web_sys::console::log_1(&JsValue::from_str($t));
    }
}

#[wasm_bindgen]
pub fn init_console() {
    #[cfg(feature = "console_error_panic_hook")]
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
