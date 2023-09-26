#![no_std]

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
// #[global_allocator]
// static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

use wasm_bindgen::prelude::*;
use half::f16;

/// Converts f16 to f32
#[wasm_bindgen]
pub fn float16(v: f32) -> u16 {
    f16::from_f32_const(v).to_bits()
}

/// Converts f32 to f16
#[wasm_bindgen]
pub fn float32(v: u16) -> f32 {
    f16::from_bits(v).to_f32_const()
}

#[test]
fn conversions() {
    const VAL_F32: f32 = 3.141592653589793; // original float32 value
    const VAL_F16: f16 = f16::from_f32_const(VAL_F32); // cast to float16
    const VAL_U16: u16 = VAL_F16.to_bits(); // bitcast to u16
    assert_eq!(float16(VAL_F32), VAL_U16);
    assert_eq!(float32(VAL_U16), VAL_F32);
    assert_eq!(float16(float32(VAL_U16)), VAL_U16);
}