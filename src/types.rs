use bitflags::bitflags;
use half::f16;
use serde::{ Serialize, Deserialize };
#[cfg(feature = "checked_types")]
use bytemuck::{Pod, Zeroable, CheckedBitPattern};

use wasm_bindgen::prelude::wasm_bindgen;
use wasm_parser_derive::StructOfArray;

/// Type of GL render primitive.
#[derive(Clone, Copy)]
#[cfg_attr(feature= "checked_types", derive(CheckedBitPattern))]
#[repr(u8)]
pub enum PrimitiveType {
    Points = 0,
    Lines = 1,
    LineLoop = 2,
    LineStrip = 3,
    Triangles = 4,
    TriangleStrip = 5,
    TriangleFan = 6,
}

/// Type of material.
#[derive(Clone, Copy)]
#[cfg_attr(feature= "checked_types", derive(CheckedBitPattern))]
#[repr(u8)]
pub enum MaterialType {
    Opaque = 0,
    OpaqueDoubleSided = 1,
    Transparent = 2,
    Elevation = 3,
}


/// Bitwise flags for which vertex attributes will be used in geometry.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature= "checked_types", derive(Pod, Zeroable))]
#[cfg_attr(feature= "checked_types", repr(transparent))]
pub struct OptionalVertexAttribute(u8);

bitflags! {
    impl OptionalVertexAttribute: u8 {
        const NORMAL = 1;
        const COLOR = 2;
        const TEX_COORD = 4;
        const PROJECTED_POS = 8;
    }
}

/// Texture semantic/purpose.
#[derive(Clone, Copy)]
#[cfg_attr(feature= "checked_types", derive(CheckedBitPattern))]
#[repr(u8)]
pub enum TextureSemantic {
    BaseColor = 0,
}


#[derive(Clone, Copy, StructOfArray)]
pub struct RgbaU8 {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

#[derive(Clone, Copy, StructOfArray)]
pub struct Half2 {
    pub x: f16,
    pub y: f16,
}

#[derive(Clone, Copy, StructOfArray)]
pub struct Half3 {
    pub x: f16,
    pub y: f16,
    pub z: f16,
}

#[derive(Clone, Copy, StructOfArray)]
pub struct Int16_3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

#[derive(Clone, Copy, StructOfArray)]
pub struct Int8_3 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
}

#[derive(Clone, Copy, StructOfArray)]
#[wasm_bindgen]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

impl std::ops::Add for Float3 {
    type Output = Float3;

    fn add(self, rhs: Self) -> Self::Output {
        Float3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl From<Double3> for Float3 {
    fn from(value: Double3) -> Self {
        Float3 {
            x: value.x as f32,
            y: value.y as f32,
            z: value.z as f32,
        }
    }
}

#[derive(Clone, Copy, StructOfArray)]
#[wasm_bindgen]
pub struct Double3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

/// 3x3 row major matrix
#[derive(Clone, Copy, StructOfArray)]
pub struct Float3x3 {
    pub e00: f32,
    pub e01: f32,
    pub e02: f32,
    pub e10: f32,
    pub e11: f32,
    pub e12: f32,
    pub e20: f32,
    pub e21: f32,
    pub e22: f32,
}

/// Axis aligned bounding box.
#[derive(Clone, Copy, StructOfArray)]
#[wasm_bindgen]
pub struct AABB {
    #[soa_nested]
    pub min: Float3,
    #[soa_nested]
    pub max: Float3,
}

/// Bounding sphere.
#[derive(Clone, Copy, StructOfArray)]
#[wasm_bindgen]
pub struct BoundingSphere {
    #[soa_nested]
    pub origo: Float3,
    pub radius: f32,
}

/// Node bounding volume.
#[derive(Clone, Copy, StructOfArray)]
#[wasm_bindgen]
pub struct Bounds {
    #[soa_nested]
    pub _box: AABB,
    #[soa_nested]
    pub sphere: BoundingSphere,
}