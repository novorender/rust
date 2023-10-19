use bitflags::bitflags;

#[cfg(feature = "checked_types")]
use bytemuck::{Pod, Zeroable, CheckedBitPattern};

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
