use bitflags::bitflags;
use half::f16;
use wasm_parser_derive::StructOfArray;

use crate::thin_slice::ThinSlice;
use crate::impl_range;
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
pub struct AABB {
    #[soa_nested]
    pub min: Float3,
    #[soa_nested]
    pub max: Float3,
}

/// Bounding sphere.
#[derive(Clone, Copy, StructOfArray)]
pub struct BoundingSphere {
    #[soa_nested]
    pub origo: Float3,
    pub radius: f32,
}

/// Node bounding volume.
#[derive(Clone, Copy, StructOfArray)]
pub struct Bounds {
    #[soa_nested]
    pub _box: AABB,
    #[soa_nested]
    pub sphere: BoundingSphere,
}

/// Information about child sub meshes used to predict cost before loading.
#[derive(Clone, StructOfArray)]
#[soa_len]
pub struct SubMeshProjection {
    pub object_id: u32,
    pub primitive_type: PrimitiveType,
    pub attributes: OptionalVertexAttribute,
    /// # of deviation vertex attributes (0-3)
    pub num_deviations: u8,
    /// zero if no index buffer
    pub num_indices: u32,
    pub num_vertices: u32,
    pub num_texture_bytes: u32,
}

/// Information about child nodes.
#[derive(Clone, StructOfArray)]
#[soa_len]
pub struct ChildInfo<'a> {
    /// Byte range into Hash bytes array. The hash, formatted as hex, is used for the filename of the child node.
    #[soa_range(HashRange)]
    pub hash: &'a [HashBytes],

    pub child_index: u8, // used to be octant mask, but tree can be non-octree also..
    /// Set of bits (max 32) for which child indices are referenced by geometry.
    pub child_mask: u32,
    /// A power of two exponent describing the error tolerance of this node, which is used to determine LOD.
    pub tolerance: i8,
    /// # uncompressed bytes total for child binary file.
    pub total_byte_size: u32,

    /// Model -> world space translation vector.
    #[soa_nested]
    pub offset: Double3,
    /// Model -> world space uniform scale factor (from unit [-1,1] vectors).
    pub scale: f32,
    /// Bounding volume (in model space).
    #[soa_nested]
    pub bounds: Bounds,

    #[soa_range(SubMeshProjectionRange)]
    pub sub_meshes: SubMeshProjectionSlice<'a>,
}

/// Mesh deviations vertex attributes
#[derive(Clone, StructOfArray)]
#[soa_len]
pub struct Deviations {
    pub a: Option<f16>,
    pub b: Option<f16>,
    pub c: Option<f16>,
    pub d: Option<f16>,
}

/// Mesh vertices
#[derive(Clone, StructOfArray)]
#[soa_len]
pub struct Vertex {
    #[soa_nested]
    pub position: Int16_3,
    #[soa_nested]
    pub normal: Option<Int8_3>,
    #[soa_nested]
    pub color: Option<RgbaU8>,
    #[soa_nested]
    pub tex_coord: Option<Half2>,
    #[soa_nested]
    pub projected_pos: Option<Int16_3>,
    #[soa_nested]
    pub deviations: Deviations,
}

/// Mesh triangles
#[derive(Clone, StructOfArray)]
#[soa_len]
pub struct Triangle {
    /// Bits [0-2] are edge flags (vertex pairs ab, bc, ca), and [3-5] are corner flags.
    /// True = edge/corner is a \"hard\", or true topological feature and should be rendered and/or
    /// snapped to.
    pub topology_flags: Option<u8>,
}

/// Mesh Textures
#[derive(Clone, StructOfArray)]
#[soa_len]
#[soa_range_index(TextureInfoRange)]
pub struct TextureInfo<'a> {
    pub semantic: TextureSemantic,
    #[soa_nested]
    pub transform: Float3x3,
    #[soa_range(PixelRange)]
    pub pixel_range: &'a [TexturePixels]
}


/// Groups of 3D primitives with common attributes. These can further be split up to form e.g. 64K
/// chunks for 16 bit indexing, i.e. there can be many submeshes with the same attributes. Groups
/// are ordered by child, object and material indices.
#[derive(Clone, StructOfArray)]
#[soa_len]
pub struct SubMesh<'a> {
    pub child_index: u8, // used to be octant, but tree can be non-octree also.
    pub object_id: u32,
    pub material_index: u8,

    // the four next properties replaces the old "meshkind"
    pub primitive_type: PrimitiveType,
    pub material_type: MaterialType,
    pub attributes: OptionalVertexAttribute,
    /// # of deviation vertex attributes (0-4)"
    pub num_deviations: u8,

    ///Vertices are local to each sub-mesh.
    #[soa_range(VertexRange)]
    pub vertices: &'a [Vertex],

    /// Triangle vertex index triplets, or line index pairs, if any, are 16-bit and relative to the local vertex range.
    #[soa_range(VertexIndexRange)]
    pub primitive_vertex_indices: &'a [VertexIndex],

    /// "Hard" edge vertex index pairs, if any, are 16-bit and relative to the local vertex range.
    #[soa_range(VertexIndexRange)]
    pub edge_vertex_indices: &'a [VertexIndex],

    /// "Hard" corner vertex indices, if any, are 16-bit and relative to the local vertex range.
    #[soa_range(VertexIndexRange)]
    pub corner_vertex_indices: &'a [VertexIndex],

    #[soa_range(TextureInfoRange)]
    pub textures: TextureInfoSlice<'a>,
}

pub type VertexIndex = u32;
pub type HashBytes = u8;
pub type TexturePixels = u8;

impl_range!{
    /// Range into texture pixel blob.
    PixelRange: u32
}

impl_range!{
    /// Range into submesh projection.
    SubMeshProjectionRange: u32
}

impl_range!{
    /// Hash bytes
    HashRange: u32
}

impl_range!{
    /// Mesh vertices
    VertexRange: u32
}

impl_range!{
    /// Mesh vertices indices
    VertexIndexRange: u32
}

impl_range!{
    /// Mesh textures
    TextureInfoRange: u8
}

pub struct Schema<'a> {
    pub child_info: ChildInfoSlice<'a>,
    pub hash_bytes: ThinSlice<'a, HashBytes>,
    pub sub_mesh_projection: SubMeshProjectionSlice<'a>,
    pub sub_mesh: SubMeshSlice<'a>,
    pub texture_info: TextureInfoSlice<'a>,
    pub vertex: VertexSlice<'a>,
    pub triangle: TriangleSlice<'a>,
    /// Mesh vertex indices, relative to each draw call, hence 16 bit.
    pub vertex_index: Option<ThinSlice<'a, u16>>,
    pub texture_pixels: ThinSlice<'a, TexturePixels>,
}
