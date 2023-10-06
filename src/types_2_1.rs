use bitflags::bitflags;
use half::f16;

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

#[derive(Clone)]
pub struct RgbaU8Slice<'a> {
    pub red: ThinSlice<'a, u8>,
    pub green: ThinSlice<'a, u8>,
    pub blue: ThinSlice<'a, u8>,
    pub alpha: ThinSlice<'a, u8>,
}

#[derive(Clone, Copy)]
pub struct RgbaU8 {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

#[derive(Clone)]
pub struct Half2Slice<'a> {
    pub x: ThinSlice<'a, f16>,
    pub y: ThinSlice<'a, f16>,
}

#[derive(Clone, Copy)]
pub struct Half2 {
    pub x: f16,
    pub y: f16,
}

#[derive(Clone)]
pub struct Half3Slice<'a> {
    pub x: ThinSlice<'a, f16>,
    pub y: ThinSlice<'a, f16>,
    pub z: ThinSlice<'a, f16>,
}

#[derive(Clone, Copy)]
pub struct Half3 {
    pub x: f16,
    pub y: f16,
    pub z: f16,
}

#[derive(Clone)]
pub struct Int16_3Slice<'a> {
    pub x: ThinSlice<'a, i16>,
    pub y: ThinSlice<'a, i16>,
    pub z: ThinSlice<'a, i16>,
}

#[derive(Clone, Copy)]
pub struct Int16_3 {
    pub x: i16,
    pub y: i16,
    pub z: i16,
}

#[derive(Clone)]
pub struct Int8_3Slice<'a> {
    pub x: ThinSlice<'a, i8>,
    pub y: ThinSlice<'a, i8>,
    pub z: ThinSlice<'a, i8>,
}

#[derive(Clone, Copy)]
pub struct Int8_3 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
}

#[derive(Clone)]
pub struct Float3Slice<'a> {
    pub x: ThinSlice<'a, f32>,
    pub y: ThinSlice<'a, f32>,
    pub z: ThinSlice<'a, f32>,
}

#[derive(Clone, Copy)]
pub struct Float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Clone)]
pub struct Double3Slice<'a> {
    pub x: ThinSlice<'a, f64>,
    pub y: ThinSlice<'a, f64>,
    pub z: ThinSlice<'a, f64>,
}

#[derive(Clone, Copy)]
pub struct Double3 {
    pub x: f64,
    pub y: f64,
    pub z: f64
}

/// 3x3 row major matrix
#[derive(Clone)]
pub struct Float3x3Slice<'a> {
    pub e00: ThinSlice<'a, f32>,
    pub e01: ThinSlice<'a, f32>,
    pub e02: ThinSlice<'a, f32>,
    pub e10: ThinSlice<'a, f32>,
    pub e11: ThinSlice<'a, f32>,
    pub e12: ThinSlice<'a, f32>,
    pub e20: ThinSlice<'a, f32>,
    pub e21: ThinSlice<'a, f32>,
    pub e22: ThinSlice<'a, f32>,
}

#[derive(Clone, Copy)]
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
#[derive(Clone)]
pub struct AABBSlice<'a> {
    pub min: Float3Slice<'a>,
    pub max: Float3Slice<'a>,
}

#[derive(Clone, Copy)]
pub struct AABB {
    pub min: Float3,
    pub max: Float3,
}

/// Bounding sphere.
#[derive(Clone)]
pub struct BoundingSphereSlice<'a> {
    pub origo: Float3Slice<'a>,
    pub radius: ThinSlice<'a, f32>,
}

#[derive(Clone, Copy)]
pub struct BoundingSphere {
    pub origo: Float3,
    pub radius: f32,
}

/// Node bounding volume.
#[derive(Clone)]
pub struct BoundsSlice<'a> {
    pub _box: AABBSlice<'a>,
    pub sphere: BoundingSphereSlice<'a>,
}

#[derive(Clone, Copy)]
pub struct Bounds {
    pub _box: AABB,
    pub sphere: BoundingSphere,
}

/// Information about child sub meshes used to predict cost before loading.
#[derive(Clone)]
pub struct SubMeshProjectionSlice<'a> {
    pub len: u32,
    pub object_id: ThinSlice<'a, u32>,
    pub primitive_type: ThinSlice<'a, PrimitiveType>,
    pub attributes: ThinSlice<'a, OptionalVertexAttribute>,
    /// # of deviation vertex attributes (0-3)
    pub num_deviations: ThinSlice<'a, u8>,
    /// zero if no index buffer
    pub num_indices: ThinSlice<'a, u32>,
    pub num_vertices: ThinSlice<'a, u32>,
    pub num_texture_bytes: ThinSlice<'a, u32>,
}

#[derive(Clone)]
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
#[derive(Clone)]
pub struct ChildInfoSlice<'a> {
    pub len: u32,
    /// Byte range into Hash bytes array. The hash, formatted as hex, is used for the filename of the child node.
    pub hash: HashRangeSlice<'a>,

    pub child_index: ThinSlice<'a, u8>, // used to be octant mask, but tree can be non-octree also..
    /// Set of bits (max 32) for which child indices are referenced by geometry.
    pub child_mask: ThinSlice<'a, u32>,
    /// A power of two exponent describing the error tolerance of this node, which is used to determine LOD.
    pub tolerance: ThinSlice<'a, i8>,
    /// # uncompressed bytes total for child binary file.
    pub total_byte_size: ThinSlice<'a, u32>,

    /// Model -> world space translation vector.
    pub offset: Double3Slice<'a>,
    /// Model -> world space uniform scale factor (from unit [-1,1] vectors).
    pub scale: ThinSlice<'a, f32>,
    /// Bounding volume (in model space).
    pub bounds: BoundsSlice<'a>,

    pub sub_meshes: SubMeshProjectionRangeSlice<'a>,
    pub descendant_object_ids: DescendantObjectIdsRangeSlice<'a>,
}

#[derive(Clone)]
pub struct ChildInfo<'a> {
    /// Byte range into Hash bytes array. The hash, formatted as hex, is used for the filename of the child node.
    pub hash: &'a [HashBytes],

    pub child_index: u8, // used to be octant mask, but tree can be non-octree also..
    /// Set of bits (max 32) for which child indices are referenced by geometry.
    pub child_mask: u32,
    /// A power of two exponent describing the error tolerance of this node, which is used to determine LOD.
    pub tolerance: i8,
    /// # uncompressed bytes total for child binary file.
    pub total_byte_size: u32,

    /// Model -> world space translation vector.
    pub offset: Double3,
    /// Model -> world space uniform scale factor (from unit [-1,1] vectors).
    pub scale: f32,
    /// Bounding volume (in model space).
    pub bounds: Bounds,

    pub sub_meshes: SubMeshProjectionSlice<'a>,
    pub descendant_object_ids: &'a [DescendantObjectIds],
}

/// Mesh deviations vertex attributes
#[derive(Clone)]
pub struct DeviationsSlice<'a> {
    pub len: u32,
    pub a: Option<ThinSlice<'a, f16>>,
    pub c: Option<ThinSlice<'a, f16>>,
    pub d: Option<ThinSlice<'a, f16>>,
    pub b: Option<ThinSlice<'a, f16>>,
}

#[derive(Clone)]
pub struct Deviations {
    pub a: Option<f16>,
    pub b: Option<f16>,
    pub c: Option<f16>,
    pub d: Option<f16>,
}

/// Mesh vertices
#[derive(Clone)]
pub struct VertexSlice<'a> {
    pub len: u32,
    pub position: Int16_3Slice<'a>,
    pub normal: Option<Int8_3Slice<'a>>,
    pub color: Option<RgbaU8Slice<'a>>,
    pub tex_coord: Option<Half2Slice<'a>>,
    pub projected_pos: Option<Int16_3Slice<'a>>,
    pub deviations: DeviationsSlice<'a>
}

#[derive(Clone)]
pub struct Vertex {
    pub position: Int16_3,
    pub normal: Option<Int8_3>,
    pub color: Option<RgbaU8>,
    pub tex_coord: Option<Half2>,
    pub projected_pos: Option<Int16_3>,
    pub deviations: Deviations,
}

/// Mesh triangles
#[derive(Clone)]
pub struct TriangleSlice<'a> {
    pub len: u32,
    /// Bits [0-2] are edge flags (vertex pairs ab, bc, ca), and [3-5] are corner flags.
    /// True = edge/corner is a \"hard\", or true topological feature and should be rendered and/or
    /// snapped to.
    pub topology_flags: Option<ThinSlice<'a, u8>>,
}

#[derive(Clone)]
pub struct Triangle {
    pub topology_flags: Option<u8>,
}

/// Mesh Textures
#[derive(Clone)]
pub struct TextureInfoSlice<'a> {
    pub len: u32,
    pub semantic: ThinSlice<'a, TextureSemantic>,
    pub transform: Float3x3Slice<'a>,
    pub pixel_range: PixelRangeSlice<'a>,
}

#[derive(Clone)]
pub struct TextureInfo<'a> {
    pub semantic: TextureSemantic,
    pub transform: Float3x3,
    pub pixel_range: &'a [TexturePixels]
}

/// Groups of 3D primitives with common attributes. These can further be split up to form e.g. 64K
/// chunks for 16 bit indexing, i.e. there can be many submeshes with the same attributes. Groups
/// are ordered by child, object and material indices.
pub struct SubMeshSlice<'a> {
    pub len: u32,
    pub child_index: ThinSlice<'a, u8>, // used to be octant, but tree can be non-octree also.
    pub object_id: ThinSlice<'a, u32>,
    pub material_index: ThinSlice<'a, u8>,
    // the four next properties replaces the old "meshkind"
    pub primitive_type: ThinSlice<'a, PrimitiveType>,
    pub material_type: ThinSlice<'a, MaterialType>,
    pub attributes: ThinSlice<'a, OptionalVertexAttribute>,
    /// # of deviation vertex attributes (0-4)"
    pub num_deviations: ThinSlice<'a, u8>,
    ///Vertices are local to each sub-mesh.
    pub vertices: VertexRangeSlice<'a>,
    /// Triangle vertex index triplets, or line index pairs, if any, are 16-bit and relative to the local vertex range.
    pub primitive_vertex_indices: VertexIndexRangeSlice<'a>,
    /// "Hard" edge vertex index pairs, if any, are 16-bit and relative to the local vertex range.
    pub edge_vertex_indices: VertexIndexRangeSlice<'a>,
    /// "Hard" corner vertex indices, if any, are 16-bit and relative to the local vertex range.
    pub corner_vertex_indices: VertexIndexRangeSlice<'a>,
    pub textures: TextureInfoRangeSlice<'a>,
}

#[derive(Clone)]
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
    pub vertices: &'a [Vertex],
    /// Triangle vertex index triplets, or line index pairs, if any, are 16-bit and relative to the local vertex range.
    pub primitive_vertex_indices: &'a [VertexIndex],
    /// "Hard" edge vertex index pairs, if any, are 16-bit and relative to the local vertex range.
    pub edge_vertex_indices: &'a [VertexIndex],
    /// "Hard" corner vertex indices, if any, are 16-bit and relative to the local vertex range.
    pub corner_vertex_indices: &'a [VertexIndex],
    pub textures: TextureInfoSlice<'a>,
}

pub type VertexIndex = u32;
pub type HashBytes = u8;
pub type TexturePixels = u8;
/// Information about descendant object ids, which may be present for branches with few object ids,
/// such as documents.
pub type DescendantObjectIds = u32;

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
    /// Range into descendantObjectIdsRange.
    DescendantObjectIdsRange: u32
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
    pub descendant_object_ids: ThinSlice<'a, DescendantObjectIds>,
    pub sub_mesh_projection: SubMeshProjectionSlice<'a>,
    pub sub_mesh: SubMeshSlice<'a>,
    pub texture_info: TextureInfoSlice<'a>,
    pub vertex: VertexSlice<'a>,
    pub triangle: TriangleSlice<'a>,
    /// Mesh vertex indices, relative to each draw call, hence 16 bit.
    pub vertex_index: Option<ThinSlice<'a, u16>>,
    pub texture_pixels: ThinSlice<'a, TexturePixels>,
}
