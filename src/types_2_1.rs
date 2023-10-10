use half::f16;
use wasm_parser_derive::StructOfArray;
#[cfg(feature = "checked_types")]
use bytemuck::{Pod, Zeroable, CheckedBitPattern};

use crate::thin_slice::ThinSlice;
use crate::impl_range;
use crate::types::*;

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

    #[soa_range(DescendantObjectIdsRange)]
    pub descendant_object_ids: &'a [DescendantObjectIds],
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
    pub vertices: VertexSlice<'a>,

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
