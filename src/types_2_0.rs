use bitflags::bitflags;
use half::f16;

use crate::thin_slice::ThinSlice;
use crate::range::Range;
use crate::impl_range_iter;
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
pub struct RgbaU8<'a> {
    pub red: ThinSlice<'a, u8>,
    pub green: ThinSlice<'a, u8>,
    pub blue: ThinSlice<'a, u8>,
    pub alpha: ThinSlice<'a, u8>,
}

#[derive(Clone)]
pub struct Half2<'a> {
    pub x: ThinSlice<'a, f16>,
    pub y: ThinSlice<'a, f16>,
}

#[derive(Clone)]
pub struct Half3<'a> {
    pub x: ThinSlice<'a, f16>,
    pub y: ThinSlice<'a, f16>,
    pub z: ThinSlice<'a, f16>,
}

#[derive(Clone)]
pub struct Int16_3<'a> {
    pub x: ThinSlice<'a, i16>,
    pub y: ThinSlice<'a, i16>,
    pub z: ThinSlice<'a, i16>,
}

#[derive(Clone)]
pub struct Int8_3<'a> {
    pub x: ThinSlice<'a, i8>,
    pub y: ThinSlice<'a, i8>,
    pub z: ThinSlice<'a, i8>,
}

#[derive(Clone)]
pub struct Float3<'a> {
    pub x: ThinSlice<'a, f32>,
    pub y: ThinSlice<'a, f32>,
    pub z: ThinSlice<'a, f32>,
}

#[derive(Clone)]
pub struct Double3<'a> {
    pub x: ThinSlice<'a, f64>,
    pub y: ThinSlice<'a, f64>,
    pub z: ThinSlice<'a, f64>,
}

/// 3x3 row major matrix
#[derive(Clone)]
pub struct Float3x3<'a> {
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

/// Axis aligned bounding box.
#[derive(Clone)]
pub struct AABB<'a> {
    pub min: Float3<'a>,
    pub max: Float3<'a>,
}

/// Bounding sphere.
#[derive(Clone)]
pub struct BoundingSphere<'a> {
    pub origo: Float3<'a>,
    pub radius: ThinSlice<'a, f32>,
}

/// Node bounding volume.
#[derive(Clone)]
pub struct Bounds<'a> {
    pub _box: AABB<'a>,
    pub sphere: BoundingSphere<'a>,
}

/// Information about child sub meshes used to predict cost before loading.
#[derive(Clone)]
pub struct SubMeshProjection<'a> {
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

// SAFETY: We know the len of the slices in this struct are all len
impl<'a> SubMeshProjection<'a> {
    pub fn object_id(&self) -> &'a [u32] {
        unsafe{ self.object_id.as_slice(self.len) }
    }

    pub fn primitive_type(&self) -> &'a [PrimitiveType] {
        unsafe{ self.primitive_type.as_slice(self.len) }
    }

    pub fn attributes(&self) -> &'a [OptionalVertexAttribute] {
        unsafe{ self.attributes.as_slice(self.len) }
    }

    pub fn num_deviations(&self) -> &'a [u8] {
        unsafe{ self.num_deviations.as_slice(self.len) }
    }

    pub fn num_indices(&self) -> &'a [u32] {
        unsafe{ self.num_indices.as_slice(self.len) }
    }

    pub fn num_vertices(&self) -> &'a [u32] {
        unsafe{ self.num_vertices.as_slice(self.len) }
    }

    pub fn num_texture_bytes(&self) -> &'a [u32] {
        unsafe{ self.num_texture_bytes.as_slice(self.len) }
    }
}

/// Information about child nodes.
#[derive(Clone)]
pub struct ChildInfo<'a> {
    pub len: u32,
    /// Byte range into Hash bytes array. The hash, formatted as hex, is used for the filename of the child node.
    pub hash: HashRange<'a>,

    pub child_index: ThinSlice<'a, u8>, // used to be octant mask, but tree can be non-octree also..
    /// Set of bits (max 32) for which child indices are referenced by geometry.
    pub child_mask: ThinSlice<'a, u32>,
    /// A power of two exponent describing the error tolerance of this node, which is used to determine LOD.
    pub tolerance: ThinSlice<'a, i8>,
    /// # uncompressed bytes total for child binary file.
    pub total_byte_size: ThinSlice<'a, u32>,

    /// Model -> world space translation vector.
    pub offset: Double3<'a>,
    /// Model -> world space uniform scale factor (from unit [-1,1] vectors).
    pub scale: ThinSlice<'a, f32>,
    /// Bounding volume (in model space).
    pub bounds: Bounds<'a>,

    pub sub_meshes: SubMeshProjectionRange<'a>,
}



/// Mesh deviations vertex attributes
#[derive(Clone)]
pub struct Deviations<'a> {
    pub len: u32,
    pub a: Option<ThinSlice<'a, f16>>,
    pub c: Option<ThinSlice<'a, f16>>,
    pub d: Option<ThinSlice<'a, f16>>,
    pub b: Option<ThinSlice<'a, f16>>,
}

/// Mesh vertices
#[derive(Clone)]
pub struct Vertex<'a> {
    pub len: u32,
    pub position: Int16_3<'a>,
    pub normal: Option<Int8_3<'a>>,
    pub color: Option<RgbaU8<'a>>,
    pub tex_coord: Option<Half2<'a>>,
    pub projected_pos: Option<Int16_3<'a>>,
    pub deviations: Deviations<'a>
}

/// Mesh triangles
#[derive(Clone)]
pub struct Triangle<'a> {
    pub len: u32,
    /// Bits [0-2] are edge flags (vertex pairs ab, bc, ca), and [3-5] are corner flags.
    /// True = edge/corner is a \"hard\", or true topological feature and should be rendered and/or
    /// snapped to.
    pub topology_flags: Option<ThinSlice<'a, u8>>,
}

/// Mesh Textures
#[derive(Clone)]
pub struct TextureInfo<'a> {
    pub len: u32,
    pub semantic: ThinSlice<'a, TextureSemantic>,
    pub transform: Float3x3<'a>,
    pub pixel_range: PixelRange<'a>,
}

/// Groups of 3D primitives with common attributes. These can further be split up to form e.g. 64K
/// chunks for 16 bit indexing, i.e. there can be many submeshes with the same attributes. Groups
/// are ordered by child, object and material indices.
pub struct SubMesh<'a> {
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
    pub vertices: VertexRange<'a>,
    /// Triangle vertex index triplets, or line index pairs, if any, are 16-bit and relative to the local vertex range.
    pub primitive_vertex_indices: VertexIndexRange<'a>,
    /// "Hard" edge vertex index pairs, if any, are 16-bit and relative to the local vertex range.
    pub edge_vertex_indices: VertexIndexRange<'a>,
    /// "Hard" corner vertex indices, if any, are 16-bit and relative to the local vertex range.
    pub corner_vertex_indices: VertexIndexRange<'a>,
    pub textures: TextureInfoRange<'a>,
}

pub type HashBytes = u8;
pub type TexturePixels = u8;
/// Information about descendant object ids, which may be present for branches with few object ids,
/// such as documents.
pub type DescendantObjectIds = u32;

/// Range into texture pixel blob.
#[derive(Clone)]
pub struct PixelRange<'a>(pub Range<'a, u32>);

impl_range_iter!(PixelRange, u32);

/// Range into submesh projection.
#[derive(Clone, Copy)]
pub struct SubMeshProjectionRange<'a>(pub Range<'a, u32>);

impl_range_iter!(SubMeshProjectionRange, u32);

/// Hash bytes
#[derive(Clone, Copy)]
pub struct HashRange<'a>(pub Range<'a, u32>);

impl_range_iter!(HashRange, u32);

/// Mesh vertices
#[derive(Clone, Copy)]

pub struct VertexRange<'a>(pub Range<'a, u32>);

impl_range_iter!(VertexRange, u32);

/// Mesh vertices indices
#[derive(Clone, Copy)]
pub struct VertexIndexRange<'a>(pub Range<'a, u32>);

impl_range_iter!(VertexIndexRange, u32);

/// Mesh textures
#[derive(Clone, Copy)]
pub struct TextureInfoRange<'a>(pub Range<'a, u8>);

impl_range_iter!(TextureInfoRange, u8);

pub struct Schema<'a> {
    pub child_info: ChildInfo<'a>,
    pub hash_bytes: ThinSlice<'a, HashBytes>,
    pub sub_mesh_projection: SubMeshProjection<'a>,
    pub sub_mesh: SubMesh<'a>,
    pub texture_info: TextureInfo<'a>,
    pub vertex: Vertex<'a>,
    pub triangle: Triangle<'a>,
    /// Mesh vertex indices, relative to each draw call, hence 16 bit.
    pub vertex_index: Option<ThinSlice<'a, u16>>,
    pub texture_pixels: ThinSlice<'a, TexturePixels>,
}
