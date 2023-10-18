use std::{io::{Read, Seek, SeekFrom}, mem::{size_of, MaybeUninit}};

use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
use half::f16;

#[derive(Clone, Deserialize)]
pub struct Range<T> {
    pub start: Vec<T>,
    pub count: Vec<T>,
}

/// Type of GL render primitive.
#[derive(Clone, Deserialize)]
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
#[derive(Clone, Deserialize)]
#[repr(u8)]
pub enum MaterialType {
    Opaque = 0,
    OpaqueDoubleSided = 1,
    Transparent = 2,
    Elevation = 3,
}

bitflags! {
    /// Bitwise flags for which vertex attributes will be used in geometry.
    #[derive(Clone, Deserialize)]
    pub struct OptionalVertexAttribute: u8 {
        const NORMAL = 1;
        const COLOR = 2;
        const TEX_COORD = 4;
        const PROJECTED_POS = 8;
    }
}

/// Texture semantic/purpose.
#[derive(Clone, Deserialize)]
#[repr(u8)]
pub enum TextureSemantic {
    BaseColor = 0,
}

#[derive(Clone, Deserialize)]
pub struct RgbaU8 {
    pub red: Vec<u8>,
    pub green: Vec<u8>,
    pub blue: Vec<u8>,
    pub alpha: Vec<u8>,
}

#[derive(Clone, Deserialize)]
pub struct Half2 {
    pub x: Vec<f16>,
    pub y: Vec<f16>,
}

#[derive(Clone, Deserialize)]
pub struct Int16_3 {
    pub x: Vec<i16>,
    pub y: Vec<i16>,
    pub z: Vec<i16>,
}

#[derive(Clone, Deserialize)]
pub struct Int8_3 {
    pub x: Vec<i8>,
    pub y: Vec<i8>,
    pub z: Vec<i8>,
}

#[derive(Clone, Deserialize)]
pub struct Float3 {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
}

#[derive(Clone, Deserialize)]
pub struct Double3 {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
}

/// 3x3 row major matrix
#[derive(Clone, Deserialize)]
pub struct Float3x3 {
    pub e00: Vec<f32>,
    pub e01: Vec<f32>,
    pub e02: Vec<f32>,
    pub e10: Vec<f32>,
    pub e11: Vec<f32>,
    pub e12: Vec<f32>,
    pub e20: Vec<f32>,
    pub e21: Vec<f32>,
    pub e22: Vec<f32>,
}

/// Axis aligned bounding box.
#[derive(Clone, Deserialize)]
pub struct AABB {
    pub min: Float3,
    pub max: Float3,
}

/// Bounding sphere.
#[derive(Clone, Deserialize)]
pub struct BoundingSphere {
    pub origo: Float3,
    pub radius: Vec<f32>,
}

/// Node bounding volume.
#[derive(Clone, Deserialize)]
pub struct Bounds {
    pub _box: AABB,
    pub sphere: BoundingSphere,
}


/// Information about child sub meshes used to predict cost before loading.
#[derive(Clone, Deserialize)]
pub struct SubMeshProjection {
    pub object_id: Vec<u32>,
    pub primitive_type: Vec<PrimitiveType>,
    pub attributes: Vec<OptionalVertexAttribute>,
    /// # of deviation vertex attributes (0-3)
    pub num_deviations: Vec<u8>,
    /// zero if no index buffer
    pub num_indices: Vec<u32>,
    pub num_vertices: Vec<u32>,
    pub num_texture_bytes: Vec<u32>,
}

/// Information about child nodes.
#[derive(Clone, Deserialize)]
pub struct ChildInfo {
    /// Byte range into Hash bytes array. The hash, formatted as hex, is used for the filename of the child node.
    pub hash: HashRange,
    pub child_index: Vec<u8>,
    /// Set of bits (max 32) for which child indices are referenced by geometry.
    pub child_mask: Vec<u32>,
    /// A power of two exponent describing the error tolerance of this node, which is used to determine LOD.
    pub tolerance: Vec<i8>,
    /// # uncompressed bytes total for child binary file.
    pub total_byte_size: Vec<u32>,
    /// Model -> world space translation vector.
    pub offset: Double3,
    /// Model -> world space uniform scale factor (from unit [-1,1] vectors).
    pub scale: Vec<f32>,
    /// Bounding volume (in model space).
    pub bounds: Bounds,
    pub sub_meshes: SubMeshProjectionRange,
    pub descendant_object_ids: DescendantObjectIdsRange,
}

/// Mesh deviations vertex attributes
#[derive(Clone, Deserialize)]
pub struct Deviations {
    pub a: Option<Vec<f16>>,
    pub b: Option<Vec<f16>>,
    pub c: Option<Vec<f16>>,
    pub d: Option<Vec<f16>>,
}

/// Mesh vertices
#[derive(Clone, Deserialize)]
pub struct Vertex {
    pub position: Int16_3,
    pub normal: Option<Int8_3>,
    pub color: Option<RgbaU8>,
    pub tex_coord: Option<Half2>,
    pub projected_pos: Option<Int16_3>,
    pub deviations: Deviations,
}

/// Mesh triangles
#[derive(Clone, Deserialize)]
pub struct Triangle {
    /// Bits [0-2] are edge flags (vertex pairs ab, bc, ca), and [3-5] are corner flags.
    /// True = edge/corner is a \"hard\", or true topological feature and should be rendered and/or
    /// snapped to.
    pub topology_flags: Option<Vec<u8>>,
}

/// Mesh Textures
#[derive(Clone, Deserialize)]
pub struct TextureInfo {
    pub semantic: Vec<TextureSemantic>,
    pub transform: Float3x3,
    pub pixel_range: PixelRange,
}

/// Groups of 3D primitives with common attributes. These can further be split up to form e.g. 64K
/// chunks for 16 bit indexing, i.e. there can be many submeshes with the same attributes. Groups
/// are ordered by child, object and material indices.
#[derive(Clone, Deserialize)]
pub struct SubMesh {
    pub child_index: Vec<u8>,
    pub object_id: Vec<u32>,
    pub material_index: Vec<u8>,
    pub primitive_type: Vec<PrimitiveType>,
    pub material_type: Vec<MaterialType>,
    pub attributes: Vec<OptionalVertexAttribute>,
    /// # of deviation vertex attributes (0-4)
    pub num_deviations: Vec<u8>,
    /// Vertices are local to each sub-mesh.
    pub vertices: VertexRange,
    /// Triangle vertex index triplets, or line index pairs, if any, are 16-bit and relative to the local vertex range.
    pub primitive_vertex_indices: VertexIndexRange,
    /// "Hard" edge vertex index pairs, if any, are 16-bit and relative to the local vertex range.
    pub edge_vertex_indices: VertexIndexRange,
    /// "Hard" corner vertex indices, if any, are 16-bit and relative to the local vertex range.
    pub corner_vertex_indices: VertexIndexRange,
    pub textures: TextureInfoRange,
}

/// Hash bytes
#[derive(Clone, Deserialize)]
pub struct HashRange(Range<u32>);

/// Range into submesh projection.
#[derive(Clone, Deserialize)]
pub struct SubMeshProjectionRange(Range<u32>);

/// Range into descendantObjectIdsRange.
#[derive(Clone, Deserialize)]
pub struct DescendantObjectIdsRange(Range<u32>);

/// Mesh vertices
#[derive(Clone, Deserialize)]
pub struct VertexRange(Range<u32>);

/// Mesh vertex indices
#[derive(Clone, Deserialize)]
pub struct VertexIndexRange(Range<u32>);

/// Mesh Textures
#[derive(Clone, Deserialize)]
pub struct TextureInfoRange(Range<u8>);

/// Range into texture pixel blob.
#[derive(Clone, Deserialize)]
pub struct PixelRange(pub Range<u32>);

pub type HashBytes = u8;
/// Information about descendant object ids, which may be present for branches with few object ids,
/// such as documents.
pub type DescendantObjectIds = u32;
pub type TexturePixels = u8;


#[derive(Clone, Deserialize)]
pub struct Schema {
    pub child_info: ChildInfo,
    pub hash_bytes: Vec<HashBytes>,
    pub descendant_object_ids: Vec<DescendantObjectIds>,
    pub sub_mesh_projection: SubMeshProjection,
    pub sub_mesh: SubMesh,
    pub texture_info: TextureInfo,
    pub vertex: Vertex,
    pub triangle: Triangle,
    /// Mesh vertex indices, relative to each draw call, hence 16 bit.
    pub vertex_index: Option<Vec<u16>>,
    pub texture_pixels: Vec<TexturePixels>,
}

use anyhow::Result;
use serde::Deserialize;

#[derive(Deserialize, Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
struct Sizes {
    child_info: u32,
    hash_bytes: u32,
    descendant_object_ids: u32,
    sub_mesh_projection: u32,
    sub_mesh: u32,
    texture_info: u32,
    vertex: u32,
    triangle: u32,
    vertex_index: u32,
    texture_pixels: u32,
}

const NUM_OPTIONALS: usize = 10;

struct Optionals {
    flags: [u8; NUM_OPTIONALS],
    next: usize
}

impl Optionals {
    fn next(&mut self) -> bool {
        let ret = self.flags[self.next];
        self.next += 1;
        ret != 0
    }
}

#[inline(always)]
fn align<T>(next: usize) -> usize {
    // let rem = next % align_of::<T>();
    // if rem != 0 { align_of::<T>() - rem } else { 0 }

    (size_of::<T>() - 1) - ((next + size_of::<T>() - 1) % size_of::<T>())
}

struct Reader<R> {
    reader: R,
}

impl<R: Read + Seek> Reader<R> {
    fn read<T: for<'de> Deserialize<'de> + Pod>(&mut self) -> T {
        // bincode::deserialize_from(&mut self.reader).unwrap()

        let mut ret = MaybeUninit::uninit();
        self.reader.read_exact(bytemuck::bytes_of_mut(unsafe{ &mut *ret.as_mut_ptr() })).unwrap();
        unsafe{ ret.assume_init() }
    }


    fn read_slice<T: for<'de> Deserialize<'de> + Pod>(&mut self, len: u32) -> Vec<T> {
        if len != 0 {
            let padding = align::<T>(self.reader.stream_position().unwrap() as usize);
            if padding != 0 {
                self.reader.seek(SeekFrom::Current(padding as i64)).unwrap();
            }
            (0..len).map(|_| self.read()).collect()
        }else{
            vec![]
        }
    }


    fn read_enum_slice<T: for<'de> Deserialize<'de> + bytemuck::Pod + Sized, E: for<'de> Deserialize<'de>>(&mut self, len: u32) -> Vec<E> {
        let ret = self.read_slice::<T>(len);
        unsafe{ core::mem::transmute(ret) }


        // self.read_slice(len)
    }

    fn range<T: for<'de> Deserialize<'de> + bytemuck::Pod>(&mut self, len: u32) -> Range<T> {
        Range {
            start: self.read_slice(len),
            count: self.read_slice(len)
        }
    }

    fn double3(&mut self, len: u32) -> Double3 {
        Double3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn float3(&mut self, len: u32) -> Float3 {
        Float3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn float3x3(&mut self, len: u32) -> Float3x3 {
        Float3x3 {
            e00: self.read_slice(len),
            e01: self.read_slice(len),
            e02: self.read_slice(len),
            e10: self.read_slice(len),
            e11: self.read_slice(len),
            e12: self.read_slice(len),
            e20: self.read_slice(len),
            e21: self.read_slice(len),
            e22: self.read_slice(len),
        }
    }

    fn int16_3(&mut self, len: u32) -> Int16_3 {
        Int16_3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn int8_3(&mut self, len: u32) -> Int8_3 {
        Int8_3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn rgba_u8(&mut self, len: u32) -> RgbaU8 {
        RgbaU8 {
            red: self.read_slice(len),
            green: self.read_slice(len),
            blue: self.read_slice(len),
            alpha: self.read_slice(len),
        }
    }

    fn half2(&mut self, len: u32) -> Half2 {
        Half2 {
            x: self.read_slice(len),
            y: self.read_slice(len),
        }
    }

    fn aabb(&mut self, len: u32) -> AABB {
        AABB {
            min: self.float3(len),
            max: self.float3(len)
        }
    }

    fn bounding_sphere(&mut self, len: u32) -> BoundingSphere {
        BoundingSphere { origo: self.float3(len), radius: self.read_slice(len) }
    }

    fn bounds(&mut self, len: u32) -> Bounds {
        Bounds {
            _box: self.aabb(len),
            sphere: self.bounding_sphere(len)
        }
    }

    fn child_info(&mut self, len: u32) -> ChildInfo {
        ChildInfo {
            hash: HashRange(self.range(len)),
            child_index: self.read_slice(len),
            child_mask: self.read_slice(len),
            tolerance: self.read_slice(len),
            total_byte_size: self.read_slice(len),
            offset: self.double3(len),
            scale: self.read_slice(len),
            bounds: self.bounds(len),
            sub_meshes: SubMeshProjectionRange(self.range(len)),
            descendant_object_ids: DescendantObjectIdsRange(self.range(len)),

        }
    }

    fn sub_mesh_projection(&mut self, len: u32) -> SubMeshProjection {
        SubMeshProjection {
            object_id: self.read_slice(len),
            primitive_type: self.read_enum_slice::<u8, _>(len),
            attributes: self.read_enum_slice::<u8, _>(len),
            num_deviations: self.read_slice(len),
            num_indices: self.read_slice(len),
            num_vertices: self.read_slice(len),
            num_texture_bytes: self.read_slice(len),
        }
    }

    fn sub_mesh(&mut self, len: u32) -> SubMesh {
        SubMesh {
            child_index: self.read_slice(len),
            object_id: self.read_slice(len),
            material_index: self.read_slice(len),
            primitive_type: self.read_enum_slice::<u8, _>(len),
            material_type: self.read_enum_slice::<u8, _>(len),
            attributes: self.read_enum_slice::<u8, _>(len),
            num_deviations: self.read_slice(len),
            vertices: VertexRange(self.range(len)),
            primitive_vertex_indices: VertexIndexRange(self.range(len)),
            edge_vertex_indices: VertexIndexRange(self.range(len)),
            corner_vertex_indices: VertexIndexRange(self.range(len)),
            textures: TextureInfoRange(self.range(len)),
        }
    }

    fn texture_info(&mut self, len: u32) -> TextureInfo  {
        TextureInfo {
            semantic: self.read_enum_slice::<u8, _>(len),
            transform: self.float3x3(len),
            pixel_range: PixelRange(self.range(len)),
        }
    }

    fn deviations(&mut self, len: u32, optionals: &mut Optionals) -> Deviations {
        Deviations {
            a: optionals.next().then(|| self.read_slice(len)),
            b: optionals.next().then(|| self.read_slice(len)),
            c: optionals.next().then(|| self.read_slice(len)),
            d: optionals.next().then(|| self.read_slice(len)),
        }
    }

    fn vertex(&mut self, len: u32, optionals: &mut Optionals) -> Vertex {
        Vertex {
            position: self.int16_3(len),
            normal: optionals.next().then(|| self.int8_3(len)),
            color: optionals.next().then(|| self.rgba_u8(len)),
            tex_coord: optionals.next().then(|| self.half2(len)),
            projected_pos: optionals.next().then(|| self.int16_3(len)),
            deviations: self.deviations(len, optionals),
        }
    }

    fn triangle(&mut self, len: u32, optionals: &mut Optionals) -> Triangle {
        Triangle { topology_flags: optionals.next().then(|| self.read_slice(len)) }
    }
}

impl Schema {
    pub fn read<R: Read + Seek>(reader: R) -> Result<Schema> {
        let mut reader = Reader { reader };
        let sizes: Sizes = reader.read();
        let mut optionals = Optionals{ flags: reader.read(), next: 0 };

        let child_info = reader.child_info(sizes.child_info);
        let hash_bytes = reader.read_slice(sizes.hash_bytes);
        let descendant_object_ids = reader.read_slice(sizes.descendant_object_ids);
        let sub_mesh_projection = reader.sub_mesh_projection(sizes.sub_mesh_projection);
        let sub_mesh = reader.sub_mesh(sizes.sub_mesh);
        let texture_info = reader.texture_info(sizes.texture_info);
        let vertex = reader.vertex(sizes.vertex, &mut optionals);
        let triangle = reader.triangle(sizes.triangle, &mut optionals);
        let vertex_index = optionals.next().then(|| reader.read_slice(sizes.vertex_index));
        let texture_pixels = reader.read_slice(sizes.texture_pixels);

        Ok(Schema {
            child_info,
            hash_bytes,
            descendant_object_ids,
            sub_mesh_projection,
            sub_mesh,
            texture_info,
            vertex,
            triangle,
            vertex_index,
            texture_pixels,
        })
    }
}

#[test]
fn test_read() {
    let then = std::time::Instant::now();
    let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
    let schema = Schema::read(std::io::BufReader::new(&mut file)).unwrap();
    dbg!(then.elapsed());
    for p in schema.sub_mesh_projection.primitive_type {
        assert!((p as u8) < 7);
    }
    assert_eq!(file.read_to_end(&mut vec![0;1024]).unwrap(),  0);
}