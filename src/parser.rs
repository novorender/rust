
use core::mem::{size_of, align_of};

use half::f16;

use crate::thin_slice::ThinSlice;
use crate::range::RangeSlice;
use crate::ktx::*;

pub struct Optionals<'a, const NUM_OPTIONALS: usize> {
    flags: &'a [u8; NUM_OPTIONALS],
    next: usize
}

impl<'a, const NUM_OPTIONALS: usize> Optionals<'a, NUM_OPTIONALS> {
    pub fn new(flags: &'a [u8; NUM_OPTIONALS]) -> Self {
        Optionals {
            flags,
            next: 0,
        }
    }

    pub fn next(&mut self) -> bool {
        let ret = self.flags[self.next];
        self.next += 1;
        ret != 0
    }
}

#[inline(always)]
fn align<T>(next: usize) -> usize {
    let rem = next % align_of::<T>();
    if rem != 0 { align_of::<T>() - rem } else { 0 }

    // (size_of::<T>() - 1) - ((next + size_of::<T>() - 1) % size_of::<T>())
}

#[cfg(not(feature = "checked_types"))]
pub trait Pod {

}

#[cfg(not(feature = "checked_types"))]
impl<T> Pod for T {}

#[cfg(not(feature = "checked_types"))]
pub trait CheckedBitPattern {

}

#[cfg(not(feature = "checked_types"))]
impl<T> CheckedBitPattern for T {}


#[cfg(feature = "checked_types")]
use bytemuck::{Pod, CheckedBitPattern};

pub struct Reader<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) next: usize
}

impl<'a> Reader<'a> {
    pub fn read<T: bytemuck::Pod>(&mut self) -> &'a T {
        let ret = bytemuck::from_bytes(&self.data[self.next..self.next + size_of::<T>()]);
        self.next += size_of::<T>();
        ret
    }

    pub fn read_checked<T: bytemuck::CheckedBitPattern>(&mut self) -> &'a T {
        let ret = bytemuck::checked::from_bytes(&self.data[self.next..self.next + size_of::<T>()]);
        self.next += size_of::<T>();
        ret
    }

    pub fn read_slice<T: Pod + 'a>(&mut self, len: u32) -> ThinSlice<'a, T> {
        if len > 0 {
            self.next += align::<T>(self.next);

            let size = size_of::<T>() * len as usize;
            let next_next = self.next + size;

            #[cfg(not(feature = "checked_types"))]
            let ret = ThinSlice::from_data_and_offset(&self.data, self.next, len);

            #[cfg(feature = "checked_types")]
            let ret = ThinSlice::from(bytemuck::cast_slice(&self.data[self.next .. next_next]));

            self.next = next_next;
            ret
        }else{
            ThinSlice::empty()
        }

    }

    pub fn read_checked_slice<T: CheckedBitPattern + 'a>(&mut self, len: u32) -> ThinSlice<'a, T> {
        if len > 0 {
            self.next += align::<T>(self.next);

            let size = size_of::<T>() * len as usize;
            let next_next = self.next + size;

            #[cfg(not(feature = "checked_types"))]
            let ret = ThinSlice::from_data_and_offset(&self.data, self.next, len);

            #[cfg(feature = "checked_types")]
            let ret = ThinSlice::from(bytemuck::checked::cast_slice(&self.data[self.next .. next_next]));

            self.next = next_next;
            ret
        }else{
            ThinSlice::empty()
        }
    }

    pub fn read_range<T: Pod + 'a>(&mut self, len: u32) -> RangeSlice<'a, T> {
        RangeSlice{ start: self.read_slice(len), count: self.read_slice(len) }
    }
}


#[derive(Default)]
struct Offsets {
    offsets: [u32; Attribute::Deviations as usize + 1],
    stride: u32,
}

impl std::ops::Index<Attribute> for Offsets {
    type Output = u32;
    fn index(&self, index: Attribute) -> &Self::Output {
        &self.offsets[index as usize]
    }
}

impl std::ops::IndexMut<Attribute> for Offsets {
    fn index_mut(&mut self, index: Attribute) -> &mut Self::Output {
        &mut self.offsets[index as usize]
    }
}

#[derive(Clone, Copy)]
enum Attribute {
    Position,
    Normal,
    Color,
    TexCoord,
    ProjectedPos,
    MaterialIndex,
    ObjectId,
    Deviations,
}

impl Attribute {
    const fn num_components(self, num_deviations: u8) -> u32 {
        match self {
            Attribute::Position => 3,
            Attribute::Color => 4,
            Attribute::Normal => 3,
            Attribute::ProjectedPos => 3,
            Attribute::TexCoord => 2,
            Attribute::MaterialIndex => 1,
            Attribute::ObjectId => 1,
            Attribute::Deviations => num_deviations as u32,
        }
    }

    const fn bytes_per_element(self) -> usize {
        match self {
            Attribute::Position => size_of::<u16>(),
            Attribute::Color => size_of::<u16>(),
            Attribute::Normal => size_of::<i8>(),
            Attribute::ProjectedPos => size_of::<u16>(),
            Attribute::TexCoord => size_of::<f16>(),
            Attribute::MaterialIndex => size_of::<u8>(),
            Attribute::ObjectId => size_of::<u32>(),
            Attribute::Deviations => size_of::<f16>(),
        }
    }
}

struct AggregateProjections{
    primitives: usize,
    gpu_bytes: usize,
}

fn compute_vertex_offsets_(attributes: impl IntoIterator<Item = Attribute>, num_deviations: u8) -> Offsets {
    let mut offset = 0;
    let mut offsets: Offsets = Offsets::default();

    fn padding(alignment: u32, offset: u32) -> u32 {
        alignment - 1 - (offset + alignment - 1) % alignment
    }

    let mut max_align = 1;
    for attribute in attributes.into_iter() {
        let count = attribute.num_components(num_deviations);
        let bytes = attribute.bytes_per_element() as u32;
        max_align = max_align.max(bytes);
        offset += padding(bytes as u32, offset);
        offsets[attribute.into()] = offset;
        offset += bytes * count;
    }
    offset += padding(max_align, offset); // align stride to largest typed array
    offsets.stride = offset;
    offsets
}

// TODO: this is allocating 3 strings
#[inline(always)]
fn to_hex(hash_bytes: &[u8]) -> String {
    hash_bytes.iter().map(|x| {
        let s = format!("{:#04x}", x).to_ascii_uppercase();
        s[s.len() - 2 ..].to_string()
    }).collect()
}

#[test]
fn test_to_hex() {
    assert_eq!(to_hex(&[1, 2, 15, 3]), "01020F03")
}

pub struct Highlights<'a> {
    pub indices: &'a [u8],
    // mutex
}

#[derive(Clone, Copy)]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MeshObjectRange {
    pub object_id: u32,
    pub begin_vertex: usize,
    pub end_vertex: usize,
    pub begin_triangle: usize,
    pub end_triangle: usize,
}

#[derive(Clone, Copy)]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DrawRange {
    pub child_index: u8,
    pub byte_offset: usize,
    pub first: usize,
    pub count: usize,
}

#[derive(Clone, Copy)]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexAttribute {
    pub kind: &'static str,
    pub buffer: i8,
    pub component_count: u8,
    pub component_type: &'static str,
    pub normalized: bool,
    pub byte_offset: u32,
    pub byte_stride: u32,
}

#[derive(Clone)]
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VertexAttributes {
    pub position: VertexAttribute,
    pub normal: Option<VertexAttribute>,
    pub material: Option<VertexAttribute>,
    pub object_id: Option<VertexAttribute>,
    pub tex_coord: Option<VertexAttribute>,
    pub color: Option<VertexAttribute>,
    pub projected_pos: Option<VertexAttribute>,
    pub deviations: Option<VertexAttribute>,
    pub triangles0: Option<VertexAttribute>,
    pub triangles1: Option<VertexAttribute>,
    pub triangles2: Option<VertexAttribute>,
    pub triangles_obj_id: Option<VertexAttribute>,
    pub highlight: VertexAttribute,
    pub highlight_tri: VertexAttribute,
}

struct PossibleBuffers {
    pos: Vec<u8>,
    primary: Vec<u8>,
    tri_pos: Option<Vec<i16>>,
    tri_id: Option<Vec<u32>>,
    highlight: Vec<u8>,
    highlight_tri: Option<Vec<u8>>,
}

struct BufIndex {
    pos: i8,
    primary: i8,
    tri_pos: i8,
    tri_id: i8,
    highlight: i8,
    highlight_tri: i8,
}

#[derive(serde::Serialize)]
#[serde(untagged)]
pub enum Indices {
    #[serde(with = "serde_bytes")]
    IndexBuffer(Vec<u8>),
    NumIndices(u32),
}

mod serde_bytes_nested {
    pub fn serialize<S>(bytes: &Vec<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;

        struct AsBytes<'a>(&'a Vec<u8>);

        impl<'a> serde::Serialize for AsBytes<'a> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where S: serde::Serializer
            {
                serde_bytes::serialize(self.0, serializer)
            }
        }

        let mut seq = serializer.serialize_seq(bytes.len().into())?;
        for bytes in bytes {
            seq.serialize_element(&AsBytes(bytes))?
        }
        seq.end()
    }
}

// use crate::types_2_0::*;
// use crate::reader_2_0::*;
// use _2_0::*;
// use crate::parser::*;

macro_rules! impl_parser {
    ($version: ident, $child_ty: ty $(, $child_extra_fields: ident)*) => { paste::paste! {
        use crate::[<types $version>]::*;
        use crate::[<reader $version>]::*;
        use crate::parser::*;
        use crate::thin_slice::ThinSliceIterator;
        use crate::interleaved::*;

        use hashbrown::{HashMap, hash_map::Entry};

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

        impl From<OptionalVertexAttribute> for Attribute {
            fn from(value: OptionalVertexAttribute) -> Self {
                match value {
                    OptionalVertexAttribute::COLOR => Attribute::Color,
                    OptionalVertexAttribute::NORMAL => Attribute::Normal,
                    OptionalVertexAttribute::PROJECTED_POS => Attribute::ProjectedPos,
                    OptionalVertexAttribute::TEX_COORD => Attribute::TexCoord,
                    _ => unreachable!()
                }
            }
        }

        pub mod float3_seq_serializer {
            pub fn serialize<S>(v: &crate::[<types $version>]::Float3, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                use serde::ser::SerializeSeq;


                let mut seq = serializer.serialize_seq(Some(3))?;
                seq.serialize_element(&v.x)?;
                seq.serialize_element(&v.y)?;
                seq.serialize_element(&v.z)?;
                seq.end()
            }
        }

        pub mod double3_seq_serializer {
            pub fn serialize<S>(v: &crate::[<types $version>]::Double3, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                use serde::ser::SerializeSeq;


                let mut seq = serializer.serialize_seq(Some(3))?;
                seq.serialize_element(&v.x)?;
                seq.serialize_element(&v.y)?;
                seq.serialize_element(&v.z)?;
                seq.end()
            }
        }

        // All attributes + position for non-separate position, not used anymore
        // fn compute_vertex_offsets(attributes: OptionalVertexAttribute, num_deviations: u8, has_materials: bool, has_object_ids: bool) -> Offsets {
        //     compute_vertex_offsets_(
        //         attributes.iter()
        //             .map(|attribute| attribute.into())
        //             .chain(Some(Attribute::Position))
        //             .chain((num_deviations > 0).then_some(Attribute::Deviations))
        //             .chain(has_materials.then_some(Attribute::MaterialIndex))
        //             .chain(has_object_ids.then_some(Attribute::ObjectId)),
        //         num_deviations
        //     )
        // }

        fn compute_vertex_position_offsets() -> Offsets {
            compute_vertex_offsets_(Some(Attribute::Position), 0)
        }

        fn compute_vertex_position_deviations_offsets(num_deviations: u8) -> Offsets {
            compute_vertex_offsets_(
                Some(Attribute::Position).into_iter()
                    .chain((num_deviations > 0).then_some(Attribute::Deviations)),
                num_deviations
            )
        }

        fn compute_vertex_attributes_offsets(attributes: OptionalVertexAttribute, num_deviations: u8, has_materials: bool, has_object_ids: bool) -> Offsets {
            compute_vertex_offsets_(
                attributes.iter()
                    .map(|attribute| attribute.into())
                    .chain((num_deviations > 0).then_some(Attribute::Deviations))
                    .chain(has_materials.then_some(Attribute::MaterialIndex))
                    .chain(has_object_ids.then_some(Attribute::ObjectId)),
                num_deviations
            )
        }

        fn compute_primitive_count(primitive_type: PrimitiveType, count: u32) -> u32{
            match primitive_type {
                PrimitiveType::Points => count,
                PrimitiveType::Lines => count / 2,
                PrimitiveType::LineLoop => count,
                PrimitiveType::LineStrip => count - 1,
                PrimitiveType::Triangles => count / 3,
                PrimitiveType::TriangleStrip => count - 2,
                PrimitiveType::TriangleFan => count - 2,
            }
        }

        impl<'a> SubMeshProjectionSlice<'a> {
            fn aggregate_projections(&self, filter: impl Fn(u32) -> bool) -> AggregateProjections {
                let mut primitives = 0usize;
                let mut total_texture_bytes = 0usize;
                let mut total_num_indices = 0usize;
                let mut total_num_vertices = 0usize;
                let mut total_num_vertex_bytes = 0usize;
                for SubMeshProjection {
                    object_id,
                    primitive_type,
                    attributes,
                    num_deviations,
                    num_indices,
                    num_vertices,
                    num_texture_bytes
                } in self.iter() {
                    if filter(object_id) {
                        let has_materials = num_texture_bytes == 0;
                        let has_object_ids = true;
                        let num_bytes_per_vertex = compute_vertex_position_offsets().stride
                            + compute_vertex_attributes_offsets(
                                attributes,
                                num_deviations,
                                has_materials,
                                has_object_ids
                            ).stride;
                        primitives += compute_primitive_count(
                            primitive_type,
                            if num_indices > 0 { num_indices } else { num_vertices }
                        ) as usize;
                        total_num_indices += num_indices as usize;
                        total_num_vertices += num_vertices as usize;
                        total_num_vertex_bytes += num_bytes_per_vertex as usize;
                        total_texture_bytes += num_texture_bytes as usize;
                    }
                }

                let idx_stride = if total_num_vertices < u32::MAX as usize { 2 } else { 4 };
                let gpu_bytes = total_texture_bytes + total_num_vertex_bytes + total_num_indices * idx_stride;
                AggregateProjections {
                    primitives,
                    gpu_bytes
                }
            }
        }

        impl<'a> ChildInfo<'a> {
            pub fn to_child(&self, filter: impl Fn(u32) -> bool) -> $child_ty {
                let id = to_hex(self.hash);
                let f32_offset = self.offset.into();
                let bounds = Bounds {
                    _box: AABB {
                        min: self.bounds._box.min + f32_offset,
                        max: self.bounds._box.max + f32_offset,
                    },
                    sphere: BoundingSphere {
                        origo: self.bounds.sphere.origo + f32_offset,
                        radius: self.bounds.sphere.radius
                    }
                };

                let AggregateProjections { primitives, gpu_bytes } = self.sub_meshes.aggregate_projections(
                    filter
                );
                let parent_primitives = 0; // This gets a value from an empty array in the original
                let primitives_delta = primitives - parent_primitives;

                Child {
                    id,
                    child_index: self.child_index,
                    child_mask: self.child_mask,
                    tolerance: self.tolerance,
                    byte_size: self.total_byte_size,
                    offset: self.offset,
                    scale: self.scale,
                    bounds,
                    primitives,
                    primitives_delta,
                    gpu_bytes,
                    $(
                        $child_extra_fields: (!self.$child_extra_fields.is_empty())
                            .then(|| bytemuck::cast_slice(self.$child_extra_fields)),
                    )*
                }
            }
        }

        impl<'a> Int8_3Slice<'a> {
            fn copy_to_interleaved_array(&self, dst: &mut [i8], component: u32, byte_offset: usize, byte_stride: usize, len: usize) {
                debug_assert_eq!(byte_offset % size_of::<i8>(), 0);
                debug_assert_eq!(byte_stride % size_of::<i8>(), 0);


                let offset = byte_offset / size_of::<i8>();
                let stride = byte_stride / size_of::<i8>();

                let mut src = match component {
                    0 => self.x.thin_iter(),
                    1 => self.y.thin_iter(),
                    2 => self.z.thin_iter(),
                    _ => unreachable!()
                };

                for dst in dst[offset..].iter_mut().step_by(stride).take(len) {
                    *dst = unsafe{ *src.next() }
                }
            }
        }

        impl<'a> VertexSlice<'a> {
            fn copy_attribute_to_interleaved_array(&self, dst: &mut [u8], attribute: Attribute, component: u32, byte_offset: usize, byte_stride: usize) {
                match attribute {
                    Attribute::Position => todo!(),
                    Attribute::Normal => if let Some(normal) = &self.normal {
                        normal.copy_to_interleaved_array(
                            bytemuck::cast_slice_mut(dst),
                            component,
                            byte_offset,
                            byte_stride,
                            self.len as usize
                        )
                    },
                    Attribute::Color => todo!(),
                    Attribute::TexCoord => todo!(),
                    Attribute::ProjectedPos => todo!(),
                    Attribute::MaterialIndex => todo!(),
                    Attribute::ObjectId => todo!(),
                    Attribute::Deviations => todo!(),
                }
            }
        }

        struct SubMesh<'a> {
            child_index: u8,
            object_id: u32,
            material_index: u8,
            material_type: MaterialType,
            primitive_type: PrimitiveType,
            attributes: OptionalVertexAttribute,
            num_deviations: u8,
            vertices: VertexSlice<'a>,
            indices: &'a [VertexIndex],
            textures: TextureInfoSlice<'a>,
            texture_range: TextureInfoRange,
        }

        #[derive(serde::Serialize)]
        #[serde(rename_all = "camelCase")]
        pub struct ReturnSubMesh {
            pub material_type: u8,
            primitive_type: PrimitiveType,
            pub num_vertices: u32,
            pub num_triangles: u32,
            object_ranges: Vec<MeshObjectRange>,
            vertex_attributes: VertexAttributes,
            #[serde(with = "serde_bytes_nested")]
            vertex_buffers: Vec<Vec<u8>>,
            indices: Indices,
            pub base_color_texture: Option<u32>,
            draw_ranges: Vec<DrawRange>,
        }

        #[derive(serde::Serialize)]
        #[serde(rename_all = "camelCase")]
        #[derive(Clone)]
        pub struct Texture {
            pub semantic: TextureSemantic,
            pub transform: Float3x3,
            pub params: TextureParameters,
        }


        impl<'a> Schema<'a> {
            pub fn parse(data: &'a [u8]) -> Schema<'a> {
                Parser::new(Reader { data, next: 0 }).read_schema()
            }

            fn sub_meshes(&self, filter: impl Fn(u32) -> bool + 'a) -> impl Iterator<Item = SubMesh> {
                // TODO: iter and thin_iter could receive schema instead of individual params?
                let mut texture_ranges = self.sub_mesh.textures.thin_iter();
                self.sub_mesh.iter(
                    self.vertex.clone(),
                    self.vertex_index.unwrap_or(ThinSlice::empty()),
                    self.vertex_index.unwrap_or(ThinSlice::empty()),
                    self.vertex_index.unwrap_or(ThinSlice::empty()),
                    self.texture_info.clone()
                )
                .filter_map(move |sub_mesh| if filter(sub_mesh.object_id) {
                    let texture_range = unsafe{ texture_ranges.next() };
                    Some(SubMesh {
                        child_index: sub_mesh.child_index,
                        object_id: sub_mesh.object_id,
                        material_index: sub_mesh.material_index,
                        material_type: if sub_mesh.material_index == u8::MAX
                            && sub_mesh.textures.len == 0
                            && (
                                sub_mesh.primitive_type == PrimitiveType::TriangleStrip
                                    || sub_mesh.primitive_type == PrimitiveType::Triangles
                                )
                        {
                            MaterialType::Elevation
                        }else{
                            sub_mesh.material_type
                        },
                        primitive_type: sub_mesh.primitive_type,
                        attributes: sub_mesh.attributes,
                        num_deviations: sub_mesh.num_deviations,
                        vertices: sub_mesh.vertices,
                        indices: sub_mesh.primitive_vertex_indices,
                        textures: sub_mesh.textures,
                        texture_range,
                    })
                }else{
                    None
                })
            }

            pub fn geometry(&self, enable_outlines: bool, highlights: Highlights, filter: impl Fn(u32) -> bool) -> (Vec<ReturnSubMesh>, Vec<Option<Texture>>){
                // TODO: This is only to check if there's a global indices buffer but maybe just
                // check sub_mesh.indices.is_empty()
                let vertex_index = &self.vertex_index;

                let mut sub_meshes = vec![];
                let mut referenced_textures = HashMap::new();

                struct Group<'a> {
                    material_type: MaterialType,
                    primitive_type: PrimitiveType,
                    attributes: OptionalVertexAttribute,
                    num_deviations: u8,
                    group_meshes: Vec<SubMesh<'a>>
                }

                #[derive(Hash, PartialEq, Eq)]
                struct Key {
                    material_type: MaterialType,
                    primitive_type: PrimitiveType,
                    attributes: OptionalVertexAttribute,
                    num_deviations: u8,
                    child_index: u8,
                }

                let mut groups = HashMap::new();
                for sub_mesh in self.sub_meshes(filter) {
                    let SubMesh {
                        material_type,
                        primitive_type,
                        attributes,
                        num_deviations,
                        child_index,
                        ..
                    } = sub_mesh;

                    let group = groups.entry(Key{
                        material_type,
                        primitive_type,
                        attributes,
                        num_deviations: num_deviations,
                        child_index: child_index,
                    }).or_insert_with(|| Group {
                        material_type,
                        primitive_type,
                        attributes,
                        num_deviations: num_deviations,
                        group_meshes: vec![]
                    });
                    group.group_meshes.push(sub_mesh);
                }

                // TODO: we don't want highlights to change during parsing, so we hold the lock for the entire file
                // most probably it won't be needed as this is a copy of the original in js anyway?
                // highlights.mutex.lock()

                for Group {
                    material_type,
                    primitive_type,
                    attributes,
                    num_deviations,
                    group_meshes
                } in groups.values() {
                    if group_meshes.is_empty() {
                        continue
                    }

                    let has_materials = group_meshes.iter().any(|m| m.material_index != u8::MAX);
                    let has_object_ids = group_meshes.iter().any(|m| m.object_id != u32::MAX);
                    let position_stride = compute_vertex_position_deviations_offsets(*num_deviations).stride as usize;
                    let triangle_pos_stride = position_stride * 3;
                    let attrib_offsets = compute_vertex_attributes_offsets(
                        *attributes,
                        *num_deviations,
                        has_materials,
                        has_object_ids
                    );
                    let vertex_stride = attrib_offsets.stride as usize;

                    let mut child_indices = group_meshes.iter()
                        .map(|mesh| mesh.child_index)
                        .collect::<Vec<_>>();
                    child_indices.sort_unstable();
                    child_indices.dedup();

                    let mut num_vertices = 0;
                    let mut num_indices = 0;
                    let mut num_triangles = 0;
                    for mesh in group_meshes {
                        let vtx_cnt = mesh.vertices.len as usize;
                        let idx_cnt = mesh.indices.len();
                        num_vertices += vtx_cnt;
                        num_indices += idx_cnt;
                        if *primitive_type == PrimitiveType::Triangles {
                            num_triangles += (if idx_cnt > 0 { idx_cnt } else { vtx_cnt } as f32 / 3.).round() as usize;
                        }
                    }

                    let mut vertex_buffer = Vec::with_capacity(num_vertices * vertex_stride);
                    unsafe{ vertex_buffer.set_len(num_vertices * vertex_stride) };

                    let mut triangle_pos_buffer;
                    let mut triangle_object_id_buffer;
                    let mut highlight_buffer_tri;
                    if enable_outlines && *primitive_type == PrimitiveType::Triangles {
                        triangle_pos_buffer = Some(Vec::with_capacity(num_triangles * triangle_pos_stride));
                        triangle_object_id_buffer = Some(vec![0; num_triangles]);
                        highlight_buffer_tri = Some(vec![0; num_triangles]);
                    }else{
                        triangle_pos_buffer = None;
                        triangle_object_id_buffer = None;
                        highlight_buffer_tri = None;
                    }

                    let mut position_buffer = Vec::with_capacity(num_vertices * position_stride);
                    unsafe{ position_buffer.set_len(num_vertices * position_stride) };

                    let index_buffer_bytes_per_element;
                    let mut index_buffer;
                     if vertex_index.is_some() {
                        let bytes_per_element = if num_vertices < u16::MAX as usize {
                            size_of::<u16>()
                        }else{
                            size_of::<u32>()
                        };
                        index_buffer_bytes_per_element = Some(bytes_per_element);
                        let mut buffer = Vec::with_capacity(num_indices * bytes_per_element);
                        unsafe{ buffer.set_len(num_indices * bytes_per_element) };
                        index_buffer = Some(buffer);
                    }else{
                        index_buffer_bytes_per_element = None;
                        index_buffer = None;
                    }

                    let mut highlight_buffer = vec![0; num_vertices];
                    let mut index_offset = 0;
                    let mut vertex_offset = 0;
                    let mut triangle_offset = 0;
                    let mut object_ranges = vec![];
                    let mut draw_ranges = vec![];

                    for child_index in child_indices {
                        let has_meshes = group_meshes.iter()
                            .any(|mesh| mesh.child_index == child_index);
                        if !has_meshes {
                            continue
                        }

                        let draw_range_begin = if index_buffer.is_some() {
                            index_offset
                        } else {
                            vertex_offset
                        };

                        for sub_mesh in group_meshes.iter()
                            .filter(|sub_mesh| sub_mesh.child_index == child_index)
                        {
                            for attrib in attributes.iter()
                                .map(|attribute| attribute.into())
                                .chain((*num_deviations > 0).then_some(Attribute::Deviations))
                                .chain(has_materials.then_some(Attribute::MaterialIndex))
                                .chain(has_object_ids.then_some(Attribute::ObjectId))
                            {
                                let num_components = attrib.num_components(*num_deviations) as usize;
                                let bytes_per_element = attrib.bytes_per_element();
                                let dst = &mut vertex_buffer[vertex_offset * vertex_stride ..];
                                for c in 0..num_components {
                                    let offset = attrib_offsets[attrib] as usize + c * bytes_per_element;
                                    match attrib {
                                        Attribute::MaterialIndex =>
                                            fill_to_interleaved_array(
                                                bytemuck::cast_slice_mut(dst),
                                                sub_mesh.material_index,
                                                offset,
                                                vertex_stride,
                                                0,
                                                sub_mesh.vertices.len as usize,
                                            ),
                                        Attribute::ObjectId =>
                                            fill_to_interleaved_array(
                                                bytemuck::cast_slice_mut(dst),
                                                sub_mesh.object_id,
                                                offset,
                                                vertex_stride,
                                                0,
                                                sub_mesh.vertices.len as usize,
                                            ),
                                        _ => sub_mesh.vertices.copy_attribute_to_interleaved_array(
                                            dst,
                                            attrib,
                                            c as u32,
                                            offset,
                                            vertex_stride
                                        )
                                    }
                                }
                            }

                            let mut num_triangles_in_submesh = 0;
                            if let (Some(triangle_pos_buffer), Some(triangle_object_id_buffer))
                                = (&mut triangle_pos_buffer, &mut triangle_object_id_buffer)
                            {
                                if vertex_index.is_some() && index_buffer.is_some() {
                                    num_triangles_in_submesh = sub_mesh.indices.len() / 3;
                                    let (x, y ,z) = (
                                        sub_mesh.vertices.position.x,
                                        sub_mesh.vertices.position.y,
                                        sub_mesh.vertices.position.z
                                    );
                                    for index in sub_mesh.indices {
                                        let index = *index as usize;
                                        // TODO: Add support for triangle strips and fans as well...
                                        triangle_pos_buffer.push(unsafe{ *x.get_unchecked(index) });
                                        triangle_pos_buffer.push(unsafe{ *y.get_unchecked(index) });
                                        triangle_pos_buffer.push(unsafe{ *z.get_unchecked(index) });
                                    }
                                }else{
                                    let mut position = sub_mesh.vertices.position.thin_iter();
                                    num_triangles_in_submesh = sub_mesh.vertices.len as usize / 3;
                                    for _ in 0..sub_mesh.vertices.len {
                                        let pos = unsafe{ position.next() };
                                        triangle_pos_buffer.push(pos.x);
                                        triangle_pos_buffer.push(pos.y);
                                        triangle_pos_buffer.push(pos.z);
                                    }
                                }
                                triangle_object_id_buffer[triangle_offset .. triangle_offset + num_triangles_in_submesh]
                                    .fill(sub_mesh.object_id);
                            }

                            copy_to_interleaved_array::<i16>(
                                bytemuck::cast_slice_mut(&mut position_buffer),
                                unsafe{ sub_mesh.vertices.position.x.as_slice(sub_mesh.vertices.len) },
                                vertex_offset * position_stride + 0,
                                position_stride,
                                0,
                                sub_mesh.vertices.len as usize,
                            );

                            copy_to_interleaved_array::<i16>(
                                bytemuck::cast_slice_mut(&mut position_buffer),
                                unsafe{ sub_mesh.vertices.position.y.as_slice(sub_mesh.vertices.len) },
                                vertex_offset * position_stride + 2,
                                position_stride,
                                0,
                                sub_mesh.vertices.len as usize,
                            );

                            copy_to_interleaved_array::<i16>(
                                bytemuck::cast_slice_mut(&mut position_buffer),
                                unsafe{ sub_mesh.vertices.position.z.as_slice(sub_mesh.vertices.len) },
                                vertex_offset * position_stride + 4,
                                position_stride,
                                0,
                                sub_mesh.vertices.len as usize,
                            );


                            // initialize index buffer (if any)
                            if let Some(index_buffer) = &mut index_buffer {
                                if num_vertices > u16::MAX as usize {
                                    for (dst, src) in bytemuck::cast_slice_mut::<_, u32>(index_buffer)[index_offset .. index_offset + sub_mesh.indices.len()]
                                        .iter_mut()
                                        .zip(sub_mesh.indices)
                                    {
                                        *dst = *src as u32 + vertex_offset as u32
                                    }
                                }else{
                                    for (dst, src) in bytemuck::cast_slice_mut::<_, u16>(index_buffer)[index_offset .. index_offset + sub_mesh.indices.len()]
                                        .iter_mut()
                                        .zip(sub_mesh.indices)
                                    {
                                        *dst = *src + vertex_offset as u16
                                    }
                                }
                                index_offset += sub_mesh.indices.len();
                            }

                            // initialize highlight buffer
                            let highlight_index = highlights.indices
                                .get(sub_mesh.object_id as usize)
                                .copied();
                            let sub_mesh_end_vertex = vertex_offset + sub_mesh.vertices.len as usize;
                            let sub_mesh_end_triangle = triangle_offset + sub_mesh.indices.len() as usize / 3;
                            if let Some(highlight_index) = highlight_index {
                                highlight_buffer[vertex_offset .. sub_mesh_end_vertex ]
                                    .fill(highlight_index);
                                if let Some(highlight_buffer_tri) = &mut highlight_buffer_tri {
                                    highlight_buffer_tri[triangle_offset .. sub_mesh_end_triangle]
                                        .fill(highlight_index);
                                }
                            }

                            // update object ranges
                            let mut new_range = true;
                            if let Some(MeshObjectRange {
                                object_id,
                                end_vertex,
                                end_triangle,
                                ..
                            }) = object_ranges.last_mut() {
                                if *object_id == sub_mesh.object_id {
                                    *end_vertex = sub_mesh_end_vertex;
                                    *end_triangle = sub_mesh_end_triangle;
                                    new_range = false;
                                }
                            }

                            if new_range {
                                object_ranges.push(MeshObjectRange {
                                    object_id: sub_mesh.object_id,
                                    begin_vertex: vertex_offset,
                                    end_vertex: sub_mesh_end_vertex,
                                    begin_triangle: triangle_offset,
                                    end_triangle: sub_mesh_end_triangle,
                                })
                            }

                            triangle_offset += num_triangles_in_submesh;
                            vertex_offset += sub_mesh.vertices.len as usize;
                        }

                        let draw_range_end = if index_buffer.is_some() {
                            index_offset
                        }else{
                            vertex_offset
                        };
                        let byte_offset = draw_range_begin * if let Some(bytes_per_element) = index_buffer_bytes_per_element {
                            bytes_per_element
                        }else{
                            vertex_stride
                        };
                        let count = draw_range_end - draw_range_begin;
                        draw_ranges.push(DrawRange {
                            child_index,
                            byte_offset,
                            first: draw_range_begin,
                            count
                        });
                    }

                    fn enumerate_buffers(possible_buffers: PossibleBuffers) -> (Vec<Vec<u8>>, BufIndex) {
                        let mut buffers = vec![];
                        let index = BufIndex {
                            primary: {
                                let id = buffers.len();
                                buffers.push(possible_buffers.primary);
                                id as i8
                            },
                            highlight: {
                                let id = buffers.len();
                                buffers.push(possible_buffers.highlight);
                                id as i8
                            },
                            pos: {
                                let id = buffers.len();
                                buffers.push(possible_buffers.pos);
                                id as i8
                            },
                            tri_pos: {
                                if let Some(tri_pos) = possible_buffers.tri_pos {
                                    let id = buffers.len();
                                    buffers.push(bytemuck::cast_slice(&tri_pos).to_vec());  // TODO: avoid this copy, probably use ArrayBuffers directly here
                                    id as i8
                                }else{
                                    -1
                                }
                            },
                            tri_id: {
                                if let Some(tri_id) = possible_buffers.tri_id {
                                    let id = buffers.len();
                                    buffers.push(bytemuck::cast_slice(&tri_id).to_vec()); // TODO: avoid this copy, probably use ArrayBuffers directly here
                                    id as i8
                                }else{
                                    -1
                                }
                            },
                            highlight_tri: {
                                if let Some(highlight_tri) = possible_buffers.highlight_tri {
                                    let id = buffers.len();
                                    buffers.push(highlight_tri);
                                    id as i8
                                }else{
                                    -1
                                }
                            }
                        };

                        (buffers, index)
                    }

                    debug_assert_eq!(vertex_offset, num_vertices);
                    debug_assert_eq!(index_offset, num_indices);
                    debug_assert_eq!(triangle_offset, triangle_object_id_buffer.as_ref().map(|buffer| buffer.len()).unwrap_or(0));

                    let has_triangle_pos_buffer = triangle_pos_buffer.is_some();
                    let has_triangle_object_id_buffer = triangle_object_id_buffer.is_some();

                    let (vertex_buffers, buf_index) = enumerate_buffers(PossibleBuffers {
                        primary: vertex_buffer,
                        highlight: highlight_buffer,
                        pos: position_buffer,
                        tri_pos: triangle_pos_buffer,
                        tri_id: triangle_object_id_buffer,
                        highlight_tri: highlight_buffer_tri
                    });

                    let indices = if let Some(index_buffer) = index_buffer {
                        Indices::IndexBuffer(index_buffer)
                    }else{
                        Indices::NumIndices(num_vertices as u32)
                    };

                    let base_color_texture;
                    let first_group_mesh = group_meshes.first().unwrap();
                    if first_group_mesh.textures.len > 0 {
                        let texture = unsafe{ first_group_mesh.textures
                            .thin_iter(self.texture_pixels)
                            .next() };
                        let texture_index = first_group_mesh.texture_range.start;
                        base_color_texture = Some(texture_index);
                        match referenced_textures.entry(texture_index) {
                            Entry::Vacant(vacant) => { vacant.insert(texture); },
                            _ => ()
                        };
                    }else{
                        base_color_texture = None
                    }

                    let stride = vertex_stride as u32;
                    let deviations_kind = if *num_deviations == 0 || *num_deviations == 1 {
                        "FLOAT".into()
                    }else if *num_deviations == 3 {
                        "FLOAT_VEC3"
                    }else if *num_deviations == 4 {
                        "FLOAT_VEC4"
                    }else{
                        unreachable!("Number of deviations can be at most 4")
                    };
                    let vertex_attributes = VertexAttributes {
                        position: VertexAttribute { kind: "FLOAT_VEC4", buffer: buf_index.pos, component_count: 3, component_type: "SHORT", normalized: true, byte_offset: attrib_offsets[Attribute::Position], byte_stride: 0 },
                        normal: attributes.contains(OptionalVertexAttribute::NORMAL).then(|| VertexAttribute { kind: "FLOAT_VEC3", buffer: buf_index.primary, component_count: 3, component_type: "BYTE", normalized: true, byte_offset: attrib_offsets[Attribute::Normal], byte_stride: stride }),
                        material: has_materials.then(|| VertexAttribute { kind: "UNSIGNED_INT", buffer: buf_index.primary, component_count: 1, component_type: "UNSIGNED_BYTE", normalized: false, byte_offset: attrib_offsets[Attribute::MaterialIndex], byte_stride: stride}),
                        object_id: has_object_ids.then(|| VertexAttribute { kind: "UNSIGNED_INT", buffer: buf_index.primary, component_count: 1, component_type: "UNSIGNED_BYTE", normalized: false, byte_offset: attrib_offsets[Attribute::ObjectId], byte_stride: stride}),
                        tex_coord: attributes.contains(OptionalVertexAttribute::TEX_COORD).then(|| VertexAttribute { kind: "FLOAT_VEC2", buffer: buf_index.primary, component_count: 2, component_type: "HALF_FLOAT", normalized: false, byte_offset: attrib_offsets[Attribute::TexCoord], byte_stride: stride}),
                        color: attributes.contains(OptionalVertexAttribute::COLOR).then(|| VertexAttribute { kind: "FLOAT_VEC4", buffer: buf_index.primary, component_count: 4, component_type: "UNSIGNED_BYTE", normalized: true, byte_offset: attrib_offsets[Attribute::Color], byte_stride: stride}),
                        projected_pos: attributes.contains(OptionalVertexAttribute::PROJECTED_POS).then(|| VertexAttribute { kind: "FLOAT_VEC4", buffer: buf_index.primary, component_count: 3, component_type: "SHORT", normalized: true, byte_offset: attrib_offsets[Attribute::ProjectedPos], byte_stride: stride}),
                        deviations: (*num_deviations > 0).then(|| VertexAttribute { kind: deviations_kind, buffer: buf_index.primary, component_count: *num_deviations, component_type: "HALF_FLOAT", normalized: false, byte_offset: attrib_offsets[Attribute::Deviations], byte_stride: stride}),
                        triangles0: has_triangle_pos_buffer.then(|| VertexAttribute { kind: "FLOAT_VEC4", buffer: buf_index.tri_pos, component_count: 3, component_type: "SHORT", normalized: true, byte_offset: 0, byte_stride: 18}),
                        triangles1: has_triangle_pos_buffer.then(|| VertexAttribute { kind: "FLOAT_VEC4", buffer: buf_index.tri_pos, component_count: 3, component_type: "SHORT", normalized: true, byte_offset: 6, byte_stride: 18}),
                        triangles2: has_triangle_pos_buffer.then(|| VertexAttribute { kind: "FLOAT_VEC4", buffer: buf_index.tri_pos, component_count: 3, component_type: "SHORT", normalized: true, byte_offset: 12, byte_stride: 18}),
                        triangles_obj_id: has_triangle_object_id_buffer.then(|| VertexAttribute { kind: "UNSIGNED_INT", buffer: buf_index.tri_id, component_count: 1, component_type: "UNSIGNED_INT", normalized: false, byte_offset: 0, byte_stride: 4}),
                        highlight: VertexAttribute { kind: "UNSIGNED_INT", buffer: buf_index.highlight, component_count: 1, component_type: "UNSIGNED_BYTE", normalized: false, byte_offset: 0, byte_stride: 0 },
                        highlight_tri: VertexAttribute { kind: "UNSIGNED_INT", buffer: buf_index.highlight_tri, component_count: 1, component_type: "UNSIGNED_BYTE", normalized: false, byte_offset: 0, byte_stride: 0 },
                    };

                    object_ranges.sort_by_key(|obj_range| obj_range.object_id);

                    sub_meshes.push(ReturnSubMesh {
                        material_type: *material_type as u8,
                        primitive_type: *primitive_type,
                        num_vertices: num_vertices as u32,
                        num_triangles: num_triangles as u32,
                        object_ranges,
                        vertex_attributes,
                        vertex_buffers,
                        indices,
                        base_color_texture,
                        draw_ranges: draw_ranges,
                    })
                }

                // TODO: highlights.mutex.unlock()

                let mut textures = vec![None; self.texture_info.len as usize];
                for (index, reference) in referenced_textures {
                    let semantic = reference.semantic;
                    let transform = reference.transform;
                    let ktx = reference.pixel_range;
                    let params = parse_ktx(ktx);
                    textures[index as usize] = Some(Texture { semantic, transform, params });
                }

                (sub_meshes, textures)
            }
        }
    }};
}

pub mod _2_0 {
    use crate::types_2_0;

    impl_parser!(_2_0, Child);

    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Child {
        pub id: String,
        pub child_index: u8,
        pub child_mask: u32,
        pub tolerance: i8,
        pub byte_size: u32,
        #[serde(with = "_2_0::double3_seq_serializer")]
        pub offset: Double3,
        pub scale: f32,
        pub bounds: Bounds,
        pub primitives: usize,
        pub primitives_delta: usize,
        pub gpu_bytes: usize,
    }

    impl<'a> types_2_0::Schema<'a> {
        pub fn children(&self, filter: impl Fn(u32) -> bool + Copy + 'a) -> impl ExactSizeIterator<Item = Child> + '_ {
            self.child_info.iter(
                self.hash_bytes,
                self.sub_mesh_projection.clone(),
            ).map(move |child_info| child_info.to_child(filter))
        }
    }
}

pub mod _2_1 {
    use crate::types_2_1;

    impl_parser!(_2_1, Child<'a>, descendant_object_ids);

    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    pub struct Child<'a> {
        id: String,
        pub child_index: u8,
        pub child_mask: u32,
        pub tolerance: i8,
        pub byte_size: u32,
        #[serde(with = "_2_1::double3_seq_serializer")]
        pub offset: Double3,
        pub scale: f32,
        pub bounds: Bounds,
        pub primitives: usize,
        pub primitives_delta: usize,
        pub gpu_bytes: usize,
        #[serde(with = "serde_bytes")]
        pub descendant_object_ids: Option<&'a [u8]>,
    }

    impl<'a> types_2_1::Schema<'a> {
        pub fn children(&self, filter: impl Fn(u32) -> bool + Copy + 'a) -> impl ExactSizeIterator<Item = Child<'a>> + '_ {
            self.child_info.iter(
                self.hash_bytes,
                self.sub_mesh_projection.clone(),
                self.descendant_object_ids
            ).map(move |child_info| child_info.to_child(filter))
        }
    }
}