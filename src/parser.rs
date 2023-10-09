use core::mem::{transmute, size_of, align_of};

use half::f16;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::thin_slice::ThinSlice;
use crate::range::RangeSlice;
use crate::types::OptionalVertexAttribute;
use crate::{types_2_1, parser_2_1, types_2_0, parser_2_0};
use crate::types::*;

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
    pub(crate) next: usize,
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

    pub fn read_slice<T: Pod>(&mut self, len: u32) -> ThinSlice<'a, T>
    where T: 'a
    {
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

    pub fn read_checked_slice<T: CheckedBitPattern>(&mut self, len: u32) -> ThinSlice<'a, T>
    where T: 'a
    {
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

    pub fn read_range<T: Pod>(&mut self, len: u32) -> RangeSlice<'a, T>
    where T: 'a
    {
        RangeSlice{ start: self.read_slice(len), count: self.read_slice(len) }
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

fn attribute_components(attribute: Attribute) -> u32 {
    match attribute {
        Attribute::Position => 3,
        Attribute::Color => 4,
        Attribute::Normal => 3,
        Attribute::ProjectedPos => 3,
        Attribute::TexCoord => 2,
        Attribute::MaterialIndex => 1,
        Attribute::ObjectId => 1,
        Attribute::Deviations => 4,
    }
}

const fn attribute_bytes_per_element(attribute: Attribute) -> usize {
    match attribute {
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

fn compute_vertex_offsets_(attributes: impl IntoIterator<Item = Attribute>) -> Offsets {
    let mut offset = 0;
    let mut offsets: Offsets = Offsets::default();

    fn padding(alignment: u32, offset: u32) -> u32 {
        alignment - 1 - (offset + alignment - 1) % alignment
    }

    let mut max_align = 1;
    for attribute in attributes.into_iter() {
        let count = attribute_components(attribute);
        let bytes = attribute_bytes_per_element(attribute) as u32;
        max_align = max_align.max(bytes);
        offset += padding(bytes as u32, offset);
        offsets[attribute.into()] = offset;
        offset += bytes * count;
    }
    offset += padding(max_align, offset); // align stride to largest typed array
    offsets.stride = offset;
    offsets
}

fn compute_vertex_offsets(
    attributes: OptionalVertexAttribute,
    num_deviations: u8,
    has_materials: bool,
    has_object_ids: bool) -> Offsets
{
    compute_vertex_offsets_(attributes.into_iter()
        .map(|attribute| attribute.into())
        .chain(Some(Attribute::Position))
        .chain((num_deviations > 0).then_some(Attribute::Deviations))
        .chain(has_materials.then_some(Attribute::MaterialIndex))
        .chain(has_object_ids.then_some(Attribute::ObjectId))
    )
}

fn compute_vertex_position_offsets() -> Offsets {
    compute_vertex_offsets_(Some(Attribute::Position))
}

fn compute_vertex_attributes_offsets(
    attributes: OptionalVertexAttribute,
    num_deviations: u8,
    has_materials: bool,
    has_object_ids: bool) -> Offsets
{
    compute_vertex_offsets_(attributes.into_iter()
        .map(|attribute| attribute.into())
        .chain((num_deviations > 0).then_some(Attribute::Deviations))
        .chain(has_materials.then_some(Attribute::MaterialIndex))
        .chain(has_object_ids.then_some(Attribute::ObjectId))
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

struct AggregateProjections{
    primitives: usize,
    gpu_bytes: usize,
}

enum SubMeshProjectionSlice<'a> {
    _2_0(&'a types_2_0::SubMeshProjectionSlice<'a>),
    _2_1(&'a types_2_1::SubMeshProjectionSlice<'a>),
}

impl<'a> SubMeshProjectionSlice<'a> {
    fn aggregate_projections(&self, filter: impl Fn(u32) -> bool) -> AggregateProjections {
        let mut primitives = 0usize;
        let mut total_texture_bytes = 0usize;
        let mut total_num_indices = 0usize;
        let mut total_num_vertices = 0usize;
        let mut total_num_vertex_bytes = 0usize;

        let mut process_sub_mesh = |
            object_id: u32,
            primitive_type: PrimitiveType,
            attributes: OptionalVertexAttribute,
            num_deviations: u8,
            num_indices: u32,
            num_vertices: u32,
            num_texture_bytes: u32
        | {
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
        };

        match self {
            Self::_2_0(sub_mesh_primitive) => {
                for types_2_0::SubMeshProjection {
                    object_id,
                    primitive_type,
                    attributes,
                    num_deviations,
                    num_indices,
                    num_vertices,
                    num_texture_bytes
                } in sub_mesh_primitive.iter() {
                    process_sub_mesh(object_id, primitive_type, attributes, num_deviations, num_indices, num_vertices, num_texture_bytes);
                }
            }
            Self::_2_1(sub_mesh_primitive) => {
                for types_2_1::SubMeshProjection {
                    object_id,
                    primitive_type,
                    attributes,
                    num_deviations,
                    num_indices,
                    num_vertices,
                    num_texture_bytes
                } in sub_mesh_primitive.iter() {
                    process_sub_mesh(object_id, primitive_type, attributes, num_deviations, num_indices, num_vertices, num_texture_bytes);
                }
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


// TODO: Using unions by now but the usage is convoluted, needs unsafe to access...
// better move the common types to types module instead of auto generated from schema?
#[derive(Clone, Copy)]
pub union Double3 {
    _2_0: types_2_0::Double3,
    _2_1: types_2_1::Double3,
}

#[derive(Clone, Copy)]
pub union Float3 {
    _2_0: types_2_0::Float3,
    _2_1: types_2_1::Float3,
}

impl std::ops::Add for Float3 {
    type Output = Float3;

    fn add(self, rhs: Self) -> Self::Output {
        Float3 { _2_0: types_2_0::Float3 {
            x: unsafe{ self._2_0.x + rhs._2_0.x },
            y: unsafe{ self._2_0.y + rhs._2_0.y },
            z: unsafe{ self._2_0.z + rhs._2_0.z },
        }}
    }
}

impl From<Double3> for Float3 {
    fn from(value: Double3) -> Self {
        Float3 { _2_0: types_2_0::Float3 {
            x: unsafe{ value._2_0.x as f32 },
            y: unsafe{ value._2_0.y as f32 },
            z: unsafe{ value._2_0.z as f32 },
        }}
    }
}


pub struct AABB {
    min: Float3,
    max: Float3,
}

pub struct BoundingSphere {
    origo: Float3,
    radius: f32,
}

pub struct Bounds {
    _box: AABB,
    sphere: BoundingSphere,
}

// End union types

pub struct Child {
    pub id: String,
    pub child_index: u8,
    pub child_mask: u32,
    pub tolerance: i8,
    pub byte_size: u32,
    pub offset: Double3,
    pub scale: f32,
    pub bounds: Bounds,
    pub primitives: usize,
    pub primitives_delta: usize,
    pub gpu_bytes: usize,
}

#[wasm_bindgen]
pub enum Version {
    _2_0,
    _2_1,
}

pub enum Schema<'a> {
    Schema2_0(types_2_0::Schema<'a>),
    Schema2_1(types_2_1::Schema<'a>),
}

impl<'a> Schema<'a> {
    pub fn parse(data: &'a [u8], version: Version) -> Schema<'a> {
        let reader = Reader { data, next: 0 };
        match version {
            Version::_2_0 => {
                let mut parser = parser_2_0::Parser::new(reader);
                Schema::Schema2_0(parser.read_schema())
            }

            Version::_2_1 => {
                let mut parser = parser_2_1::Parser::new(reader);
                Schema::Schema2_1(parser.read_schema())
            }
        }
    }

    pub fn children(&self, filter: impl Fn(u32) -> bool + Copy + 'a) -> Vec<Child> {
        let parse_child_info = |
            hash: &[types_2_1::HashBytes],
            child_index: u8,
            child_mask: u32,
            tolerance: i8,
            total_byte_size: u32,
            offset: Double3,
            scale: f32,
            bounds: &Bounds,
            sub_meshes: SubMeshProjectionSlice,
        | {
            let id = to_hex(hash);
            let f32_offset = offset.into();
            let bounds = Bounds {
                _box: AABB {
                    min: bounds._box.min + f32_offset,
                    max: bounds._box.max + f32_offset,
                },
                sphere: BoundingSphere {
                    origo: bounds.sphere.origo + f32_offset,
                    radius: bounds.sphere.radius
                }
            };

            let AggregateProjections { primitives, gpu_bytes } = sub_meshes.aggregate_projections(
                filter
            );
            let parent_primitives = 0; // This gets a value from an empty array in the original
            let primitives_delta = primitives - parent_primitives;
            Child {
                id,
                child_index: child_index,
                child_mask: child_mask,
                tolerance: tolerance,
                byte_size: total_byte_size,
                offset,
                scale,
                bounds,
                primitives,
                primitives_delta,
                gpu_bytes,
            }
        };

        match self {
            Self::Schema2_0(schema) => schema.child_info
                .iter(schema.hash_bytes, schema.sub_mesh_projection.clone())
                .map(move |child_info| parse_child_info(
                    child_info.hash,
                    child_info.child_index,
                    child_info.child_mask,
                    child_info.tolerance,
                    child_info.total_byte_size,
                    Double3{ _2_0: child_info.offset },
                    child_info.scale,
                    unsafe{ transmute(&child_info.bounds) },
                    SubMeshProjectionSlice::_2_0(&child_info.sub_meshes),
                )).collect::<Vec<_>>(),

            Self::Schema2_1(schema) => schema.child_info
                .iter(schema.hash_bytes, schema.sub_mesh_projection.clone(), schema.descendant_object_ids)
                .map(move |child_info| parse_child_info(
                    child_info.hash,
                    child_info.child_index,
                    child_info.child_mask,
                    child_info.tolerance,
                    child_info.total_byte_size,
                    Double3{ _2_1: child_info.offset },
                    child_info.scale,
                    unsafe{ transmute(&child_info.bounds) },
                    SubMeshProjectionSlice::_2_1(&child_info.sub_meshes),
                )).collect::<Vec<_>>(),
        }
    }
}