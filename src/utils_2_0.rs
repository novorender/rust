/// Types SoA to AoS

use core::mem::size_of;

use half::f16;

use crate::types_2_0::*;
use crate::thin_slice::{ThinSliceIter, ThinSliceIterator};
use crate::range::RangeInstance;

#[derive(Clone, Copy)]
pub struct Float3Instance {
    x: f32,
    y: f32,
    z: f32
}

impl<'a> Float3<'a> {
    pub unsafe fn get_unchecked(&self, index: usize) -> Float3Instance {
        Float3Instance {
            x: *self.x.get_unchecked(index),
            y: *self.y.get_unchecked(index),
            z: *self.z.get_unchecked(index)
        }
    }

    pub fn iter(&self) -> Float3Iter {
        Float3Iter { x: self.x.iter(), y: self.y.iter(), z: self.z.iter() }
    }
}

pub struct Float3Iter<'a> {
    x: ThinSliceIter<'a, f32>,
    y: ThinSliceIter<'a, f32>,
    z: ThinSliceIter<'a, f32>,
}

impl<'a> ThinSliceIterator for Float3Iter<'a> {
    type Item = Float3Instance;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        Float3Instance {
            x: unsafe{ *self.x.next() },
            y: unsafe{ *self.y.next() },
            z: unsafe{ *self.z.next() },
        }
    }
}

impl std::ops::Add for Float3Instance {
    type Output = Float3Instance;

    fn add(self, rhs: Self) -> Self::Output {
        Float3Instance {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl From<Double3Instance> for Float3Instance {
    fn from(value: Double3Instance) -> Self {
        Float3Instance {
            x: value.x as f32,
            y: value.y as f32,
            z: value.z as f32,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Double3Instance {
    x: f64,
    y: f64,
    z: f64
}

impl<'a> Double3<'a> {
    pub unsafe fn get_unchecked(&self, index: usize) -> Double3Instance {
        Double3Instance {
            x: *self.x.get_unchecked(index),
            y: *self.y.get_unchecked(index),
            z: *self.z.get_unchecked(index)
        }
    }

    pub fn iter(&self) -> Double3Iter {
        Double3Iter { x: self.x.iter(), y: self.y.iter(), z: self.z.iter() }
    }
}

pub struct Double3Iter<'a> {
    x: ThinSliceIter<'a, f64>,
    y: ThinSliceIter<'a, f64>,
    z: ThinSliceIter<'a, f64>,
}

impl<'a> ThinSliceIterator for Double3Iter<'a> {
    type Item = Double3Instance;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        Double3Instance {
            x: unsafe{ *self.x.next() },
            y: unsafe{ *self.y.next() },
            z: unsafe{ *self.z.next() },
        }
    }
}

pub struct AABBInstance {
    min: Float3Instance,
    max: Float3Instance,
}

impl<'a> AABB<'a> {
    pub unsafe fn get_unchecked(&self, index: usize) -> AABBInstance {
        AABBInstance {
            min: self.min.get_unchecked(index),
            max: self.max.get_unchecked(index)
        }
    }
}

pub struct BoundingSphereInstance {
    origo: Float3Instance,
    radius: f32,
}

impl<'a> BoundingSphere<'a> {
    pub unsafe fn get_unchecked(&self, index: usize) -> BoundingSphereInstance {
        BoundingSphereInstance {
            origo: self.origo.get_unchecked(index),
            radius: *self.radius.get_unchecked(index)
        }
    }
}

pub struct BoundsInstance {
    _box: AABBInstance,
    sphere: BoundingSphereInstance,
}

impl<'a> Bounds<'a> {
    pub unsafe fn get_unchecked(&self, index: usize) -> BoundsInstance {
        BoundsInstance {
            _box: self._box.get_unchecked(index),
            sphere: self.sphere.get_unchecked(index)
        }
    }

    pub fn iter(&self) -> BoundsIter {
        BoundsIter {
            box_min: self._box.min.iter(),
            box_max: self._box.max.iter(),
            origo_iter: self.sphere.origo.iter(),
            radius_iter: self.sphere.radius.iter()
        }
    }
}

pub struct BoundsIter<'a> {
    box_min: Float3Iter<'a>,
    box_max: Float3Iter<'a>,
    origo_iter: Float3Iter<'a>,
    radius_iter: ThinSliceIter<'a, f32>,
}

impl<'a> ThinSliceIterator for BoundsIter<'a> {
    type Item = BoundsInstance;

    #[inline(always)]
    unsafe fn next(&mut self) -> Self::Item {
        BoundsInstance {
            _box: AABBInstance {
                min: unsafe{ self.box_min.next() },
                max: unsafe{ self.box_max.next() },
            },
            sphere: BoundingSphereInstance {
                origo: unsafe{ self.origo_iter.next() },
                radius: unsafe{ *self.radius_iter.next() },
            },
        }
    }
}

pub struct SubMeshProjectionInstance {
    object_id: u32,
    primitive_type: PrimitiveType,
    attributes: OptionalVertexAttribute,
    num_deviations: u8,
    num_indices: u32,
    num_vertices: u32,
    num_texture_bytes: u32,
}

pub struct SubMeshProjectionIter<'a> {
    len: usize,
    object_id: ThinSliceIter<'a, u32>,
    primitive_type: ThinSliceIter<'a, PrimitiveType>,
    attributes: ThinSliceIter<'a, OptionalVertexAttribute>,
    num_deviations: ThinSliceIter<'a, u8>,
    num_indices: ThinSliceIter<'a, u32>,
    num_vertices: ThinSliceIter<'a, u32>,
    num_texture_bytes: ThinSliceIter<'a, u32>,
}

impl<'a> Iterator for SubMeshProjectionIter<'a> {
    type Item = SubMeshProjectionInstance;

    // SAFETY: We check len before calling next on the thin slice iterators which all have the
    // same size
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(unsafe{ SubMeshProjectionInstance{
            object_id: *self.object_id.next(),
            primitive_type: *self.primitive_type.next(),
            attributes: *self.attributes.next(),
            num_deviations: *self.num_deviations.next(),
            num_indices: *self.num_indices.next(),
            num_vertices: *self.num_vertices.next(),
            num_texture_bytes: *self.num_texture_bytes.next(),
        }})
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a> ExactSizeIterator for SubMeshProjectionIter<'a> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}


impl<'a> SubMeshProjection<'a> {
    pub unsafe fn range(&self, range: RangeInstance<u32>) -> SubMeshProjection<'a> {
        SubMeshProjection {
            len: range.count,
            object_id: self.object_id.range(range),
            primitive_type: self.primitive_type.range(range),
            attributes: self.attributes.range(range),
            num_deviations: self.num_deviations.range(range),
            num_indices: self.num_indices.range(range),
            num_vertices: self.num_vertices.range(range),
            num_texture_bytes: self.num_texture_bytes.range(range),
        }
    }

    pub fn iter(&self) -> SubMeshProjectionIter<'a> {
        SubMeshProjectionIter {
            len: self.len as usize,
            object_id: self.object_id.iter(),
            primitive_type: self.primitive_type.iter(),
            attributes: self.attributes.iter(),
            num_deviations: self.num_deviations.iter(),
            num_indices: self.num_indices.iter(),
            num_vertices: self.num_vertices.iter(),
            num_texture_bytes: self.num_texture_bytes.iter(),
        }
    }
}

impl<'a> ChildInfo<'a> {
    pub fn iter(&'a self, schema: &'a Schema<'a>) -> ChildInfoIter<'a> {
        ChildInfoIter {
            schema,
            len: self.len as usize,
            hash: self.hash.iter(),
            child_index: self.child_index.iter(),
            child_mask: self.child_mask.iter(),
            tolerance: self.tolerance.iter(),
            total_byte_size: self.total_byte_size.iter(),
            offset: self.offset.iter(),
            scale: self.scale.iter(),
            bounds: self.bounds.iter(),
            sub_meshes: self.sub_meshes.iter(),
        }
    }
}

pub struct ChildInfoInstance<'a> {
    pub hash: &'a [HashBytes],
    pub child_index: u8,
    pub child_mask: u32,
    pub tolerance: i8,
    pub total_byte_size: u32,
    pub offset: Double3Instance,
    pub scale: f32,
    pub bounds: BoundsInstance,
    pub sub_meshes: SubMeshProjection<'a>,
}

pub struct ChildInfoIter<'a> {
    schema: &'a Schema<'a>,
    len: usize,
    hash: HashRangeIter<'a>,
    child_index: ThinSliceIter<'a, u8>,
    child_mask: ThinSliceIter<'a, u32>,
    tolerance: ThinSliceIter<'a, i8>,
    total_byte_size: ThinSliceIter<'a, u32>,
    offset: Double3Iter<'a>,
    scale: ThinSliceIter<'a, f32>,
    bounds: BoundsIter<'a>,
    sub_meshes: SubMeshProjectionRangeIter<'a>,
}

impl<'a> Iterator for ChildInfoIter<'a> {
    type Item = ChildInfoInstance<'a>;

    // SAFETY: We check len before calling next on the thin slice iterators which all have the
    // same size
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;

        let hash_range = unsafe{ self.hash.next() };
        let hash = unsafe{ self.schema.hash_bytes.slice_range(hash_range.0) };

        let sub_meshes_range = unsafe{ self.sub_meshes.next() };
        let sub_meshes = unsafe{ self.schema.sub_mesh_projection.range(sub_meshes_range.0) };

        Some(ChildInfoInstance {
            hash,
            child_index: unsafe{ *self.child_index.next() },
            child_mask: unsafe{ *self.child_mask.next() },
            tolerance: unsafe{ *self.tolerance.next() },
            total_byte_size: unsafe{ *self.total_byte_size.next() },
            offset: unsafe{ self.offset.next() },
            scale: unsafe{ *self.scale.next() },
            bounds: unsafe{ self.bounds.next() },
            sub_meshes,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a> ExactSizeIterator for ChildInfoIter<'a> {
    fn len(&self) -> usize {
        self.len
    }
}


// Types utils

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

fn compute_vertex_offsets(attributes: OptionalVertexAttribute, num_deviations: u8, has_materials: bool, has_object_ids: bool) -> Offsets {
    compute_vertex_offsets_(attributes.iter()
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

fn compute_vertex_attributes_offsets(attributes: OptionalVertexAttribute, num_deviations: u8, has_materials: bool, has_object_ids: bool) -> Offsets {
    compute_vertex_offsets_(attributes.iter()
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

impl<'a> SubMeshProjection<'a> {
    fn aggregate_projections(&self, separate_positions_buffer: bool, filter: impl Fn(u32) -> bool) -> AggregateProjections {
        let mut primitives = 0usize;
        let mut total_texture_bytes = 0usize;
        let mut total_num_indices = 0usize;
        let mut total_num_vertices = 0usize;
        let mut total_num_vertex_bytes = 0usize;
        for SubMeshProjectionInstance {
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
                let num_bytes_per_vertex = if separate_positions_buffer {
                    compute_vertex_position_offsets().stride
                        + compute_vertex_attributes_offsets(
                            attributes,
                            num_deviations,
                            has_materials,
                            has_object_ids
                        ).stride
                }else{
                    compute_vertex_offsets(
                        attributes,
                        num_deviations,
                        has_materials,
                        has_object_ids
                    ).stride
                };
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

pub struct Child {
    pub id: String,
    pub child_index: u8,
    pub child_mask: u32,
    pub tolerance: i8,
    pub byte_size: u32,
    pub offset: Double3Instance,
    pub scale: f32,
    pub bounds: BoundsInstance,
    pub primitives: usize,
    pub primitives_delta: usize,
    pub gpu_bytes: usize,
}

impl<'a> Schema<'a> {
    pub fn children(&self, separate_positions_buffer: bool, filter: impl Fn(u32) -> bool + Copy + 'a) -> impl ExactSizeIterator<Item = Child> + '_ {
        self.child_info.iter(self).map(move |child_info| {
            let id = to_hex(child_info.hash);
            let f32_offset = child_info.offset.into();
            let bounds = BoundsInstance {
                _box: AABBInstance {
                    min: child_info.bounds._box.min + f32_offset,
                    max: child_info.bounds._box.max + f32_offset,
                },
                sphere: BoundingSphereInstance {
                    origo: child_info.bounds.sphere.origo + f32_offset,
                    radius: child_info.bounds.sphere.radius
                }
            };

            let AggregateProjections { primitives, gpu_bytes } = child_info.sub_meshes.aggregate_projections(
                separate_positions_buffer,
                filter
            );
            let parent_primitives = 0; // This gets a value from an empty array in the original
            let primitives_delta = primitives - parent_primitives;
            Child {
                id,
                child_index: child_info.child_index,
                child_mask: child_info.child_mask,
                tolerance: child_info.tolerance,
                byte_size: child_info.total_byte_size,
                offset: child_info.offset,
                scale: child_info.scale,
                bounds,
                primitives,
                primitives_delta,
                gpu_bytes,
            }
        })
    }
}