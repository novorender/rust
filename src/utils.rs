use core::{ptr::NonNull, marker::PhantomData, slice, mem::size_of};

use half::f16;

#[derive(Clone, Copy)]
pub struct ThinSlice<'a, T> {
    start: NonNull<T>,
    #[cfg(debug_assertions)]
    len: usize,
    marker: PhantomData<&'a ()>,
}

impl<'a, T> ThinSlice<'a, T> {
    pub fn from_data_and_offset(data: &'a [u8], offset: usize, len: u32) -> ThinSlice<'a, T> {
        let ptr = unsafe{ NonNull::new_unchecked(data[offset..].as_ptr() as *const T as *mut T) };
        ThinSlice {
            start: ptr,
            #[cfg(debug_assertions)]
            len: len as usize,
            marker: PhantomData,
        }
    }

    pub fn empty() -> ThinSlice<'a, T>
    where T: 'a
    {
        Self::from(&[])
    }

    pub unsafe fn as_slice(self, len: u32) -> &'a [T] {
        #[cfg(debug_assertions)]
        debug_assert_eq!(self.len, len as usize);

        NonNull::slice_from_raw_parts(self.start, len as usize).as_ref()
    }

    // pub unsafe fn range(self, range: Range<u32>, i: usize) -> &'a [T] {
    //     let start = *range.start.get(i) as usize;
    //     let count = *range.count.get(i) as usize;

    //     #[cfg(debug_assertions)]
    //     debug_assert!(self.len > start + count);

    //     slice::from_raw_parts(self.start.as_ptr().add(start), count).as_ref()
    // }

    pub unsafe fn slice_range(self, range: std::ops::Range<u32>) -> &'a [T] {
        #[cfg(debug_assertions)]
        debug_assert!(self.len > range.end as usize);

        slice::from_raw_parts(self.start.as_ptr().add(range.start as usize), range.len() as usize).as_ref()
    }

    pub unsafe fn range(self, range: RangeInstance<u32>) -> ThinSlice<'a, T>
    where T: 'a
    {
        #[cfg(debug_assertions)]
        debug_assert!(self.len > (range.start + range.count) as usize);

        self.slice_range(range.into()).into()
    }

    pub unsafe fn get(self, index: usize) -> &'a T {
        #[cfg(debug_assertions)]
        debug_assert!(index < self.len);

        &*self.start.as_ptr().add(index)
    }
}

impl<'a, T> From<&'a [T]> for ThinSlice<'a, T> {
    fn from(value: &'a [T]) -> Self {
        let ptr = value.as_ptr() as *mut T;
        ThinSlice {
            // SAFETY: We are referencing into a slice so it can't be null
            start: unsafe{ NonNull::new_unchecked(ptr) },
            #[cfg(debug_assertions)]
            len: value.len(),
            marker: PhantomData,
        }
    }
}

impl<'a, T> From<&'a [T;0]> for ThinSlice<'a, T> {
    fn from(value: &'a [T;0]) -> Self {
        let ptr = value.as_ptr() as *mut T;
        ThinSlice {
            // SAFETY: We are referencing into a slice so it can't be null
            start: unsafe{ NonNull::new_unchecked(ptr) },
            #[cfg(debug_assertions)]
            len: value.len(),
            marker: PhantomData,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Range<'a, T> {
    pub start: ThinSlice<'a, T>,
    pub count: ThinSlice<'a, T>,
}

#[derive(Clone, Copy)]
pub struct RangeInstance<T> {
    pub start: T,
    pub count: T,
}

impl<'a, T: Copy> Range<'a, T> {
    pub unsafe fn get(&self, index: usize) -> RangeInstance<T> {
        RangeInstance{ start: *self.start.get(index), count: *self.count.get(index)}
    }
}

impl<T: std::ops::Add<Output = T> + Copy> Into<std::ops::Range<T>> for RangeInstance<T> {
    fn into(self) -> std::ops::Range<T> {
        self.start .. self.start + self.count
    }
}

// Types SoA to AoS
use crate::types::*;

#[derive(Clone, Copy)]
pub struct Float3Instance {
    x: f32,
    y: f32,
    z: f32
}

impl<'a> Float3<'a> {
    unsafe fn get(&self, index: usize) -> Float3Instance {
        Float3Instance { x: *self.x.get(index), y: *self.y.get(index), z: *self.z.get(index) }
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
    unsafe fn get(&self, index: usize) -> Double3Instance {
        Double3Instance { x: *self.x.get(index), y: *self.y.get(index), z: *self.z.get(index) }
    }
}

pub struct AABBInstance {
    min: Float3Instance,
    max: Float3Instance,
}

impl<'a> AABB<'a> {
    unsafe fn get(&self, index: usize) -> AABBInstance {
        AABBInstance { min: self.min.get(index), max: self.max.get(index) }
    }
}

struct BoundingSphereInstance {
    origo: Float3Instance,
    radius: f32,
}

impl<'a> BoundingSphere<'a> {
    unsafe fn get(&self, index: usize) -> BoundingSphereInstance {
        BoundingSphereInstance { origo: self.origo.get(index), radius: *self.radius.get(index) }
    }
}

pub struct BoundsInstance {
    _box: AABBInstance,
    sphere: BoundingSphereInstance,
}

impl<'a> Bounds<'a> {
    unsafe fn get(&self, index: usize) -> BoundsInstance {
        BoundsInstance { _box: self._box.get(index), sphere: self.sphere.get(index) }
    }
}

pub struct SubMeshProjectionIter<'a> {
    len: u32,
    object_id: slice::Iter<'a, u32>,
    primitive_type: slice::Iter<'a, PrimitiveType>,
    attributes: slice::Iter<'a, OptionalVertexAttribute>,
    num_deviations: slice::Iter<'a, u8>,
    num_indices: slice::Iter<'a, u32>,
    num_vertices: slice::Iter<'a, u32>,
    num_texture_bytes: slice::Iter<'a, u32>,
}

impl<'a> Iterator for SubMeshProjectionIter<'a> {
    type Item = (u32, PrimitiveType, OptionalVertexAttribute, u8, u32, u32, u32);
    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(unsafe{(
            *self.object_id.next().unwrap_unchecked(),
            *self.primitive_type.next().unwrap_unchecked(),
            *self.attributes.next().unwrap_unchecked(),
            *self.num_deviations.next().unwrap_unchecked(),
            *self.num_indices.next().unwrap_unchecked(),
            *self.num_vertices.next().unwrap_unchecked(),
            *self.num_texture_bytes.next().unwrap_unchecked(),
        )})
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
            len: self.len,
            object_id: unsafe{ self.object_id.as_slice(self.len).iter() },
            primitive_type: unsafe{ self.primitive_type.as_slice(self.len).iter() },
            attributes: unsafe{ self.attributes.as_slice(self.len).iter() },
            num_deviations: unsafe{ self.num_deviations.as_slice(self.len).iter() },
            num_indices: unsafe{ self.num_indices.as_slice(self.len).iter() },
            num_vertices: unsafe{ self.num_vertices.as_slice(self.len).iter() },
            num_texture_bytes: unsafe{ self.num_texture_bytes.as_slice(self.len).iter() }
        }
    }
}


// TODO: Try with iterators per field instead of copies
struct ChildInfoIter<'a> {
    schema: &'a Schema<'a>,
    next: usize,
}

struct ChildInfoInstance<'a> {
    pub hash: &'a [HashBytes],
    pub child_index: u8,
    pub child_mask: u32,
    pub tolerance: i8,
    pub total_byte_size: u32,
    pub offset: Double3Instance,
    pub scale: f32,
    pub bounds: BoundsInstance,
    pub sub_meshes: SubMeshProjection<'a>,
    pub descendant_object_ids: &'a [DescendantObjectIds]
}

impl<'a> Iterator for ChildInfoIter<'a> {
    type Item = ChildInfoInstance<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.next;
        let child_info = &self.schema.child_info;
        if index == child_info.len as usize {
            return None
        }
        self.next += 1;
        let hash = unsafe{ self.schema.hash_bytes.slice_range(child_info.hash.0.get(index).into()) };
        let sub_meshes = unsafe { self.schema.sub_mesh_projection.range(child_info.sub_meshes.0.get(index).into()) };
        let descendant_object_ids = unsafe { self.schema.descendant_object_ids.slice_range(child_info.sub_meshes.0.get(index).into()) };
        Some(ChildInfoInstance {
            hash,
            child_index: unsafe{ *child_info.child_index.get(index) },
            child_mask: unsafe{ *child_info.child_mask.get(index) },
            tolerance: unsafe{ *child_info.tolerance.get(index) },
            total_byte_size: unsafe{ *child_info.total_byte_size.get(index) },
            offset: unsafe{ child_info.offset.get(index) },
            scale: unsafe{ *child_info.scale.get(index) },
            bounds: unsafe{ child_info.bounds.get(index) },
            sub_meshes,
            descendant_object_ids,
        })
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
        for (
            object_id,
            primitive_type,
            attributes,
            num_deviations,
            num_indices,
            num_vertices,
            num_texture_bytes
        ) in self.iter()
        {
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

pub struct Child<'a> {
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
    pub descendant_object_ids: &'a [u32]
}

impl<'a> Schema<'a> {
    pub fn children(&self, separate_positions_buffer: bool, filter: impl Fn(u32) -> bool + Copy) -> impl Iterator<Item = Child> {
        ChildInfoIter {
            schema: self,
            next: 0,
        }.map(move |child_info| {
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
                descendant_object_ids: child_info.descendant_object_ids,
            }
        })
    }
}