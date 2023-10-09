use core::mem::{size_of, align_of};

use half::f16;

use crate::thin_slice::ThinSlice;
use crate::range::RangeSlice;
use crate::types_2_0::*;
use crate::reader_2_0::*;

pub struct Optionals<'a> {
    flags: &'a [u8; NUM_OPTIONALS],
    next: usize
}

impl<'a> Optionals<'a> {
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

impl<'a> SubMeshProjectionSlice<'a> {
    fn aggregate_projections(&self, separate_positions_buffer: bool, filter: impl Fn(u32) -> bool) -> AggregateProjections {
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
    pub offset: Double3,
    pub scale: f32,
    pub bounds: Bounds,
    pub primitives: usize,
    pub primitives_delta: usize,
    pub gpu_bytes: usize,
}

// pub struct Highlights<'a> {
//     indices: &'a [u8],
//     // mutex
// }

impl<'a> Schema<'a> {
    pub fn parse(data: &'a [u8]) -> Schema<'a> {
        let mut reader = Reader { data, next: 0 };
        let sizes: &Sizes = reader.read();
        let mut optionals = Optionals{ flags: reader.read_checked(), next: 0 };

        let child_info = reader.read_child_info(sizes.child_info);
        let hash_bytes = reader.read_slice(sizes.hash_bytes);
        let sub_mesh_projection = reader.read_sub_mesh_projection(sizes.sub_mesh_projection);
        let sub_mesh = reader.read_sub_mesh(sizes.sub_mesh);
        let texture_info = reader.read_texture_info(sizes.texture_info);
        let vertex = reader.read_vertex(sizes.vertex, &mut optionals);
        let triangle = reader.read_triangle(sizes.triangle, &mut optionals);
        let vertex_index = optionals.next().then(|| reader.read_slice(sizes.vertex_index));
        let texture_pixels = reader.read_slice(sizes.texture_pixels);

        debug_assert_eq!(reader.next, reader.data.len());

        Schema {
            child_info,
            hash_bytes,
            sub_mesh_projection,
            sub_mesh,
            texture_info,
            vertex,
            triangle,
            vertex_index,
            texture_pixels,
        }
    }

    pub fn children(&self, separate_positions_buffer: bool, filter: impl Fn(u32) -> bool + Copy + 'a) -> impl ExactSizeIterator<Item = Child> + '_ {
        self.child_info.iter(self.hash_bytes, self.sub_mesh_projection.clone()).map(move |child_info| {
            let id = to_hex(child_info.hash);
            let f32_offset = child_info.offset.into();
            let bounds = Bounds {
                _box: AABB {
                    min: child_info.bounds._box.min + f32_offset,
                    max: child_info.bounds._box.max + f32_offset,
                },
                sphere: BoundingSphere {
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

    // fn sub_meshes(&self, filter: impl Fn(u32) -> bool) -> impl Iterator<Item = SubMesh> {
    //     self.sub_mesh.iter()
    //         .filter(|sub_mesh|  filter(sub_mesh.object_id))
    // }

    // pub fn geometry(&self, separate_positions_buffer: bool, enable_outlines: bool, highlights: Highlights, filter: impl Fn(u32) -> bool) {
    //     let (vertex, vertex_index) = (&self.vertex, &self.vertex_index);

    //     let filtered_sub_meshes = self.sub_meshes(filter);
    // }
}