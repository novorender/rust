
use core::mem::{size_of, align_of};

use half::f16;

use crate::thin_slice::ThinSlice;
use crate::range::RangeSlice;
use crate::ktx::*;
use wasm_bindgen::prelude::*;

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

#[wasm_bindgen]
#[derive(Default)]
pub struct Highlights {
    pub(crate) indices: Vec<u8>,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct MeshObjectRange {
    #[wasm_bindgen(js_name = objectId)]
    pub object_id: u32,
    #[wasm_bindgen(js_name = beginVertex)]
    pub begin_vertex: usize,
    #[wasm_bindgen(js_name = endVertex)]
    pub end_vertex: usize,
    #[wasm_bindgen(js_name = beginTriangle)]
    pub begin_triangle: usize,
    #[wasm_bindgen(js_name = endTriangle)]
    pub end_triangle: usize,
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct DrawRange {
    #[wasm_bindgen(js_name = childIndex)]
    pub child_index: u8,
    #[wasm_bindgen(js_name = byteOffset)]
    pub byte_offset: usize,
    pub first: usize,
    pub count: usize,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Array<DrawRange>")]
    pub type ArrayDrawRange;
    #[wasm_bindgen(typescript_type = "Array<MeshObjectRange>")]
    pub type ArrayObjectRange;
    #[wasm_bindgen(typescript_type = "Array<Uint8Array>")]
    pub type ArrayUint8Array;
    #[wasm_bindgen(typescript_type = r#""POINTS" | "LINES" | "LINE_LOOP" | "LINE_STRIP" | "TRIANGLES" | "TRIANGLE_STRIP" | "TRIANGLE_FAN""#)]
    pub type PrimitiveTypeStr;
    #[wasm_bindgen(typescript_type = "ShaderAttributeType")]
    pub type ShaderAttributeType;
    #[wasm_bindgen(typescript_type = "ComponentType")]
    pub type ComponentType;
    #[wasm_bindgen(typescript_type = "1 | 2 | 3 | 4")]
    pub type ComponentCount;
    #[wasm_bindgen(typescript_type = "readonly number[]")]
    pub type Float3x3AsArray;
    #[wasm_bindgen(typescript_type = "TextureParams")]
    pub type TextureParams;
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct VertexAttribute {
    #[wasm_bindgen(skip)]
    pub kind: &'static str,
    pub buffer: i8,
    component_count: u8,
    #[wasm_bindgen(skip)]
    pub component_type: &'static str,
    pub normalized: bool,
    #[wasm_bindgen(js_name = byteOffset)]
    pub byte_offset: u32,
    #[wasm_bindgen(js_name = byteStride)]
    pub byte_stride: u32,
}

#[wasm_bindgen]
impl VertexAttribute {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> ShaderAttributeType {
        let js_value: JsValue = self.kind.into();
        js_value.into()
    }

    #[wasm_bindgen(js_name = componentType)]
    #[wasm_bindgen(getter)]
    pub fn component_type(&self) -> ComponentType {
        let js_value: JsValue = self.component_type.into();
        js_value.into()
    }

    #[wasm_bindgen(js_name = componentCount)]
    #[wasm_bindgen(getter)]
    pub fn component_count(&self) -> ComponentCount {
        let js_value: JsValue = self.component_count.into();
        js_value.into()
    }
}

#[wasm_bindgen]
#[derive(Clone)]
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
    tri_pos: Option<Vec<u8>>,
    tri_id: Option<Vec<u8>>,
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

pub enum Indices {
    IndexBuffer(Vec<u8>),
    NumIndices(u32),
}

// substitute with macro_rules! ... to get proper autcomplete on the implementation
// use crate::types_2_0::*;
// use crate::reader_2_0::*;
// use _2_0::*;

macro_rules! impl_parser {
    ($version: ident, $child_ty: ty $(, $child_extra_fields: ident)*) => { paste::paste! {
        use crate::[<types $version>]::*;
        use crate::[<reader $version>]::*;
        use crate::parser::*;
        use crate::thin_slice::ThinSliceIterator;
        use crate::interleaved::*;

        use hashbrown::{HashMap, hash_map::Entry};
        use js_sys::Array;
        use wasm_bindgen::JsValue;

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
                        total_num_vertex_bytes += num_vertices as usize * num_bytes_per_vertex as usize;
                        total_texture_bytes += num_texture_bytes as usize;
                    }
                }

                let idx_stride = if total_num_vertices < u16::MAX as usize { 2 } else { 4 };
                let gpu_bytes = total_texture_bytes + total_num_vertex_bytes + total_num_indices * idx_stride;
                AggregateProjections {
                    primitives,
                    gpu_bytes
                }
            }
        }

        impl<'a> ChildInfo<'a> {
            pub fn to_child(&self, filter: impl Fn(u32) -> bool) -> Child {
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
                        // $child_extra_fields: unsafe{ Uint32Array::view(self.$child_extra_fields) },
                        $child_extra_fields: self.$child_extra_fields.into(),
                    )*
                }
            }
        }

        impl<'a> SubMesh<'a> {
            fn interleave_attribute(&self, dst: &mut [u8], attribute: Attribute, num_deviations: u8, byte_offset: usize, byte_stride: usize) {
                // SAFETY: all the slices have len self.len so calling as_slice(self.len) is safe
                match attribute {
                    Attribute::Position => interleave_three::<i16>(
                        bytemuck::cast_slice_mut(dst),
                        unsafe{ self.vertices.position.x.as_slice(self.vertices.len) },
                        unsafe{ self.vertices.position.y.as_slice(self.vertices.len) },
                        unsafe{ self.vertices.position.z.as_slice(self.vertices.len) },
                        byte_offset,
                        byte_stride,
                    ),
                    Attribute::Normal => if let Some(normal) = &self.vertices.normal {
                        interleave_three(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ normal.x.as_slice(self.vertices.len) },
                            unsafe{ normal.y.as_slice(self.vertices.len) },
                            unsafe{ normal.z.as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        )
                    },
                    Attribute::Color => if let Some(color) = &self.vertices.color {
                        interleave_four(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ color.red.as_slice(self.vertices.len) },
                            unsafe{ color.green.as_slice(self.vertices.len) },
                            unsafe{ color.blue.as_slice(self.vertices.len) },
                            unsafe{ color.alpha.as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        )
                    },
                    Attribute::TexCoord => if let Some(tex_coord) = &self.vertices.tex_coord {
                        interleave_two(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ tex_coord.x.as_slice(self.vertices.len) },
                            unsafe{ tex_coord.y.as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        )
                    },
                    Attribute::ProjectedPos => if let Some(projected_pos) = &self.vertices.projected_pos {
                        interleave_three(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ projected_pos.x.as_slice(self.vertices.len) },
                            unsafe{ projected_pos.y.as_slice(self.vertices.len) },
                            unsafe{ projected_pos.z.as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        )
                    },
                    Attribute::MaterialIndex => fill_to_interleaved_array(
                        bytemuck::cast_slice_mut(dst),
                        self.material_index,
                        byte_offset,
                        byte_stride,
                        0,
                        self.vertices.len as usize,
                    ),
                    Attribute::ObjectId => fill_to_interleaved_array(
                        bytemuck::cast_slice_mut(dst),
                        self.object_id,
                        byte_offset,
                        byte_stride,
                        0,
                        self.vertices.len as usize,
                    ),
                    Attribute::Deviations => match num_deviations {
                        1 => interleave_one(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ self.vertices.deviations.a.unwrap().as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        ),
                        2 => interleave_two(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ self.vertices.deviations.a.unwrap().as_slice(self.vertices.len) },
                            unsafe{ self.vertices.deviations.b.unwrap().as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        ),
                        3 => interleave_three(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ self.vertices.deviations.a.unwrap().as_slice(self.vertices.len) },
                            unsafe{ self.vertices.deviations.b.unwrap().as_slice(self.vertices.len) },
                            unsafe{ self.vertices.deviations.c.unwrap().as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        ),
                        4 => interleave_four(
                            bytemuck::cast_slice_mut(dst),
                            unsafe{ self.vertices.deviations.a.unwrap().as_slice(self.vertices.len) },
                            unsafe{ self.vertices.deviations.b.unwrap().as_slice(self.vertices.len) },
                            unsafe{ self.vertices.deviations.c.unwrap().as_slice(self.vertices.len) },
                            unsafe{ self.vertices.deviations.d.unwrap().as_slice(self.vertices.len) },
                            byte_offset,
                            byte_stride,
                        ),
                        _ => (),
                    }
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

        #[wasm_bindgen(js_name = [<ReturnSubMesh $version>])]
        pub struct ReturnSubMesh {
            #[wasm_bindgen(js_name = "materialType")]
            pub material_type: MaterialType,
            #[wasm_bindgen(skip)]
            pub primitive_type: PrimitiveType,
            #[wasm_bindgen(js_name = "numVertices")]
            pub num_vertices: u32,
            #[wasm_bindgen(js_name = "numTriangles")]
            pub num_triangles: u32,
            #[wasm_bindgen(skip)]
            pub object_ranges: Option<Vec<MeshObjectRange>>,
            #[wasm_bindgen(skip)]
            pub vertex_attributes: Option<VertexAttributes>,
            #[wasm_bindgen(skip)]
            pub vertex_buffers: Vec<Vec<u8>>,
            #[wasm_bindgen(skip)]
            pub indices: Indices,
            #[wasm_bindgen(js_name = "baseColorTexture")]
            pub base_color_texture: Option<u32>,
            #[wasm_bindgen(skip)]
            pub draw_ranges: Option<Vec<DrawRange>>,
        }

        #[wasm_bindgen(js_class = [<ReturnSubMesh $version>])]
        impl ReturnSubMesh {
            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "primitiveType")]
            pub fn primitive_type(&self) -> PrimitiveTypeStr {
                let js_value: JsValue = format!("{:?}", self.primitive_type)
                    .to_ascii_uppercase()
                    .into();
                js_value.into()
            }

            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "objectRanges")]
            pub fn object_ranges(&mut self) -> ArrayObjectRange {
                let array: Array = self.object_ranges.take().unwrap()
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                let js_value: JsValue = array.into();
                js_value.into()
            }

            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "vertexAttributes")]
            pub fn vertex_attributes(&mut self) -> VertexAttributes {
                self.vertex_attributes.take().unwrap()
            }

            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "vertexBuffers")]
            pub fn vertex_buffers(&mut self) -> ArrayUint8Array {
                let array: Array = self.vertex_buffers.iter()
                    .map(|buffer| {
                        let typed_array = js_sys::Uint8Array::new_with_length(buffer.len() as u32);
                        typed_array.copy_from(buffer);
                        typed_array.buffer()
                    })
                    .collect();
                let js_value: JsValue = array.into();
                js_value.into()
            }

            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "drawRanges")]
            pub fn draw_ranges(&mut self) -> ArrayDrawRange {
                let array: Array = self.draw_ranges.take().unwrap()
                    .into_iter()
                    .map(JsValue::from)
                    .collect();
                let js_value: JsValue = array.into();
                js_value.into()
            }

            #[wasm_bindgen(getter)]
            #[wasm_bindgen(js_name = "indices")]
            pub fn indices(&self) -> JsValue {
                match &self.indices {
                    Indices::IndexBuffer(buffer) => if self.num_vertices > u16::MAX as u32 {
                        let buffer: &[u32] = bytemuck::cast_slice(buffer);
                        let typed_array = js_sys::Uint32Array::new_with_length(buffer.len() as u32);
                        typed_array.copy_from(buffer);
                        typed_array.into()
                    }else{
                        let buffer: &[u16] = bytemuck::cast_slice(buffer);
                        let typed_array = js_sys::Uint16Array::new_with_length(buffer.len() as u32);
                        typed_array.copy_from(buffer);
                        typed_array.into()
                    },
                    Indices::NumIndices(num_indices) => (*num_indices).into(),
                }
            }
        }

        #[wasm_bindgen(js_name = [<Texture $version>])]
        #[derive(Clone)]
        pub struct Texture {
            pub semantic: TextureSemantic,
            pub transform: Float3x3,
            params: TextureParameters,
        }

        #[wasm_bindgen(js_class = [<Texture $version>])]
        impl Texture {
            #[wasm_bindgen(getter)]
            pub fn params(&self) -> TextureParams {
                // TODO: Is this correct? also try to avoid clone
                let js_value: JsValue = self.params.clone().into();
                js_value.into()
            }
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

            pub fn geometry(&self, enable_outlines: bool, highlights: &Highlights, filter: impl Fn(u32) -> bool) -> (Vec<ReturnSubMesh>, Vec<Option<Texture>>){
                let mut sub_meshes = Vec::with_capacity(self.sub_mesh.len as usize);
                let mut referenced_textures = HashMap::new();

                struct Group<'a> {
                    material_type: MaterialType,
                    primitive_type: PrimitiveType,
                    attributes: OptionalVertexAttribute,
                    num_deviations: u8,
                    group_meshes: Vec<SubMesh<'a>>,
                    has_materials: bool,
                    has_object_ids: bool,
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
                        material_index,
                        object_id,
                        ..
                    } = sub_mesh;

                    let group = groups.entry(Key{
                        material_type,
                        primitive_type,
                        attributes,
                        num_deviations,
                        child_index,
                    }).or_insert_with(|| Group {
                        material_type,
                        primitive_type,
                        attributes,
                        num_deviations,
                        has_materials: false,
                        has_object_ids: false,
                        group_meshes: Vec::with_capacity(self.sub_mesh.len as usize)
                    });
                    group.has_materials |= material_index != u8::MAX;
                    group.has_object_ids |= object_id != u32::MAX;
                    group.group_meshes.push(sub_mesh);
                }

                for Group {
                    material_type,
                    primitive_type,
                    attributes,
                    num_deviations,
                    group_meshes,
                    has_materials,
                    has_object_ids,
                } in groups.values() {
                    if group_meshes.is_empty() {
                        continue
                    }

                    let position_stride = compute_vertex_position_deviations_offsets(*num_deviations).stride as usize;
                    let triangle_pos_stride = position_stride * 3;
                    let attrib_offsets = compute_vertex_attributes_offsets(
                        *attributes,
                        *num_deviations,
                        *has_materials,
                        *has_object_ids
                    );
                    let vertex_stride = attrib_offsets.stride as usize;

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

                    let mut triangle_pos_buffer: Option<Vec<u8>>;
                    let mut triangle_object_id_buffer: Option<Vec<u8>>;
                    let mut highlight_buffer_tri;
                    if enable_outlines && *primitive_type == PrimitiveType::Triangles {
                        let mut buffer = Vec::with_capacity(num_triangles * triangle_pos_stride * size_of::<i16>());
                        unsafe{ buffer.set_len(num_triangles * triangle_pos_stride * size_of::<i16>()) };
                        triangle_pos_buffer = Some(buffer);

                        let mut buffer = Vec::with_capacity(num_triangles * size_of::<u32>());
                        unsafe{ buffer.set_len(num_triangles * size_of::<u32>()) };
                        triangle_object_id_buffer = Some(buffer);

                        let mut buffer = Vec::with_capacity(num_triangles);
                        unsafe{ buffer.set_len(num_triangles) };
                        highlight_buffer_tri = Some(buffer);
                    }else{
                        triangle_pos_buffer = None;
                        triangle_object_id_buffer = None;
                        highlight_buffer_tri = None;
                    }

                    let mut position_buffer = Vec::with_capacity(num_vertices * position_stride);
                    unsafe{ position_buffer.set_len(num_vertices * position_stride) };

                    let index_buffer_bytes_per_element;
                    let mut index_buffer;
                     if num_indices != 0 {
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
                    let mut object_ranges = Vec::with_capacity(group_meshes.len());
                    let mut draw_ranges = Vec::with_capacity(group_meshes.len());

                    let draw_range_begin = if index_buffer.is_some() {
                        index_offset
                    } else {
                        vertex_offset
                    };

                    for sub_mesh in group_meshes.iter() {
                        for attrib in attributes.iter()
                            .map(|attribute| attribute.into())
                            .chain((*num_deviations > 0).then_some(Attribute::Deviations))
                            .chain(has_materials.then_some(Attribute::MaterialIndex))
                            .chain(has_object_ids.then_some(Attribute::ObjectId))
                        {
                            let dst = &mut vertex_buffer[vertex_offset * vertex_stride ..];
                            let offset = attrib_offsets[attrib] as usize;
                            sub_mesh.interleave_attribute(
                                dst,
                                attrib,
                                *num_deviations,
                                offset,
                                vertex_stride
                            )
                        }

                        let mut num_triangles_in_submesh = 0;
                        if let (Some(triangle_pos_buffer), Some(triangle_object_id_buffer))
                            = (&mut triangle_pos_buffer, &mut triangle_object_id_buffer)
                        {
                            let triangle_pos_buffer = &mut bytemuck::cast_slice_mut(triangle_pos_buffer)[triangle_offset ..];
                            if index_buffer.is_some() {
                                num_triangles_in_submesh = sub_mesh.indices.len() / 3;
                                let (x, y ,z) = (
                                    sub_mesh.vertices.position.x,
                                    sub_mesh.vertices.position.y,
                                    sub_mesh.vertices.position.z
                                );
                                for (index, triangle) in sub_mesh.indices.iter()
                                    .zip(triangle_pos_buffer.chunks_mut(3))
                                {
                                    let index = *index as usize;
                                    // TODO: Add support for triangle strips and fans as well...
                                    triangle[0] = unsafe{ *x.get_unchecked(index) };
                                    triangle[1] = unsafe{ *y.get_unchecked(index) };
                                    triangle[2] = unsafe{ *z.get_unchecked(index) };
                                }
                            }else{
                                let mut position = sub_mesh.vertices.position.thin_iter();
                                num_triangles_in_submesh = sub_mesh.vertices.len as usize / 3;
                                for triangle in triangle_pos_buffer.chunks_mut(3)
                                    .take(sub_mesh.vertices.len as usize)
                                {
                                    let pos = unsafe{ position.next() };
                                    triangle[0] = pos.x;
                                    triangle[1] = pos.y;
                                    triangle[2] = pos.z;
                                }
                            }
                            bytemuck::cast_slice_mut(triangle_object_id_buffer)[triangle_offset .. triangle_offset + num_triangles_in_submesh]
                                .fill(sub_mesh.object_id);
                        }

                        sub_mesh.interleave_attribute(
                            &mut position_buffer,
                            Attribute::Position,
                            *num_deviations,
                            vertex_offset * position_stride,
                            position_stride,
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
                        child_index: group_meshes[0].child_index,
                        byte_offset,
                        first: draw_range_begin,
                        count
                    });

                    fn enumerate_buffers(possible_buffers: PossibleBuffers) -> (Vec<Vec<u8>>, BufIndex) {
                        let mut buffers = Vec::with_capacity(6);
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
                                    buffers.push(tri_pos);
                                    id as i8
                                }else{
                                    -1
                                }
                            },
                            tri_id: {
                                if let Some(tri_id) = possible_buffers.tri_id {
                                    let id = buffers.len();
                                    buffers.push(tri_id);
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
                        let texture = unsafe{
                            first_group_mesh.textures
                                .thin_iter(self.texture_pixels)
                                .next()
                        };
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
                        "FLOAT"
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

                    object_ranges.sort_unstable_by_key(|obj_range| obj_range.object_id);

                    sub_meshes.push(ReturnSubMesh{
                        material_type: *material_type,
                        primitive_type: *primitive_type,
                        num_vertices: num_vertices as u32,
                        num_triangles: num_triangles as u32,
                        object_ranges: Some(object_ranges),
                        vertex_attributes: Some(vertex_attributes),
                        vertex_buffers,
                        indices,
                        base_color_texture,
                        draw_ranges: Some(draw_ranges),
                    })
                }

                let mut textures = vec![None; self.texture_info.len as usize];
                for (index, reference) in referenced_textures {
                    let semantic = reference.semantic;
                    let transform = reference.transform;
                    let ktx = reference.pixel_range;
                    let params = parse_ktx(ktx).unwrap().texture_parameters();
                    textures[index as usize] = Some(Texture {
                        semantic,
                        transform,
                        params,
                    });
                }

                (sub_meshes, textures)
            }
        }
    }};
}

pub mod _2_0 {
    use crate::types_2_0;

    impl_parser!(_2_0, Child);

    #[wasm_bindgen(js_name = Child_2_0)]
    pub struct Child {
        pub(crate) id: String,
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

    #[wasm_bindgen(js_class = Child_2_0)]
    impl Child {
        pub fn id(&self) -> String {
            self.id.clone()
        }
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
    #[cfg(target_family = "wasm")]
    use js_sys::Uint32Array;

    use crate::types_2_1;

    impl_parser!(_2_1, Child, descendant_object_ids);

    #[wasm_bindgen(js_name = Child_2_1)]
    pub struct Child {
        id: String,
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
        #[cfg(target_family = "wasm")]
        descendant_object_ids: Uint32Array,
        #[cfg(not(target_family = "wasm"))]
        descendant_object_ids: Vec<u32>,
    }

    #[wasm_bindgen(js_class = Child_2_1)]
    impl Child {
        pub fn id(&self) -> String {
            self.id.clone()
        }

        #[cfg(target_family = "wasm")]
        pub fn descendant_object_ids(&self) -> JsValue {
            // self.descendant_object_ids.subarray(0, self.descendant_object_ids.byte_length() / size_of::<u32>() as u32)
            if self.descendant_object_ids.byte_length() > 0 {
                self.descendant_object_ids.clone().into()
            }else{
                JsValue::NULL
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    impl Child {
        pub fn descendant_object_ids(&self) -> &[u32] {
            &self.descendant_object_ids
        }
    }

    impl<'a> types_2_1::Schema<'a> {
        pub fn children(&self, filter: impl Fn(u32) -> bool + Copy + 'a) -> impl ExactSizeIterator<Item = Child> + '_ {
            self.child_info.iter(
                self.hash_bytes,
                self.sub_mesh_projection.clone(),
                self.descendant_object_ids
            ).map(move |child_info| child_info.to_child(filter))
        }
    }
}