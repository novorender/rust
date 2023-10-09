use crate::types_2_1::*;
use crate::parser_2_1::{Reader, Optionals};

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub struct Sizes {
    pub child_info: u32,
    pub hash_bytes: u32,
    pub descendant_object_ids: u32,
    pub sub_mesh_projection: u32,
    pub sub_mesh: u32,
    pub texture_info: u32,
    pub vertex: u32,
    pub triangle: u32,
    pub vertex_index: u32,
    pub texture_pixels: u32,
}

pub const NUM_OPTIONALS: usize = 10;

impl<'a> Reader<'a> {
    pub fn read_double3(&mut self, len: u32) -> Double3Slice<'a> {
        Double3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    pub fn read_float3(&mut self, len: u32) -> Float3Slice<'a> {
        Float3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    pub fn read_float3x3(&mut self, len: u32) -> Float3x3Slice<'a> {
        Float3x3Slice {
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

    pub fn read_int16_3(&mut self, len: u32) -> Int16_3Slice<'a> {
        Int16_3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    pub fn read_int8_3(&mut self, len: u32) -> Int8_3Slice<'a> {
        Int8_3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    pub fn read_rgba_u8(&mut self, len: u32) -> RgbaU8Slice<'a> {
        RgbaU8Slice {
            red: self.read_slice(len),
            green: self.read_slice(len),
            blue: self.read_slice(len),
            alpha: self.read_slice(len),
        }
    }

    pub fn read_half2(&mut self, len: u32) -> Half2Slice<'a> {
        Half2Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
        }
    }

    pub fn read_aabb(&mut self, len: u32) -> AABBSlice<'a> {
        AABBSlice { min: self.read_float3(len), max: self.read_float3(len) }
    }

    pub fn read_bounding_sphere(&mut self, len: u32) -> BoundingSphereSlice<'a> {
        BoundingSphereSlice { origo: self.read_float3(len), radius: self.read_slice(len) }
    }

    pub fn read_bounds(&mut self, len: u32) -> BoundsSlice<'a> {
        BoundsSlice {
            _box: self.read_aabb(len),
            sphere: self.read_bounding_sphere(len),
        }
    }

    pub fn read_child_info(&mut self, len: u32) -> ChildInfoSlice<'a> {
        ChildInfoSlice {
            len,
            hash: HashRangeSlice(self.range(len)),
            child_index: self.read_slice(len),
            child_mask: self.read_slice(len),
            tolerance: self.read_slice(len),
            total_byte_size: self.read_slice(len),
            offset: self.read_double3(len),
            scale: self.read_slice(len),
            bounds: self.read_bounds(len),
            sub_meshes: SubMeshProjectionRangeSlice(self.range(len)),
            descendant_object_ids: DescendantObjectIdsRangeSlice(self.range(len)),

        }
    }

    pub fn read_sub_mesh_projection(&mut self, len: u32) -> SubMeshProjectionSlice<'a> {
        SubMeshProjectionSlice {
            len,
            object_id: self.read_slice(len),
            primitive_type: self.read_checked_slice(len),
            attributes: self.read_slice(len),
            num_deviations: self.read_slice(len),
            num_indices: self.read_slice(len),
            num_vertices: self.read_slice(len),
            num_texture_bytes: self.read_slice(len),
        }
    }

    pub fn read_sub_mesh(&mut self, len: u32) -> SubMeshSlice<'a> {
        SubMeshSlice {
            len,
            child_index: self.read_slice(len),
            object_id: self.read_slice(len),
            material_index: self.read_slice(len),
            primitive_type: self.read_checked_slice(len),
            material_type: self.read_checked_slice(len),
            attributes: self.read_slice(len),
            num_deviations: self.read_slice(len),
            vertices: VertexRangeSlice(self.range(len)),
            primitive_vertex_indices: VertexIndexRangeSlice(self.range(len)),
            edge_vertex_indices: VertexIndexRangeSlice(self.range(len)),
            corner_vertex_indices: VertexIndexRangeSlice(self.range(len)),
            textures: TextureInfoRangeSlice(self.range(len)),
        }
    }

    pub fn read_texture_info(&mut self, len: u32) -> TextureInfoSlice<'a>  {
        TextureInfoSlice {
            len,
            semantic: self.read_checked_slice(len),
            transform: self.read_float3x3(len),
            pixel_range: PixelRangeSlice(self.range(len)),
        }
    }

    pub fn read_deviations(&mut self, len: u32, optionals: &mut Optionals) -> DeviationsSlice<'a> {
        DeviationsSlice {
            len,
            a: optionals.next().then(|| self.read_slice(len)),
            c: optionals.next().then(|| self.read_slice(len)),
            d: optionals.next().then(|| self.read_slice(len)),
            b: optionals.next().then(|| self.read_slice(len)),
        }
    }

    pub fn read_vertex(&mut self, len: u32, optionals: &mut Optionals) -> VertexSlice<'a> {
        VertexSlice {
            len,
            position: self.read_int16_3(len),
            normal: optionals.next().then(|| self.read_int8_3(len)),
            color: optionals.next().then(|| self.read_rgba_u8(len)),
            tex_coord: optionals.next().then(|| self.read_half2(len)),
            projected_pos: optionals.next().then(|| self.read_int16_3(len)),
            deviations: self.read_deviations(len, optionals),
        }
    }

    pub fn read_triangle(&mut self, len: u32, optionals: &mut Optionals) -> TriangleSlice<'a> {
        TriangleSlice { len, topology_flags: optionals.next().then(|| self.read_slice(len)) }
    }
}
