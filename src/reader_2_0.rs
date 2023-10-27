use crate::types_2_0::*;
use crate::parser::{Reader, Optionals};


#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
#[repr(C)]
pub struct Sizes {
    pub child_info: u32,
    pub hash_bytes: u32,
    pub sub_mesh_projection: u32,
    pub sub_mesh: u32,
    pub texture_info: u32,
    pub vertex: u32,
    pub triangle: u32,
    pub vertex_index: u32,
    pub texture_pixels: u32,
}

const NUM_OPTIONALS: usize = 10;

pub struct Parser<'a> {
    reader: Reader<'a>,
    sizes: &'a Sizes,
    optionals: Optionals<'a, NUM_OPTIONALS>,
}

impl<'a> Parser<'a> {
    pub fn new(mut reader: Reader<'a>) -> Parser<'a> {
        let sizes: &Sizes = reader.read();
        let optionals = Optionals::new(reader.read_checked());
        Parser { reader, sizes, optionals }
    }

    pub fn read_double3(&mut self, len: u32) -> Double3Slice<'a> {
        Double3Slice {
            x: self.reader.read_slice(len),
            y: self.reader.read_slice(len),
            z: self.reader.read_slice(len),
        }
    }

    pub fn read_float3(&mut self, len: u32) -> Float3Slice<'a> {
        Float3Slice {
            x: self.reader.read_slice(len),
            y: self.reader.read_slice(len),
            z: self.reader.read_slice(len),
        }
    }

    pub fn read_float3x3(&mut self, len: u32) -> Float3x3Slice<'a> {
        Float3x3Slice {
            e00: self.reader.read_slice(len),
            e01: self.reader.read_slice(len),
            e02: self.reader.read_slice(len),
            e10: self.reader.read_slice(len),
            e11: self.reader.read_slice(len),
            e12: self.reader.read_slice(len),
            e20: self.reader.read_slice(len),
            e21: self.reader.read_slice(len),
            e22: self.reader.read_slice(len),
        }
    }

    pub fn read_int16_3(&mut self, len: u32) -> Int16_3Slice<'a> {
        Int16_3Slice {
            x: self.reader.read_slice(len),
            y: self.reader.read_slice(len),
            z: self.reader.read_slice(len),
        }
    }

    pub fn read_int8_3(&mut self, len: u32) -> Int8_3Slice<'a> {
        Int8_3Slice {
            x: self.reader.read_slice(len),
            y: self.reader.read_slice(len),
            z: self.reader.read_slice(len),
        }
    }

    pub fn read_rgba_u8(&mut self, len: u32) -> RgbaU8Slice<'a> {
        RgbaU8Slice {
            red: self.reader.read_slice(len),
            green: self.reader.read_slice(len),
            blue: self.reader.read_slice(len),
            alpha: self.reader.read_slice(len),
        }
    }

    pub fn read_half2(&mut self, len: u32) -> Half2Slice<'a> {
        Half2Slice {
            x: self.reader.read_slice(len),
            y: self.reader.read_slice(len),
        }
    }

    pub fn read_aabb(&mut self, len: u32) -> AABBSlice<'a> {
        AABBSlice { min: self.read_float3(len), max: self.read_float3(len) }
    }

    pub fn read_bounding_sphere(&mut self, len: u32) -> BoundingSphereSlice<'a> {
        BoundingSphereSlice { origo: self.read_float3(len), radius: self.reader.read_slice(len) }
    }

    pub fn read_bounds(&mut self, len: u32) -> BoundsSlice<'a> {
        BoundsSlice {
            box_: self.read_aabb(len),
            sphere: self.read_bounding_sphere(len),
        }
    }

    pub fn read_child_info(&mut self, len: u32) -> ChildInfoSlice<'a> {
        ChildInfoSlice {
            len,
            hash: HashRangeSlice(self.reader.read_range(len)),
            child_index: self.reader.read_slice(len),
            child_mask: self.reader.read_slice(len),
            tolerance: self.reader.read_slice(len),
            total_byte_size: self.reader.read_slice(len),
            offset: self.read_double3(len),
            scale: self.reader.read_slice(len),
            bounds: self.read_bounds(len),
            sub_meshes: SubMeshProjectionRangeSlice(self.reader.read_range(len)),
        }
    }

    pub fn read_sub_mesh_projection(&mut self, len: u32) -> SubMeshProjectionSlice<'a> {
        SubMeshProjectionSlice {
            len,
            object_id: self.reader.read_slice(len),
            primitive_type: self.reader.read_checked_slice(len),
            attributes: self.reader.read_slice(len),
            num_deviations: self.reader.read_slice(len),
            num_indices: self.reader.read_slice(len),
            num_vertices: self.reader.read_slice(len),
            num_texture_bytes: self.reader.read_slice(len),
        }
    }

    pub fn read_sub_mesh(&mut self, len: u32) -> SubMeshSlice<'a> {
        SubMeshSlice {
            len,
            child_index: self.reader.read_slice(len),
            object_id: self.reader.read_slice(len),
            material_index: self.reader.read_slice(len),
            primitive_type: self.reader.read_checked_slice(len),
            material_type: self.reader.read_checked_slice(len),
            attributes: self.reader.read_slice(len),
            num_deviations: self.reader.read_slice(len),
            vertices: VertexRangeSlice(self.reader.read_range(len)),
            primitive_vertex_indices: VertexIndexRangeSlice(self.reader.read_range(len)),
            edge_vertex_indices: VertexIndexRangeSlice(self.reader.read_range(len)),
            corner_vertex_indices: VertexIndexRangeSlice(self.reader.read_range(len)),
            textures: TextureInfoRangeSlice(self.reader.read_range(len)),
        }
    }

    pub fn read_texture_info(&mut self, len: u32) -> TextureInfoSlice<'a>  {
        TextureInfoSlice {
            len,
            semantic: self.reader.read_checked_slice(len),
            transform: self.read_float3x3(len),
            pixel_range: PixelRangeSlice(self.reader.read_range(len)),
        }
    }

    pub fn read_deviations(&mut self, len: u32) -> DeviationsSlice<'a> {
        DeviationsSlice {
            len,
            a: self.optionals.next().then(|| self.reader.read_slice(len)),
            c: self.optionals.next().then(|| self.reader.read_slice(len)),
            d: self.optionals.next().then(|| self.reader.read_slice(len)),
            b: self.optionals.next().then(|| self.reader.read_slice(len)),
        }
    }

    pub fn read_vertex(&mut self, len: u32) -> VertexSlice<'a> {
        VertexSlice {
            len,
            position: self.read_int16_3(len),
            normal: self.optionals.next().then(|| self.read_int8_3(len)),
            color: self.optionals.next().then(|| self.read_rgba_u8(len)),
            tex_coord: self.optionals.next().then(|| self.read_half2(len)),
            projected_pos: self.optionals.next().then(|| self.read_int16_3(len)),
            deviations: self.read_deviations(len),
        }
    }

    pub fn read_triangle(&mut self, len: u32) -> TriangleSlice<'a> {
        TriangleSlice { len, topology_flags: self.optionals.next().then(|| self.reader.read_slice(len)) }
    }

    pub fn read_schema(&mut self) -> Schema<'a> {
        let child_info = self.read_child_info(self.sizes.child_info);
        let hash_bytes = self.reader.read_slice(self.sizes.hash_bytes);
        let sub_mesh_projection = self.read_sub_mesh_projection(self.sizes.sub_mesh_projection);
        let sub_mesh = self.read_sub_mesh(self.sizes.sub_mesh);
        let texture_info = self.read_texture_info(self.sizes.texture_info);
        let vertex = self.read_vertex(self.sizes.vertex);
        let triangle = self.read_triangle(self.sizes.triangle);
        let vertex_index = self.optionals.next().then(|| self.reader.read_slice(self.sizes.vertex_index));
        let texture_pixels = self.reader.read_slice(self.sizes.texture_pixels);

        debug_assert_eq!(self.reader.next, self.reader.data.len());

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
}