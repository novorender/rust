use core::mem::{size_of, align_of};

use anyhow::Result;

use crate::thin_slice::ThinSlice;
use crate::range::RangeSlice;
use crate::types_2_1::*;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, Debug)]
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

struct Optionals<'a> {
    flags: &'a [u8; NUM_OPTIONALS],
    next: usize
}

impl<'a> Optionals<'a> {
    fn next(&mut self) -> bool {
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
trait Pod {

}

#[cfg(not(feature = "checked_types"))]
impl<T> Pod for T {}

#[cfg(not(feature = "checked_types"))]
trait CheckedBitPattern {

}

#[cfg(not(feature = "checked_types"))]
impl<T> CheckedBitPattern for T {}


#[cfg(feature = "checked_types")]
use bytemuck::{Pod, CheckedBitPattern};

struct Reader<'a> {
    data: &'a [u8],
    next: usize
}

impl<'a> Reader<'a> {
    fn read<T: bytemuck::Pod>(&mut self) -> &'a T {
        let ret = bytemuck::from_bytes(&self.data[self.next..self.next + size_of::<T>()]);
        self.next += size_of::<T>();
        ret
    }

    fn read_checked<T: bytemuck::CheckedBitPattern>(&mut self) -> &'a T {
        let ret = bytemuck::checked::from_bytes(&self.data[self.next..self.next + size_of::<T>()]);
        self.next += size_of::<T>();
        ret
    }

    fn read_slice<T: Pod>(&mut self, len: u32) -> ThinSlice<'a, T>
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

    fn read_checked_slice<T: CheckedBitPattern>(&mut self, len: u32) -> ThinSlice<'a, T>
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

    fn range<T: Pod>(&mut self, len: u32) -> RangeSlice<'a, T>
    where T: 'a
    {
        RangeSlice{ start: self.read_slice(len), count: self.read_slice(len) }
    }

    fn double3(&mut self, len: u32) -> Double3Slice<'a> {
        Double3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn float3(&mut self, len: u32) -> Float3Slice<'a> {
        Float3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn float3x3(&mut self, len: u32) -> Float3x3Slice<'a> {
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

    fn int16_3(&mut self, len: u32) -> Int16_3Slice<'a> {
        Int16_3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn int8_3(&mut self, len: u32) -> Int8_3Slice<'a> {
        Int8_3Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn rgba_u8(&mut self, len: u32) -> RgbaU8Slice<'a> {
        RgbaU8Slice {
            red: self.read_slice(len),
            green: self.read_slice(len),
            blue: self.read_slice(len),
            alpha: self.read_slice(len),
        }
    }

    fn half2(&mut self, len: u32) -> Half2Slice<'a> {
        Half2Slice {
            x: self.read_slice(len),
            y: self.read_slice(len),
        }
    }

    fn aabb(&mut self, len: u32) -> AABBSlice<'a> {
        AABBSlice { min: self.float3(len), max: self.float3(len) }
    }

    fn bounding_sphere(&mut self, len: u32) -> BoundingSphereSlice<'a> {
        BoundingSphereSlice { origo: self.float3(len), radius: self.read_slice(len) }
    }

    fn bounds(&mut self, len: u32) -> BoundsSlice<'a> {
        BoundsSlice {
            _box: self.aabb(len),
            sphere: self.bounding_sphere(len),
        }
    }

    fn child_info(&mut self, len: u32) -> ChildInfoSlice<'a> {
        ChildInfoSlice {
            len,
            hash: HashRangeSlice(self.range(len)),
            child_index: self.read_slice(len),
            child_mask: self.read_slice(len),
            tolerance: self.read_slice(len),
            total_byte_size: self.read_slice(len),
            offset: self.double3(len),
            scale: self.read_slice(len),
            bounds: self.bounds(len),
            sub_meshes: SubMeshProjectionRangeSlice(self.range(len)),
            descendant_object_ids: DescendantObjectIdsRangeSlice(self.range(len)),

        }
    }

    fn sub_mesh_projection(&mut self, len: u32) -> SubMeshProjectionSlice<'a> {
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

    fn sub_mesh(&mut self, len: u32) -> SubMeshSlice<'a> {
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

    fn texture_info(&mut self, len: u32) -> TextureInfoSlice<'a>  {
        TextureInfoSlice {
            len,
            semantic: self.read_checked_slice(len),
            transform: self.float3x3(len),
            pixel_range: PixelRangeSlice(self.range(len)),
        }
    }

    fn deviations(&mut self, len: u32, optionals: &mut Optionals) -> DeviationsSlice<'a> {
        DeviationsSlice {
            len,
            a: optionals.next().then(|| self.read_slice(len)),
            c: optionals.next().then(|| self.read_slice(len)),
            d: optionals.next().then(|| self.read_slice(len)),
            b: optionals.next().then(|| self.read_slice(len)),
        }
    }

    fn vertex(&mut self, len: u32, optionals: &mut Optionals) -> VertexSlice<'a> {
        VertexSlice {
            len,
            position: self.int16_3(len),
            normal: optionals.next().then(|| self.int8_3(len)),
            color: optionals.next().then(|| self.rgba_u8(len)),
            tex_coord: optionals.next().then(|| self.half2(len)),
            projected_pos: optionals.next().then(|| self.int16_3(len)),
            deviations: self.deviations(len, optionals),
        }
    }

    fn triangle(&mut self, len: u32, optionals: &mut Optionals) -> TriangleSlice<'a> {
        TriangleSlice { len, topology_flags: optionals.next().then(|| self.read_slice(len)) }
    }
}

impl<'a> Schema<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Schema<'a>> {
        let mut reader = Reader { data, next: 0 };
        let sizes: &Sizes = reader.read();
        let mut optionals = Optionals{ flags: reader.read_checked(), next: 0 };

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

        debug_assert_eq!(reader.next, reader.data.len());

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
