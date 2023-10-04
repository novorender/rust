use core::mem::size_of;

use anyhow::Result;
use bytemuck::{Pod, Zeroable, CheckedBitPattern};

use crate::utils::{ThinSlice, Range};
use crate::types::*;

#[derive(Pod, Zeroable, Copy, Clone, Debug)]
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

#[derive(Clone, Copy)]
#[repr(transparent)]
struct OptionalsData([bool; NUM_OPTIONALS]);

unsafe impl CheckedBitPattern for OptionalsData {
    type Bits = [u8; NUM_OPTIONALS];

    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        bits.iter().all(|b| *b == 0 || *b == 1)
    }
}

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
    // let rem = next % align_of::<T>();
    // if rem != 0 { align_of::<T>() - rem } else { 0 }

    (size_of::<T>() - 1) - ((next + size_of::<T>() - 1) % size_of::<T>())
}

struct Reader<'a> {
    data: &'a [u8],
    next: usize
}

impl<'a> Reader<'a> {
    fn read<T: Pod>(&mut self) -> &'a T {
        let ret = bytemuck::from_bytes(&self.data[self.next..self.next + size_of::<T>()]);
        self.next += size_of::<T>();
        ret
    }

    fn read_checked<T: CheckedBitPattern>(&mut self) -> &'a T {
        let ret = bytemuck::checked::from_bytes(&self.data[self.next..self.next + size_of::<T>()]);
        self.next += size_of::<T>();
        ret
    }

    fn read_slice<T>(&mut self, len: u32) -> ThinSlice<'a, T>
    where T: 'a
    {
        if len > 0 {
            self.next += align::<T>(self.next);

            let size = size_of::<T>() * len as usize;
            let next_next = self.next + size;
            let ret = ThinSlice::from_data_and_offset(&self.data, self.next, len);
            // let ret = ThinSlice::from(bytemuck::cast_slice(&self.data[self.next .. next_next]));
            self.next = next_next;
            ret
        }else{
            ThinSlice::empty()
        }

    }

    fn read_checked_slice<T>(&mut self, len: u32) -> ThinSlice<'a, T>
    where T: 'a
    {
        if len > 0 {
            self.next += align::<T>(self.next);

            let size = size_of::<T>() * len as usize;
            let next_next = self.next + size;
            let ret = ThinSlice::from_data_and_offset(&self.data, self.next, len);
            // let ret = ThinSlice::from(bytemuck::checked::cast_slice(&self.data[self.next .. next_next]));
            self.next = next_next;
            ret
        }else{
            ThinSlice::empty()
        }
    }

    fn range<T: Pod>(&mut self, len: u32) -> Range<'a, T> {
        Range{ start: self.read_slice(len), count: self.read_slice(len) }
    }

    fn double3(&mut self, len: u32) -> Double3<'a> {
        Double3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn float3(&mut self, len: u32) -> Float3<'a> {
        Float3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn float3x3(&mut self, len: u32) -> Float3x3<'a> {
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

    fn int16_3(&mut self, len: u32) -> Int16_3<'a> {
        Int16_3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn int8_3(&mut self, len: u32) -> Int8_3<'a> {
        Int8_3 {
            x: self.read_slice(len),
            y: self.read_slice(len),
            z: self.read_slice(len),
        }
    }

    fn rgba_u8(&mut self, len: u32) -> RgbaU8<'a> {
        RgbaU8 {
            red: self.read_slice(len),
            green: self.read_slice(len),
            blue: self.read_slice(len),
            alpha: self.read_slice(len),
        }
    }

    fn half2(&mut self, len: u32) -> Half2<'a> {
        Half2 {
            x: self.read_slice(len),
            y: self.read_slice(len),
        }
    }

    fn aabb(&mut self, len: u32) -> AABB<'a> {
        AABB { min: self.float3(len), max: self.float3(len) }
    }

    fn bounding_sphere(&mut self, len: u32) -> BoundingSphere<'a> {
        BoundingSphere { origo: self.float3(len), radius: self.read_slice(len) }
    }

    fn bounds(&mut self, len: u32) -> Bounds<'a> {
        Bounds {
            _box: self.aabb(len),
            sphere: self.bounding_sphere(len),
        }
    }

    fn child_info(&mut self, len: u32) -> ChildInfo<'a> {
        ChildInfo {
            len,
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

    fn sub_mesh_projection(&mut self, len: u32) -> SubMeshProjection<'a> {
        SubMeshProjection {
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

    fn sub_mesh(&mut self, len: u32) -> SubMesh<'a> {
        SubMesh {
            len,
            child_index: self.read_slice(len),
            object_id: self.read_slice(len),
            material_index: self.read_slice(len),
            primitive_type: self.read_checked_slice(len),
            material_type: self.read_checked_slice(len),
            attributes: self.read_slice(len),
            num_deviations: self.read_slice(len),
            vertices: VertexRange(self.range(len)),
            primitive_vertex_indices: VertexIndexRange(self.range(len)),
            edge_vertex_indices: VertexIndexRange(self.range(len)),
            corner_vertex_indices: VertexIndexRange(self.range(len)),
            textures: TextureInfoRange(self.range(len)),
        }
    }

    fn texture_info(&mut self, len: u32) -> TextureInfo<'a>  {
        TextureInfo {
            len,
            semantic: self.read_checked_slice(len),
            transform: self.float3x3(len),
            pixel_range: PixelRange(self.range(len)),
        }
    }

    fn deviations(&mut self, len: u32, optionals: &mut Optionals) -> Deviations<'a> {
        Deviations {
            len,
            a: optionals.next().then(|| self.read_slice(len)),
            c: optionals.next().then(|| self.read_slice(len)),
            d: optionals.next().then(|| self.read_slice(len)),
            b: optionals.next().then(|| self.read_slice(len)),
        }
    }

    fn vertex(&mut self, len: u32, optionals: &mut Optionals) -> Vertex<'a> {
        Vertex {
            len,
            position: self.int16_3(len),
            normal: optionals.next().then(|| self.int8_3(len)),
            color: optionals.next().then(|| self.rgba_u8(len)),
            tex_coord: optionals.next().then(|| self.half2(len)),
            projected_pos: optionals.next().then(|| self.int16_3(len)),
            deviations: self.deviations(len, optionals),
        }
    }

    fn triangle(&mut self, len: u32, optionals: &mut Optionals) -> Triangle<'a> {
        Triangle { len, topology_flags: optionals.next().then(|| self.read_slice(len)) }
    }
}

impl<'a> Schema<'a> {
    pub fn read(data: &'a [u8]) -> Result<Schema<'a>> {
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

        Ok(Schema {
            version: "2.1",
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
    use std::io::Read;

    let then = std::time::Instant::now();
    let mut file = std::fs::File::open("8AC1B48A77DC9D0E0AE8DDC366379FFF").unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();

    let schema = Schema::read(&data).unwrap();
    dbg!(then.elapsed());

    for p in schema.sub_mesh_projection.primitive_type() {
        assert!((*p as u8) < 7);
    }
    for p in schema.sub_mesh_projection.attributes() {
        let mut p = *p;
        p.remove(OptionalVertexAttribute::NORMAL);
        p.remove(OptionalVertexAttribute::COLOR);
        p.remove(OptionalVertexAttribute::TEX_COORD);
        p.remove(OptionalVertexAttribute::PROJECTED_POS);
        assert!(p.is_empty());
    }
    // assert_eq!(file.read_to_end(&mut vec![0;1024]).unwrap(),  0);
}