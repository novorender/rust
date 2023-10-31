
use crate::{types_2_1::*, parser::{_2_1::{NodeData, ReturnSubMesh}, Indices}};
use densevec::DenseVec;
use na::*;
use na::simd::WideF32x4;
use js_sys::{Uint16Array, Int16Array, Uint32Array, Float32Array};
use wasm_bindgen::prelude::wasm_bindgen;

#[derive(Clone, Copy)]
pub struct OctreeNode<'a> {
    data: &'a NodeData,
    geometry: &'a [ReturnSubMesh],
}

impl<'a> OctreeNode<'a> {
    pub fn new(data: &'a NodeData, geometry: &'a [ReturnSubMesh]) -> OctreeNode<'a> {
        OctreeNode { data, geometry }
    }

    #[inline(always)]
    pub fn center4(self) -> Vec4 {
        let center = self.data.bounds.sphere.origo;
        vec4(center.x, center.y, center.z, 0.)
    }

    #[inline(always)]
    fn radius(self) -> f32 {
        self.data.bounds.sphere.radius
    }

    #[inline(always)]
    pub fn corners(self) -> [Vec4; 8] {
        let AABB {
            min: Float3 { x: x0, y: y0, z: z0 },
            max: Float3 { x: x1, y: y1, z: z1 }
        } = self.data.bounds.box_;

        [
            vec4(x0, y0, z0, 1.),
            vec4(x1, y0, z0, 1.),
            vec4(x0, y1, z0, 1.),
            vec4(x1, y1, z0, 1.),
            vec4(x0, y0, z1, 1.),
            vec4(x1, y0, z1, 1.),
            vec4(x0, y1, z1, 1.),
            vec4(x1, y1, z1, 1.),
        ]
    }

    pub fn model_local_matrix(self, local_space_translation: Vec3) -> Mat4 {
        let scale = self.data.scale;
        // TODO: we are losing precision here, offset is f64
        let (ox, oy, oz) = (self.data.offset.x as f32, self.data.offset.y as f32, self.data.offset.z as f32);
        let (tx, ty, tz) = (local_space_translation.x, local_space_translation.y, local_space_translation.z);
        Mat4::new(
            scale, 0., 0., 0.,
            0., scale, 0., 0.,
            0., 0., scale, 0.,
            ox - tx, oy - ty, oz - tz, 1.,
        )
    }

    pub fn intersects_plane(&self, plane: Vec4) -> bool {
        let distance = plane.dot(&self.center4());
        if distance.abs() > self.radius() {
            return false
        }

        let mut side = 0.;
        self.corners().iter().any(|corner| {
            let distance = plane.dot(corner);
            let dist_sign = distance.signum();
            if side != 0. && dist_sign != side {
                return true
            }
            if distance != 0. {
                side = dist_sign
            }
            false
        })
    }
}

pub struct LineClusterRef<'a> {
    pub object_id: u32,
    pub vertices: &'a [f32],
}

pub struct LineCluster {
    pub object_id: u32,
    pub vertices: Vec<f32>,
}

pub struct RenderNode<'a> {
    pub mask: u32,
    pub node: OctreeNode<'a>,
}

const DENORM_MATRIX: Mat4 = {
    let s = 1. / 32767.5;
    let o = 0.5 * s;
    Mat4::new(
        s, 0., 0., 0.,
        0., s, 0., 0.,
        0., 0., s, 0.,
        o, o, o, 1.,
    )
};


struct NodeIntersectionBuilder<'a>{
    clusters: Vec<LineClusterRef<'a>>,
}

pub struct OutlineRenderer {
    clip_plane: Vec4,
    local_space_translation: Vec3,
    local_plane_matrix: Mat4
}

impl OutlineRenderer {
    pub fn intersect_nodes_triangles<'buffer>(&self, render_nodes: & [RenderNode], mut buffer: &'buffer mut [f32]) -> impl ExactSizeIterator<Item = LineCluster> + 'buffer {
        //TODO: Aren't clusters already clustered by object_id??
        let mut line_cluster_arrays: DenseVec<Vec<&[f32]>> = DenseVec::new();
        let clip_plane = self.clip_plane;
        for node in render_nodes.iter() {
            if !node.node.intersects_plane(clip_plane) {
                continue;
            }

            let (node_clusters, new_buffer) = self.create_node_line_clusters(node.node, buffer);
            buffer = new_buffer;
            for (child_index, clusters) in node_clusters.iter() {
                if (1 << child_index) & node.mask == 0 {
                    continue;
                }

                for cluster in clusters {
                    line_cluster_arrays.entry(cluster.object_id as usize)
                        .or_insert_with(|| vec![])
                        .push(cluster.vertices);
                }
            }
        }

        line_cluster_arrays.into_iter().map(move |(object_id, vertices)| {
            LineCluster  {
                object_id: object_id as u32,
                vertices: vertices.into_iter().flat_map(|vertices| vertices).copied().collect()
            }
        })
    }

    pub fn create_node_line_clusters<'buffer>(&self, node: OctreeNode, mut buffer: &'buffer mut [f32]) -> (DenseVec<Vec<LineClusterRef<'buffer>>>, &'buffer mut [f32]){
        let mut child_builders = DenseVec::new();

        let model_local_matrix = node.model_local_matrix(self.local_space_translation);
        let model_to_plane_mat = self.local_plane_matrix * model_local_matrix * DENORM_MATRIX;

        for mesh in node.geometry {
            if mesh.num_triangles == 0
                || mesh.primitive_type != PrimitiveType::Triangles
                || mesh.base_color_texture.is_some()
                || !mesh.indices.is_buffer()
            {
                continue;
            }

            for draw_range in mesh.draw_ranges.as_ref().unwrap() {
                let child_builder = child_builders
                    .entry(draw_range.child_index as usize)
                    .or_insert_with(|| NodeIntersectionBuilder { clusters: vec![] });

                let begin_triangle = draw_range.first / 3;
                let end_triangle = begin_triangle + draw_range.count / 3;


                for object_range in mesh.object_ranges.as_ref().unwrap() {
                    if object_range.begin_triangle < begin_triangle
                        || object_range.end_triangle > end_triangle
                    {
                        continue
                    }

                    let begin = object_range.begin_triangle * 3;
                    let end = object_range.end_triangle * 3;

                    let lines = match &mesh.indices {
                        Indices::IndexBuffer16(idx_buff) => {
                            intersect_triangles(
                                &idx_buff[begin .. end],
                                &mesh.possible_buffers.pos,
                                model_to_plane_mat,
                                buffer
                            )
                        }
                        Indices::IndexBuffer32(idx_buff) => {
                            intersect_triangles(
                                &idx_buff[begin .. end],
                                &mesh.possible_buffers.pos,
                                model_to_plane_mat,
                                buffer
                            )
                        }
                        _ => unreachable!()
                    };

                    if lines > 0 {
                        let end_offset = lines as usize * 4;
                        let (vertices, new_buffer) = buffer.split_at_mut(end_offset);
                        buffer = new_buffer;
                        let line_cluster = LineClusterRef{
                            object_id: object_range.object_id,
                            vertices,
                        };
                        child_builder.clusters.push(line_cluster)
                    }
                }
            }
        }

        let clusters = child_builders.into_iter()
            .map(|(child_index, builder)| (child_index, builder.clusters))
            .collect();
        (clusters, buffer)
    }
}

// TODO: usize can't be converted from u32 directly, we are doing two conversions for u16 -> u32 -> usize
// surely optimized by the compiler but check
pub fn intersect_triangles<I: Into<u32> + Copy>(idx: &[I], pos: &[i16], model_to_plane_mat: Mat4, output: &mut [f32]) -> u32 {
    let mut offset = 0;
    let mut emit = move |x, y| {
        output[offset + 0] = x;
        output[offset + 1] = y;
        offset += 2;
    };

    debug_assert_eq!(idx.len() % 3, 0);

    let model_to_plane_matx4: Mat4<WideF32x4> = Mat4::from_fn(|i,j| WideF32x4::splat(model_to_plane_mat[(i,j)]));

    let mut n = 0;
    for triangle in idx.chunks_exact(3*4) {
        let i0 = triangle[0].into() as usize;
        let i1 = triangle[1].into() as usize;
        let i2 = triangle[2].into() as usize;
        let i3 = triangle[3].into() as usize;
        let i4 = triangle[4].into() as usize;
        let i5 = triangle[5].into() as usize;
        let i6 = triangle[6].into() as usize;
        let i7 = triangle[7].into() as usize;
        let i8 = triangle[8].into() as usize;
        let i9 = triangle[9].into() as usize;
        let i10 = triangle[10].into() as usize;
        let i11 = triangle[11].into() as usize;

        let p0 = vec4(
            [pos[i0 * 3 + 0] as f32, pos[i3 * 3 + 0] as f32, pos[i6 * 3 + 0] as f32, pos[i9 * 3 + 0] as f32].into(),
            [pos[i0 * 3 + 1] as f32, pos[i3 * 3 + 1] as f32, pos[i6 * 3 + 1] as f32, pos[i9 * 3 + 1] as f32].into(),
            [pos[i0 * 3 + 2] as f32, pos[i3 * 3 + 2] as f32, pos[i6 * 3 + 2] as f32, pos[i9 * 3 + 2] as f32].into(),
            WideF32x4::splat(1.),
        );
        let p1 = vec4(
            [pos[i1 * 3 + 0] as f32, pos[i4 * 3 + 0] as f32, pos[i7 * 3 + 0] as f32, pos[i10 * 3 + 0] as f32].into(),
            [pos[i1 * 3 + 1] as f32, pos[i4 * 3 + 1] as f32, pos[i7 * 3 + 1] as f32, pos[i10 * 3 + 1] as f32].into(),
            [pos[i1 * 3 + 2] as f32, pos[i4 * 3 + 2] as f32, pos[i7 * 3 + 2] as f32, pos[i10 * 3 + 2] as f32].into(),
            WideF32x4::splat(1.),
        );
        let p2 = vec4(
            [pos[i2 * 3 + 0] as f32, pos[i5 * 3 + 0] as f32, pos[i8 * 3 + 0] as f32, pos[i11 * 3 + 0] as f32].into(),
            [pos[i2 * 3 + 1] as f32, pos[i5 * 3 + 1] as f32, pos[i8 * 3 + 1] as f32, pos[i11 * 3 + 1] as f32].into(),
            [pos[i2 * 3 + 2] as f32, pos[i5 * 3 + 2] as f32, pos[i8 * 3 + 2] as f32, pos[i11 * 3 + 2] as f32].into(),
            WideF32x4::splat(1.),
        );

        let p0 = model_to_plane_matx4 * p0;
        let p1 = model_to_plane_matx4 * p1;
        let p2 = model_to_plane_matx4 * p2;

        // for i in 0..4 {
        //     let p0 = unsafe{ vec3(p0.x.extract_unchecked(i), p0.y.extract_unchecked(i), p0.z.extract_unchecked(i)) };
        //     let p1 = unsafe{ vec3(p1.x.extract_unchecked(i), p1.y.extract_unchecked(i), p1.z.extract_unchecked(i)) };
        //     let p2 = unsafe{ vec3(p2.x.extract_unchecked(i), p2.y.extract_unchecked(i), p2.z.extract_unchecked(i)) };

        //     let gt0 = p0.z > 0.; let gt1 = p1.z > 0.; let gt2 = p2.z > 0.;
        //     let lt0 = p0.z < 0.; let lt1 = p1.z < 0.; let lt2 = p2.z < 0.;
        //     if (gt0 || gt1 || gt2) && (lt0 || lt1 || lt2) {
        //         if intersect_edge(p0, p1, &mut emit) { n+=1 };
        //         if intersect_edge(p1, p2, &mut emit) { n+=1 };
        //         if intersect_edge(p2, p0, &mut emit) { n+=1 };
        //         debug_assert_eq!(n % 2, 0)
        //     }
        // }


        let zeros = WideF32x4::splat(0.);
        let gt0 = p0.z.simd_gt(zeros); let gt1 = p1.z.simd_gt(zeros); let gt2 = p2.z.simd_gt(zeros);
        let lt0 = p0.z.simd_lt(zeros); let lt1 = p1.z.simd_lt(zeros); let lt2 = p2.z.simd_lt(zeros);
        let cond = ((gt0 | gt1 | gt2) & (lt0 | lt1 | lt2)).bitmask();
        if cond != 0 {
            let cond0 = ((p0.z.simd_le(zeros) & p1.z.simd_gt(zeros)) | (p1.z.simd_le(zeros) & p0.z.simd_gt(zeros))).bitmask() & cond;
            let ab0 = if cond0 != 0 {
                Some(intersect_edge_x4(p0, p1))
            }else{
                None
            };

            let cond1 = ((p1.z.simd_le(zeros) & p2.z.simd_gt(zeros)) | (p2.z.simd_le(zeros) & p1.z.simd_gt(zeros))).bitmask() & cond;
            let ab1 = if cond1 != 0 {
                Some(intersect_edge_x4(p1, p2))
            }else{
                None
            };

            let cond2 = ((p2.z.simd_le(zeros) & p0.z.simd_gt(zeros)) | (p0.z.simd_le(zeros) & p2.z.simd_gt(zeros))).bitmask() & cond;
            let ab2 = if cond2 != 0 {
                Some(intersect_edge_x4(p2, p0))
            }else{
                None
            };

            for i in 0..4 {
                // let p0 = unsafe{ vec3(p0.x.extract_unchecked(i), p0.y.extract_unchecked(i), p0.z.extract_unchecked(i)) };
                // let p1 = unsafe{ vec3(p1.x.extract_unchecked(i), p1.y.extract_unchecked(i), p1.z.extract_unchecked(i)) };
                // let p2 = unsafe{ vec3(p2.x.extract_unchecked(i), p2.y.extract_unchecked(i), p2.z.extract_unchecked(i)) };
                // let gt0 = p0.z > 0.; let gt1 = p1.z > 0.; let gt2 = p2.z > 0.;
                // let lt0 = p0.z < 0.; let lt1 = p1.z < 0.; let lt2 = p2.z < 0.;
                // if (gt0 || gt1 || gt2) && (lt0 || lt1 || lt2) {
                // if (cond >> i) & 0x1 != 0 {
                    // if (p0.z <= 0. && p1.z > 0.) || (p1.z <= 0. && p0.z > 0.) {
                    if (cond0 >> i) & 0x1 != 0 {
                        let (a0, b0) = unsafe{ ab0.unwrap_unchecked() };
                        unsafe{ emit(a0.extract_unchecked(i), b0.extract_unchecked(i)) }
                        n += 1;
                    }
                    // if (p1.z <= 0. && p2.z > 0.) || (p2.z <= 0. && p1.z > 0.) {
                    if (cond1 >> i) & 0x1 != 0 {
                        let (a1, b1) = unsafe{ ab1.unwrap_unchecked() };
                        unsafe{ emit(a1.extract_unchecked(i), b1.extract_unchecked(i)) }
                        n += 1;
                    }
                    // if (p2.z <= 0. && p0.z > 0.) || (p0.z <= 0. && p2.z > 0.) {
                    if (cond2 >> i) & 0x1 != 0 {
                        let (a2, b2) = unsafe{ ab2.unwrap_unchecked() };
                        unsafe{ emit(a2.extract_unchecked(i), b2.extract_unchecked(i)) }
                        n += 1;
                    }


                // }
            }
        }
        debug_assert_eq!(n % 2, 0)
    }

    for triangle in idx[idx.len() - idx.len() % 12 ..].chunks(3) {
        let i0 = triangle[0].into() as usize;
        let i1 = triangle[1].into() as usize;
        let i2 = triangle[2].into() as usize;
        let p0 = vec4(pos[i0 * 3 + 0] as f32, pos[i0 * 3 + 1] as f32, pos[i0 * 3 + 2] as f32, 1.);
        let p1 = vec4(pos[i1 * 3 + 0] as f32, pos[i1 * 3 + 1] as f32, pos[i1 * 3 + 2] as f32, 1.);
        let p2 = vec4(pos[i2 * 3 + 0] as f32, pos[i2 * 3 + 1] as f32, pos[i2 * 3 + 2] as f32, 1.);

        let p0 = model_to_plane_mat * p0;
        let p1 = model_to_plane_mat * p1;
        let p2 = model_to_plane_mat * p2;

        let gt0 = p0.z > 0.; let gt1 = p1.z > 0.; let gt2 = p2.z > 0.;
        let lt0 = p0.z < 0.; let lt1 = p1.z < 0.; let lt2 = p2.z < 0.;
        if (gt0 || gt1 || gt2) && (lt0 || lt1 || lt2) {
            if intersect_edge(p0.xyz(), p1.xyz(), &mut emit) { n+=1 };
            if intersect_edge(p1.xyz(), p2.xyz(), &mut emit) { n+=1 };
            if intersect_edge(p2.xyz(), p0.xyz(), &mut emit) { n+=1 };
            debug_assert_eq!(n % 2, 0)
        }
    }

    n / 2
}

#[wasm_bindgen]
pub struct WasmMat4(Mat4);

#[wasm_bindgen]
pub fn allocate_mat4(mat: &[f32]) -> WasmMat4 {
    WasmMat4(Mat4::from_column_slice(mat))
}

#[wasm_bindgen]
pub fn intersect_triangles_u16(
    idx: Uint16Array,
    pos: Int16Array,
    model_to_plane_mat: &WasmMat4,
    output: Float32Array) -> u32
{
    let idx = unsafe{ std::slice::from_raw_parts(
        idx.byte_offset() as usize as *const u16,
        idx.length() as usize
    )};
    let pos = unsafe{ std::slice::from_raw_parts(
        pos.byte_offset() as usize as *const i16,
        pos.length() as usize
    )};
    let output: &mut [f32] = unsafe{ std::slice::from_raw_parts_mut(
        output.byte_offset() as usize as *mut f32,
        output.length() as usize
    )};

    intersect_triangles(idx, pos, model_to_plane_mat.0, output)
}

#[wasm_bindgen]
pub fn intersect_triangles_u32(
    idx: Uint32Array,
    pos: Int16Array,
    model_to_plane_mat: &WasmMat4,
    output: Float32Array) -> u32
{
    let idx = unsafe{ std::slice::from_raw_parts(
        idx.byte_offset() as usize as *const u32,
        idx.length() as usize
    )};
    let pos = unsafe{ std::slice::from_raw_parts(
        pos.byte_offset() as usize as *const i16,
        pos.length() as usize
    )};
    let output: &mut [f32] = unsafe{ std::slice::from_raw_parts_mut(
        output.byte_offset() as usize as *mut f32,
        output.length() as usize
    )};

    intersect_triangles(idx, pos, model_to_plane_mat.0, output)
}

#[inline(always)]
fn intersect_edge(p0: Vec3, p1: Vec3, mut emit: impl FnMut(f32, f32)) -> bool {
    if (p0.z <= 0. && p1.z > 0.) || (p1.z <= 0. && p0.z > 0.) {
        let t = -p0.z / (p1.z - p0.z);
        emit(lerp(p0.x, p1.x, t), lerp(p0.y, p1.y, t));
        true
    }else{
        false
    }
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline(always)]
fn intersect_edge_x4(p0: Vec4<WideF32x4>, p1: Vec4<WideF32x4>) -> (WideF32x4, WideF32x4) {
    let t = -p0.z / (p1.z - p0.z);
    let a = lerp_x4(p0.x, p1.x, t);
    let b = lerp_x4(p0.y, p1.y, t);
    (a, b)
}

#[inline(always)]
fn lerp_x4(a: WideF32x4, b: WideF32x4, t: WideF32x4) -> WideF32x4 {
    a + (b - a) * t
}