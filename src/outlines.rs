
use std::borrow::Cow;

use crate::{types_2_1::*, parser::{_2_1::{Child, ReturnSubMesh}, Indices}};
use densevec::DenseVec;
use glam::*;

#[derive(Clone, Copy)]
pub struct OctreeNode<'a> {
    data: &'a Child,
    geometry: &'a [ReturnSubMesh]
}

impl<'a> OctreeNode<'a> {
    pub fn new(data: &'a Child, geometry: &'a [ReturnSubMesh]) -> OctreeNode<'a> {
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
        mat4(
            vec4(scale, 0., 0., 0.),
            vec4(0., scale, 0., 0.),
            vec4(0., 0., scale, 0.),
            vec4(ox - tx, oy - ty, oz - tz, 1.),
        )
    }

    pub fn intersects_plane(&self, plane: Vec4) -> bool {
        let distance = plane.dot(self.center4());
        if distance.abs() > self.radius() {
            return false
        }

        let mut side = 0.;
        self.corners().into_iter().any(|corner| {
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
    mat4(
        vec4(s, 0., 0., 0.),
        vec4(0., s, 0., 0.),
        vec4(0., 0., s, 0.),
        vec4(o, o, o, 1.),
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

        let mut lines = 0;
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
                            self.intersect_triangles(
                                &idx_buff[begin .. end],
                                &mesh.possible_buffers.pos,
                                model_to_plane_mat,
                                buffer
                            )
                        }
                        Indices::IndexBuffer32(idx_buff) => {
                            self.intersect_triangles(
                                &idx_buff[begin .. end],
                                &mesh.possible_buffers.pos,
                                model_to_plane_mat,
                                buffer
                            )
                        }
                        _ => unreachable!()
                    };

                    if lines > 0 {
                        let end_offset = lines as usize * 2;
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

    // TODO: usize can't be converted from u32 directly, we are doing two conversions for u16 -> u32 -> usize
    // surely optimized by the compiler but check
    pub fn intersect_triangles<I: Into<u32> + Copy>(&self, idx: &[I], pos: &[i16], model_to_plane_mat: Mat4, output: &mut [f32]) -> u32 {
        let mut offset = 0;
        let mut emit = move |x, y| {
            output[offset + 0] = x;
            output[offset + 1] = y;
            offset += 2;
        };

        debug_assert_eq!(idx.len() % 3, 0);

        let mut n = 0;
        for triangle in idx.chunks(3) {
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
                if intersect_edge(p0, p1, &mut emit) { n+=1 };
                if intersect_edge(p0, p1, &mut emit) { n+=1 };
                if intersect_edge(p0, p1, &mut emit) { n+=1 };
                debug_assert_eq!(n % 2, 0)
            }
        }

        n
    }
}

#[inline(always)]
fn intersect_edge(p0: Vec4, p1: Vec4, mut emit: impl FnMut(f32, f32)) -> bool {
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