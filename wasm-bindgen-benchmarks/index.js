import * as wasm from "../pkg";
const { mat4, vec3, vec4 } = glMatrix;

wasm.init_console();

function wasmBench() {
    wasm.bench_intersections();
}

function intersectTriangles(output /* Float32Array*/, offset/*: number*/, idx/*: Uint16Array | Uint32Array*/, pos/*: Int16Array*/, modelToPlaneMatrix/*: ReadonlyMat4*/) {
    const p0 = vec3.create(); const p1 = vec3.create(); const p2 = vec3.create();
    let n = 0;
    function emit(x/*: number*/, y/*: number*/) {
        output[offset++] = x;
        output[offset++] = y;
        n++;
    }

    // for each triangle...
    // console.assert(idx.length % 3 == 0); // assert that we are dealing with triangles.
    for (let i = 0; i < idx.length; i += 3) {
        const i0 = idx[i + 0]; const i1 = idx[i + 1]; const i2 = idx[i + 2];
        vec3.set(p0, pos[i0 * 3 + 0], pos[i0 * 3 + 1], pos[i0 * 3 + 2]);
        vec3.set(p1, pos[i1 * 3 + 0], pos[i1 * 3 + 1], pos[i1 * 3 + 2]);
        vec3.set(p2, pos[i2 * 3 + 0], pos[i2 * 3 + 1], pos[i2 * 3 + 2]);
        // transform positions into clipping plane space, i.e. xy on plane, z above or below
        vec3.transformMat4(p0, p0, modelToPlaneMatrix);
        vec3.transformMat4(p1, p1, modelToPlaneMatrix);
        vec3.transformMat4(p2, p2, modelToPlaneMatrix);
        // check if z-coords are greater and less than 0
        const z0 = p0[2]; const z1 = p1[2]; const z2 = p2[2];
        const gt0 = z0 > 0; const gt1 = z1 > 0; const gt2 = z2 > 0;
        const lt0 = z0 < 0; const lt1 = z1 < 0; const lt2 = z2 < 0;
        // does triangle intersect plane?
        // this test is not just a possible optimization, but also excludes problematic triangles that straddles the plane along an edge
        if ((gt0 || gt1 || gt2) && (lt0 || lt1 || lt2)) { // SIMD: any()?
            // check for edge intersections
            intersectEdge(emit, p0, p1);
            intersectEdge(emit, p1, p2);
            intersectEdge(emit, p2, p0);
            // console.assert(n % 2 == 0); // check that there are always pairs of vertices
        }
    }
    return n / 2;
}

function* range(start = 0, end = Infinity, step = 1) {
    let iterationCount = 0;
    for (let i = start; i < end; i += step) {
      iterationCount++;
      yield i;
    }
    return iterationCount;
}

function randomI16() {
    return Math.floor(Math.random() * 0xFFFF);
}

function getModelLocalMatrix(localSpaceTranslation/*: ReadonlyVec3*/, offset, scale) {
    const [ox, oy, oz] = offset;
    const [tx, ty, tz] = localSpaceTranslation;
    const modelLocalMatrix = mat4.fromValues(
        scale, 0, 0, 0,
        0, scale, 0, 0,
        0, 0, scale, 0,
        ox - tx, oy - ty, oz - tz, 1
    );
    return modelLocalMatrix;
}

function orthoNormalBasisMatrixFromPlane(plane/*: ReadonlyVec4*/)/*: ReadonlyMat4*/ {
    const [nx, ny, nz, offs] = plane;
    const axisZ = vec3.fromValues(nx, ny, nz);
    const minI = Math.abs(nx) < Math.abs(ny) && Math.abs(nx) < Math.abs(nz) ? 0 : Math.abs(ny) < Math.abs(nz) ? 1 : 2;
    const axisY = vec3.fromValues(0, 0, 0);
    axisY[minI] = 1;
    const axisX = vec3.cross(vec3.create(), axisY, axisZ);
    vec3.cross(axisX, axisY, axisZ);
    vec3.normalize(axisX, axisX);
    vec3.cross(axisY, axisZ, axisX);
    vec3.normalize(axisY, axisY);
    const [bx, by, bz] = axisX;
    const [tx, ty, tz] = axisY;
    return mat4.fromValues(
        bx, by, bz, 0,
        tx, ty, tz, 0,
        nx, ny, nz, 0,
        nx * -offs, ny * -offs, nz * -offs, 1
    );
}

function planeMatrices(plane/*: ReadonlyVec4*/, localSpaceTranslation/*: ReadonlyVec3*/) {
    const [x, y, z, o] = plane;
    const normal = vec3.fromValues(x, y, z);
    const distance = -o - vec3.dot(localSpaceTranslation, normal);
    const planeLS = vec4.fromValues(normal[0], normal[1], normal[2], -distance);
    const planeLocalMatrix = orthoNormalBasisMatrixFromPlane(planeLS);
    const localPlaneMatrix = mat4.invert(mat4.create(), planeLocalMatrix);
    return { planeLocalMatrix, localPlaneMatrix };// as const;
}

function normInt16ToFloatMatrix() {
    // Positions in model (node) space are given in 16 bit signed normalized ints.
    // Prior to opengl 4.2, this means mapping [-0x8000, 0x7fff] to [-1, 1] respectively: https://www.khronos.org/opengl/wiki/Normalized_Integer
    // This roughly equates to f = (v + 0.5) / 32767.5
    const s = 1 / 32767.5;
    const o = 0.5 * s;
    return mat4.fromValues(
        s, 0, 0, 0,
        0, s, 0, 0,
        0, 0, s, 0,
        o, o, o, 1,
    );
}

function jsBench() {
    const INTERSECTIONS_LEN = 1000000;
    const pos = new Int16Array([...range(0, INTERSECTIONS_LEN)].flatMap((i) => {
        return [
            randomI16(), randomI16(), randomI16(),
            randomI16(), randomI16(), randomI16(),
            randomI16(), randomI16(), randomI16()
        ];
    }));

    const sequentialIdx = new Uint32Array([...range(0, INTERSECTIONS_LEN)].flatMap((i) => (
        [i * 3 + 0, i * 3 + 1, i * 3 + 2]
    )));

    const randomOffsetIdx = new Uint32Array([...range(0, INTERSECTIONS_LEN)].flatMap((i) => {
        const idxOffset = Math.floor((Math.random() * 2. - 1.) * 100.);
        i = Math.min(i + idxOffset, INTERSECTIONS_LEN - 1);
        return [i * 3 + 0, i * 3 + 1, i * 3 + 2];
    }));

    const randomIdx = new Uint32Array([...range(0, INTERSECTIONS_LEN)].flatMap((i) => {
        i = Math.floor(Math.random() * INTERSECTIONS_LEN);
        return [i * 3 + 0, i * 3 + 1, i * 3 + 2];
    }));

    const localSpaceTranslation = vec3.fromValues(100., 10., 1000.);
    const offset = vec3.fromValues(10., 1000., 10000.);
    const scale = 3.;
    const planeNormal = vec3.fromValues(1., 3., 5.);
    vec3.normalize(planeNormal, planeNormal);
    const planeOffset = 2000.;
    const plane = vec4.fromValues(planeNormal.x, planeNormal.y, planeNormal.z, planeOffset);
    const modelLocalMatrix = getModelLocalMatrix(localSpaceTranslation, offset, scale);
    const { planeLocalMatrix, localPlaneMatrix } = planeMatrices(plane, localSpaceTranslation);
    const modelToPlaneMatrix = localPlaneMatrix * modelLocalMatrix * normInt16ToFloatMatrix();
    const output = new Float32Array(INTERSECTIONS_LEN * 10);

    const BENCH_RUN_TIME = 10000.;
    let start = performance.now();
    let end = start;
    let num_iterations = 0;
    while(end - start < BENCH_RUN_TIME) {
        intersectTriangles(output, 0, sequentialIdx, pos, modelToPlaneMatrix);
        end = performance.now();
        num_iterations += 1;
    }
    console.log(`sequential indices: ${(end - start)  / num_iterations}ms num iterations: ${num_iterations}`)


    start = performance.now();
    end = start;
    num_iterations = 0;
    while(end - start < BENCH_RUN_TIME) {
        intersectTriangles(output, 0, randomOffsetIdx, pos, modelToPlaneMatrix);
        end = performance.now();
        num_iterations += 1;
    }
    console.log(`random offset indices: ${(end - start)  / num_iterations}ms num iterations: ${num_iterations}`)


    start = performance.now();
    end = start;
    num_iterations = 0;
    while(end - start < BENCH_RUN_TIME) {
        intersectTriangles(output, 0, randomIdx, pos, modelToPlaneMatrix);
        end = performance.now();
        num_iterations += 1;
    }
    console.log(`random indices: ${(end - start)  / num_iterations}ms num iterations: ${num_iterations}`)
}

wasmBench()
jsBench();