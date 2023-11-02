use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Array<DrawRange>")]
    pub type ArrayDrawRange;
    #[wasm_bindgen(typescript_type = "Array<MeshObjectRange>")]
    pub type ArrayObjectRange;
    #[wasm_bindgen(typescript_type = "Array<Uint8Array>")]
    pub type ArrayUint8Array;
    #[wasm_bindgen(typescript_type = "Array<Uint8Array> | undefined")]
    pub type ArrayUint8ArrayOrUndefined;
    #[wasm_bindgen(typescript_type = "Array<Uint16Array>")]
    pub type ArrayUint16Array;
    #[wasm_bindgen(typescript_type = "Array<Uint32Array>")]
    pub type ArrayUint32Array;
    #[wasm_bindgen(typescript_type = "Uint8Array | undefined")]
    pub type Uint8ArrayOrUndefined;
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