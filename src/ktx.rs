use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct TextureParameters {
    #[wasm_bindgen(skip)]
    pub kind: &'static str,
    #[wasm_bindgen(skip)]
    pub internal_format: &'static str,
    pub ty: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    #[wasm_bindgen(skip)]
    pub image_data: Vec<u8>,
}

pub fn parse_ktx(ktx: &[u8]) -> TextureParameters {
    // TODO:
    TextureParameters {
        kind: "TEXTURE_2D",
        internal_format: "GL_RGBA8",
        ty: gl::BYTE,
        width: 1,
        height: 1,
        depth: 1,
        image_data: vec![0, 0, 0, 1],
    }
}