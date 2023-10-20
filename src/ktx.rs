use std::{mem::size_of, ops::Range, fmt::Debug};

use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;
use anyhow::{Result, anyhow};
use endianness::ByteOrder;

use crate::parser::ArrayUint8Array;

pub enum TextureData<'a> {
    CubeLevels(&'a[&'a[u8]]),
    Levels(&'a[u8]),
}

#[derive(serde::Serialize)]
#[wasm_bindgen]
#[derive(Clone)]
pub struct TextureParameters {
    #[wasm_bindgen(skip)]
    pub kind: &'static str,
    #[wasm_bindgen(skip)]
    pub internal_format: &'static str,
    #[wasm_bindgen(skip)]
    pub ty: Option<&'static str>,
    #[wasm_bindgen(skip)]
    pub has_mips: bool,
    #[wasm_bindgen(skip)]
    pub width: u32,
    #[wasm_bindgen(skip)]
    pub height: u32,
    #[wasm_bindgen(skip)]
    pub depth: u32,
    #[serde(skip)]
    image_data: Array,
}

#[wasm_bindgen]
impl TextureParameters {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        unreachable!()
    }

    #[wasm_bindgen(getter, js_name = "internalFormat")]
    pub fn internal_format(&self) -> String {
        unreachable!()
    }

    #[wasm_bindgen(getter, js_name = "type")]
    pub fn ty(&self) -> Option<String> {
        unreachable!()
    }

    #[wasm_bindgen(getter)]
    pub fn image(&self) -> ArrayUint8Array {
        if self.has_mips {
            JsValue::undefined().into()
        }else{
            let js_value: JsValue = self.image_data.clone().into();
            js_value.into()
        }
    }

    #[wasm_bindgen(getter, js_name = "mipMaps")]
    pub fn mip_maps(&self) -> ArrayUint8Array {
        if self.has_mips {
            let js_value: JsValue = self.image_data.clone().into();
            js_value.into()
        }else{
            JsValue::undefined().into()
        }
    }
}

#[derive(Clone)]
pub struct Ktx<'a> {
    pub kind: &'static str,
    pub internal_format: &'static str,
    pub ty: Option<&'static str>,
    pub image_data: &'a [u8],
    pub header: Header,
}

impl Ktx<'_> {
    pub fn image_data(&self) -> Array {
        let mips = self.header.num_levels.max(1);
        if self.header.num_faces == 6 {
            let images = Array::new_with_length(mips);
            images.fill(&Array::new().into(), 0, mips);
            for image in ImageIterator::new(&self.image_data, &self.header) {
                let mip: Array = images.get(image.mip).into();
                mip.set(image.face, unsafe{ Uint8Array::view(image.buffer) }.into())
            }
            images
        }else{
            let images = Array::new_with_length(mips);
            for image in ImageIterator::new(&self.image_data, &self.header) {
                images.set(image.mip, unsafe{ Uint8Array::view(image.buffer) }.into())
            }
            images
        }
    }

    pub fn texture_parameters(&self) -> TextureParameters {
        TextureParameters {
            kind: self.kind,
            internal_format: self.internal_format,
            ty: self.ty,
            has_mips: self.header.num_array_elements > 0,
            width: self.header.pixel_width,
            height: self.header.pixel_height,
            depth: self.header.pixel_depth,
            image_data: self.image_data()
        }
    }
}

static IDENTIFIER: [u8;12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
static HEADER_LEN: usize = 12 + (13 * 4); // identifier + header elements (not including key value meta-data pairs)


#[derive(Clone)]
pub struct Header {
    pub gl_type: u32,
    pub gl_type_size: u32,
    pub gl_format: u32,
    pub gl_internal_format: u32,
    pub gl_base_internal_format: u32,
    pub pixel_width: u32,
    pub pixel_height: u32,
    pub pixel_depth: u32,
    pub num_array_elements: u32,
    pub num_faces: u32,
    pub num_levels: u32,
    pub bytes_key_value_data: u32,
    pub endianness: ByteOrder,
}

impl Debug for Header {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Header")
            .field("gl_type", &texture_data_type(self.gl_type))
            .field("gl_type_size", &self.gl_type_size)
            .field("gl_format", &self.gl_format)
            .field("gl_internal_rormat", &texture_internal_format(self.gl_internal_format))
            .field("gl_base_internal_format", &self.gl_base_internal_format)
            .field("pixel_width", &self.pixel_width)
            .field("pixel_height", &self.pixel_height)
            .field("pixel_depth", &self.pixel_depth)
            .field("num_array_elements", &self.num_array_elements)
            .field("num_faces", &self.num_faces)
            .field("num_levels", &self.num_levels)
            .field("bytes_key_value_data", &self.bytes_key_value_data)
            .field("endianness", &self.endianness)
            .finish()
    }
}

fn texture_internal_format(gl_internal_format: u32) -> &'static str {
    match gl_internal_format {
        gl::R8 => "R8",
        gl::R8_SNORM => "R8_SNORM",
        gl::RG8 => "RG8",
        gl::RG8_SNORM => "RG8_SNORM",
        gl::RGB8 => "RGB8",
        gl::RGB8_SNORM => "RGB8_SNORM",
        gl::RGB565 => "RGB565",
        gl::RGBA4 => "RGBA4",
        gl::RGB5_A1 => "RGB5_A1",
        gl::RGBA8 => "RGBA8",
        gl::RGBA8_SNORM => "RGBA8_SNORM",
        gl::RGB10_A2 => "RGB10_A2",
        gl::RGB10_A2UI => "RGB10_A2UI",
        gl::SRGB8 => "SRGB8",
        gl::SRGB8_ALPHA8 => "SRGB8_ALPHA8",
        gl::R16F => "R16F",
        gl::RG16F => "RG16F",
        gl::RGB16F => "RGB16F",
        gl::RGBA16F => "RGBA16F",
        gl::R32F => "R32F",
        gl::RG32F => "RG32F",
        gl::RGB32F => "RGB32F",
        gl::RGBA32F => "RGBA32F",
        gl::R11F_G11F_B10F => "R11F_G11F_B10F",
        gl::RGB9_E5 => "RGB9_E5",
        gl::R8I => "R8I",
        gl::R8UI => "R8UI",
        gl::R16I => "R16I",
        gl::R16UI => "R16UI",
        gl::R32I => "R32I",
        gl::R32UI => "R32UI",
        gl::RG8I => "RG8I",
        gl::RG8UI => "RG8UI",
        gl::RG16I => "RG16I",
        gl::RG16UI => "RG16UI",
        gl::RG32I => "RG32I",
        gl::RG32UI => "RG32UI",
        gl::RGB8I => "RGB8I",
        gl::RGB8UI => "RGB8UI",
        gl::RGB16I => "RGB16I",
        gl::RGB16UI => "RGB16UI",
        gl::RGB32I => "RGB32I",
        gl::RGB32UI => "RGB32UI",
        gl::RGBA8I => "RGBA8I",
        gl::RGBA8UI => "RGBA8UI",
        gl::RGBA16I => "RGBA16I",
        gl::RGBA16UI => "RGBA16UI",
        gl::RGBA32I => "RGBA32I",
        gl::RGBA32UI => "RGBA32UI",
        // gl::COMPRESSED_RGB_S3TC_DXT1_EXT => "COMPRESSED_RGB_S3TC_DXT1_EXT",
        // gl::COMPRESSED_RGBA_S3TC_DXT1_EXT => "COMPRESSED_RGBA_S3TC_DXT1_EXT",
        // gl::COMPRESSED_RGBA_S3TC_DXT3_EXT => "COMPRESSED_RGBA_S3TC_DXT3_EXT",
        // gl::COMPRESSED_RGBA_S3TC_DXT5_EXT => "COMPRESSED_RGBA_S3TC_DXT5_EXT",
        gl::COMPRESSED_R11_EAC => "COMPRESSED_R11_EAC",
        gl::COMPRESSED_SIGNED_R11_EAC => "COMPRESSED_SIGNED_R11_EAC",
        gl::COMPRESSED_RG11_EAC => "COMPRESSED_RG11_EAC",
        gl::COMPRESSED_SIGNED_RG11_EAC => "COMPRESSED_SIGNED_RG11_EAC",
        gl::COMPRESSED_RGB8_ETC2 => "COMPRESSED_RGB8_ETC2",
        gl::COMPRESSED_RGBA8_ETC2_EAC => "COMPRESSED_RGBA8_ETC2_EAC",
        gl::COMPRESSED_SRGB8_ETC2 => "COMPRESSED_SRGB8_ETC2",
        gl::COMPRESSED_SRGB8_ALPHA8_ETC2_EAC => "COMPRESSED_SRGB8_ALPHA8_ETC2_EAC",
        gl::COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2 => "COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2",
        gl::COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2 => "COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2",
        // gl::COMPRESSED_RGB_PVRTC_4BPPV1_IMG => "COMPRESSED_RGB_PVRTC_4BPPV1_IMG",
        // gl::COMPRESSED_RGBA_PVRTC_4BPPV1_IMG => "COMPRESSED_RGBA_PVRTC_4BPPV1_IMG",
        // gl::COMPRESSED_RGB_PVRTC_2BPPV1_IMG => "COMPRESSED_RGB_PVRTC_2BPPV1_IMG",
        // gl::COMPRESSED_RGBA_PVRTC_2BPPV1_IMG => "COMPRESSED_RGBA_PVRTC_2BPPV1_IMG",
        // gl::COMPRESSED_RGB_ETC1_WEBGL => "COMPRESSED_RGB_ETC1_WEBGL",
        _ => unreachable!()
    }
}

fn texture_data_type(gl_type: u32) -> &'static str {
    match gl_type {
        gl::UNSIGNED_BYTE => "UNSIGNED_BYTE",
        gl::UNSIGNED_SHORT_5_6_5 => "UNSIGNED_SHORT_5_6_5",
        gl::UNSIGNED_SHORT_4_4_4_4 => "UNSIGNED_SHORT_4_4_4_4",
        gl::UNSIGNED_SHORT_5_5_5_1 => "UNSIGNED_SHORT_5_5_5_1",
        gl::HALF_FLOAT => "HALF_FLOAT",
        // gl::HALF_FLOAT_OES =>  "HALF_FLOAT_OES",
        gl::FLOAT => "FLOAT",
        gl::UNSIGNED_SHORT => "UNSIGNED_SHORT",
        gl::UNSIGNED_INT => "UNSIGNED_INT",
        gl::UNSIGNED_INT_24_8 => "UNSIGNED_INT_24_8",
        gl::BYTE => "BYTE",
        gl::SHORT => "SHORT",
        gl::INT => "INT",
        // gl::FLOAT_32_UNSIGNED_INT_24_8_REV => "FLOAT_32_UNSIGNED_INT_24_8_REV",
        gl::UNSIGNED_INT_5_9_9_9_REV => "UNSIGNED_INT_5_9_9_9_REV",
        gl::UNSIGNED_INT_2_10_10_10_REV => "UNSIGNED_INT_2_10_10_10_REV",
        gl::UNSIGNED_INT_10F_11F_11F_REV => "UNSIGNED_INT_10F_11F_11F_REV",
        _ => unreachable!()
    }
}


fn parse_header(ktx: &[u8]) -> Result<Header> {
    for (id, ktx) in IDENTIFIER.iter().zip(ktx[0..12].iter()) {
        if ktx != id {
            return Err(anyhow!("Texture missing KTX identifier"))
        }
    }

    let data_size = size_of::<u32>();
    let endianness_test = ktx[12 .. 12 + data_size].to_vec();
    let endianness = if *bytemuck::from_bytes::<u32>(&endianness_test) == 0x04030201 {
        ByteOrder::LittleEndian
    }else{
        ByteOrder::BigEndian
    };

    Ok(Header {
        gl_type: endianness::read_u32(&ktx[12 + data_size * 1 ..], endianness)?,
        gl_type_size: endianness::read_u32(&ktx[12 + data_size * 2 ..], endianness)?,
        gl_format: endianness::read_u32(&ktx[12 + data_size * 3 ..], endianness)?,
        gl_internal_format: endianness::read_u32(&ktx[12 + data_size * 4 ..], endianness)?,
        gl_base_internal_format: endianness::read_u32(&ktx[12 + data_size * 5 ..], endianness)?,
        pixel_width: endianness::read_u32(&ktx[12 + data_size * 6 ..], endianness)?,
        pixel_height: endianness::read_u32(&ktx[12 + data_size * 7 ..], endianness)?,
        pixel_depth: endianness::read_u32(&ktx[12 + data_size * 8 ..], endianness)?,
        num_array_elements: endianness::read_u32(&ktx[12 + data_size * 9 ..], endianness)?,
        num_faces: endianness::read_u32(&ktx[12 + data_size * 10 ..], endianness)?,
        num_levels: endianness::read_u32(&ktx[12 + data_size * 11 ..], endianness)?,
        bytes_key_value_data: endianness::read_u32(&ktx[12 + data_size * 12 ..], endianness)?,
        endianness,
    })
}

pub struct Image<'a> {
    pub mip: u32,
    pub element: u32,
    pub face: u32,
    pub width: u32,
    pub height: u32,
    pub blob_range: Range<u32>,
    pub buffer: &'a [u8]
}

pub struct ImageIterator<'a> {
    data: &'a [u8],
    mip: u32,
    element: u32,
    face: u32,
    z_slice: u32,
    mip_width: u32,
    mip_height: u32,
    num_mips: u32,
    num_elements: u32,
    num_faces: u32,
    depth: u32,
    data_offset: u32,
    image_size_denom: u32,
    image_stride: u32,
    endianness: ByteOrder,
}

impl ImageIterator<'_> {
    fn new<'data>(data: &'data [u8], header: &Header) -> ImageIterator<'data> {
        let num_mips = header.num_levels.max(1);
        let num_elements = header.num_array_elements.max(1);
        let num_faces = header.num_faces;
        let depth = header.pixel_depth.max(1);
        let mut data_offset = header.bytes_key_value_data;
        let image_size_denom = if header.num_faces == 6 && header.num_array_elements == 0 {
            1
        }else{
            num_elements * num_faces * depth
        };
        let image_size = endianness::read_u32(data, header.endianness).unwrap();
        data_offset += 4;
        let image_stride = image_size / image_size_denom;
        debug_assert_eq!(image_stride % 4, 0);

        ImageIterator {
            data: &data[data_offset as usize ..],
            mip: 0,
            element: 0,
            face: 0,
            z_slice: 0,
            mip_width: header.pixel_width,
            mip_height: header.pixel_height,
            num_mips,
            num_elements,
            num_faces,
            depth,
            data_offset,
            image_size_denom,
            image_stride,
            endianness: header.endianness,
        }
    }
}

impl<'a> Iterator for ImageIterator<'a> {
    type Item = Image<'a>;
    fn next(&mut self) -> Option<Self::Item> {

        if self.z_slice == self.depth {
            self.z_slice = 0;
            self.face += 1;
        }

        if self.face == self.num_faces {
            self.face = 0;
            self.element += 1;
        }

        if self.element == self.num_elements {
            self.element = 0;
            self.mip += 1;
            if self.mip < self.num_mips {
                let image_size = endianness::read_u32(self.data, self.endianness).unwrap();
                self.image_stride = image_size / self.image_size_denom;
                debug_assert_eq!(self.image_stride % 4, 0);
                self.mip_width = self.mip_width >> 1;
                self.mip_height = self.mip_height >> 1;
                self.data = &self.data[4..];
                self.data_offset += 4;
            }
        }

        if self.mip == self.num_mips {
            return None
        }


        let begin = self.data_offset;
        self.data_offset += self.image_stride;
        let buffer = &self.data[.. self.image_stride as usize];
        self.data = &self.data[self.image_stride as usize ..];
        let end = self.data_offset;
        Some(Image {
            mip: self.mip,
            element: self.element,
            face: self.face,
            width: self.mip_width,
            height: self.mip_height,
            blob_range: begin .. end,
            buffer,
        })
    }
}

pub fn parse_ktx(ktx: &[u8]) -> Result<Ktx> {
    let header = parse_header(ktx)?;

    let kind = if header.num_array_elements > 0 {
        "TEXTURE_ARRAY"
    }else if header.num_faces == 6 {
        "TEXTURE_CUBE_MAP"
    }else if header.pixel_depth > 0 {
        "TEXTURE_3D"
    }else{
        "TEXTURE_2D"
    };

    let internal_format = texture_internal_format(header.gl_internal_format);

    let ty = if header.gl_type > 0 {
        Some(texture_data_type(header.gl_type))
    }else{
        None
    };

    Ok(Ktx {
        kind,
        internal_format,
        ty,
        header,
        image_data: &ktx[HEADER_LEN..],
    })
}

// #[test]
// fn test() {
//     let data = include_bytes!("../skybox.ktx");
//     let ktx = parse_ktx(data).unwrap();
//     assert_eq!(ktx.header.num_faces, 6);
//     assert_eq!(ktx.header.num_array_elements, 0);
//     assert_eq!(ktx.header.num_levels, 11);
//     dbg!(ktx.header);
// }