#[cfg(test)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use core::mem::size_of;

use wasm_bindgen::JsValue;


#[test]
fn test_copy_interleaved() {
    let src = [0f32, 1., 2., 3., 4., 5., 6.];
    let mut dst = [0.; 14];
    wasm_parser::copy_to_interleaved_array_f32(&mut dst, &src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
    assert_eq!(dst, [0., 0., 1., 0., 2., 0., 3., 0., 4., 0., 5., 0., 6., 0.]);
}

#[test]
fn test_fill_interleaved() {
    let src = 1.;
    let mut dst = [0f32; 14];
    wasm_parser::fill_to_interleaved_array_f32(&mut dst, src, size_of::<f32>() * 2, size_of::<f32>() * 2, 1, 7);
    assert_eq!(dst, [0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]);
}

#[cfg(not(target_family = "wasm"))]
#[test]
fn test_parser() -> anyhow::Result<()>{
    use wasm_parser::types_2_0::{Schema, OptionalVertexAttribute};

    let response = reqwest::blocking::get("https://api.novorender.com/assets/scenes/18f56c98c1e748feb8369a6d32fde9ef/webgl2_bin/1CC58DC2F443F89F7021A675640029D7")?;
    if !response.status().is_success(){
        return Err(anyhow::anyhow!("Downloading scene file:\n{}", response.text()?));
    }
    let data = response.bytes()?;

    let then = std::time::Instant::now();
    let schema = Schema::parse(&data);
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

    let _children = schema
        .children(|_| true)
        .collect::<Vec<_>>();

    // TODO: test something about the children

    Ok(())
}


#[wasm_bindgen_test::wasm_bindgen_test]
async fn test_parser_wasm() -> Result<(), JsValue> {
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response};
    use wasm_parser::types_2_0::{Schema, OptionalVertexAttribute};
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;

    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);

    let url = "https://api.novorender.com/assets/scenes/18f56c98c1e748feb8369a6d32fde9ef/webgl2_bin/1CC58DC2F443F89F7021A675640029D7";
    let request = Request::new_with_str_and_init(&url, &opts)?;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into().unwrap();

    let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
    let data = Uint8Array::new(&array_buffer).to_vec();

    let performance = web_sys::window().unwrap().performance().unwrap();
    performance.mark("start schema")?;
    let schema = Schema::parse(&data);
    performance.mark("end schema")?;
    performance.measure_with_start_mark_and_end_mark("schema", "start schema", "end schema")?;

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

    performance.mark("start children")?;
    let _children = schema
        .children(|_| true)
        .collect::<Vec<_>>();
    performance.mark("end children")?;
    performance.measure_with_start_mark_and_end_mark("children", "start children", "end children")?;

    // TODO: test something about the children

    Ok(())
}
