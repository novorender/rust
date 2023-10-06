extern crate proc_macro;

use proc_macro2::{TokenStream, Ident, Span};
use quote::{TokenStreamExt, ToTokens};
use smallvec::SmallVec;
use syn::{parse_macro_input, DeriveInput, Meta, Type, PathArguments, GenericArgument, TypeReference};
use quote::quote;
use syn::parse::Parse;

// fn impl_enum(name: Ident, data: DataEnum) -> proc_macro::TokenStream {
//     let ident = data.
// }


#[proc_macro_derive(StructOfArray, attributes(soa_len, soa_nested, soa_range, soa_range_index))]
pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // TODO: function to detect wasm-parser or other
    let crate_name = Ident::new("crate", Span::call_site());

    let ast = parse_macro_input!(input as DeriveInput);
    let name = ast.ident;
    let vis = ast.vis;
    let ty_generics = ast.generics;
    // let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    let slice_name = Ident::new(&format!("{}Slice", name.to_token_stream()), Span::call_site());
    let iter_name = Ident::new(&format!("{}Iter", name.to_token_stream()), Span::call_site());
    let thin_iter_name = Ident::new(&format!("{}ThinIter", name.to_token_stream()), Span::call_site());

    let struct_data = match ast.data{
		syn::Data::Enum(enum_data) => return syn::Error::new(
            enum_data.enum_token.span,
            "StructOfArray can't be derived for enums"
        ).to_compile_error().into(),
		syn::Data::Struct(struct_data) => struct_data,
		syn::Data::Union(union_data) => return syn::Error::new(
            union_data.union_token.span,
            "StructOfArray can't be derived for unions"
        ).to_compile_error().into(),
	};


    let fields = struct_data.fields;

    let iter_field_name = fields.iter()
        .map(|field| &field.ident)
        .collect::<SmallVec<[_;10]>>();
    let iter_field_name = &iter_field_name;
    let thin_iter_field_name = fields.iter()
        .map(|field| &field.ident)
        .collect::<SmallVec<[_;10]>>();
    let thin_iter_field_name = &thin_iter_field_name;

    // Detect primitive from type and add exceptions to type that look like non primitive
    // let field_is_primitive = fields.iter().map(|field| {
    //     let is_primitive_attr = field.attrs.iter().any(|attr| match &attr.meta {
    //         Meta::Path(path) => path.segments.len() == 1 && path.segments[0].ident == "soa_primitive",
    //         _ => false,
    //     });
    //     let looks_like_primitive_type = match &field.ty {
    //         Type::Path(path) => !path.path.segments.last().unwrap().ident
    //             .to_string()
    //             .chars()
    //             .next()
    //             .unwrap()
    //             .is_uppercase(),
    //         _ => false,
    //     };
    //     is_primitive_attr || looks_like_primitive_type
    // }).collect::<SmallVec<[bool;10]>>();

    let field_attrs = fields.iter().map(|field| {
        let mut is_primitive = !field.attrs.iter().any(|attr| match &attr.meta {
            Meta::Path(path) => path.segments.len() == 1 && path.segments[0].ident == "soa_nested",
            _ => false,
        });

        let range = field.attrs.iter().filter_map(|attr| match &attr.meta {
            Meta::List(meta_list) => if meta_list.path.segments.len() == 1
                && meta_list.path.segments[0].ident == "soa_range"
            {
                Some(meta_list.tokens.clone())
            }else{
                None
            },
            _ => None,
        }).next();

        let is_range = range.is_some();

        let (is_optional, slice_type, range_data_type, range_is_slice) = match &field.ty {
            Type::Path(path) => if !path.path.segments.is_empty() {
                if path.path.segments[0].ident.to_string() == "Option" {
                    match &path.path.segments[0].arguments {
                        PathArguments::AngleBracketed(args) => match &args.args[0] {
                            GenericArgument::Type(ty) => (true, quote!(#ty), None, false),
                            _ => todo!()
                        }
                        _ => todo!()
                    }
                }else if let Some(range) = range{
                    is_primitive = false;
                    // let range_slice_ty = Ident::new(&format!("{}Slice", range), Span::call_site());
                    let range_slice_ty = &field.ty;
                    (false, range, Some(quote!(#range_slice_ty)), false)
                }else{
                    let ty = &field.ty;
                    (false, quote!(#ty), None, false)
                }
            }else{
                todo!()
            },

            Type::Reference(reference) => if let Some(range) = range {
                is_primitive = false;
                if let Type::Slice(slice) = &*reference.elem {
                    let range_data_type = &*slice.elem;
                    (false, range, Some(quote!(#crate_name::thin_slice::ThinSlice<'a, #range_data_type>)), true)
                }else{
                    let ty = &field.ty;
                    (false, quote!(#ty), None, false)
                }
            }else{
                let ty = &field.ty;
                (false, quote!(#ty), None, false)
            }

            other => (false, quote!(#other), None, false),
        };

        (is_primitive, is_optional, is_range, slice_type, range_data_type, range_is_slice)
    }).collect::<SmallVec<[(bool, bool, bool, TokenStream, Option<TokenStream>, bool);10]>>();

    let slice_field_ty = field_attrs.iter()
        .map(|(is_primitive, is_optional, is_range, ty, range_data_type, range_is_slice)|  {
            if *is_primitive && !*is_optional {
                quote!(#crate_name::thin_slice::ThinSlice<'a, #ty>)
            }else if *is_primitive && *is_optional {
                quote!(Option<#crate_name::thin_slice::ThinSlice<'a, #ty>>)
            }else if !*is_primitive && !*is_optional {
                let ty = Ident::new(&format!("{}Slice", ty.to_token_stream()), Span::call_site());
                quote!(#ty <'a>)
            }else{ // !is_primitive && is_optional
                let ty = Ident::new(&format!("{}Slice", ty.to_token_stream()), Span::call_site());
                quote!(Option<#ty <'a>>)
            }
        });

    let iter_field_ty = field_attrs.iter()
        .map(|(is_primitive, is_optional, is_range, ty, range_data_type, range_is_slice)|  {
            if *is_primitive && !*is_optional {
                quote!(#crate_name::thin_slice::ThinSliceIter<'a, #ty>)
            }else if *is_primitive && *is_optional {
                quote!(Option<#crate_name::thin_slice::ThinSliceIter<'a, #ty>>)
            }else if !*is_primitive && !*is_optional {
                let ty = Ident::new(&format!("{}ThinIter", ty.to_token_stream()), Span::call_site());
                quote!(#ty <'a>)
            }else{ // !is_primitive && is_optional
                let ty = Ident::new(&format!("{}ThinIter", ty.to_token_stream()), Span::call_site());
                quote!(Option<#ty <'a>>)
            }
        }).collect::<SmallVec<[_;10]>>();

    let iter_field_value = fields.iter()
        .zip(&field_attrs)
        .map(|(field, (is_primitive, is_optional, is_range, _, range_data_type, range_is_slice))|  {
            let field_name = &field.ident;
            if !*is_range {
                if *is_primitive && !*is_optional {
                    quote!(*self.#field_name.next())
                }else if *is_primitive && *is_optional {
                    quote!(self.#field_name.as_mut().map(|#field_name| *#field_name.next()))
                }else if !*is_primitive && !*is_optional {
                    quote!(self.#field_name.next())
                }else{ // !is_primitive && is_optional
                    quote!(self.#field_name.as_mut().map(|#field_name| #field_name.next()))
                }
            }else{
                let field_data_name = Ident::new(&format!("{}_data", field.ident.to_token_stream()), Span::call_site());
                if !*is_optional && *range_is_slice {
                    quote!(self.#field_data_name.slice_range(&self.#field_name.next()))
                }else if !*is_optional && !*range_is_slice{
                    quote!(self.#field_data_name.range(&self.#field_name.next()))
                }else{
                    todo!()
                }
            }
        }).collect::<SmallVec<[_;10]>>();

    let slice_field_thin_iter = fields.iter()
        .zip(&field_attrs)
        .map(|(field, (_, is_optional, is_range, _, range_data_type, range_is_slice))|  {
            let field_name = &field.ident;
            if !*is_optional {
                quote!(self.#field_name.thin_iter())
            }else if !*is_optional {
                quote!(self.#field_name.thin_iter())
            }else{
                quote!(self.#field_name.as_ref().map(|#field_name| #field_name.thin_iter()))
            }
        }).collect::<SmallVec<[_;10]>>();

    let iter_data_field_ty = fields.iter()
        .zip(&field_attrs)
        .filter_map(|(field, (_, is_optional, is_range, _, range_data_type, range_is_slice))|  {
            if *is_range {
                Some(quote!(#range_data_type))
            }else{
                None
            }
        }).collect::<SmallVec<[_;10]>>();

    let iter_data_field_name = fields.iter()
        .zip(&field_attrs)
        .filter_map(|(field, (_, is_optional, is_range, _, range_data_type, range_is_slice))|  {
            if *is_range {
                let field_data_name = Ident::new(&format!("{}_data", field.ident.to_token_stream()), Span::call_site());
                Some(quote!(#field_data_name))
            }else{
                None
            }
        }).collect::<SmallVec<[_;10]>>();

    let has_len = ast.attrs.iter().any(|attr| match &attr.meta {
        Meta::Path(path) => path.segments.len() == 1 && path.segments[0].ident == "soa_len",
        _ => false,
    });

    // let range_index_type = ast.attrs.iter().filter_map(|attr| match &attr.meta {
    //     Meta::List(meta_list) => if meta_list.path.segments.len() == 1
    //         && meta_list.path.segments[0].ident == "soa_range_index"
    //     {
    //         Some(&meta_list.tokens)
    //     }else{
    //         None
    //     },
    //     _ => None,
    // }).next();

    let slice_field_name = fields.iter().map(|field| &field.ident);
    // let range_impl = if let Some(range_index_type) = range_index_type {
    //     quote!{
    //         #vis fn range(&self, range: &#range_index_type) -> Self {
    //             Self {
    //                 len: range.count,
    //                 #(
    //                     #slice_field_name: self.#slice_field_name.range(range),
    //                 )*
    //             }
    //         }
    //     }
    // }else{
    //     quote!()
    // };
    let range_impl_calls = fields.iter()
        .zip(&field_attrs)
        .map(|(field, (_, is_optional, is_range, _, range_data_type, range_is_slice))|  {
            let field_name = &field.ident;
            if !*is_optional {
                quote!(#field_name: self.#field_name.range(range))
            }else{
                quote!(#field_name: self.#field_name.as_ref().map(|#field_name| #field_name.range(range)))
            }
        });

    let slice_len_field = if has_len {
        quote!(#vis len: u32,)
    }else{
        quote!()
    };

    let slice_field_name = fields.iter().map(|field| &field.ident);
    let slice_struct = quote! {
        #[derive(Clone)]
        #vis struct #slice_name <'a> {
            #slice_len_field
            #(
                #vis #slice_field_name: #slice_field_ty,
            )*
        }
    };

    let iterator = if has_len {
        quote! {
            #vis struct #iter_name <'a> {
                len: usize,
                #(
                    #iter_field_name: #iter_field_ty,
                )*
                #(
                    #iter_data_field_name: #iter_data_field_ty,
                )*
            }

            #vis struct #thin_iter_name <'a> {
                #(
                    #thin_iter_field_name: #iter_field_ty,
                )*
                #(
                    #iter_data_field_name: #iter_data_field_ty,
                )*
            }
        }
    }else{
        quote! {
            #vis struct #thin_iter_name <'a> {
                #(
                    #thin_iter_field_name: #iter_field_ty,
                )*
                #(
                    #iter_data_field_name: #iter_data_field_ty,
                )*
            }
        }
    };

    let slice_primitive_field_accessor = fields.iter()
        .zip(&field_attrs)
        .map(|(field, (is_primitive, is_optional, is_range, field_ty, range_data_type, range_is_slice))| {
            let slice_field_name = &field.ident;

            if *is_primitive && !*is_optional {
                quote! {
                    #vis fn #slice_field_name (&self) -> &'a [#field_ty] {
                        unsafe{ self.#slice_field_name.as_slice(self.len) }
                    }
                }
            }else if *is_primitive && *is_optional {
                quote! {
                    #vis fn #slice_field_name (&self) -> Option<&'a [#field_ty]> {
                        Some(unsafe{ self.#slice_field_name?.as_slice(self.len) })
                    }
                }
            }else{
                quote!()
            }
        });

    let implementation = if has_len {
        quote! {
            impl<'a> #slice_name <'a> {
                #( #slice_primitive_field_accessor )*

                #vis fn iter(&self #(, #iter_data_field_name: #iter_data_field_ty )*) -> #iter_name <'a> {
                    #iter_name {
                        len: self.len as usize,
                        #(
                            #iter_field_name: unsafe{ #slice_field_thin_iter },
                        )*
                        #(
                            #iter_data_field_name,
                        )*
                    }
                }

                #vis fn thin_iter(&self #(, #iter_data_field_name: #iter_data_field_ty )*) -> #thin_iter_name <'a> {
                    #thin_iter_name {
                        #(
                            #iter_field_name: #slice_field_thin_iter,
                        )*
                        #(
                            #iter_data_field_name,
                        )*
                    }
                }

                #vis unsafe fn range(&self, range: &#crate_name::range::Range<u32>) -> Self {
                    Self {
                        len: range.count,
                        #(
                            #range_impl_calls,
                        )*
                    }
                }
            }

            impl<'a> Iterator for #iter_name <'a> {
                type Item = #name #ty_generics;

                // SAFETY: We check len before calling next on the thin slice iterators which all have the
                // same size
                #[inline(always)]
                fn next(&mut self) -> Option<Self::Item> {
                    use #crate_name::thin_slice::ThinSliceIterator;

                    if self.len == 0 {
                        return None;
                    }

                    self.len -= 1;

                    Some(unsafe{#name {
                        #(
                            #iter_field_name: unsafe { #iter_field_value },
                        )*
                    }})
                }

                #[inline(always)]
                fn size_hint(&self) -> (usize, Option<usize>) {
                    (self.len, Some(self.len))
                }
            }

            impl<'a> ExactSizeIterator for #iter_name <'a> {
                #[inline(always)]
                fn len(&self) -> usize {
                    self.len
                }
            }

            impl<'a> #crate_name::thin_slice::ThinSliceIterator for #thin_iter_name <'a> {
                type Item = #name #ty_generics;

                #[inline(always)]
                unsafe fn next(&mut self) -> Self::Item {
                    use #crate_name::thin_slice::ThinSliceIterator;

                    #name {
                        #(
                            #thin_iter_field_name: unsafe { #iter_field_value },
                        )*
                    }
                }
            }
        }
    }else{
        quote! {
            impl<'a> #slice_name <'a> {
                #vis fn thin_iter(&self, #( #iter_data_field_name: #iter_data_field_ty )*) -> #thin_iter_name <'a> {
                    #thin_iter_name {
                        #(
                            #iter_field_name: self.#iter_field_name.thin_iter(),
                        )*
                    }
                }

                #vis unsafe fn range(&self, range: &#crate_name::range::Range<u32>) -> Self {
                    Self {
                        #(
                            #range_impl_calls,
                        )*
                    }
                }
            }

            impl<'a> #crate_name::thin_slice::ThinSliceIterator for #thin_iter_name <'a> {
                type Item = #name #ty_generics;

                #[inline(always)]
                unsafe fn next(&mut self) -> Self::Item {
                    use #crate_name::thin_slice::ThinSliceIterator;

                    #name {
                        #(
                            #thin_iter_field_name: unsafe { #iter_field_value },
                        )*
                    }
                }
            }
        }
    };

    // eprintln!("{}", implementation);

    let mut generated = TokenStream::new();
    generated.append_all(slice_struct);
    generated.append_all(iterator);
    generated.append_all(implementation);
    generated.into()

}
