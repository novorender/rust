extern crate proc_macro;

use proc_macro2::{TokenStream, Ident, Span};
use quote::{TokenStreamExt, ToTokens};
use smallvec::SmallVec;
use syn::{parse_macro_input, DeriveInput, Meta, Type, PathArguments, GenericArgument};
use quote::quote;

enum Range<'a> {
    PrimitiveSlice {
        /// The data type referenced by this range
        data_type: TokenStream,
        /// The actual range type
        range_type: &'a TokenStream
    },
    NestedSlice {
        /// The data type referenced by this range
        data_type: &'a Type,
        /// The actual range type
        range_type: &'a TokenStream
    },
}

enum FieldType<'a> {
    Primitive(&'a Type),
    OptionalPrimitive(&'a Type),
    Nested(&'a Type),
    OptionalNested(&'a Type),
    Range(Range<'a>),
}

struct Field<'a> {
    ident: &'a Ident,
    ty: FieldType<'a>,
}

impl<'a> Field<'a> {
    fn is_range(&self) -> bool {
        if let FieldType::Range(_) = &self.ty { true } else { false }
    }

    fn is_optional(&self) -> bool {
        if let FieldType::OptionalPrimitive(_) | FieldType::OptionalNested(_)
            = &self.ty { true } else { false }
    }
}

#[proc_macro_derive(StructOfArray, attributes(soa_len, soa_nested, soa_range))]
pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // TODO: function to detect wasm-parser or other
    let crate_name = Ident::new("crate", Span::call_site());

    let ast = parse_macro_input!(input as DeriveInput);
    let name = ast.ident;
    let vis = ast.vis;
    let ty_generics = ast.generics;
    // let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

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

    let field_name = fields.iter()
        .map(|field| &field.ident)
        .collect::<SmallVec<[_;10]>>();

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


    // Fields that have the soa_len attribute will get a len field and implement Iterator
    let has_len = ast.attrs.iter().any(|attr| match &attr.meta {
        Meta::Path(path) => path.segments.len() == 1 && path.segments[0].ident == "soa_len",
        _ => false,
    });

    let fields = fields.iter().map(|field| {
        let is_primitive = !field.attrs.iter().any(|attr| match &attr.meta {
            Meta::Path(path) => path.segments.len() == 1 && path.segments[0].ident == "soa_nested",
            _ => false,
        });

        let range_type = field.attrs.iter().filter_map(|attr| match &attr.meta {
            Meta::List(meta_list) => if meta_list.path.segments.len() == 1
                && meta_list.path.segments[0].ident == "soa_range"
            {
                Some(&meta_list.tokens)
            }else{
                None
            },
            _ => None,
        }).next();

        let is_range = range_type.is_some();

        let field_ty = match &field.ty {
            Type::Path(path) => if path.path.segments[0].ident.to_string() == "Option" {
                match &path.path.segments[0].arguments {
                    PathArguments::AngleBracketed(args) => match &args.args[0] {
                        GenericArgument::Type(ty) => if is_range {
                            return Err(syn::Error::new(
                                field.ident.as_ref().unwrap().span(),
                                "Optional range fields are not supported yet"
                            ))
                        }else if is_primitive {
                            FieldType::OptionalPrimitive(ty)
                        }else{
                            FieldType::OptionalNested(ty)
                        }
                        _ => unreachable!()
                    }
                    _ => unreachable!()
                }
            }else if let Some(range_type) = range_type {
                FieldType::Range(Range::NestedSlice { data_type: &field.ty, range_type })
            }else if is_primitive {
                FieldType::Primitive(&field.ty)
            }else{
                FieldType::Nested(&field.ty)
            },

            Type::Reference(reference) => if let Some(range_type) = range_type {
                if let Type::Slice(slice) = &*reference.elem {
                    let range_data_type = &*slice.elem;
                    FieldType::Range(Range::PrimitiveSlice{
                        data_type: quote!(#crate_name::thin_slice::ThinSlice<'a, #range_data_type>),
                        range_type,
                    })
                }else{
                    return Err(syn::Error::new(
                        field.ident.as_ref().unwrap().span(),
                        "Ranges can only be references to slices or nested thin slices"
                    ))
                }
            }else{
                return Err(syn::Error::new(
                    field.ident.as_ref().unwrap().span(),
                    "Primitive type references are not supported yet"
                ))
            }

            _ => if is_primitive {
                FieldType::Primitive(&field.ty)
            }else{
                FieldType::Nested(&field.ty)
            }
        };

        Ok(Field {
            ident: field.ident.as_ref().unwrap(),
            ty: field_ty
        })
    }).collect::<Result<SmallVec<[Field;10]>, syn::Error>>();

    let fields = match fields {
        Ok(fields) => fields,
        Err(err) => return err.to_compile_error().into()
    };

    // Field types for the Slice struct
    let slice_field_ty = fields.iter()
        .map(|field|  match &field.ty {
            FieldType::Primitive(ty) =>
                quote!(#crate_name::thin_slice::ThinSlice<'a, #ty>),
            FieldType::OptionalPrimitive(ty) =>
                quote!(Option<#crate_name::thin_slice::ThinSlice<'a, #ty>>),
            FieldType::Nested(ty) => {
                let ty = Ident::new(&format!("{}Slice", ty.to_token_stream()), Span::call_site());
                quote!(#ty <'a>)
            }
            FieldType::OptionalNested(ty) => {
                let ty = Ident::new(&format!("{}Slice", ty.to_token_stream()), Span::call_site());
                quote!(Option<#ty <'a>>)
            }
            FieldType::Range(Range::PrimitiveSlice { range_type, .. })
                | FieldType::Range(Range::NestedSlice { range_type, .. }) =>
            {
                // TypeRange -> TypeRangeSlice generated in impl_range
                let ty = Ident::new(&format!("{}Slice", range_type), Span::call_site());
                quote!(#ty <'a>)
            }
        });

    // Field types for the ThinIter struct
    let iter_field_ty = fields.iter()
        .map(|field| match &field.ty {
            FieldType::Primitive(ty) =>
                quote!(#crate_name::thin_slice::ThinSliceIter<'a, #ty>),
            FieldType::OptionalPrimitive(ty) =>
                quote!(Option<#crate_name::thin_slice::ThinSliceIter<'a, #ty>>),
            FieldType::Nested(ty) => {
                // Type -> TypeThinIter which is generated for every type in this macro
                let ty = Ident::new(&format!("{}ThinIter", ty.to_token_stream()), Span::call_site());
                quote!(#ty <'a>)
            }
            FieldType::OptionalNested(ty) => {
                // Option<Type> -> Option<TypeThinIter> which is generated for every type in this macro
                let ty = Ident::new(&format!("{}ThinIter", ty.to_token_stream()), Span::call_site());
                quote!(Option<#ty <'a>>)
            }
            FieldType::Range(Range::NestedSlice { range_type, .. }) =>{
                // TypeRange => TypeRangeThinIter which is generated by the impl_range macro
                let ty = Ident::new(&format!("{}ThinIter", range_type), Span::call_site());
                quote!(#ty <'a>)
            }
            FieldType::Range(Range::PrimitiveSlice { range_type, .. }) => {
                // TypeRange => TypeRangeThinIter which is generated by the impl_range macro
                let ty = Ident::new(&format!("{}ThinIter", range_type), Span::call_site());
                quote!(#ty <'a>)

            }
        }).collect::<SmallVec<[_;10]>>();

    // Data field types for the ThinIter struct (those fields used to retrieve the actual data from
    // the range) the type is just the slice type, ThinSlice<_> for primitives or TypeSlice for
    // nested
    let iter_data_field_ty = fields.iter()
        .filter_map(|field|  {
            match &field.ty {
                FieldType::Range(range) => match range {
                    Range::PrimitiveSlice { data_type, .. }
                        => Some(quote!(#data_type)),
                    Range::NestedSlice { data_type, .. }
                        => Some(quote!(#data_type)),
                }
                _ => None
            }
        }).collect::<SmallVec<[_;10]>>();

    // Data field names for the ThinIter struct
    let iter_data_field_name = fields.iter()
        .filter_map(|field|  {
            if field.is_range() {
                let field_data_name = Ident::new(&format!("{}_data", field.ident.to_token_stream()), Span::call_site());
                Some(quote!(#field_data_name))
            }else{
                None
            }
        }).collect::<SmallVec<[_;10]>>();

    // Field value for the ThinIter Output on next
    let iter_field_value = fields.iter()
        .map(|field|  {
            let field_name = field.ident;
            match &field.ty {
                FieldType::Primitive(_) => quote!(*self.#field_name.next()),
                FieldType::Nested(_) => quote!(self.#field_name.next()),
                FieldType::OptionalPrimitive(_) =>
                    quote!(self.#field_name.as_mut().map(|#field_name| *#field_name.next())),
                FieldType::OptionalNested(_) =>
                    quote!(self.#field_name.as_mut().map(|#field_name| #field_name.next())),
                FieldType::Range(Range::PrimitiveSlice { .. }) => {
                    let field_data_name = Ident::new(&format!("{}_data", field_name.to_token_stream()), Span::call_site());
                    quote!(self.#field_data_name.slice_range(&self.#field_name.next()))
                }
                FieldType::Range(Range::NestedSlice { .. }) => {
                    let field_data_name = Ident::new(&format!("{}_data", field_name.to_token_stream()), Span::call_site());
                    quote!(self.#field_data_name.range(&self.#field_name.next()))
                }
            }
        }).collect::<SmallVec<[_;10]>>();

    // Field value for the Iter and ThinIter types when instanciating on iter() and thin_iter()
    let slice_field_thin_iter = fields.iter()
        .map(|field|  {
            let field_name = &field.ident;
            if field.is_optional() {
                quote!(self.#field_name.as_ref().map(|#field_name| #field_name.thin_iter()))
            }else{
                quote!(self.#field_name.thin_iter())
            }
        }).collect::<SmallVec<[_;10]>>();

    // Field name and value on range calls on the Slice types
    let range_impl_calls = fields.iter()
        .map(|field|  {
            let field_name = &field.ident;
            if field.is_optional() {
                quote!(#field_name: self.#field_name.as_ref().map(|#field_name| #field_name.range(range)))
            }else{
                quote!(#field_name: self.#field_name.range(range))
            }
        });

    // Len field on the Slice type for types with soa_len attribute
    let slice_len_field = if has_len {
        quote!(#vis len: u32,)
    }else{
        quote!()
    };

    // The Slice type declaration
    let slice_name: Ident = Ident::new(&format!("{}Slice", name.to_token_stream()), Span::call_site());
    let slice_struct = quote! {
        #[derive(Clone)]
        #vis struct #slice_name <'a> {
            #slice_len_field
            #(
                #vis #field_name: #slice_field_ty,
            )*
        }
    };

    // The Iter and ThinIter types for soa_len or ThinIter declaration for the rest of the types
    let iter_name = Ident::new(&format!("{}Iter", name.to_token_stream()), Span::call_site());
    let thin_iter_name = Ident::new(&format!("{}ThinIter", name.to_token_stream()), Span::call_site());
    let iterator = if has_len {
        quote! {
            #vis struct #iter_name <'a> {
                len: usize,
                #(
                    #field_name: #iter_field_ty,
                )*
                #(
                    #iter_data_field_name: #iter_data_field_ty,
                )*
            }

            #vis struct #thin_iter_name <'a> {
                #(
                    #field_name: #iter_field_ty,
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
                    #field_name: #iter_field_ty,
                )*
                #(
                    #iter_data_field_name: #iter_data_field_ty,
                )*
            }
        }
    };

    // Accessor methods for each primitive or optional primitive fields on Slice types
    // Nested and nested ranges can't be accessed as a slice as they are not contiguous
    // TODO: Primitive ranges would need data but could be implemented if needed
    let slice_primitive_field_accessor = fields.iter()
        .map(|field| {
            let slice_field_name = &field.ident;
            match &field.ty {
                FieldType::Primitive(ty) => quote! {
                    #vis fn #slice_field_name (&self) -> &'a [#ty] {
                        unsafe{ self.#slice_field_name.as_slice(self.len) }
                    }
                },
                FieldType::OptionalPrimitive(ty) => quote! {
                    #vis fn #slice_field_name (&self) -> Option<&'a [#ty]> {
                        Some(unsafe{ self.#slice_field_name?.as_slice(self.len) })
                    }
                },
                FieldType::Nested(_) | FieldType::OptionalNested(_)=> quote!(),
                FieldType::Range(_) => quote!()
            }
        });

    // Implementations for Slice, Iter and ThinIter types
    let implementation = if has_len {
        quote! {
            impl<'a> #slice_name <'a> {
                #( #slice_primitive_field_accessor )*

                #vis fn iter(&self #(, #iter_data_field_name: #iter_data_field_ty )*) -> #iter_name <'a> {
                    #iter_name {
                        len: self.len as usize,
                        #(
                            #field_name: unsafe{ #slice_field_thin_iter },
                        )*
                        #(
                            #iter_data_field_name,
                        )*
                    }
                }

                #vis fn thin_iter(&self #(, #iter_data_field_name: #iter_data_field_ty )*) -> #thin_iter_name <'a> {
                    #thin_iter_name {
                        #(
                            #field_name: #slice_field_thin_iter,
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
                            #field_name: unsafe { #iter_field_value },
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
                            #field_name: unsafe { #iter_field_value },
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
                            #field_name: unsafe{ #slice_field_thin_iter },
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
                            #field_name: unsafe { #iter_field_value },
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
