use crate::tools::py_err;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};
use pyo3::{intern, Bound, PyResult};
use std::collections::HashSet;

const CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY: &str = "pydantic.internal.union_discriminator";

macro_rules! get {
    ($dict: expr, $key: expr) => {
        $dict.get_item(intern!($dict.py(), $key))?
    };
}

macro_rules! traverse_key_fn {
    ($key: expr, $func: expr, $dict: expr, $ctx: expr) => {{
        if let Some(v) = get!($dict, $key) {
            $func(v.downcast_exact()?, $ctx)?
        }
    }};
}

macro_rules! traverse {
    ($($key:expr => $func:expr),*; $dict: expr, $ctx: expr) => {{
        $(traverse_key_fn!($key, $func, $dict, $ctx);)*
        traverse_key_fn!("serialization", gather_schema, $dict, $ctx);
        gather_meta($dict, $ctx)
    }}
}

macro_rules! defaultdict_list_append {
    ($dict: expr, $key: expr, $value: expr) => {{
        match $dict.get_item($key)? {
            None => {
                let list = PyList::empty_bound($dict.py());
                list.append($value)?;
                $dict.set_item($key, list)?;
            }
            // Safety: we know that the value is a PyList as we just created it above
            Some(list) => unsafe { list.downcast_unchecked::<PyList>() }.append($value)?,
        };
    }};
}

fn gather_definition_ref(schema_ref_dict: &Bound<'_, PyDict>, ctx: &mut GatherCtx) -> PyResult<()> {
    if let Some(schema_ref) = get!(schema_ref_dict, "schema_ref") {
        let schema_ref_pystr = schema_ref.downcast_exact::<PyString>()?;
        let schema_ref_str = schema_ref_pystr.to_str()?;

        if !ctx.recursively_seen_refs.contains(schema_ref_str) {
            defaultdict_list_append!(&ctx.def_refs, schema_ref_pystr, schema_ref_dict);

            // TODO should py_err! when not found. That error can be used to detect the missing defs in cleaning side
            if let Some(definition) = ctx.definitions_dict.get_item(schema_ref_pystr)? {
                ctx.recursively_seen_refs.insert(schema_ref_str.to_string());

                gather_schema(definition.downcast_exact::<PyDict>()?, ctx)?;
                traverse_key_fn!("serialization", gather_schema, schema_ref_dict, ctx);
                gather_meta(schema_ref_dict, ctx)?;

                ctx.recursively_seen_refs.remove(schema_ref_str);
            }
        } else {
            ctx.recursive_def_refs.add(schema_ref_pystr)?;
            for seen_ref in &ctx.recursively_seen_refs {
                let seen_ref_pystr = PyString::new_bound(schema_ref.py(), seen_ref);
                ctx.recursive_def_refs.add(seen_ref_pystr)?;
            }
        }
        Ok(())
    } else {
        py_err!(PyKeyError; "Invalid definition-ref, missing schema_ref")
    }
}

fn gather_meta(schema: &Bound<'_, PyDict>, ctx: &mut GatherCtx) -> PyResult<()> {
    if let Some(meta) = get!(schema, "metadata") {
        let meta_dict = meta.downcast_exact::<PyDict>()?;
        if let Some(discriminator) = get!(meta_dict, CORE_SCHEMA_METADATA_DISCRIMINATOR_PLACEHOLDER_KEY) {
            let schema_discriminator = PyTuple::new_bound(schema.py(), vec![schema.as_any(), &discriminator]);
            ctx.discriminators.append(schema_discriminator)?;
        }
    }
    Ok(())
}

fn gather_list(schema_list: &Bound<'_, PyList>, ctx: &mut GatherCtx) -> PyResult<()> {
    for v in schema_list.iter() {
        gather_schema(v.downcast_exact()?, ctx)?;
    }
    Ok(())
}

fn gather_dict(schemas_by_key: &Bound<'_, PyDict>, ctx: &mut GatherCtx) -> PyResult<()> {
    for (_, v) in schemas_by_key.iter() {
        gather_schema(v.downcast_exact()?, ctx)?;
    }
    Ok(())
}

fn gather_union_choices(schema_list: &Bound<'_, PyList>, ctx: &mut GatherCtx) -> PyResult<()> {
    for v in schema_list.iter() {
        if let Ok(tup) = v.downcast_exact::<PyTuple>() {
            gather_schema(tup.get_item(0)?.downcast_exact()?, ctx)?;
        } else {
            gather_schema(v.downcast_exact()?, ctx)?;
        }
    }
    Ok(())
}

fn gather_arguments(arguments: &Bound<'_, PyList>, ctx: &mut GatherCtx) -> PyResult<()> {
    for v in arguments.iter() {
        traverse_key_fn!("schema", gather_schema, v.downcast_exact::<PyDict>()?, ctx);
    }
    Ok(())
}

// Has 100% coverage in Pydantic side. This is exclusively used there
#[cfg_attr(has_coverage_attribute, coverage(off))]
fn gather_schema(schema: &Bound<'_, PyDict>, ctx: &mut GatherCtx) -> PyResult<()> {
    let type_ = get!(schema, "type");
    if type_.is_none() {
        return py_err!(PyKeyError; "Schema type missing");
    }
    match type_.unwrap().downcast_exact::<PyString>()?.to_str()? {
        "definition-ref" => gather_definition_ref(schema, ctx),
        "definitions" => traverse!("schema" => gather_schema, "definitions" => gather_list; schema, ctx),
        "list" | "set" | "frozenset" | "generator" => traverse!("items_schema" => gather_schema; schema, ctx),
        "tuple" => traverse!("items_schema" => gather_list; schema, ctx),
        "dict" => traverse!("keys_schema" => gather_schema, "values_schema" => gather_schema; schema, ctx),
        "union" => traverse!("choices" => gather_union_choices; schema, ctx),
        "tagged-union" => traverse!("choices" => gather_dict; schema, ctx),
        "chain" => traverse!("steps" => gather_list; schema, ctx),
        "lax-or-strict" => traverse!("lax_schema" => gather_schema, "strict_schema" => gather_schema; schema, ctx),
        "json-or-python" => traverse!("json_schema" => gather_schema, "python_schema" => gather_schema; schema, ctx),
        "model-fields" | "typed-dict" => traverse!(
            "extras_schema" => gather_schema, "computed_fields" => gather_list, "fields" => gather_dict; schema, ctx
        ),
        "dataclass-args" => traverse!("computed_fields" => gather_list, "fields" => gather_list; schema, ctx),
        "arguments" => traverse!(
            "arguments_schema" => gather_arguments,
            "var_args_schema" => gather_schema,
            "var_kwargs_schema" => gather_schema;
            schema, ctx
        ),
        "call" => traverse!("arguments_schema" => gather_schema, "return_schema" => gather_schema; schema, ctx),
        "computed-field" | "function-plain" => traverse!("return_schema" => gather_schema; schema, ctx),
        "function-wrap" => traverse!("return_schema" => gather_schema, "schema" => gather_schema; schema, ctx),
        _ => traverse!("schema" => gather_schema; schema, ctx),
    }
}

pub struct GatherCtx<'a, 'py> {
    pub definitions_dict: &'a Bound<'py, PyDict>,
    pub def_refs: Bound<'py, PyDict>,
    pub recursive_def_refs: Bound<'py, PySet>,
    pub discriminators: Bound<'py, PyList>,
    recursively_seen_refs: HashSet<String>,
}

#[pyfunction(signature = (schema, definitions))]
pub fn gather_schemas_for_cleaning<'py>(
    schema: &Bound<'py, PyAny>,
    definitions: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = schema.py();
    let mut ctx = GatherCtx {
        definitions_dict: definitions.downcast_exact()?,
        def_refs: PyDict::new_bound(py),
        recursive_def_refs: PySet::empty_bound(py)?,
        discriminators: PyList::empty_bound(py),
        recursively_seen_refs: HashSet::new(),
    };
    gather_schema(schema.downcast_exact::<PyDict>()?, &mut ctx)?;

    let res = PyDict::new_bound(py);
    res.set_item(intern!(py, "definition_refs"), ctx.def_refs)?;
    res.set_item(intern!(py, "recursive_refs"), ctx.recursive_def_refs)?;
    res.set_item(intern!(py, "deferred_discriminators"), ctx.discriminators)?;
    Ok(res)
}
