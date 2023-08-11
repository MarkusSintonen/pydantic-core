use std::collections::{HashMap};
use pyo3::{intern, PyResult, Python};
use pyo3::types::{PyDict, PyList, PyString};
use crate::core_schema_utils::walk_core_schema::{CoreSchemaVisitor, invalid_schema};
use crate::tools::{py_err, SchemaDict};
use pyo3::prelude::*;
use pyo3::exceptions::{PyAssertionError};

fn make_definitions_result<'s, 'py: 's>(py: Python<'py>, schema: &'s PyDict, definitions: &PyList) -> PyResult<&'s PyDict> {
    if definitions.len() > 0 {
        let definitions_schema = PyDict::new(py);
        definitions_schema.set_item(intern!(py, "type"), intern!(py, "definitions"))?;
        definitions_schema.set_item(intern!(py, "schema"), schema)?;
        definitions_schema.set_item(intern!(py, "definitions"), definitions)?;
        Ok(definitions_schema)
    } else {
        Ok(schema)
    }
}

fn _collect_refs_and_clone<'s, 'py: 's>(py: Python<'py>, schema: &'s PyDict) -> PyResult<(&'s PyDict, &'s PyDict)> {
    struct V<'v> { valid_defs: &'v PyDict, invalid_defs: &'v PyDict }
    impl<'v> CoreSchemaVisitor<()> for V<'v> {
        const SET_SCHEMA_KEYS: bool = true;

        fn visit<'d, 'py: 'd>(&self, py: Python<'py>, schema: &'d PyDict, context: &mut ()) -> PyResult<&'d PyDict> {
            let type_: &str = schema.get_as_req::<&str>(intern!(py, "type"))?;
            let new_schema = if type_ == "definitions" {
                let definitions: &PyList = schema.get_as_req(intern!(py, "definitions"))?;
                for v in definitions {
                    let definition: &PyDict = v.downcast()?;
                    let ref_: &PyString = definition.get_as_req(intern!(py, "ref"))?;
                    let def_schema = self.visit(py, definition, context)?.copy()?;
                    if invalid_schema(py, schema)? {
                        self.invalid_defs.set_item(ref_, def_schema)?;
                    } else {
                        self.valid_defs.set_item(ref_, def_schema)?;
                    }
                }
                schema.get_as_req::<&PyDict>(intern!(py, "schema"))?.copy()?
            } else {
                let res = schema.copy()?;
                if let Some(ref_) = res.get_as::<&PyString>(intern!(py, "ref"))? {
                    if invalid_schema(py, res)? {
                        self.invalid_defs.set_item(ref_, res)?;
                    } else {
                        self.valid_defs.set_item(ref_, res)?;
                    }
                }
                res
            };
            self.recurse(py, new_schema, context)?;
            Ok(new_schema)
        }
    }

    let visitor = V { valid_defs: PyDict::new(py), invalid_defs: PyDict::new(py) };
    let new_schema = visitor.visit(py, schema, &mut ())?;

    visitor.invalid_defs.update(visitor.valid_defs.as_mapping())?;
    Ok((new_schema, visitor.invalid_defs))
}

#[pyfunction]
pub fn collect_refs<'s, 'py: 's>(py: Python<'py>, schema: &'s PyDict) -> PyResult<&'s PyDict> {
    let (new_schema, all_defs) = _collect_refs_and_clone(py, schema)?;
    let res = PyDict::new(py);
    res.set_item("schema", new_schema)?;
    res.set_item("all_defs", all_defs)?;
    Ok(res)
}

fn _flatten_refs<'s, 'py: 's>(
    py: Python<'py>, schema: &'s PyDict, all_defs: &'s PyDict
) -> PyResult<()> {
    struct V<'d> { all_defs: &'d PyDict }
    impl<'d> CoreSchemaVisitor<()> for V<'d> {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s PyDict, context: &mut ()) -> PyResult<&'s PyDict> {
            if schema.get_as_req::<&PyString>(intern!(py, "type"))?.to_str()? == "definitions" {
                return py_err!(PyAssertionError; "Should not encounter definitions as they are handled by collect_refs");
            }

            self.recurse(py, schema, context)?;

            if let Some(ref_) = schema.get_as::<&PyString>(intern!(py, "ref"))? {
                if self.all_defs.contains(ref_)? {
                    self.all_defs.set_item(ref_, schema.copy()?)?;

                    schema.clear();
                    schema.set_item(intern!(py, "type"), intern!(py, "definition-ref"))?;
                    schema.set_item(intern!(py, "schema_ref"), ref_)?;
                }
            }
            Ok(schema)
        }
    }

    let visitor = V { all_defs };
    visitor.visit(py, schema, &mut ())?;
    Ok(())
}

#[pyfunction]
pub fn flatten_refs<'s, 'py: 's>(py: Python<'py>, schema: &'s PyDict, all_defs: &'s PyDict) -> PyResult<&'s PyDict> {
    _flatten_refs(py, schema, all_defs)?;
    Ok(schema)
}

#[derive(Default)]
struct RefCounts {
    ref_count: u32,
    involved_in_recursion: bool,
    recursion_ref_count: u32,
}

fn count_refs<'s, 'py: 's>(
    py: Python<'py>, schema: &'s PyDict, all_defs: &'s PyDict
) -> PyResult<HashMap<String, RefCounts>> {
    struct C {
        ref_counts: HashMap<String, RefCounts>,
    }
    struct V<'d> { all_defs: &'d PyDict }
    impl<'d> CoreSchemaVisitor<C> for V<'d> {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s PyDict, context: &mut C) -> PyResult<&'s PyDict> {
            if schema.get_as_req::<&PyString>(intern!(py, "type"))?.to_str()? != "definition-ref" {
                self.recurse(py, schema, context)?;
                return Ok(schema);
            }

            let ref_str: &PyString = schema.get_as_req(intern!(py, "schema_ref"))?;
            let ref_: &str = ref_str.extract()?;
            let refs = context.ref_counts.entry(ref_.to_string()).or_insert_with(|| RefCounts::default());
            refs.ref_count += 1;

            if refs.recursion_ref_count != 0 {
                refs.involved_in_recursion = true;
                Ok(schema)
            } else {
                refs.recursion_ref_count += 1;
                self.visit(py, self.all_defs.get_as_req(ref_str)?, context)?;
                context.ref_counts.get_mut(ref_).unwrap().recursion_ref_count -= 1;
                Ok(schema)
            }
        }
    }

    let mut context = C { ref_counts: HashMap::new() };
    V { all_defs }.visit(py, schema, &mut context)?;

    for v in context.ref_counts.values() {
        if v.recursion_ref_count != 0 {
            return py_err!(PyAssertionError; "this is a bug! please report it")
        }
    }

    Ok(context.ref_counts)
}

fn inline_refs<'s, 'py: 's>(
    py: Python<'py>, schema: &'s PyDict, all_defs: &'s PyDict, ref_counts: &mut HashMap<String, RefCounts>
) -> PyResult<()> {
    struct C<'m> {
        ref_counts: &'m mut HashMap<String, RefCounts>,
    }
    struct V<'m> {
        all_defs: &'m PyDict,
    }
    impl<'m> CoreSchemaVisitor<C<'m>> for V<'m> {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s PyDict, context: &mut C<'m>) -> PyResult<&'s PyDict> {
            if schema.get_as_req::<&PyString>(intern!(py, "type"))?.to_str()? == "definition-ref" {
                let ref_str: &PyString = schema.get_as_req(intern!(py, "schema_ref"))?;
                let ref_: &str = ref_str.extract()?;
                // Check if the reference is only used once and not involved in recursion
                let c = context.ref_counts.entry(ref_.to_string()).or_insert_with(|| RefCounts::default());
                if c.ref_count <= 1 && !c.involved_in_recursion {
                    let serialization = schema.get_item(intern!(py, "serialization"));

                    // Inline the reference by replacing the reference with the actual schema
                    schema.clear();
                    schema.update(self.all_defs.pop::<&PyDict>(ref_str)?.as_mapping())?;
                    schema.del_item(intern!(py, "ref"))?;
                    // put all other keys that were on the def-ref schema into the inlined version
                    // in particular this is needed for `serialization`
                    if let Some(ser_dict) = serialization {
                        schema.set_item("serialization", ser_dict)?;
                    }

                    c.ref_count -= 1; // because we just replaced it!
                }
            }
            self.recurse(py, schema, context)?;
            Ok(schema)
        }
    }

    let mut context = C { ref_counts };
    V { all_defs }.visit(py, schema, &mut context)?;
    Ok(())
}

#[pyfunction]
pub fn simplify_schema_references<'s, 'py: 's>(py: Python<'py>, schema: &'s PyDict, inline: bool) -> PyResult<&'s PyDict> {
    let (new_schema, all_defs) = _collect_refs_and_clone(py, schema)?;

    _flatten_refs(py, new_schema, &all_defs)?;

    for k in all_defs.keys() {
        let def: &PyDict = all_defs.get_as_req(k.downcast()?)?;
        _flatten_refs(py, def, &all_defs)?;
    }

    if inline {
        let mut ref_counts = count_refs(py, new_schema, &all_defs)?;
        inline_refs(py, new_schema, &all_defs, &mut ref_counts)?;

        let res_defs = PyList::empty(py);
        for (_, v) in all_defs {
            let ref_ = v.downcast::<PyDict>()?.get_as_req::<&str>(intern!(py, "ref"))?;
            if let Some(c) = ref_counts.get(ref_) {
                if c.ref_count > 0 {
                    res_defs.append(v)?;
                }
            }
        }
        make_definitions_result(py, new_schema, res_defs)
    } else {
        make_definitions_result(py, new_schema, all_defs.values())
    }
}
