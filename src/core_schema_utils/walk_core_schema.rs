use crate::tools::SchemaDict;
use pyo3::{intern, PyResult, Python};
use pyo3::types::{PyDict, PyList, PySet, PyString};
use pyo3::prelude::*;

pub trait CoreSchemaVisitor<C> {
    const SET_SCHEMA_KEYS: bool;

    fn visit<'s, 'py: 's>(
        &self,
        py: Python<'py>,
        schema: &'s Bound<'py, PyDict>,
        context: &mut C,
    ) -> PyResult<Option<Bound<'py, PyDict>>>;

    fn recurse<'py>(
        &self,
        py: Python<'py>,
        schema: &Bound<'py, PyDict>,
        context: &mut C,
    ) -> PyResult<()> {
        let type_: Bound<'_, PyString> = schema.get_as_req(intern!(py, "type"))?;
        match type_.to_str()? {
            "definitions" => self.handle_definitions_schema(py, schema, context)?,
            "list" => self.walk_dict(py, intern!(py, "items_schema"), schema, context)?,
            "set" => self.walk_dict(py, intern!(py, "items_schema"), schema, context)?,
            "frozenset" => self.walk_dict(py, intern!(py, "items_schema"), schema, context)?,
            "generator" => self.walk_dict(py, intern!(py, "items_schema"), schema, context)?,
            "tuple-variable" => self.walk_dict(py, intern!(py, "items_schema"), schema, context)?,
            "tuple-positional" => {
                self.walk_list(py, intern!(py, "items_schema"), schema, context)?;
                self.walk_dict(py, intern!(py, "extra_schema"), schema, context)?;
            },
            "dict" => {
                self.walk_dict(py, intern!(py, "keys_schema"), schema, context)?;
                self.walk_dict(py, intern!(py, "values_schema"), schema, context)?;
            },
            "function" => self.walk_dict(py, intern!(py, "schema"), schema, context)?,
            "union" => self.walk_list(py, intern!(py, "choices"), schema, context)?,
            "tagged-union" => self.handle_tagged_union_schema(py, schema, context)?,
            "chain" => self.walk_list(py, intern!(py, "steps"), schema, context)?,
            "lax-or-strict" => {
                self.walk_dict(py, intern!(py, "lax_schema"), schema, context)?;
                self.walk_dict(py, intern!(py, "strict_schema"), schema, context)?;
            },
            "json-or-python" => {
                self.walk_dict(py, intern!(py, "json_schema"), schema, context)?;
                self.walk_dict(py, intern!(py, "python_schema"), schema, context)?;
            },
            "model-fields" => {
                self.walk_dict(py, intern!(py, "extra_validator"), schema, context)?;
                self.walk_computed_fields(py, schema, context)?;
                self.walk_model_fields(py, schema, context)?;
            },
            "typed-dict" => {
                self.walk_dict(py, intern!(py, "extra_validator"), schema, context)?;
                self.walk_computed_fields(py, schema, context)?;
                self.walk_model_fields(py, schema, context)?;
            },
            "dataclass-args" => self.handle_dataclass_args_schema(py, schema, context)?,
            "arguments" => self.handle_arguments_schema(py, schema, context)?,
            "call" => {
                self.walk_dict(py, intern!(py, "arguments_schema"), schema, context)?;
                self.walk_dict(py, intern!(py, "return_schema"), schema, context)?;
            },
            _ => self.walk_dict(py, intern!(py, "schema"), schema, context)?,
        };

        if let Some(ser_schema) = schema.get_as::<Bound<'_, PyDict>>(intern!(py, "serialization"))? {
            if Self::SET_SCHEMA_KEYS {
                let new_ser_schema = ser_schema.copy()?;
                self.walk_dict(py, intern!(py, "schema"), &new_ser_schema, context)?;
                self.walk_dict(py, intern!(py, "return_schema"), &new_ser_schema, context)?;
                schema.set_item(intern!(py, "serialization"), new_ser_schema)?;
            } else {
                self.walk_dict(py, intern!(py, "schema"), &ser_schema, context)?;
                self.walk_dict(py, intern!(py, "return_schema"), &ser_schema, context)?;
            };
        }

        Ok(())
    }

    #[inline]
    fn walk_dict(&self, py: Python, key: &Bound<'_, PyString>, schema: &Bound<'_, PyDict>, context: &mut C) -> PyResult<()> {
        if let Some(dict) = schema.get_as::<Bound<'_, PyDict>>(key)? {
            let new_dict = self.visit(py, &dict, context)?;
            if Self::SET_SCHEMA_KEYS {
                if let Some(r) = new_dict {
                    schema.set_item(key, r)?;
                }
            }
        }
        Ok(())
    }

    fn walk_list(&self, py: Python, key: &Bound<'_, PyString>, schema: &Bound<'_, PyDict>, context: &mut C) -> PyResult<()> {
        let items: Bound<'_, PyList> = schema.get_as_req(key)?;
        if Self::SET_SCHEMA_KEYS {
            let new_items = PyList::empty_bound(py);
            for item in items {
                let new_item = self.visit(py, item.downcast()?, context)?;
                if let Some(r) = new_item {
                    new_items.append(r)?;
                }
            }
            schema.set_item(key, new_items)?;
        } else {
            for item in items {
                self.visit(py, item.downcast()?, context)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn walk_field<'d, 'py: 'd>(&self, py: Python<'py>, key: &Bound<'py, PyString>, field: Bound<'py, PyDict>, context: &mut C) -> PyResult<Bound<'py, PyDict>> {
        let field_schema: Bound<'_, PyDict> = (&field).get_as_req(key)?;
        let res_schema = self.visit(py, &field_schema, context)?;
        if Self::SET_SCHEMA_KEYS {
            if let Some(r) = res_schema {
                let new_field = field.copy()?;
                new_field.set_item(key, r)?;
                Ok(new_field)
            } else {
                Ok(field)
            }
        } else {
            Ok(field)
        }
    }

    fn walk_computed_fields(&self, py: Python, schema: &Bound<'_, PyDict>, context: &mut C) -> PyResult<()> {
        let fields_key = intern!(py, "computed_fields");
        let schema_key = intern!(py, "return_schema");
        if let Some(computed_fields) = schema.get_as::<Bound<'_, PyList>>(fields_key)? {
            if Self::SET_SCHEMA_KEYS {
                let new_fields = PyList::empty_bound(py);
                for v in computed_fields {
                    let new_field = self.walk_field(py, schema_key, v.extract()?, context)?;
                    new_fields.append(new_field)?;
                }
                schema.set_item(fields_key, new_fields)?;
            } else {
                for v in computed_fields {
                    self.walk_field(py, schema_key, v.extract()?, context)?;
                }
            }
        }
        Ok(())
    }

    fn walk_model_fields(&self, py: Python, schema: &Bound<'_, PyDict>, context: &mut C) -> PyResult<()> {
        let fields_key = intern!(py, "fields");
        let schema_key = intern!(py, "schema");
        let fields: Bound<'_, PyDict> = schema.get_as_req(fields_key)?;
        if Self::SET_SCHEMA_KEYS {
            let new_fields = PyDict::new_bound(py);
            for (k, v) in fields.iter() {
                let new_field = self.walk_field(py, schema_key, v.extract()?, context)?;
                new_fields.set_item(k, new_field)?;
            }
            schema.set_item(fields_key, new_fields)?;
        } else {
            for (_k, v) in fields.iter() {
                self.walk_field(py, schema_key, v.extract()?, context)?;
            }
        }
        Ok(())
    }

    fn handle_definitions_schema(
        &self,
        py: Python,
        schema: &Bound<'_, PyDict>,
        context: &mut C,
    ) -> PyResult<()> {
        let mut schema_definitions: Bound<'_, PyList> = schema.get_as_req(intern!(py, "definitions"))?;

        if Self::SET_SCHEMA_KEYS {
            let new_schema_definitions = PyList::empty_bound(py);
            for v in &schema_definitions {
                let updated_definition = self.visit(py, v.downcast()?, context)?;
                if let Some(r) = updated_definition {
                    if r.contains(intern!(py, "ref"))? {
                        // If the updated definition schema doesn't have a 'ref', it shouldn't go in the definitions
                        // This is most likely to happen due to replacing something with a definition reference, in
                        // which case it should certainly not go in the definitions list
                        new_schema_definitions.append(r)?
                    }
                }
            }
            schema_definitions = new_schema_definitions
        } else {
            for definition in &schema_definitions {
                let def = self.visit(py, definition.downcast()?, context)?;
                if let Some(r) = def {
                    if !r.contains(intern!(py, "ref"))? {
                        panic!("asdasd");
                    }
                }
            }
        }

        let inner_schema: Bound<'_, PyDict> = schema.get_as_req(intern!(py, "schema"))?;
        let new_inner_schema = self.visit(py, &inner_schema, context)?;
        if Self::SET_SCHEMA_KEYS {
            schema.set_item(intern!(py, "definitions"), schema_definitions)?;
            if let Some(r) = new_inner_schema {
                schema.set_item(intern!(py, "schema"), r)?;
            }
        }

        return Ok(());
    }

    fn handle_tagged_union_schema(
        &self, py: Python, schema: &Bound<'_, PyDict>, context: &mut C,
    ) -> PyResult<()> {
        let choices: Bound<'_, PyDict> = schema.get_as_req(intern!(py, "choices"))?;

        if Self::SET_SCHEMA_KEYS {
            let new_choices = PyDict::new_bound(py);
            for (k, v) in choices.iter() {
                if let Ok(choice) = v.downcast::<PyDict>() {
                    if let Some(r) = self.visit(py, choice, context)? {
                        new_choices.set_item(k, r)?;
                    }
                } else {
                    new_choices.set_item(k, v)?;
                }
            }
            schema.set_item(intern!(py, "choices"), new_choices)?;
        } else {
            for (_k, v) in choices.iter() {
                if let Ok(choice) = v.downcast::<PyDict>() {
                    self.visit(py, choice, context)?;
                }
            }
        }
        return Ok(());
    }

    fn handle_dataclass_args_schema(
        &self, py: Python, schema: &Bound<'_, PyDict>, context: &mut C,
    ) -> PyResult<()> {
        self.walk_computed_fields(py, schema, context)?;

        let fields_key = intern!(py, "fields");
        let schema_key = intern!(py, "schema");
        let fields: Bound<'_, PyList> = schema.get_as_req(fields_key)?;
        if Self::SET_SCHEMA_KEYS {
            let new_fields = PyList::empty_bound(py);
            for v in fields {
                let new_field = self.walk_field(py, schema_key, v.extract()?, context)?;
                new_fields.append(new_field)?;
            }
            schema.set_item(fields_key, new_fields)?;
        } else {
            for v in fields {
                self.walk_field(py, schema_key, v.extract()?, context)?;
            }
        }

        return Ok(());
    }

    fn handle_arguments_schema(
        &self, py: Python, schema: &Bound<'_, PyDict>, context: &mut C,
    ) -> PyResult<()> {
        let args_key = intern!(py, "arguments_schema");
        let schema_key = intern!(py, "schema");
        let arguments_schema: Bound<'_, PyList> = schema.get_as_req(args_key)?;
        if Self::SET_SCHEMA_KEYS {
            let new_args = PyList::empty_bound(py);
            for v in arguments_schema {
                let new_arg = self.walk_field(py, schema_key, v.extract()?, context)?;
                new_args.append(new_arg)?;
            }
            schema.set_item(args_key, new_args)?;
        } else {
            for v in arguments_schema {
                self.walk_field(py, schema_key, v.extract()?, context)?;
            }
        }

        self.walk_dict(py, intern!(py, "var_args_schema"), schema, context)?;
        self.walk_dict(py, intern!(py, "var_kwargs_schema"), schema, context)?;
        return Ok(());
    }
}

#[pyfunction]
pub fn collect_definitions<'py>(py: Python<'py>, schema: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    struct C<'p> { defs: Bound<'p, PyDict> }
    struct V {}
    impl<'p> CoreSchemaVisitor<C<'p>> for V {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s Bound<'py, PyDict>, context: &mut C) -> PyResult<Option<Bound<'py, PyDict>>> {
            if let Some(r) = schema.get_as::<Bound<'_, PyString>>(intern!(py, "ref"))? {
                context.defs.set_item(r, schema)?;
            }
            self.recurse(py, schema, context)?;
            Ok(None)
        }
    }

    let mut context = C { defs: PyDict::new_bound(py) };
    V {}.visit(py, schema, &mut context)?;
    Ok(context.defs)
}

#[pyfunction]
pub fn collect_ref_names<'py>(py: Python<'py>, schema: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PySet>> {
    struct C<'a, 'p: 'a> { refs: &'a Bound<'p, PySet> }
    struct V {}
    impl<'a, 'p> CoreSchemaVisitor<C<'a, 'p>> for V {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s Bound<'py, PyDict>, context: &mut C) -> PyResult<Option<Bound<'py, PyDict>>> {
            if let Some(r) = schema.get_as::<Bound<'_, PyString>>(intern!(py, "ref"))? {
                context.refs.add(r)?;
            }
            self.recurse(py, schema, context)?;
            Ok(None)
        }
    }

    let refs = PySet::empty_bound(py)?;
    let mut context = C { refs: &refs };
    V {}.visit(py, schema, &mut context)?;
    Ok(refs)
}


#[inline]
pub fn invalid_schema<'py>(py: Python<'py>, schema: &Bound<'py, PyDict>) -> PyResult<bool> {
    if let Some(meta) = schema.get_as::<Bound<'_, PyDict>>(intern!(py, "metadata"))? {
        Ok(meta.contains(intern!(py, "invalid"))?)
    } else {
        Ok(false)
    }
}


#[pyfunction]
pub fn collect_invalid_schemas<'py, 'd: 'py>(py: Python<'py>, schema: &'d Bound<'py, PyDict>) -> PyResult<Bound<'py, PyList>> {
    struct C<'a, 'p: 'a> { invalid_schemas: &'a Bound<'p, PyList> }
    struct V {}
    impl<'a, 'p> CoreSchemaVisitor<C<'a, 'p>> for V {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s Bound<'py, PyDict>, context: &mut C) -> PyResult<Option<Bound<'py, PyDict>>> {
            if invalid_schema(py, schema)? {
                context.invalid_schemas.append(schema)?;
            }
            self.recurse(py, schema, context)?;
            Ok(None)
        }
    }

    let invalid_schemas = PyList::empty_bound(py);
    let mut context = C { invalid_schemas: &invalid_schemas };
    V {}.visit(py, schema, &mut context)?;
    Ok(invalid_schemas)
}

#[pyfunction]
pub fn apply_discriminators<'py>(py: Python<'py>, schema: &Bound<'py, PyDict>, apply_callback: PyObject) -> PyResult<()> {
    struct V { callback: PyObject }
    impl CoreSchemaVisitor<()> for V {
        const SET_SCHEMA_KEYS: bool = false;

        fn visit<'s, 'py: 's>(&self, py: Python<'py>, schema: &'s Bound<'py, PyDict>, _: &mut ()) -> PyResult<Option<Bound<'py, PyDict>>> {
            self.recurse(py, schema, &mut ())?;

            if schema.get_as_req::<Bound<'_, PyString>>(intern!(py, "type"))?.to_str()? == "tagged-union" {
                return Ok(None)
            }

            if let Some(metadata) = schema.get_as::<Bound<'_, PyDict>>(intern!(py, "metadata"))? {
                if let Some(discriminator) = metadata.get_as::<Bound<'_, PyString>>(intern!(py, "pydantic.internal.union_discriminator"))? {
                    let new_schema = self.callback.call1(py, (schema, discriminator,))?;
                    schema.clear();
                    schema.update(new_schema.downcast_bound::<PyDict>(py)?.as_mapping())?;
                }
            }
            Ok(None)
        }
    }

    V { callback: apply_callback }.visit(py, schema, &mut ())?;
    Ok(())
}
