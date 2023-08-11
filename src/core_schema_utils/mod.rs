mod walk_core_schema;
mod simplify_schema;

pub use self::walk_core_schema::collect_definitions;
pub use self::walk_core_schema::collect_ref_names;
pub use self::walk_core_schema::collect_invalid_schemas;
pub use self::walk_core_schema::apply_discriminators;
pub use self::simplify_schema::simplify_schema_references;

pub use self::simplify_schema::collect_refs;
pub use self::simplify_schema::flatten_refs;
