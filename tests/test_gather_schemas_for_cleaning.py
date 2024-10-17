from pydantic_core import core_schema, gather_schemas_for_cleaning


def test_no_refs():
    p1 = core_schema.arguments_parameter('a', core_schema.int_schema())
    p2 = core_schema.arguments_parameter('b', core_schema.int_schema())
    schema = core_schema.arguments_schema([p1, p2])
    res = gather_schemas_for_cleaning(schema, definitions={})
    assert res['definition_refs'] == {}
    assert res['recursive_refs'] == set()
    assert res['deferred_discriminators'] == []


def test_simple_ref_schema():
    schema = core_schema.definition_reference_schema('ref1')
    definitions = {'ref1': core_schema.int_schema(ref='ref1')}

    res = gather_schemas_for_cleaning(schema, definitions)
    assert res['definition_refs'] == {'ref1': [schema]} and res['definition_refs']['ref1'][0] is schema
    assert res['recursive_refs'] == set()
    assert res['deferred_discriminators'] == []


def test_deep_ref_schema():
    class Model:
        pass

    ref11 = core_schema.definition_reference_schema('ref1')
    ref12 = core_schema.definition_reference_schema('ref1')
    ref2 = core_schema.definition_reference_schema('ref2')

    union = core_schema.union_schema([core_schema.int_schema(), (ref11, 'ref_label')])
    tup = core_schema.tuple_schema([ref12, core_schema.str_schema()])
    schema = core_schema.model_schema(
        Model,
        core_schema.model_fields_schema(
            {'a': core_schema.model_field(union), 'b': core_schema.model_field(ref2), 'c': core_schema.model_field(tup)}
        ),
    )
    definitions = {'ref1': core_schema.str_schema(ref='ref1'), 'ref2': core_schema.bytes_schema(ref='ref2')}

    res = gather_schemas_for_cleaning(schema, definitions)
    assert res['definition_refs'] == {'ref1': [ref11, ref12], 'ref2': [ref2]}
    assert res['definition_refs']['ref1'][0] is ref11 and res['definition_refs']['ref1'][1] is ref12
    assert res['definition_refs']['ref2'][0] is ref2
    assert res['recursive_refs'] == set()
    assert res['deferred_discriminators'] == []


def test_ref_in_serialization_schema():
    ref = core_schema.definition_reference_schema('ref1')
    schema = core_schema.str_schema(
        serialization=core_schema.plain_serializer_function_ser_schema(lambda v: v, return_schema=ref),
    )
    res = gather_schemas_for_cleaning(schema, definitions={'ref1': core_schema.str_schema()})
    assert res['definition_refs'] == {'ref1': [ref]} and res['definition_refs']['ref1'][0] is ref
    assert res['recursive_refs'] == set()
    assert res['deferred_discriminators'] == []


def test_recursive_ref_schema():
    ref1 = core_schema.definition_reference_schema('ref1')
    res = gather_schemas_for_cleaning(ref1, definitions={'ref1': ref1})
    assert res['definition_refs'] == {'ref1': [ref1]} and res['definition_refs']['ref1'][0] is ref1
    assert res['recursive_refs'] == {'ref1'}
    assert res['deferred_discriminators'] == []


def test_deep_recursive_ref_schema():
    ref1 = core_schema.definition_reference_schema('ref1')
    ref2 = core_schema.definition_reference_schema('ref2')
    ref3 = core_schema.definition_reference_schema('ref3')

    res = gather_schemas_for_cleaning(
        core_schema.union_schema([ref1, core_schema.int_schema()]),
        definitions={
            'ref1': core_schema.union_schema([core_schema.int_schema(), ref2]),
            'ref2': core_schema.union_schema([ref3, core_schema.float_schema()]),
            'ref3': core_schema.union_schema([ref1, core_schema.str_schema()]),
        },
    )
    assert res['definition_refs'] == {'ref1': [ref1], 'ref2': [ref2], 'ref3': [ref3]}
    assert res['recursive_refs'] == {'ref1', 'ref2', 'ref3'}
    assert res['definition_refs']['ref1'][0] is ref1
    assert res['definition_refs']['ref2'][0] is ref2
    assert res['definition_refs']['ref3'][0] is ref3
    assert res['deferred_discriminators'] == []


def test_discriminator_meta():
    class Model:
        pass

    ref1 = core_schema.definition_reference_schema('ref1')

    field1 = core_schema.model_field(core_schema.str_schema())
    field1['metadata'] = {'pydantic.internal.union_discriminator': 'foobar1'}

    field2 = core_schema.model_field(core_schema.int_schema())
    field2['metadata'] = {'pydantic.internal.union_discriminator': 'foobar2'}

    schema = core_schema.model_schema(Model, core_schema.model_fields_schema({'a': field1, 'b': ref1}))
    res = gather_schemas_for_cleaning(schema, definitions={'ref1': field2})
    assert res['definition_refs'] == {'ref1': [ref1]} and res['definition_refs']['ref1'][0] is ref1
    assert res['recursive_refs'] == set()
    assert res['deferred_discriminators'] == [(field1, 'foobar1'), (field2, 'foobar2')]
    assert res['deferred_discriminators'][0][0] is field1 and res['deferred_discriminators'][1][0] is field2
