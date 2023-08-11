from pydantic_core import PydanticWalkCoreSchema


def test_walk():
    w: PydanticWalkCoreSchema

    def cb(s: dict) -> dict:
        print(s)
        w.recurse(s)
        return s

    s = {
        'type': 'tagged-union',
        'discriminator': 'foo',
        'from_attributes': False,
        'choices': {
            'apple': {
                'type': 'typed-dict',
                'fields': {
                    'foo': {'type': 'typed-dict-field', 'schema': {'type': 'str'}},
                    'bar': {'type': 'typed-dict-field', 'schema': {'type': 'int'}},
                },
            },
            'banana': {
                'type': 'typed-dict',
                'fields': {
                    'foo': {'type': 'typed-dict-field', 'schema': {'type': 'str'}},
                    'spam': {
                        'type': 'typed-dict-field',
                        'schema': {'type': 'list', 'items_schema': {'type': 'int'}},
                    },
                },
            },
        },
    }

    w = PydanticWalkCoreSchema(cb)
    res = w.visit(s)
    assert res == s
