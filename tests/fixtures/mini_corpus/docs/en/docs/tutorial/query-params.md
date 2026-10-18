# Query Parameters { #query-parameters }

When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.

{* ../../docs_src/query_params/tutorial001_py310.py hl[9] *}

The query is the set of key-value pairs that go after the `?` in a URL, separated by `&` characters.

## Defaults { #defaults }

As query parameters are not a fixed part of a path, they can be optional and can have default values.

| Parameter | Type | Default |
|-----------|------|---------|
| `skip` | `int` | `0` |
| `limit` | `int` | `10` |

/// tip

You could also use `Enum`s the same way as with [Path Parameters](path-params.md#predefined-values).

///
