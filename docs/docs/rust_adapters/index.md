# Rust Adapters

An adapter maps a Trustfall schema to data. Each resolver receives query contexts and returns one
outcome per context, in the same order. That order is part of the contract: it lets the engine
join properties, edges, filters, and folds without materializing the whole query.

Use `BasicAdapter` for most synchronous adapters. Use `Adapter` only when resolver hints or
`Arc<str>` field names are useful. The async equivalents are `AsyncBasicAdapter` and
`AsyncAdapter`.

## Error handling

Every adapter declares an associated `Error` type. Resolver outputs contain `Result` values, so
an adapter can report failures without panicking or erasing its error type.

```rust
impl<'vertex> BasicAdapter<'vertex> for CachedAdapter {
    type Vertex = Vertex;
    type Error = Infallible;

    // Implement the four resolver methods.
}
```

Set `Error = Infallible` when data access cannot fail. The resolver helpers infer that type, so
infallible resolvers can return plain values. `execute_query()` still yields `QueryResult` values;
map them through `IntoRow::into_row` to recover rows without `unwrap`.

```rust
# use trustfall::{IntoRow, QueryResult};
# use std::{collections::BTreeMap, convert::Infallible, sync::Arc};
# let rows: Vec<QueryResult<Infallible>> = Vec::new();
let rows = rows.into_iter().map(IntoRow::into_row);
```

For a fallible adapter, return its concrete error in the relevant resolver outcome. The execution
iterator yields completed rows first, then one `ExecutionError::Adapter(error)`, and ends. Parse
and argument errors remain the outer `anyhow::Result` from `execute_query()`.

## Asynchronous adapters

Enable the `async` feature and call `execute_query_async()`. The async traits use `Stream` in the
same places where the synchronous traits use `Iterator`. They preserve the same one-outcome-per-
context and input-order requirements.

```toml
[dependencies]
trustfall = { version = "0.8", features = ["async"] }
```

`AsyncBasicAdapter` is the default choice. Its resolvers use `&str` names and handle
`__typename` automatically. Use `AsyncAdapter` when you need `ResolveInfo` or `ResolveEdgeInfo`.

The async API is runtime-agnostic and does not require `Send`. Drive the returned stream on the
executor that owns the adapter. If a resolver fetches multiple contexts concurrently, preserve the
input order; the `async_helpers` module provides ordered buffered helpers for that case.

## Resolver rules

- Starting-edge resolvers return fallible vertices.
- Property and coercion resolvers return one `(context, Result<...>)` pair per input context.
- Edge resolvers return either a fallible neighbor stream or one context-level error.
- A context without an active vertex comes from a missing `@optional` edge. Return `Null` for a
  property, no neighbors for an edge, and `false` for a coercion.

The [`trustfall` examples](https://github.com/obi1kenobi/trustfall/tree/main/trustfall/examples)
show complete synchronous `BasicAdapter` implementations.
