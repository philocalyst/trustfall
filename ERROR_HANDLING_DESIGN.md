# Track A — First-class adapter error handling (working design)

Status: in progress on branch `feat/adapter-error-handling`.
This file is the shared source of truth while the change is in flight. Delete before merge.

## Goal

Give adapters a first-class, fail-fast error channel without forcing infallible
adapters (`BasicAdapter`) to ever write `Result`/`Ok`, and without threading
`Result` through the hot interpreter internals.

## Core mechanism (the "why it's surgical")

Every engine internal (`EdgeExpander`, `RecursiveEdgeExpander`, `collect_fold_elements`,
the fns in `execution.rs` / `filtering.rs` / `hints/dynamic.rs`) consumes **plain,
non-`Result`** iterators. So we:

1. Make the public `Adapter` trait **fallible**.
2. Keep an internal `pub(crate) trait RawAdapter` with today's **infallible** signatures.
   The engine fns swap their bound `Adapter` -> `RawAdapter`; their bodies are unchanged.
3. Wrap the user's fallible `Adapter` in `ErrorTrackingAdapter`, which implements
   `RawAdapter` by draining the first `Err` into a shared slot and *fusing* the iterator.
4. The outermost results iterator from `interpret_ir` checks the slot after each pull;
   if set, it discards the in-flight row, yields exactly one `Err`, then `None` forever.

Pairing invariant (adapter yields exactly one outcome per input context, in order) is
preserved because for `resolve_neighbors` we fuse the **inner** vertex iterator, leaving
the outer `(context, neighbors)` stream 1:1.

## Exact public trait shape

```rust
pub trait Adapter<'vertex> {
    type Vertex: Clone + Debug + 'vertex;

    /// Infallible adapters set this to `std::convert::Infallible`.
    type Error: std::error::Error + 'static;

    fn resolve_starting_vertices(
        &self, edge_name: &Arc<str>, parameters: &EdgeParameters, resolve_info: &ResolveInfo,
    ) -> VertexIterator<'vertex, Result<Self::Vertex, Self::Error>>;

    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self, contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>, property_name: &Arc<str>, resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, Result<FieldValue, Self::Error>>;

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self, contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>, edge_name: &Arc<str>, parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<
        'vertex, V,
        VertexIterator<'vertex, Result<Self::Vertex, Self::Error>>, // per-neighbor fallible
    >;

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self, contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>, coerce_to_type: &Arc<str>, resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, Result<bool, Self::Error>>;
}
```

Note the shape: property/coercion carry `Result` in the *outcome slot*; neighbors carry
`Result` on each *neighbor vertex* (outer stream stays 1:1); starting vertices are bare
`Result<Vertex, Error>`.

`Error: 'static + std::error::Error` only — **no `Send`/`Sync`** on the base trait so
`trustfall_wasm` (error wraps `JsValue`, `!Send`) still complies. The `trustfall::execute_query`
facade adds `where A::Error: Send + Sync + 'static` for its `anyhow` conversion.

## Mechanical rule for INFALLIBLE direct `Adapter` impls (numbers, filesystem, etc.)

- add `type Error = std::convert::Infallible;`
- `resolve_starting_vertices`: wrap the produced vertices in `Ok`
  (`.map(Ok)` on the iterator, or return `Box::new(inner.map(Ok))`).
- `resolve_property` / `resolve_coercion`: map the outcome `(ctx, v)` -> `(ctx, Ok(v))`.
- `resolve_neighbors`: map `(ctx, neighbors)` -> `(ctx, Box::new(neighbors.map(Ok)))`.

`BasicAdapter` implementors: **no change** — the blanket impl does the `Ok`-wrapping.

## `interpret_ir` new signature

```rust
pub fn interpret_ir<'query, AdapterT: Adapter<'query> + 'query>(
    adapter: Arc<AdapterT>, indexed_query: Arc<IndexedQuery>,
    arguments: Arc<BTreeMap<Arc<str>, FieldValue>>,
) -> Result<
    Box<dyn Iterator<Item = Result<
        BTreeMap<Arc<str>, FieldValue>,
        ExecutionError<AdapterT::Error>,
    >> + 'query>,
    QueryArgumentsError,
>
```

## `ExecutionError`

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutionError<E: std::error::Error + 'static> {
    #[error("the adapter reported an error during query execution: {0}")]
    Adapter(E),
}
```
`#[non_exhaustive]` so engine-detected contract violations (today's panics) can migrate in later.

## Facade `trustfall::execute_query`

Item type becomes `Result<BTreeMap<Arc<str>, FieldValue>, anyhow::Error>`; requires
`A::Error: Send + Sync + 'static`; maps `ExecutionError<E>` through `anyhow`.

## Test-callsite migration

`interpret_ir(...).unwrap()` sites that then `.collect()` / iterate must handle the new
`Result` item — for infallible adapters use `.map(|r| r.expect("infallible adapter"))`
(or `.map(Result::unwrap)`).

## Out of scope for Track A (do NOT do here)

Async, `FallibleBasicAdapter`, `try_resolve_*` helpers, stubgen `--async`, bindings rewrite.
Those are later phases. Track A only makes the trait fallible + threads errors to the caller.
