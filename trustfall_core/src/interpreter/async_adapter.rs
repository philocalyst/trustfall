//! Stream-based adapter traits.
//!
//! [`AsyncAdapter`] is the asynchronous counterpart of [`Adapter`](super::Adapter). It replaces
//! lazy [`Iterator`]s with lazy [`Stream`]s, allowing adapters to suspend for I/O while preserving
//! context order. The traits neither require a particular runtime nor impose `Send`.

use std::{fmt::Debug, pin::Pin, sync::Arc};

use futures_core::Stream;

use crate::ir::{EdgeParameters, FieldValue};

use super::{AsVertex, DataContext, ResolveEdgeInfo, ResolveInfo};

/// A pinned, boxed [`Stream`] of `T`.
///
/// The interpreter owns the stream's lifetime. Adapters normally create this with `Box::pin()`.
pub type VertexStream<'vertex, VertexT> = Pin<Box<dyn Stream<Item = VertexT> + 'vertex>>;

/// Query contexts passed to an async adapter resolver.
///
/// The interpreter handles errors from earlier stages, so a resolver receives only successful
/// contexts. It may still return errors for the resolution it performs itself.
pub type ContextStream<'vertex, VertexT> = VertexStream<'vertex, DataContext<VertexT>>;

/// `(context, outcome)` pairs returned by an async adapter resolver.
///
/// Each input context must produce one pair, in input order. The outcome may contain the adapter
/// error for that context; use [`NeighborResolutionStream`] for the two error layers of an edge.
pub type ContextOutcomeStream<'vertex, VertexT, OutcomeT> =
    VertexStream<'vertex, (DataContext<VertexT>, OutcomeT)>;

/// An edge resolution: a stream of fallible neighbors or a context-level error.
///
/// The outer error means the edge could not be resolved for this context. The inner stream lets
/// an adapter report an error while producing a particular neighbor. Both end query execution.
pub type NeighborResolutionStream<'vertex, VertexT, ErrorT> =
    Result<VertexStream<'vertex, Result<VertexT, ErrorT>>, ErrorT>;

/// An asynchronous data provider for streaming query execution.
///
/// It has the same resolver contract as [`Adapter`](super::Adapter), with streams in place of
/// iterators. Resolver methods are called while the query result stream is polled; they should
/// preserve context order and remain lazy where their data source permits it.
///
/// Implement [`AsyncBasicAdapter`](super::async_basic_adapter::AsyncBasicAdapter) when its
/// smaller, infallible per-vertex API is sufficient.
pub trait AsyncAdapter<'vertex> {
    /// The type of vertices this adapter queries.
    type Vertex: Clone + Debug + 'vertex;

    /// The error type this adapter may report.
    type Error: std::error::Error + 'static;

    /// Resolve a starting edge into vertices.
    ///
    /// Each item becomes one root query context. An error terminates the query result stream.
    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexStream<'vertex, Result<Self::Vertex, Self::Error>>;

    /// Resolve a property for each input context.
    ///
    /// Return one outcome for every input context, in the same order. Contexts whose active vertex
    /// is absent because of `@optional` must resolve this property as [`FieldValue::Null`].
    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextStream<'vertex, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeStream<'vertex, V, Result<FieldValue, Self::Error>>;

    /// Resolve an edge for each input context.
    ///
    /// Return one outcome for every input context, in the same order. For a missing optional
    /// vertex, return an empty neighbor stream. The outer `Result` reports failure to resolve the
    /// edge as a whole; items in the inner stream report failures encountered among its neighbors.
    #[allow(clippy::type_complexity)]
    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextStream<'vertex, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeStream<
        'vertex,
        V,
        NeighborResolutionStream<'vertex, Self::Vertex, Self::Error>,
    >;

    /// Test each input vertex for a requested subtype.
    ///
    /// Return one outcome for every input context, in the same order. A missing optional vertex
    /// must resolve to `false`; the interpreter preserves it as an absent optional context.
    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextStream<'vertex, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeStream<'vertex, V, Result<bool, Self::Error>>;
}
