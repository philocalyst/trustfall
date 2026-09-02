//! Stream-based adapter traits.
//!
//! [`AsyncAdapter`] is the asynchronous counterpart of [`Adapter`](super::Adapter). It replaces
//! lazy [`Iterator`]s with lazy [`Stream`]s, allowing adapters to overlap I/O while preserving
//! context order. The traits do not require a runtime or `Send`.

use std::{fmt::Debug, pin::Pin, sync::Arc};

use futures_core::Stream;

use crate::ir::{EdgeParameters, FieldValue};

use super::{AsVertex, DataContext, ResolveEdgeInfo, ResolveInfo};

/// A pinned, boxed [`Stream`] of `T`.
pub type VertexStream<'vertex, VertexT> = Pin<Box<dyn Stream<Item = VertexT> + 'vertex>>;

/// Query contexts passed to an async adapter resolver.
///
/// The engine handles upstream errors. Resolvers receive only contexts.
pub type ContextStream<'vertex, VertexT> = VertexStream<'vertex, DataContext<VertexT>>;

/// `(context, outcome)` pairs returned by an async adapter resolver.
pub type ContextOutcomeStream<'vertex, VertexT, OutcomeT> =
    Pin<Box<dyn Stream<Item = (DataContext<VertexT>, OutcomeT)> + 'vertex>>;

/// An edge resolution: a stream of fallible neighbors or a context-level error.
pub type NeighborResolutionStream<'vertex, VertexT, ErrorT> =
    Result<VertexStream<'vertex, Result<VertexT, ErrorT>>, ErrorT>;

/// An asynchronous data provider for streaming query execution.
///
/// It has the same resolver contract as [`Adapter`](super::Adapter), with streams in place of
/// iterators. Implement [`AsyncBasicAdapter`](super::async_basic_adapter::AsyncBasicAdapter) when
/// its smaller API is sufficient.
pub trait AsyncAdapter<'vertex> {
    /// The type of vertices this adapter queries.
    type Vertex: Clone + Debug + 'vertex;

    /// The error type this adapter may report.
    type Error: std::error::Error + 'static;

    /// Resolve a starting edge into vertices.
    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexStream<'vertex, Result<Self::Vertex, Self::Error>>;

    /// Resolve a property for each input context.
    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextStream<'vertex, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeStream<'vertex, V, Result<FieldValue, Self::Error>>;

    /// Resolve an edge for each input context.
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
    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextStream<'vertex, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeStream<'vertex, V, Result<bool, Self::Error>>;
}
