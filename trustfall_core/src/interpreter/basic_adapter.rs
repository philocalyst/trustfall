use std::fmt::Debug;

use crate::ir::{EdgeParameters, FieldValue};

use super::{
    Adapter, AsVertex, ContextIterator, ContextOutcomeIterator, NeighborResolution,
    ResolveEdgeInfo, ResolveInfo, Typename, VertexIterator, helpers::resolve_property_with,
};

/// A simplified variant of the [`Adapter`] trait.
///
/// Implementing `BasicAdapter` provides a "free" [`Adapter`] implementation.
/// `BasicAdapter` gives up a bit of [`Adapter`]'s flexibility in exchange for being
/// as simple as possible to implement:
/// - `&str` instead of `&Arc<str>` for all names of types, properties, and edges.
/// - Simplified function signatures, with only the minimum necessary arguments.
/// - Automatic handling of the `__typename` special property.
///
/// The easiest way to implement this trait is with the `Vertex` associated type set
/// to an enum that is `#[derive(Debug, Clone, TrustfallEnumVertex)]`.
///
/// # Errors
///
/// This trait carries the same [`Adapter::Error`] channel as [`Adapter`] itself.
/// Adapters that cannot fail should declare `type Error = std::convert::Infallible`:
/// combined with the [`resolve_property_with`]-family of helpers (whose error type is
/// inferred from these signatures), infallible implementations never write `Result`
/// or `Ok` — the helpers produce the fallible outcomes on their behalf.
pub trait BasicAdapter<'vertex> {
    /// The type of vertices in the dataset this adapter queries.
    /// It's frequently a good idea to use an Rc<...> type for cheaper cloning here.
    type Vertex: Typename + Clone + Debug + 'vertex;

    /// The error type this adapter may report. See [`Adapter::Error`].
    ///
    /// Infallible adapters set this to [`std::convert::Infallible`].
    type Error: std::error::Error + 'static;

    /// Produce an iterator of vertices for the specified starting edge.
    ///
    /// Starting edges are ones where queries are allowed to begin.
    /// They are defined directly on the root query type of the schema.
    /// For example, `User` is the starting edge of the following query:
    /// ```graphql
    /// query {
    ///     User {
    ///         name @output
    ///     }
    /// }
    /// ```
    ///
    /// The caller guarantees that:
    /// - The specified edge is a starting edge in the schema being queried.
    /// - Any parameters the edge requires per the schema have values provided.
    fn resolve_starting_vertices(
        &self,
        edge_name: &str,
        parameters: &EdgeParameters,
    ) -> VertexIterator<'vertex, Result<Self::Vertex, Self::Error>>;

    /// Resolve the value of a vertex property over an iterator of query contexts.
    ///
    /// Each [`DataContext`] in the `contexts` argument has an active vertex,
    /// which is either `None`, or a `Some(Self::Vertex)` value representing a vertex
    /// of type `type_name` defined in the schema.
    ///
    /// This method resolves the property value on that active vertex.
    ///
    /// Unlike the [`Adapter::resolve_property`] method, this method does not
    /// handle the special `__typename` property. Instead, that property is resolved
    /// by the [`BasicAdapter::resolve_typename`] method, which has a default implementation
    /// using the [`Typename`] trait implemented by `Self::Vertex`.
    ///
    /// The caller guarantees that:
    /// - `type_name` is a type or interface defined in the schema.
    /// - `property_name` is a property field on `type_name` defined in the schema.
    /// - When the active vertex is `Some(...)`, it's a vertex of type `type_name`:
    ///   either its type is exactly `type_name`, or `type_name` is an interface that
    ///   the vertex's type implements.
    ///
    /// The returned iterator must satisfy these properties:
    /// - Produce `(context, outcome)` tuples with the property's value (or an error)
    ///   for that context.
    /// - Produce contexts in the same order as the input `contexts` iterator produced them.
    /// - Produce property values whose type matches the property's type defined in the schema.
    /// - When a context's active vertex is `None`, its property outcome is `Ok(FieldValue::Null)`.
    ///
    /// [`DataContext`]: super::DataContext
    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &str,
        property_name: &str,
    ) -> ContextOutcomeIterator<'vertex, V, Result<FieldValue, Self::Error>>;

    /// Resolve the neighboring vertices across an edge, for each query context in an iterator.
    ///
    /// Each [`DataContext`](super::DataContext) in the `contexts` argument has an active vertex,
    /// which is either `None`, or a `Some(Self::Vertex)` value representing a vertex
    /// of type `type_name` defined in the schema.
    ///
    /// This method resolves the neighboring vertices for that active vertex.
    ///
    /// If the schema this adapter covers has no edges aside from starting edges,
    /// then this method will never be called and may be implemented as `unreachable!()`.
    ///
    /// The caller guarantees that:
    /// - `type_name` is a type or interface defined in the schema.
    /// - `edge_name` is an edge field on `type_name` defined in the schema.
    /// - Any parameters the edge requires per the schema have values provided.
    /// - When the active vertex is `Some(...)`, it's a vertex of type `type_name`:
    ///   either its type is exactly `type_name`, or `type_name` is an interface that
    ///   the vertex's type implements.
    ///
    /// The returned iterator must satisfy these properties:
    /// - Produce `(context, resolution)` tuples: either an iterator of (individually
    ///   fallible) neighbor vertices, or an error that failed the edge resolution for
    ///   that context as a whole.
    /// - Produce contexts in the same order as the input `contexts` iterator produced them.
    /// - Each successfully resolved neighbor is of the type specified for that edge in the schema.
    /// - When a context's active vertex is None, it has an empty neighbors iterator.
    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &str,
        edge_name: &str,
        parameters: &EdgeParameters,
    ) -> ContextOutcomeIterator<'vertex, V, NeighborResolution<'vertex, Self::Vertex, Self::Error>>;

    /// Attempt to coerce vertices to a subtype, over an iterator of query contexts.
    ///
    /// In this example query, the starting vertices of type `File` are coerced to `AudioFile`:
    /// ```graphql
    /// query {
    ///     File {
    ///         ... on AudioFile {
    ///             duration @output
    ///         }
    ///     }
    /// }
    /// ```
    /// The `... on AudioFile` operator causes only `AudioFile` vertices to be retained,
    /// filtering out all other kinds of `File` vertices.
    ///
    /// Each [`DataContext`](super::DataContext) in the `contexts` argument has an active vertex,
    /// which is either `None`, or a `Some(Self::Vertex)` value representing a vertex
    /// of type `type_name` defined in the schema.
    ///
    /// This method checks whether the active vertex is of the specified subtype.
    ///
    /// If this adapter's schema contains no subtyping, then no type coercions are possible:
    /// this method will never be called and may be implemented as `unreachable!()`.
    ///
    /// The caller guarantees that:
    /// - `type_name` is an interface defined in the schema.
    /// - `coerce_to_type` is a type or interface that implements `type_name` in the schema.
    /// - When the active vertex is `Some(...)`, it's a vertex of type `type_name`:
    ///   either its type is exactly `type_name`, or `type_name` is an interface that
    ///   the vertex's type implements.
    ///
    /// The returned iterator must satisfy these properties:
    /// - Produce `(context, outcome)` tuples showing if the coercion succeeded (or an error).
    /// - Produce contexts in the same order as the input `contexts` iterator produced them.
    /// - When a context's active vertex is `None`, its coercion outcome is `Ok(false)`.
    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &str,
        coerce_to_type: &str,
    ) -> ContextOutcomeIterator<'vertex, V, Result<bool, Self::Error>>;

    /// Resolve the `__typename` special property over an iterator of query contexts.
    ///
    /// Each [`DataContext`] in the `contexts` argument has an active vertex,
    /// which is either `None`, or a `Some(Self::Vertex)` value representing a vertex
    /// of type `type_name` defined in the schema.
    ///
    /// This method resolves the name of the type of that active vertex. That type may not always
    /// be the same as the value of the `type_name` parameter, due to inheritance in the schema.
    /// For example, consider a schema with types `interface Message` and
    /// `type Email implements Message`, and a query like the following:
    /// ```graphql
    /// query {
    ///     Message {
    ///         __typename @output
    ///     }
    /// }
    /// ```
    /// The resulting `resolve_typename()` call here would have `type_name = "Message"`.
    /// However, some of the messages read by this query may be emails!
    /// For those messages, outputting `__typename` would produce the value `"Email"`.
    ///
    /// The default implementation uses the [`Typename`] trait implemented by `Self::Vertex`
    /// to get each vertex's type name.
    ///
    /// The caller guarantees that:
    /// - `type_name` is a type or interface defined in the schema.
    /// - When the active vertex is `Some(...)`, it's a vertex of type `type_name`:
    ///   either its type is exactly `type_name`, or `type_name` is an interface that
    ///   the vertex's type implements.
    ///
    /// The returned iterator must satisfy these properties:
    /// - Produce `(context, outcome)` tuples with the property's value (or an error)
    ///   for that context.
    /// - Produce contexts in the same order as the input `contexts` iterator produced them.
    /// - Produce property values whose type matches the property's type defined in the schema.
    /// - When a context's active vertex is `None`, its property outcome is `Ok(FieldValue::Null)`.
    ///
    /// # Overriding the default implementation
    ///
    /// Some adapters may be able to implement this method more efficiently than the provided
    /// default implementation.
    ///
    /// For example: adapters having access to a [`Schema`] can use
    /// the [`interpreter::helpers::resolve_typename`](super::helpers::resolve_typename) method,
    /// which implements a "fast path" for types that don't have any subtypes per the schema.
    ///
    /// [`DataContext`]: super::DataContext
    /// [`Schema`]: crate::schema::Schema
    fn resolve_typename<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        _type_name: &str,
    ) -> ContextOutcomeIterator<'vertex, V, Result<FieldValue, Self::Error>> {
        resolve_property_with(contexts, |vertex| vertex.typename().into())
    }
}

impl<'vertex, T> Adapter<'vertex> for T
where
    T: BasicAdapter<'vertex>,
{
    type Vertex = T::Vertex;
    type Error = T::Error;

    // The `BasicAdapter` layer's outcome shapes are identical to `Adapter`'s, so this blanket
    // implementation only adapts names (`&str` -> `&Arc<str>`) and arguments (dropping the
    // resolver hints) — including dispatching the `__typename` special property.
    fn resolve_starting_vertices(
        &self,
        edge_name: &std::sync::Arc<str>,
        parameters: &EdgeParameters,
        _resolve_info: &ResolveInfo,
    ) -> VertexIterator<'vertex, Result<Self::Vertex, Self::Error>> {
        <Self as BasicAdapter>::resolve_starting_vertices(self, edge_name.as_ref(), parameters)
    }

    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &std::sync::Arc<str>,
        property_name: &std::sync::Arc<str>,
        _resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, Result<FieldValue, Self::Error>> {
        if property_name.as_ref() == "__typename" {
            self.resolve_typename(contexts, type_name.as_ref())
        } else {
            <Self as BasicAdapter>::resolve_property(
                self,
                contexts,
                type_name.as_ref(),
                property_name.as_ref(),
            )
        }
    }

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &std::sync::Arc<str>,
        edge_name: &std::sync::Arc<str>,
        parameters: &EdgeParameters,
        _resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'vertex, V, NeighborResolution<'vertex, Self::Vertex, Self::Error>>
    {
        <Self as BasicAdapter>::resolve_neighbors(
            self,
            contexts,
            type_name.as_ref(),
            edge_name.as_ref(),
            parameters,
        )
    }

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &std::sync::Arc<str>,
        coerce_to_type: &std::sync::Arc<str>,
        _resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, Result<bool, Self::Error>> {
        <Self as BasicAdapter>::resolve_coercion(
            self,
            contexts,
            type_name.as_ref(),
            coerce_to_type.as_ref(),
        )
    }
}
