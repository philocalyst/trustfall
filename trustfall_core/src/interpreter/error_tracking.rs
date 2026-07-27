//! Internal machinery that bridges the fallible public [`Adapter`] trait to the
//! interpreter's infallible internals.
//!
//! The public [`Adapter`] trait is fallible: its resolver methods yield `Result`s.
//! The interpreter internals (in `execution.rs`, `filtering.rs`, and `hints/dynamic.rs`),
//! however, are written against plain, infallible iterators — this keeps the hottest and
//! subtlest code in the crate free of `Result`-threading.
//!
//! The bridge is [`ErrorTrackingAdapter`]: it wraps a fallible [`Adapter`] and implements
//! the infallible [`RawAdapter`] trait that the internals are generic over. When an
//! adapter yields its first `Err`, the wrapper stashes it into a shared [`ErrorSlot`] and
//! *fuses* the offending iterator (stops yielding). The interpreter keeps running on the
//! truncated, infallible streams; the outermost results iterator (see [`surface_errors`])
//! notices the slot is set, discards the in-flight row, and yields exactly one error.

use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Deref,
    sync::{Arc, Mutex},
};

use crate::ir::{EdgeParameters, FieldValue};

use super::{
    Adapter, AsVertex, ContextIterator, ContextOutcomeIterator, ResolveEdgeInfo, ResolveInfo,
    VertexIterator, error::ExecutionError,
};

/// Shared, single-writer slot holding the first adapter error seen during execution.
///
/// Execution is single-threaded per query, but we use `Arc<Mutex<..>>` rather than
/// `Rc<Cell<..>>` so the slot stays `Send`-compatible for the eventual async core.
/// The mutex is effectively never contended, so its cost is noise next to an adapter call.
pub(crate) type ErrorSlot<E> = Arc<Mutex<Option<E>>>;

pub(crate) fn new_error_slot<E>() -> ErrorSlot<E> {
    Arc::new(Mutex::new(None))
}

/// Records `error` into `slot` if the slot is empty (keeps only the *first* error).
fn record_error<E>(slot: &ErrorSlot<E>, error: E) {
    let mut guard = slot.lock().expect("error slot mutex poisoned");
    if guard.is_none() {
        *guard = Some(error);
    }
}

/// The interpreter-internal, infallible mirror of the public [`Adapter`] trait.
///
/// Its method signatures are exactly those of the pre-error-handling `Adapter`: plain
/// iterators with no `Result`. The interpreter internals are generic over this trait,
/// so their bodies are unchanged by the introduction of fallible adapters.
///
/// The only production implementor is [`ErrorTrackingAdapter`], but test/tracing wrappers
/// that used to implement `Adapter` directly may also target it.
pub(crate) trait RawAdapter<'vertex> {
    type Vertex: Clone + Debug + 'vertex;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexIterator<'vertex, Self::Vertex>;

    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, FieldValue>;

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'vertex, V, VertexIterator<'vertex, Self::Vertex>>;

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, bool>;
}

/// Iterator adapter that yields the `Ok` values of an underlying `Result` iterator, and on
/// the first `Err` records it into a shared [`ErrorSlot`] and then fuses (never yields again).
pub(crate) struct FuseOnError<I, T, E> {
    inner: I,
    slot: ErrorSlot<E>,
    done: bool,
    _marker: PhantomData<fn() -> T>,
}

impl<I, T, E> FuseOnError<I, T, E> {
    fn new(inner: I, slot: ErrorSlot<E>) -> Self {
        Self { inner, slot, done: false, _marker: PhantomData }
    }
}

impl<I, T, E> Iterator for FuseOnError<I, T, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.done {
            return None;
        }
        match self.inner.next() {
            Some(Ok(value)) => Some(value),
            Some(Err(error)) => {
                record_error(&self.slot, error);
                self.done = true;
                None
            }
            None => None,
        }
    }
}

/// Wraps a fallible [`Adapter`] and exposes it to the interpreter as an infallible
/// [`RawAdapter`], routing the first error into a shared [`ErrorSlot`].
///
/// The `Holder` type parameter is any smart pointer/reference that derefs to the adapter,
/// so callers can wrap either an owned `Arc<A>` (the [`interpret_ir`](super::execution::interpret_ir)
/// path) or a borrowed `&A` (the [`DynamicallyResolvedValue`](super::DynamicallyResolvedValue)
/// path). It defaults to `Arc<A>`.
pub(crate) struct ErrorTrackingAdapter<'vertex, A, Holder = Arc<A>>
where
    A: Adapter<'vertex>,
    Holder: Deref<Target = A>,
{
    inner: Holder,
    slot: ErrorSlot<A::Error>,
    // Only the lifetime is phantom — deliberately not `&'vertex A`, so wrapping a borrowed
    // `&A` (the `DynamicallyResolvedValue` path) does not spuriously require `A: 'vertex`.
    _marker: PhantomData<&'vertex ()>,
}

impl<'vertex, A, Holder> ErrorTrackingAdapter<'vertex, A, Holder>
where
    A: Adapter<'vertex>,
    Holder: Deref<Target = A>,
{
    pub(crate) fn new(inner: Holder) -> Self {
        Self { inner, slot: new_error_slot(), _marker: PhantomData }
    }

    /// A handle to the shared error slot, for the outermost results iterator to inspect.
    pub(crate) fn error_slot(&self) -> ErrorSlot<A::Error> {
        self.slot.clone()
    }
}

impl<'vertex, A, Holder> RawAdapter<'vertex> for ErrorTrackingAdapter<'vertex, A, Holder>
where
    A: Adapter<'vertex>,
    Holder: Deref<Target = A>,
{
    type Vertex = A::Vertex;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexIterator<'vertex, Self::Vertex> {
        let inner = self.inner.resolve_starting_vertices(edge_name, parameters, resolve_info);
        Box::new(FuseOnError::new(inner, self.slot.clone()))
    }

    fn resolve_property<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, FieldValue> {
        // Scalar outcomes record-and-substitute rather than fuse: on error we keep the
        // one-outcome-per-context pairing (substituting `Null`) and let the recorded error
        // abort the stream. The substituted row/candidate is discarded on abort. Preserving
        // the pairing is what lets the `DynamicallyResolvedValue` path attach the error to a
        // context instead of silently dropping it.
        let slot = self.slot.clone();
        let inner = self
            .inner
            .resolve_property(contexts, type_name, property_name, resolve_info)
            .map(move |(ctx, result)| match result {
                Ok(value) => (ctx, value),
                Err(error) => {
                    record_error(&slot, error);
                    (ctx, FieldValue::Null)
                }
            });
        Box::new(inner)
    }

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'vertex, V, VertexIterator<'vertex, Self::Vertex>> {
        let slot = self.slot.clone();
        let inner =
            self.inner.resolve_neighbors(contexts, type_name, edge_name, parameters, resolve_info);
        // The outer (context, neighbors) stream stays 1:1 with the input contexts — the
        // pairing invariant the interpreter relies on. Only the *inner* neighbor iterator
        // is fused on error.
        Box::new(inner.map(move |(ctx, neighbors)| {
            let fused: VertexIterator<'vertex, Self::Vertex> =
                Box::new(FuseOnError::new(neighbors, slot.clone()));
            (ctx, fused)
        }))
    }

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'vertex>(
        &self,
        contexts: ContextIterator<'vertex, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'vertex, V, bool> {
        // Record-and-substitute `false` on error (see `resolve_property`). A `false` coercion
        // simply filters the context out; the recorded error aborts the stream regardless.
        let slot = self.slot.clone();
        let inner = self
            .inner
            .resolve_coercion(contexts, type_name, coerce_to_type, resolve_info)
            .map(move |(ctx, result)| match result {
                Ok(value) => (ctx, value),
                Err(error) => {
                    record_error(&slot, error);
                    (ctx, false)
                }
            });
        Box::new(inner)
    }
}

/// Iterator adapter placed at the very end of the interpreter pipeline.
///
/// It yields `Ok(item)` for each successful result. Before and after each pull it checks the
/// shared [`ErrorSlot`]: if an adapter error was recorded, it discards the in-flight item,
/// yields exactly one `Err`, and then ends (fail-fast, DataFusion-style).
struct SurfaceErrors<'query, Item, E> {
    inner: Box<dyn Iterator<Item = Item> + 'query>,
    slot: ErrorSlot<E>,
    done: bool,
}

impl<'query, Item, E: std::error::Error + 'static> Iterator for SurfaceErrors<'query, Item, E> {
    type Item = Result<Item, ExecutionError<E>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // An error may have been recorded during construction of a *previous* row that was
        // nonetheless partially yielded; catch it before pulling anything new.
        if let Some(error) = self.slot.lock().expect("error slot mutex poisoned").take() {
            self.done = true;
            return Some(Err(ExecutionError::Adapter(error)));
        }

        let item = self.inner.next();

        // Producing this item may have pulled the erroring adapter value. If so, the item is
        // built from truncated data and must be discarded in favor of the error.
        if let Some(error) = self.slot.lock().expect("error slot mutex poisoned").take() {
            self.done = true;
            return Some(Err(ExecutionError::Adapter(error)));
        }

        item.map(Ok)
    }
}

/// Wrap the interpreter's final (infallible) results iterator so it surfaces any recorded
/// adapter error as a terminal `Err`.
pub(crate) fn surface_errors<'query, Item: 'query, E: std::error::Error + 'static>(
    inner: Box<dyn Iterator<Item = Item> + 'query>,
    slot: ErrorSlot<E>,
) -> Box<dyn Iterator<Item = Result<Item, ExecutionError<E>>> + 'query> {
    Box::new(SurfaceErrors { inner, slot, done: false })
}

/// Attach recorded adapter errors onto a per-context outcome stream, as `Result`s in the
/// outcome slot, for the [`DynamicallyResolvedValue`](super::DynamicallyResolvedValue) path.
///
/// For each `(context, value)` produced by `inner`, this yields `(context, Ok(value))` — unless
/// an adapter error has been recorded in `slot`, in which case it yields `(context, Err(error))`
/// for that context and then ends. Because scalar resolution is record-and-substitute (1:1), the
/// erroring context is still present in `inner`, so its error is never silently dropped.
struct SurfaceErrorsPaired<'query, C, T, E> {
    inner: Box<dyn Iterator<Item = (C, T)> + 'query>,
    slot: ErrorSlot<E>,
    done: bool,
}

impl<'query, C, T, E> Iterator for SurfaceErrorsPaired<'query, C, T, E> {
    type Item = (C, Result<T, E>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let (context, value) = self.inner.next()?;
        if let Some(error) = self.slot.lock().expect("error slot mutex poisoned").take() {
            self.done = true;
            Some((context, Err(error)))
        } else {
            Some((context, Ok(value)))
        }
    }
}

/// See [`SurfaceErrorsPaired`].
pub(crate) fn surface_errors_paired<'query, C: 'query, T: 'query, E: 'query>(
    inner: Box<dyn Iterator<Item = (C, T)> + 'query>,
    slot: ErrorSlot<E>,
) -> Box<dyn Iterator<Item = (C, Result<T, E>)> + 'query> {
    Box::new(SurfaceErrorsPaired { inner, slot, done: false })
}
