//! End-to-end tests for adapter errors.
//!
//! A fault-injecting `NumbersAdapter` verifies the fail-fast contract: preceding rows are
//! yielded, then one error, then the result iterator ends.

use std::{
    collections::BTreeMap,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::{
    frontend::parse,
    interpreter::{
        Adapter, AsVertex, ContextIterator, ContextOutcomeIterator, NeighborResolution,
        ResolveEdgeInfo, ResolveInfo, VertexIterator, error::ExecutionError,
        execution::interpret_ir,
    },
    ir::{EdgeParameters, FieldValue},
    numbers_interpreter::NumbersAdapter,
};

type Row = BTreeMap<Arc<str>, FieldValue>;

/// Deliberately local-only, as errors may wrap JavaScript values in WASM adapters.
#[derive(Debug)]
pub(super) struct TestError {
    _local: std::rc::Rc<()>,
}

impl TestError {
    pub(super) fn new() -> Self {
        Self { _local: std::rc::Rc::new(()) }
    }
}

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "injected test error")
    }
}

impl std::error::Error for TestError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fault {
    StartingVertices,
    Property,
    Neighbors,
    Coercion,
}

/// Wraps `NumbersAdapter` and injects an error after a configurable number of values.
pub(super) struct FaultyAdapter {
    inner: NumbersAdapter,
    fault: Fault,
    remaining: Arc<AtomicUsize>,
    error_emitted: Arc<AtomicBool>,
}

impl FaultyAdapter {
    pub(super) fn new(fault: Fault, fail_after: usize) -> Self {
        Self {
            inner: NumbersAdapter::new(),
            fault,
            remaining: Arc::new(AtomicUsize::new(fail_after)),
            error_emitted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The schema of the wrapped [`NumbersAdapter`].
    pub(super) fn schema(&self) -> crate::schema::Schema {
        self.inner.schema().clone()
    }
}

fn should_error(remaining: &AtomicUsize, error_emitted: &AtomicBool) -> bool {
    let current = remaining.load(Ordering::SeqCst);
    if current == 0 {
        assert!(
            !error_emitted.swap(true, Ordering::SeqCst),
            "adapter was polled after its first error"
        );
        true
    } else {
        remaining.store(current - 1, Ordering::SeqCst);
        false
    }
}

impl<'a> Adapter<'a> for FaultyAdapter {
    type Vertex = <NumbersAdapter as Adapter<'a>>::Vertex;
    type Error = TestError;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexIterator<'a, Result<Self::Vertex, Self::Error>> {
        let inner = self
            .inner
            .resolve_starting_vertices(edge_name, parameters, resolve_info)
            .map(unwrap_ok);
        if self.fault == Fault::StartingVertices {
            let remaining = self.remaining.clone();
            let error_emitted = self.error_emitted.clone();
            Box::new(inner.map(move |v| {
                if should_error(&remaining, &error_emitted) { Err(TestError::new()) } else { Ok(v) }
            }))
        } else {
            Box::new(inner.map(Ok))
        }
    }

    fn resolve_property<V: AsVertex<Self::Vertex> + 'a>(
        &self,
        contexts: ContextIterator<'a, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'a, V, Result<FieldValue, Self::Error>> {
        let inner = self
            .inner
            .resolve_property(contexts, type_name, property_name, resolve_info)
            .map(|(ctx, value)| (ctx, unwrap_ok(value)));
        if self.fault == Fault::Property {
            let remaining = self.remaining.clone();
            let error_emitted = self.error_emitted.clone();
            Box::new(inner.map(move |(ctx, value)| {
                if should_error(&remaining, &error_emitted) {
                    (ctx, Err(TestError::new()))
                } else {
                    (ctx, Ok(value))
                }
            }))
        } else {
            Box::new(inner.map(|(ctx, value)| (ctx, Ok(value))))
        }
    }

    #[allow(clippy::type_complexity)]
    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'a>(
        &self,
        contexts: ContextIterator<'a, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'a, V, NeighborResolution<'a, Self::Vertex, Self::Error>> {
        let faulted = self.fault == Fault::Neighbors;
        let remaining = self.remaining.clone();
        let error_emitted = self.error_emitted.clone();
        let inner =
            self.inner.resolve_neighbors(contexts, type_name, edge_name, parameters, resolve_info);
        Box::new(inner.map(move |(ctx, resolution)| {
            let neighbors = unwrap_ok(resolution);
            let neighbors: Box<dyn Iterator<Item = _> + 'a> = Box::new(neighbors.map(unwrap_ok));
            let out: VertexIterator<'a, Result<Self::Vertex, Self::Error>> = if faulted {
                let remaining = remaining.clone();
                let error_emitted = error_emitted.clone();
                Box::new(neighbors.map(move |v| {
                    if should_error(&remaining, &error_emitted) {
                        Err(TestError::new())
                    } else {
                        Ok(v)
                    }
                }))
            } else {
                Box::new(neighbors.map(Ok))
            };
            (ctx, Ok(out))
        }))
    }

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'a>(
        &self,
        contexts: ContextIterator<'a, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'a, V, Result<bool, Self::Error>> {
        let inner = self
            .inner
            .resolve_coercion(contexts, type_name, coerce_to_type, resolve_info)
            .map(|(ctx, value)| (ctx, unwrap_ok(value)));
        if self.fault == Fault::Coercion {
            let remaining = self.remaining.clone();
            let error_emitted = self.error_emitted.clone();
            Box::new(inner.map(move |(ctx, value)| {
                if should_error(&remaining, &error_emitted) {
                    (ctx, Err(TestError::new()))
                } else {
                    (ctx, Ok(value))
                }
            }))
        } else {
            Box::new(inner.map(|(ctx, value)| (ctx, Ok(value))))
        }
    }
}

/// `NumbersAdapter` is infallible, so its outcomes are `Result<_, Infallible>`.
fn unwrap_ok<T>(result: Result<T, std::convert::Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

pub(super) fn run(
    fault: Fault,
    fail_after: usize,
    query: &str,
) -> Vec<Result<Row, ExecutionError<TestError>>> {
    let adapter = Arc::new(FaultyAdapter::new(fault, fail_after));
    let schema = adapter.inner.schema().clone();
    let indexed = parse(&schema, query).expect("query failed to parse");
    interpret_ir(adapter, indexed, Arc::new(BTreeMap::new()))
        .expect("unexpected query arguments error")
        .collect()
}

/// Baseline row count for a query run against a never-erroring adapter.
fn baseline_row_count(query: &str) -> usize {
    let results = run(Fault::Property, usize::MAX, query);
    assert!(results.iter().all(Result::is_ok), "baseline run unexpectedly errored");
    results.len()
}

fn assert_fail_fast_exact(results: &[Result<Row, ExecutionError<TestError>>], expected_ok: usize) {
    assert_eq!(
        results.len(),
        expected_ok + 1,
        "expected {expected_ok} Ok rows then exactly one Err; got {} items",
        results.len()
    );
    for (i, r) in results.iter().take(expected_ok).enumerate() {
        assert!(r.is_ok(), "result {i} should be Ok, was {r:?}");
    }
    assert!(
        matches!(results[expected_ok], Err(ExecutionError::Adapter(TestError { .. }))),
        "final item should be the injected adapter error, was {:?}",
        results[expected_ok],
    );
}

fn assert_fail_fast_terminal(results: &[Result<Row, ExecutionError<TestError>>]) {
    assert!(!results.is_empty(), "expected at least the terminal error");
    let (last, rest) = results.split_last().unwrap();
    assert!(
        matches!(last, Err(ExecutionError::Adapter(TestError { .. }))),
        "last item should be the injected adapter error, was {last:?}"
    );
    for (i, r) in rest.iter().enumerate() {
        assert!(r.is_ok(), "item {i} before the error should be Ok, was {r:?}");
    }
}

pub(super) const FLAT: &str = r#"{ Number(min: 0, max: 50) { value @output } }"#;
pub(super) const SUCCESSOR: &str = r#"{ Number(min: 0, max: 50) { successor { value @output } } }"#;

#[test]
fn error_in_resolve_starting_vertices_is_fail_fast() {
    assert!(baseline_row_count(FLAT) > 5);
    let results = run(Fault::StartingVertices, 5, FLAT);
    assert_fail_fast_exact(&results, 5);
}

#[test]
fn error_in_resolve_property_is_fail_fast() {
    assert!(baseline_row_count(FLAT) > 3);
    let results = run(Fault::Property, 3, FLAT);
    assert_fail_fast_exact(&results, 3);
}

#[test]
fn error_in_resolve_neighbors_is_fail_fast() {
    assert!(baseline_row_count(SUCCESSOR) > 4);
    let results = run(Fault::Neighbors, 4, SUCCESSOR);
    assert_fail_fast_exact(&results, 4);
}

#[test]
fn error_after_zero_rows_yields_only_the_error() {
    let results = run(Fault::StartingVertices, 0, FLAT);
    assert_fail_fast_exact(&results, 0);
}

#[test]
fn no_error_when_budget_exceeds_work() {
    let total = baseline_row_count(FLAT);
    let results = run(Fault::Property, total + 10, FLAT);
    assert_eq!(results.len(), total);
    assert!(results.iter().all(Result::is_ok));
}

#[test]
fn error_in_coercion_is_fail_fast() {
    let query = r#"{ Number(min: 0, max: 50) { successor { ... on Prime { value @output } } } }"#;
    let results = run(Fault::Coercion, 4, query);
    assert_fail_fast_terminal(&results);
}

#[test]
fn error_inside_fold_terminates() {
    let query = r#"{
        Number(min: 1, max: 50) {
            value @output
            multiple(max: 30) @fold {
                factor: value @output
            }
        }
    }"#;
    let results = run(Fault::Neighbors, 5, query);
    assert_fail_fast_terminal(&results);
}

#[test]
fn scalar_error_inside_materialized_fold_stops_adapter_polls() {
    let query = r#"{
        Number(min: 1, max: 50) {
            multiple(max: 30) @fold {
                factor: value @output
            }
        }
    }"#;
    let results = run(Fault::Property, 5, query);
    assert_fail_fast_terminal(&results);
}

#[test]
fn error_inside_recurse_terminates() {
    let query = r#"{
        Number(min: 0, max: 10) {
            value @output
            successor @recurse(depth: 3) {
                succ: value @output
            }
        }
    }"#;
    let results = run(Fault::Neighbors, 5, query);
    assert_fail_fast_terminal(&results);
}

#[test]
fn error_inside_optional_terminates() {
    let query = r#"{
        Number(min: 0, max: 50) {
            value @output
            predecessor @optional {
                pred: value @output
            }
        }
    }"#;
    let results = run(Fault::Neighbors, 5, query);
    assert_fail_fast_terminal(&results);
}

/// Injects a context-level neighbor-resolution error.
struct ContextFaultyAdapter {
    inner: NumbersAdapter,
    remaining: Arc<AtomicUsize>,
    error_emitted: Arc<AtomicBool>,
    code: u32,
}

impl ContextFaultyAdapter {
    fn new(fail_after: usize, code: u32) -> Self {
        Self {
            inner: NumbersAdapter::new(),
            remaining: Arc::new(AtomicUsize::new(fail_after)),
            error_emitted: Arc::new(AtomicBool::new(false)),
            code,
        }
    }
}

impl<'a> Adapter<'a> for ContextFaultyAdapter {
    type Vertex = <NumbersAdapter as Adapter<'a>>::Vertex;
    type Error = CodedError;

    fn resolve_starting_vertices(
        &self,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveInfo,
    ) -> VertexIterator<'a, Result<Self::Vertex, Self::Error>> {
        Box::new(
            self.inner
                .resolve_starting_vertices(edge_name, parameters, resolve_info)
                .map(|v| Ok(unwrap_ok(v))),
        )
    }

    fn resolve_property<V: AsVertex<Self::Vertex> + 'a>(
        &self,
        contexts: ContextIterator<'a, V>,
        type_name: &Arc<str>,
        property_name: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'a, V, Result<FieldValue, Self::Error>> {
        Box::new(
            self.inner
                .resolve_property(contexts, type_name, property_name, resolve_info)
                .map(|(ctx, value)| (ctx, Ok(unwrap_ok(value)))),
        )
    }

    fn resolve_neighbors<V: AsVertex<Self::Vertex> + 'a>(
        &self,
        contexts: ContextIterator<'a, V>,
        type_name: &Arc<str>,
        edge_name: &Arc<str>,
        parameters: &EdgeParameters,
        resolve_info: &ResolveEdgeInfo,
    ) -> ContextOutcomeIterator<'a, V, NeighborResolution<'a, Self::Vertex, Self::Error>> {
        let remaining = self.remaining.clone();
        let error_emitted = self.error_emitted.clone();
        let code = self.code;
        let inner =
            self.inner.resolve_neighbors(contexts, type_name, edge_name, parameters, resolve_info);
        Box::new(inner.map(move |(ctx, resolution)| {
            if should_error(&remaining, &error_emitted) {
                (ctx, Err(CodedError(code)))
            } else {
                let neighbors = unwrap_ok(resolution);
                let neighbors: VertexIterator<'a, Result<Self::Vertex, Self::Error>> =
                    Box::new(neighbors.map(|vertex| Ok(unwrap_ok(vertex))));
                (ctx, Ok(neighbors))
            }
        }))
    }

    fn resolve_coercion<V: AsVertex<Self::Vertex> + 'a>(
        &self,
        contexts: ContextIterator<'a, V>,
        type_name: &Arc<str>,
        coerce_to_type: &Arc<str>,
        resolve_info: &ResolveInfo,
    ) -> ContextOutcomeIterator<'a, V, Result<bool, Self::Error>> {
        Box::new(
            self.inner
                .resolve_coercion(contexts, type_name, coerce_to_type, resolve_info)
                .map(|(ctx, value)| (ctx, Ok(unwrap_ok(value)))),
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
struct CodedError(u32);

impl fmt::Display for CodedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "coded error {}", self.0)
    }
}

impl std::error::Error for CodedError {}

fn run_context_fault(
    fail_after: usize,
    code: u32,
    query: &str,
) -> Vec<Result<Row, ExecutionError<CodedError>>> {
    let adapter = Arc::new(ContextFaultyAdapter::new(fail_after, code));
    let schema = adapter.inner.schema().clone();
    let indexed = parse(&schema, query).expect("query failed to parse");
    interpret_ir(adapter, indexed, Arc::new(BTreeMap::new()))
        .expect("unexpected query arguments error")
        .collect()
}

#[test]
fn context_level_neighbor_failure_is_fail_fast_and_identifiable() {
    let results = run_context_fault(3, 4242, SUCCESSOR);
    assert!(!results.is_empty());
    let (last, rest) = results.split_last().unwrap();
    for (i, r) in rest.iter().enumerate() {
        assert!(r.is_ok(), "item {i} before the error should be Ok, was {r:?}");
    }
    match last {
        Err(ExecutionError::Adapter(error)) => assert_eq!(error.0, 4242),
        other => panic!("expected the injected context-level error, got {other:?}"),
    }
}

#[test]
fn context_level_neighbor_failure_at_zero_budget() {
    let results = run_context_fault(0, 7, SUCCESSOR);
    assert_eq!(results.len(), 1);
    assert!(matches!(&results[0], Err(ExecutionError::Adapter(e)) if e.0 == 7));
}
