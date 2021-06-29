use crate::common::*;
use ordered_float::NotNan;
use petgraph::Incoming;
use slotmap::SparseSecondaryMap;
use std::{cell::RefCell, ops};

#[derive(Clone, Copy)]
pub struct Array<'g> {
    node_id: OpNodeId,
    graph: &'g Graph,
}

#[derive(Clone, Copy)]
pub struct DualArray<'g> {
    value_node_id: OpNodeId,
    grad_node_id: OpNodeId,
    graph: &'g Graph,
}

impl<'g> Array<'g> {
    pub fn graph(&self) -> &'g Graph {
        self.graph
    }

    fn broadcast(self, shape: &Shape) -> Self {
        self.graph.with_state(|state| {
            let self_shape = state.ops.graph[self.node_id].shape.clone();
            if &self_shape == shape {
                self
            } else {
                Array {
                    node_id: state.ops.new_node(
                        shape.clone(),
                        Op::View(View::broadcast(&self_shape, &shape)),
                        &[self.node_id],
                    ),
                    graph: self.graph,
                }
            }
        })
    }

    fn unary_op(self, op: UnaryOp) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id].shape.clone();
            Array {
                node_id: state.ops.new_node(shape, Op::Unary(op), &[self.node_id]),
                graph: self.graph,
            }
        })
    }

    fn binary_op(self, rhs: Array, op: BinaryOp) -> Self {
        let op_shape = self.graph.with_state(|state| {
            state.ops.graph[self.node_id]
                .shape
                .match_with_broadcast(&state.ops.graph[rhs.node_id].shape)
        });

        let lhs = self.broadcast(&op_shape).node_id;
        let rhs = rhs.broadcast(&op_shape).node_id;

        self.graph.with_state(|state| Array {
            node_id: state.ops.new_node(op_shape, Op::Binary(op), &[lhs, rhs]),
            graph: self.graph,
        })
    }

    fn compare_and_select(
        self,
        compare_mode: CompareMode,
        rhs: Array,
        pass: Array,
        fail: Array,
    ) -> Self {
        let op_shape = self.graph.with_state(|state| {
            state.ops.graph[self.node_id]
                .shape
                .match_with_broadcast(&state.ops.graph[rhs.node_id].shape)
                .match_with_broadcast(&state.ops.graph[pass.node_id].shape)
                .match_with_broadcast(&state.ops.graph[fail.node_id].shape)
        });

        let lhs = self.broadcast(&op_shape).node_id;
        let rhs = rhs.broadcast(&op_shape).node_id;
        let pass = pass.broadcast(&op_shape).node_id;
        let fail = fail.broadcast(&op_shape).node_id;

        self.graph.with_state(|state| Array {
            node_id: state.ops.new_node(
                op_shape,
                Op::CompareAndSelect(compare_mode),
                &[lhs, rhs, pass, fail],
            ),
            graph: self.graph,
        })
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: Axis) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id].shape.reduce(axis);
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::Reduce { reduce_op, axis }, &[self.node_id]),
                graph: self.graph,
            }
        })
    }

    pub fn one_hot(self, count: usize) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id].shape.one_hot(count);
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::Unary(UnaryOp::OneHot), &[self.node_id]),
                graph: self.graph,
            }
        })
    }

    pub fn reduce_max(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Max, self.shape().axis(axis))
    }
    pub fn reduce_sum(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Sum, self.shape().axis(axis))
    }

    pub fn argmax(self, axis: isize) -> Self {
        // implement with reduce_max for now
        let coord_or_zero = self.select_eq(
            self.reduce_max(axis),
            self.coord(axis),
            self.graph.literal(0.0),
        );
        coord_or_zero.reduce_max(axis)
    }

    fn reduce_onto_per_element(self, shape: &Shape) -> Self {
        let mut output = self;
        while let Some(axis) = output.shape().reduce_axis_onto_per_element(shape) {
            output = output.reduce_op(ReduceOp::Sum, axis);
        }
        output
    }

    pub fn coord(self, axis: isize) -> Self {
        self.graph.coord(self.shape(), axis)
    }

    pub fn select_eq(self, rhs: Array, pass: Array, fail: Array) -> Self {
        self.compare_and_select(CompareMode::Eq, rhs, pass, fail)
    }
    pub fn select_gt(self, rhs: Array, pass: Array, fail: Array) -> Self {
        self.compare_and_select(CompareMode::Gt, rhs, pass, fail)
    }

    pub fn sqrt(self) -> Self {
        self.unary_op(UnaryOp::Sqrt)
    }
    pub fn exp(self) -> Self {
        self.unary_op(UnaryOp::Exp)
    }
    pub fn log(self) -> Self {
        self.unary_op(UnaryOp::Log)
    }

    pub fn matmul(self, rhs: Array) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id]
                .shape
                .matrix_multiply(&state.ops.graph[rhs.node_id].shape);
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::MatMul, &[self.node_id, rhs.node_id]),
                graph: self.graph,
            }
        })
    }

    pub fn transpose(self) -> Self {
        self.graph.with_state(|state| {
            let input_shape = &state.ops.graph[self.node_id].shape;
            let view = input_shape.identity_view().transposed();
            let output_shape = input_shape.transposed();
            Array {
                node_id: state
                    .ops
                    .new_node(output_shape, Op::View(view), &[self.node_id]),
                graph: self.graph,
            }
        })
    }

    pub fn shape(&self) -> Shape {
        self.graph
            .with_state(|state| state.ops.graph[self.node_id].shape.clone())
    }

    pub fn accumulate(&self, src: Array) {
        self.graph.with_state(|state| {
            assert_eq!(state.ops.graph[self.node_id].op, Op::Accumulate);
            assert_eq!(
                state.ops.graph[self.node_id].shape,
                state.ops.graph[src.node_id].shape
            );
            let arg = state
                .ops
                .graph
                .edges_directed(self.node_id, Incoming)
                .count();
            state.ops.graph.add_edge(
                src.node_id,
                self.node_id,
                OpEdge {
                    arg,
                    view: state.ops.graph[src.node_id].shape.identity_view(),
                },
            );
        })
    }
}

impl<'g> ops::Add for Array<'g> {
    type Output = Array<'g>;
    fn add(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Add)
    }
}
impl<'g> ops::Add<f32> for Array<'g> {
    type Output = Array<'g>;
    fn add(self, rhs: f32) -> Self::Output {
        let rhs = self.graph.literal(rhs);
        self.binary_op(rhs, BinaryOp::Add)
    }
}

impl<'g> ops::Sub for Array<'g> {
    type Output = Array<'g>;
    fn sub(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Sub)
    }
}
impl<'g> ops::Sub<Array<'g>> for f32 {
    type Output = Array<'g>;
    fn sub(self, rhs: Array<'g>) -> Self::Output {
        let lhs = rhs.graph.literal(self);
        lhs.binary_op(rhs, BinaryOp::Sub)
    }
}

impl<'g> ops::Mul for Array<'g> {
    type Output = Array<'g>;
    fn mul(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Mul)
    }
}
impl<'g> ops::Mul<f32> for Array<'g> {
    type Output = Array<'g>;
    fn mul(self, rhs: f32) -> Self::Output {
        let rhs = self.graph.literal(rhs);
        self.binary_op(rhs, BinaryOp::Mul)
    }
}

impl<'g> ops::Div for Array<'g> {
    type Output = Array<'g>;
    fn div(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Div)
    }
}
impl<'g> ops::Neg for Array<'g> {
    type Output = Array<'g>;
    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOp::Neg)
    }
}

impl<'g> ops::Mul<Array<'g>> for f32 {
    type Output = Array<'g>;
    fn mul(self, rhs: Array<'g>) -> Self::Output {
        let lhs = rhs.graph.literal(self);
        lhs.binary_op(rhs, BinaryOp::Mul)
    }
}
impl<'g> ops::Div<f32> for Array<'g> {
    type Output = Array<'g>;
    fn div(self, rhs: f32) -> Self::Output {
        let rhs = self.graph.literal(rhs);
        self.binary_op(rhs, BinaryOp::Div)
    }
}

impl<'g> DualArray<'g> {
    pub fn new(value: Array<'g>, grad: Array<'g>) -> Self {
        Self {
            value_node_id: value.node_id,
            grad_node_id: grad.node_id,
            graph: value.graph,
        }
    }

    pub fn value(self) -> Array<'g> {
        Array {
            node_id: self.value_node_id,
            graph: self.graph,
        }
    }

    pub fn grad(self) -> Array<'g> {
        Array {
            node_id: self.grad_node_id,
            graph: self.graph,
        }
    }

    pub fn shape(&self) -> Shape {
        self.value().shape()
    }

    pub fn graph(&self) -> &'g Graph {
        self.graph
    }

    pub fn leaky_relu(self, leakiness: f32) -> Self {
        let a = self.value();
        let da = self.grad();

        let zero = self.graph.literal(0.0);
        let b = a.select_gt(zero, a, a * leakiness);

        let db = self.graph.accumulator(b.shape());
        da.accumulate(a.select_gt(zero, db, db * leakiness));

        Self::new(b, db)
    }

    pub fn matmul(self, rhs: DualArray) -> Self {
        let a = self.value();
        let da = self.grad();
        let b = rhs.value();
        let db = rhs.grad();

        let c = a.matmul(b);

        let dc = self.graph.accumulator(c.shape());
        da.accumulate(dc.matmul(b.transpose()));
        db.accumulate(a.transpose().matmul(dc));

        Self::new(c, dc)
    }
}

impl<'g> ops::Add for DualArray<'g> {
    type Output = DualArray<'g>;
    fn add(self, rhs: DualArray<'g>) -> Self::Output {
        let a = self.value();
        let da = self.grad();
        let b = rhs.value();
        let db = rhs.grad();

        let c = a + b;
        let dc = self.graph.accumulator(c.shape());

        da.accumulate(dc.reduce_onto_per_element(&a.shape()));
        db.accumulate(dc.reduce_onto_per_element(&b.shape()));

        Self::new(c, dc)
    }
}

struct OpGraphState {
    graph: OpGraph,
    next_colour: usize,
}

impl OpGraphState {
    fn new_node(&mut self, shape: impl Into<Shape>, op: Op, inputs: &[OpNodeId]) -> OpNodeId {
        let shape = shape.into();
        let node_id = self.graph.add_node(OpNode {
            colour: self.next_colour,
            shape,
            op,
            cluster_id: None,
        });
        for (index, input_id) in inputs.iter().copied().enumerate() {
            self.graph.add_edge(
                input_id,
                node_id,
                OpEdge {
                    arg: index,
                    view: self.graph[input_id].shape.identity_view(),
                },
            );
        }
        node_id
    }
}

#[derive(Clone, Copy)]
struct GraphInput {
    value_node_id: OpNodeId,
    grad_node_id: Option<OpNodeId>,
}

struct GraphState {
    ops: OpGraphState,
    variables: SharedVariables,
    inputs: SparseSecondaryMap<VariableId, GraphInput>,
    outputs: SparseSecondaryMap<VariableId, OpNodeId>,
}

pub struct Graph {
    state: RefCell<GraphState>,
}

impl Graph {
    pub(crate) fn new(variables: SharedVariables) -> Self {
        Self {
            state: RefCell::new(GraphState {
                ops: OpGraphState {
                    graph: Default::default(),
                    next_colour: 0,
                },
                variables,
                inputs: SparseSecondaryMap::new(),
                outputs: SparseSecondaryMap::new(),
            }),
        }
    }

    fn with_state<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut GraphState) -> T,
    {
        let mut data = self.state.borrow_mut();
        f(&mut data)
    }

    pub fn literal(&self, value: f32) -> Array {
        self.with_state(|state| Array {
            node_id: state
                .ops
                .new_node([1], Op::Literal(NotNan::new(value).unwrap()), &[]),
            graph: self,
        })
    }

    pub fn coord(&self, shape: impl Into<Shape>, axis: isize) -> Array {
        self.with_state(|state| {
            let shape = shape.into();
            let axis = shape.axis(axis);
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::BuiltIn(BuiltInOp::Coord { axis }), &[]),
                graph: self,
            }
        })
    }

    fn input(&self, variable: &Variable) -> GraphInput {
        self.with_state(|state| {
            let variable_id = variable.checked_id(&state.variables);
            let shape = state
                .variables
                .borrow()
                .get(variable_id)
                .unwrap()
                .shape
                .clone();
            let ops = &mut state.ops;
            *state
                .inputs
                .entry(variable_id)
                .unwrap()
                .or_insert_with(|| GraphInput {
                    value_node_id: ops.new_node(shape.clone(), Op::Input { variable_id }, &[]),
                    grad_node_id: Some(ops.new_node(shape, Op::Accumulate, &[])),
                })
        })
    }

    pub fn parameter(&self, variable: &Variable) -> DualArray {
        let input = self.input(variable);
        DualArray {
            value_node_id: input.value_node_id,
            grad_node_id: input.grad_node_id.unwrap(),
            graph: self,
        }
    }

    pub fn read_variable(&self, variable: &Variable) -> Array {
        let input = self.input(variable);
        Array {
            node_id: input.value_node_id,
            graph: self,
        }
    }

    pub fn write_variable(&self, variable: &Variable, rhs: Array) {
        self.with_state(|state| {
            let variable_id = variable.checked_id(&state.variables);
            let shape = state.ops.graph[rhs.node_id].shape.clone();
            assert_eq!(
                state
                    .variables
                    .borrow()
                    .get(variable_id)
                    .unwrap()
                    .shape
                    .clone(),
                shape
            );

            // update the output node for this variable (remove any old one)
            let node_id = state
                .ops
                .new_node(shape, Op::Output { variable_id }, &[rhs.node_id]);
            if let Some(node_id) = state.outputs.insert(variable_id, node_id) {
                state.ops.graph.remove_node(node_id);
            }

            // ensure that if we read this variable again we read the latest value
            state.inputs.insert(
                variable_id,
                GraphInput {
                    value_node_id: rhs.node_id,
                    grad_node_id: None,
                },
            );
        });
    }

    pub fn update_variable<'g>(
        &'g self,
        variable: &Variable,
        f: impl FnOnce(Array<'g>) -> Array<'g>,
    ) -> Array<'g> {
        let result = f(self.read_variable(variable));
        self.write_variable(variable, result);
        result
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> Array {
        self.with_state(|state| Array {
            node_id: state.ops.new_node(shape, Op::Accumulate, &[]),
            graph: self,
        })
    }

    pub fn next_colour(&self) {
        self.with_state(|state| {
            state.ops.next_colour += 1;
        })
    }

    pub fn build_schedule(self) -> Schedule {
        self.with_state(|state| {
            Schedule::new(
                SharedVariables::clone(&state.variables),
                state.ops.graph.clone(),
            )
        })
    }
}
