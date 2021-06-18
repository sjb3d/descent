use crate::common::*;
use petgraph::Incoming;
use slotmap::SparseSecondaryMap;
use std::{cell::RefCell, ops};

#[derive(Clone, Copy)]
pub struct Array<'builder> {
    index: OpNodeIndex,
    builder: &'builder GraphBuilder,
}

#[derive(Clone, Copy)]
struct TensorNodeIndexPair {
    value: OpNodeIndex,
    grad: OpNodeIndex,
}

#[derive(Clone, Copy)]
pub struct Tensor<'builder> {
    index: TensorNodeIndexPair,
    builder: &'builder GraphBuilder,
}

impl<'builder> Array<'builder> {
    fn unary_op(self, op: UnaryOp) -> Self {
        self.builder.with_state(|state| {
            let shape = state.ops.graph[self.index].shape.clone();
            Array {
                index: state.ops.new_node(shape, Op::Unary(op), &[self.index]),
                builder: self.builder,
            }
        })
    }

    fn binary_op(self, rhs: Array, op: BinaryOp) -> Self {
        self.builder.with_state(|state| {
            let lhs_shape = state.ops.graph[self.index].shape.clone();
            let rhs_shape = state.ops.graph[rhs.index].shape.clone();
            let op_shape = lhs_shape.match_with_broadcast(&rhs_shape);

            let lhs = if op_shape == lhs_shape {
                self.index
            } else {
                state.ops.new_node(
                    op_shape.clone(),
                    Op::View(View::broadcast(&lhs_shape, &op_shape)),
                    &[self.index],
                )
            };
            let rhs = if op_shape == rhs_shape {
                rhs.index
            } else {
                state.ops.new_node(
                    op_shape.clone(),
                    Op::View(View::broadcast(&rhs_shape, &op_shape)),
                    &[rhs.index],
                )
            };

            Array {
                index: state.ops.new_node(op_shape, Op::Binary(op), &[lhs, rhs]),
                builder: self.builder,
            }
        })
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: isize) -> Self {
        self.builder.with_state(|state| {
            let shape = state.ops.graph[self.index].shape.reduce(axis);
            Array {
                index: state
                    .ops
                    .new_node(shape, Op::Reduce { reduce_op, axis }, &[self.index]),
                builder: self.builder,
            }
        })
    }

    pub fn one_hot(self, count: isize) -> Self {
        self.builder.with_state(|state| {
            let shape = state.ops.graph[self.index].shape.one_hot(count);
            Array {
                index: state
                    .ops
                    .new_node(shape, Op::Unary(UnaryOp::OneHot), &[self.index]),
                builder: self.builder,
            }
        })
    }

    pub fn reduce_max(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Max, axis)
    }
    pub fn reduce_sum(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Sum, axis)
    }

    fn reduce_onto_per_element(self, shape: &Shape) -> Self {
        let mut output = self;
        while let Some(axis) = output.shape().reduce_axis_onto_per_element(shape) {
            output = output.reduce_sum(axis);
        }
        output
    }

    pub fn exp(self) -> Self {
        self.unary_op(UnaryOp::Exp)
    }
    pub fn log(self) -> Self {
        self.unary_op(UnaryOp::Log)
    }

    pub fn matmul(self, rhs: Array) -> Self {
        self.builder.with_state(|state| {
            let shape = state.ops.graph[self.index]
                .shape
                .matrix_multiply(&state.ops.graph[rhs.index].shape);
            Array {
                index: state
                    .ops
                    .new_node(shape, Op::MatMul, &[self.index, rhs.index]),
                builder: self.builder,
            }
        })
    }

    pub fn transpose(self) -> Self {
        self.builder.with_state(|state| {
            let input_shape = &state.ops.graph[self.index].shape;
            let view = input_shape.identity_view().transposed();
            let output_shape = input_shape.transposed();
            Array {
                index: state
                    .ops
                    .new_node(output_shape, Op::View(view), &[self.index]),
                builder: self.builder,
            }
        })
    }

    pub fn shape(&self) -> Shape {
        self.builder
            .with_state(|state| state.ops.graph[self.index].shape.clone())
    }

    pub fn accumulate(&self, src: Array) {
        self.builder.with_state(|state| {
            assert_eq!(state.ops.graph[self.index].op, Op::Accumulate);
            assert_eq!(
                state.ops.graph[self.index].shape,
                state.ops.graph[src.index].shape
            );
            let arg = state.ops.graph.edges_directed(self.index, Incoming).count();
            state.ops.graph.add_edge(
                src.index,
                self.index,
                OpEdge {
                    arg,
                    view: state.ops.graph[src.index].shape.identity_view(),
                },
            );
        })
    }
}

impl<'builder> ops::Add for Array<'builder> {
    type Output = Array<'builder>;
    fn add(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Add)
    }
}
impl<'builder> ops::Sub for Array<'builder> {
    type Output = Array<'builder>;
    fn sub(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Sub)
    }
}
impl<'builder> ops::Mul for Array<'builder> {
    type Output = Array<'builder>;
    fn mul(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Mul)
    }
}
impl<'builder> ops::Div for Array<'builder> {
    type Output = Array<'builder>;
    fn div(self, rhs: Array) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Div)
    }
}
impl<'builder> ops::Neg for Array<'builder> {
    type Output = Array<'builder>;
    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOp::Neg)
    }
}

impl<'builder> ops::Mul<Array<'builder>> for f32 {
    type Output = Array<'builder>;
    fn mul(self, rhs: Array<'builder>) -> Self::Output {
        let lhs = rhs.builder.literal(self);
        lhs.binary_op(rhs, BinaryOp::Mul)
    }
}
impl<'builder> ops::Div<f32> for Array<'builder> {
    type Output = Array<'builder>;
    fn div(self, rhs: f32) -> Self::Output {
        let rhs = self.builder.literal(rhs);
        self.binary_op(rhs, BinaryOp::Div)
    }
}

impl<'builder> Tensor<'builder> {
    pub fn new(value: Array<'builder>, grad: Array<'builder>) -> Self {
        Self {
            index: TensorNodeIndexPair {
                value: value.index,
                grad: grad.index,
            },
            builder: value.builder,
        }
    }

    pub fn value(self) -> Array<'builder> {
        Array {
            index: self.index.value,
            builder: self.builder,
        }
    }

    pub fn grad(self) -> Array<'builder> {
        Array {
            index: self.index.grad,
            builder: self.builder,
        }
    }

    pub fn matmul(self, rhs: Tensor) -> Self {
        let a = self.value();
        let da = self.grad();
        let b = rhs.value();
        let db = rhs.grad();

        let c = a.matmul(b);
        let dc = self.builder.accumulator(c.shape());

        da.accumulate(dc.matmul(b.transpose()));
        db.accumulate(a.transpose().matmul(dc));

        Self::new(c, dc)
    }
}

impl<'builder> ops::Add for Tensor<'builder> {
    type Output = Tensor<'builder>;
    fn add(self, rhs: Tensor<'builder>) -> Self::Output {
        let a = self.value();
        let da = self.grad();
        let b = rhs.value();
        let db = rhs.grad();

        let c = a + b;
        let dc = self.builder.accumulator(c.shape());

        da.accumulate(dc.reduce_onto_per_element(&a.shape()));
        db.accumulate(dc.reduce_onto_per_element(&b.shape()));

        Self::new(c, dc)
    }
}

struct OpGraphBuilder {
    graph: OpGraph,
    colour: usize,
}

impl OpGraphBuilder {
    fn new_node(&mut self, shape: impl Into<Shape>, op: Op, inputs: &[OpNodeIndex]) -> OpNodeIndex {
        let shape = shape.into();
        let node_index = self.graph.add_node(OpNode {
            colour: self.colour,
            shape,
            op,
            cluster_id: None,
        });
        for (index, input) in inputs.iter().copied().enumerate() {
            self.graph.add_edge(
                input,
                node_index,
                OpEdge {
                    arg: index,
                    view: self.graph[input].shape.identity_view(),
                },
            );
        }
        node_index
    }
}

struct GraphBuilderState {
    ops: OpGraphBuilder,
    variables: SharedVariables,
    inputs: SparseSecondaryMap<Variable, TensorNodeIndexPair>,
    outputs: SparseSecondaryMap<Variable, OpNodeIndex>,
}

pub struct GraphBuilder {
    state: RefCell<GraphBuilderState>,
}

impl GraphBuilder {
    pub(crate) fn new(variables: SharedVariables) -> Self {
        Self {
            state: RefCell::new(GraphBuilderState {
                ops: OpGraphBuilder {
                    graph: Default::default(),
                    colour: 0,
                },
                variables,
                inputs: SparseSecondaryMap::new(),
                outputs: SparseSecondaryMap::new(),
            }),
        }
    }

    fn with_state<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut GraphBuilderState) -> T,
    {
        let mut data = self.state.borrow_mut();
        f(&mut data)
    }

    fn literal(&self, value: f32) -> Array {
        self.with_state(|state| Array {
            index: state.ops.new_node([1], Op::Literal(value), &[]),
            builder: self,
        })
    }

    pub fn input(&self, variable: Variable) -> Tensor {
        let index = self.with_state(|state| {
            let shape = state
                .variables
                .borrow()
                .get(variable)
                .unwrap()
                .shape
                .clone();
            let ops = &mut state.ops;
            *state
                .inputs
                .entry(variable)
                .unwrap()
                .or_insert_with(|| TensorNodeIndexPair {
                    value: ops.new_node(shape.clone(), Op::Input { variable }, &[]),
                    grad: ops.new_node(shape, Op::Accumulate, &[]),
                })
        });
        Tensor {
            index,
            builder: self,
        }
    }

    pub fn output(&self, variable: Variable, rhs: Array) {
        self.with_state(|state| {
            let shape = state.ops.graph[rhs.index].shape.clone();
            assert_eq!(
                state
                    .variables
                    .borrow()
                    .get(variable)
                    .unwrap()
                    .shape
                    .clone(),
                shape
            );

            // update the output node for this variable (remove any old one)
            let node_index =
                state
                    .ops
                    .new_node(shape.clone(), Op::Output { variable }, &[rhs.index]);
            if let Some(node_index) = state.outputs.insert(variable, node_index) {
                state.ops.graph.remove_node(node_index);
            }

            // ensure that if we read this variable again we read the latest value
            state.inputs.insert(
                variable,
                TensorNodeIndexPair {
                    value: rhs.index,
                    grad: state.ops.new_node(shape, Op::Accumulate, &[]),
                },
            );
        });
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> Array {
        self.with_state(|state| Array {
            index: state.ops.new_node(shape, Op::Accumulate, &[]),
            builder: self,
        })
    }

    pub fn next_colour(&self) {
        self.with_state(|state| {
            state.ops.colour += 1;
        })
    }

    pub fn build(self) -> Graph {
        self.with_state(|state| {
            Graph::new(
                SharedVariables::clone(&state.variables),
                state.ops.graph.clone(),
            )
        })
    }
}
