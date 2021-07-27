use crate::common::*;
use ordered_float::NotNan;
use petgraph::prelude::*;
use slotmap::SparseSecondaryMap;
use std::{cell::RefCell, convert::TryInto, ops};

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

pub trait IntoArray<'g> {
    fn into_array(self, graph: &'g Graph) -> Array<'g>;
}
impl<'g> IntoArray<'g> for Array<'g> {
    fn into_array(self, _graph: &'g Graph) -> Array<'g> {
        self
    }
}
impl<'g> IntoArray<'g> for f32 {
    fn into_array(self, graph: &'g Graph) -> Array<'g> {
        graph.literal(self)
    }
}
impl<'g> IntoArray<'g> for &Variable {
    fn into_array(self, graph: &'g Graph) -> Array<'g> {
        graph.read_variable(self)
    }
}

pub trait IntoDualArray<'g> {
    fn into_dual_array(self, graph: &'g Graph) -> DualArray<'g>;
}
impl<'g> IntoDualArray<'g> for DualArray<'g> {
    fn into_dual_array(self, _graph: &'g Graph) -> DualArray<'g> {
        self
    }
}
impl<'g> IntoDualArray<'g> for &Variable {
    fn into_dual_array(self, graph: &'g Graph) -> DualArray<'g> {
        graph.parameter(self)
    }
}

impl<'g> Array<'g> {
    pub fn graph(&self) -> &'g Graph {
        self.graph
    }

    pub fn clone_as_accumulator(&self) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id].shape;
            Array {
                node_id: state.ops.new_node(shape, Op::Unary(UnaryOp::Mov), &[]),
                graph: self.graph,
            }
        })
    }

    fn view(self, view: View) -> Self {
        if view.is_identity() {
            self
        } else {
            self.graph.with_state(|state| {
                let node_id = state
                    .ops
                    .new_node(view.output_shape, Op::Unary(UnaryOp::Mov), &[]);
                state
                    .ops
                    .graph
                    .add_edge(self.node_id, node_id, OpEdge { arg: 0, view });
                Array {
                    node_id,
                    graph: self.graph,
                }
            })
        }
    }

    fn broadcast(self, shape: &Shape) -> Self {
        self.view(View::broadcast(&self.shape(), shape))
    }

    fn unary_op(self, op: UnaryOp) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id].shape;
            Array {
                node_id: state.ops.new_node(shape, Op::Unary(op), &[self.node_id]),
                graph: self.graph,
            }
        })
    }

    fn binary_op(self, rhs: impl IntoArray<'g>, op: BinaryOp) -> Self {
        let rhs = rhs.into_array(self.graph);
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
        rhs: impl IntoArray<'g>,
        pass: impl IntoArray<'g>,
        fail: impl IntoArray<'g>,
    ) -> Self {
        let rhs = rhs.into_array(self.graph);
        let pass = pass.into_array(self.graph);
        let fail = fail.into_array(self.graph);

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
        let shape = self.shape().one_hot(count);
        self.graph.coord(shape, -1).select_eq(self, 1.0, 0.0)
    }

    pub fn reduce_max(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Max, self.shape().axis(axis))
    }
    pub fn reduce_sum(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Sum, self.shape().axis(axis))
    }

    pub fn argmax(self, axis: isize) -> Self {
        // implement with reduce_max for now
        let coord_or_zero = self.select_eq(self.reduce_max(axis), self.coord(axis), 0.0);
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

    pub fn select_eq(
        self,
        rhs: impl IntoArray<'g>,
        pass: impl IntoArray<'g>,
        fail: impl IntoArray<'g>,
    ) -> Self {
        self.compare_and_select(CompareMode::Eq, rhs, pass, fail)
    }
    pub fn select_gt(
        self,
        rhs: impl IntoArray<'g>,
        pass: impl IntoArray<'g>,
        fail: impl IntoArray<'g>,
    ) -> Self {
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
        let result = self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id]
                .shape
                .matmul(&state.ops.graph[rhs.node_id].shape);
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::MatMul, &[self.node_id, rhs.node_id]),
                graph: self.graph,
            }
        });
        let [r, m, n]: [usize; 3] = result.shape().as_slice().try_into().unwrap();
        if r == 1 {
            result.reshape([m, n])
        } else {
            result.reduce_sum(0)
        }
    }

    fn image_to_windows(self, filter: (usize, usize), pad: usize, stride: (usize, usize)) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id]
                .shape
                .image_to_windows(filter, pad, stride);
            Array {
                node_id: state.ops.new_node(
                    shape,
                    Op::ImageToWindows { stride, pad },
                    &[self.node_id],
                ),
                graph: self.graph,
            }
        })
    }

    fn windows_to_image(self, pad: usize, stride: (usize, usize)) -> Self {
        self.graph.with_state(|state| {
            let shape = state.ops.graph[self.node_id]
                .shape
                .windows_to_image(pad, stride);
            Array {
                node_id: state.ops.new_node(
                    shape,
                    Op::WindowsToImage { pad, stride },
                    &[self.node_id],
                ),
                graph: self.graph,
            }
        })
    }

    pub fn reshape(self, shape: impl Into<Shape>) -> Self {
        self.graph.with_state(|state| {
            let shape = shape.into();
            assert_eq!(
                state.ops.graph[self.node_id].shape.element_count(),
                shape.element_count()
            );
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::Unary(UnaryOp::Mov), &[self.node_id]),
                graph: self.graph,
            }
        })
    }

    pub fn transpose(self) -> Self {
        self.view(self.shape().identity_view().transposed())
    }

    pub fn shape(&self) -> Shape {
        self.graph
            .with_state(|state| state.ops.graph[self.node_id].shape)
    }

    pub fn max_pool(&self, axis: isize, size: usize) -> Self {
        let self_shape = self.shape();
        let self_axis = self_shape.axis(axis);

        let pool_shape = self_shape.pool(axis, size);
        let pool_axis = self_axis.inner();

        let mut output_shape = self_shape;
        output_shape[self_axis] /= size;

        self.reshape(pool_shape)
            .reduce_op(ReduceOp::Max, pool_axis)
            .reshape(output_shape)
    }

    pub fn accumulate(&self, src: Array) {
        self.graph.with_state(|state| {
            assert_eq!(state.ops.graph[self.node_id].op, Op::Unary(UnaryOp::Mov));
            assert_eq!(
                state.ops.graph[self.node_id].shape,
                state.ops.graph[src.node_id].shape
            );
            let src_id = if let Some(edge_ref) = state
                .ops
                .graph
                .edges_directed(self.node_id, Incoming)
                .next()
            {
                // remove the edge from the current source to this move
                let prev_edge_id = edge_ref.id();
                let prev_src_id = edge_ref.source();
                state.ops.graph.remove_edge(prev_edge_id);

                // accumulate with the given array
                state.ops.new_node(
                    state.ops.graph[src.node_id].shape,
                    Op::Binary(BinaryOp::Add),
                    &[prev_src_id, src.node_id],
                )
            } else {
                src.node_id
            };

            // add the edge to the move
            state.ops.graph.add_edge(
                src_id,
                self.node_id,
                OpEdge {
                    arg: 0,
                    view: state.ops.graph[src.node_id].shape.identity_view(),
                },
            );
        })
    }

    fn set_loss_grad(&self) {
        let grad_shape = self.shape();
        let mini_batch_size = *grad_shape.first().unwrap();
        let mini_batch_scale = self
            .graph
            .literal(1.0 / (mini_batch_size as f32))
            .broadcast(&grad_shape);
        self.graph.with_state(|state| {
            assert_eq!(state.ops.graph[self.node_id].op, Op::Unary(UnaryOp::Mov));
            assert_eq!(
                state
                    .ops
                    .graph
                    .edges_directed(self.node_id, Incoming)
                    .count(),
                0
            );
            state.ops.graph.add_edge(
                mini_batch_scale.node_id,
                self.node_id,
                OpEdge {
                    arg: 0,
                    view: grad_shape.identity_view(),
                },
            );
        })
    }
}

impl<'g, T> ops::Add<T> for Array<'g>
where
    T: IntoArray<'g>,
{
    type Output = Array<'g>;
    fn add(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Add)
    }
}
impl<'g> ops::Add<Array<'g>> for f32 {
    type Output = Array<'g>;
    fn add(self, rhs: Array<'g>) -> Self::Output {
        self.into_array(rhs.graph).binary_op(rhs, BinaryOp::Add)
    }
}

impl<'g, T> ops::Sub<T> for Array<'g>
where
    T: IntoArray<'g>,
{
    type Output = Array<'g>;
    fn sub(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Sub)
    }
}
impl<'g> ops::Sub<Array<'g>> for f32 {
    type Output = Array<'g>;
    fn sub(self, rhs: Array<'g>) -> Self::Output {
        self.into_array(rhs.graph).binary_op(rhs, BinaryOp::Sub)
    }
}

impl<'g, T> ops::Mul<T> for Array<'g>
where
    T: IntoArray<'g>,
{
    type Output = Array<'g>;
    fn mul(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Mul)
    }
}
impl<'g> ops::Mul<Array<'g>> for f32 {
    type Output = Array<'g>;
    fn mul(self, rhs: Array<'g>) -> Self::Output {
        self.into_array(rhs.graph).binary_op(rhs, BinaryOp::Mul)
    }
}

impl<'g, T> ops::Div<T> for Array<'g>
where
    T: IntoArray<'g>,
{
    type Output = Array<'g>;
    fn div(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Div)
    }
}
impl<'g> ops::Div<Array<'g>> for f32 {
    type Output = Array<'g>;
    fn div(self, rhs: Array<'g>) -> Self::Output {
        self.into_array(rhs.graph).binary_op(rhs, BinaryOp::Div)
    }
}

impl<'g> ops::Neg for Array<'g> {
    type Output = Array<'g>;
    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOp::Neg)
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

    pub fn into_inner(self) -> (Array<'g>, Array<'g>) {
        (self.value(), self.grad())
    }

    pub fn shape(&self) -> Shape {
        self.value().shape()
    }

    pub fn graph(&self) -> &'g Graph {
        self.graph
    }

    pub fn leaky_relu(self, leakiness: f32) -> Self {
        let (a, da) = self.into_inner();

        let b = a.select_gt(0.0, a, a * leakiness);

        let db = b.clone_as_accumulator();
        da.accumulate(a.select_gt(0.0, db, db * leakiness));

        Self::new(b, db)
    }

    pub fn matmul(self, rhs: impl IntoDualArray<'g>) -> Self {
        let rhs = rhs.into_dual_array(self.graph);

        let (a, da) = self.into_inner();
        let (b, db) = rhs.into_inner();

        let c = a.matmul(b);

        let dc = c.clone_as_accumulator();
        da.accumulate(dc.matmul(b.transpose()));
        db.accumulate(a.transpose().matmul(dc));

        Self::new(c, dc)
    }

    pub fn transpose(self) -> Self {
        let (a, da) = self.into_inner();

        let b = a.transpose();

        let db = b.clone_as_accumulator();
        da.accumulate(db.transpose());

        Self::new(b, db)
    }

    pub fn reshape(self, shape: impl Into<Shape>) -> Self {
        let old_shape = self.shape();
        let new_shape = shape.into();

        let (a, da) = self.into_inner();

        let b = a.reshape(new_shape);

        let db = b.clone_as_accumulator();
        da.accumulate(db.reshape(old_shape));

        Self::new(b, db)
    }

    fn image_to_windows(self, filter: (usize, usize), pad: usize, stride: (usize, usize)) -> Self {
        let (a, da) = self.into_inner();

        let b = a.image_to_windows(filter, pad, stride);

        let db = b.clone_as_accumulator();
        da.accumulate(db.windows_to_image(pad, stride));

        Self::new(b, db)
    }

    pub fn conv2d(
        self,
        filter: impl IntoDualArray<'g>,
        pad: usize,
        stride: (usize, usize),
    ) -> Self {
        let filter = filter.into_dual_array(self.graph);

        // copy and pad the input into windows that match the filter size
        let self_shape = self.shape();
        let filter_shape = filter.shape();
        assert_eq!(self_shape.len(), 4);
        assert_eq!(filter_shape.len(), 4);
        let [m, _input_h, _input_w, input_c]: [usize; 4] =
            self_shape.as_slice().try_into().unwrap();
        let [filter_oc, filter_h, filter_w, filter_ic]: [usize; 4] =
            filter_shape.as_slice().try_into().unwrap();
        assert_eq!(input_c, filter_ic);
        let windows = self.image_to_windows((filter_w, filter_h), pad, stride);

        // apply the filter using a matrix multiplication
        let windows_shape = windows.shape();
        let [windows_m, output_h, output_w, windows_fh, windows_fw, windows_ic]: [usize; 6] =
            windows_shape.as_slice().try_into().unwrap();
        assert_eq!(m, windows_m);
        assert_eq!(filter_h, windows_fh);
        assert_eq!(filter_w, windows_fw);
        assert_eq!(filter_ic, windows_ic);
        let a = windows.reshape([m * output_h * output_w, filter_h * filter_w * filter_ic]);
        let b = filter.reshape([filter_oc, filter_h * filter_w * filter_ic]);
        let c = a.matmul(b.transpose());

        // reshape output back to 4D
        c.reshape([m, output_h, output_w, filter_oc])
    }

    pub fn max_pool2d(self, filter: (usize, usize), stride: (usize, usize)) -> Self {
        let windows = self.image_to_windows(filter, 0, stride);

        let [m, output_h, output_w, filter_h, filter_w, input_c]: [usize; 6] =
            windows.shape().as_slice().try_into().unwrap();

        windows
            .reshape([m * output_h * output_w, filter_h * filter_w, input_c])
            .reduce_max(1)
            .reshape([m, output_h, output_w, input_c])
    }

    pub fn reduce_max(self, axis: isize) -> Self {
        let (a, da) = self.into_inner();

        let b = a.reduce_max(axis);

        let db = b.clone_as_accumulator();
        da.accumulate(a.select_eq(b, db, 0.0));

        Self::new(b, db)
    }

    pub fn set_loss(self) -> Array<'g> {
        self.grad().set_loss_grad();
        self.value()
    }
}

impl<'g, T> ops::Add<T> for DualArray<'g>
where
    T: IntoDualArray<'g>,
{
    type Output = DualArray<'g>;
    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into_dual_array(self.graph);

        let (a, da) = self.into_inner();
        let (b, db) = rhs.into_inner();

        let c = a + b;

        let dc = c.clone_as_accumulator();
        da.accumulate(dc.reduce_onto_per_element(&a.shape()));
        db.accumulate(dc.reduce_onto_per_element(&b.shape()));

        Self::new(c, dc)
    }
}

struct OpGraphState {
    graph: OpGraph,
    next_colour: usize,
    next_rand_uid: usize,
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
                    next_rand_uid: 0,
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

    pub fn rand(&self, shape: impl Into<Shape>) -> Array {
        self.with_state(|state| {
            let shape = shape.into();
            let uid = state.ops.next_rand_uid;
            state.ops.next_rand_uid += 1;
            Array {
                node_id: state
                    .ops
                    .new_node(shape, Op::BuiltIn(BuiltInOp::Rand { uid }), &[]),
                graph: self,
            }
        })
    }

    fn input(&self, variable: &Variable) -> GraphInput {
        self.with_state(|state| {
            let variable_id = variable.checked_id(&state.variables);
            let shape = state.variables.borrow().get(variable_id).unwrap().shape;
            let ops = &mut state.ops;
            *state
                .inputs
                .entry(variable_id)
                .unwrap()
                .or_insert_with(|| GraphInput {
                    value_node_id: ops.new_node(shape, Op::Input { variable_id }, &[]),
                    grad_node_id: Some(ops.new_node(shape, Op::Unary(UnaryOp::Mov), &[])),
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
            let shape = state.ops.graph[rhs.node_id].shape;
            assert_eq!(
                state.variables.borrow().get(variable_id).unwrap().shape,
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
            node_id: state.ops.new_node(shape, Op::Unary(UnaryOp::Mov), &[]),
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
