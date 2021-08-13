use crate::common::{Graph, *};
use ordered_float::NotNan;
use petgraph::prelude::*;
use slotmap::SparseSecondaryMap;
use std::{cell::RefCell, convert::TryInto, ops};
use tinyvec::ArrayVec as TinyVec;

#[derive(Clone, Copy)]
pub struct Array<'s> {
    node_id: OpNodeId,
    scope: &'s Scope,
}

#[derive(Clone, Copy)]
pub struct DualArray<'s> {
    value_node_id: OpNodeId,
    grad_node_id: OpNodeId,
    scope: &'s Scope,
}

pub trait IntoArray<'s> {
    fn into_array(self, scope: &'s Scope) -> Array<'s>;
}
impl<'s> IntoArray<'s> for Array<'s> {
    fn into_array(self, _scope: &'s Scope) -> Array<'s> {
        self
    }
}
impl<'s> IntoArray<'s> for f32 {
    fn into_array(self, scope: &'s Scope) -> Array<'s> {
        scope.literal(self)
    }
}
impl<'s> IntoArray<'s> for &Variable {
    fn into_array(self, scope: &'s Scope) -> Array<'s> {
        scope.read_variable(self)
    }
}

pub trait IntoDualArray<'s> {
    fn into_dual_array(self, scope: &'s Scope) -> DualArray<'s>;
}
impl<'s> IntoDualArray<'s> for DualArray<'s> {
    fn into_dual_array(self, _graph: &'s Scope) -> DualArray<'s> {
        self
    }
}
impl<'s> IntoDualArray<'s> for &Variable {
    fn into_dual_array(self, scope: &'s Scope) -> DualArray<'s> {
        scope.parameter(self)
    }
}

impl<'s> Array<'s> {
    pub fn scope(&self) -> &'s Scope {
        self.scope
    }

    pub fn clone_as_accumulator(&self) -> Self {
        self.scope.with_state(|state| {
            let shape = state.ops[self.node_id].shape;
            Array {
                node_id: state
                    .ops
                    .new_node(state.next_colour, shape, Op::Unary(UnaryOp::Mov), &[]),
                scope: self.scope,
            }
        })
    }

    fn view(self, view: View) -> Self {
        self.scope.with_state(|state| {
            let node_id = state.ops.new_node(
                state.next_colour,
                view.output_shape,
                Op::Unary(UnaryOp::Mov),
                &[],
            );
            state
                .ops
                .add_edge(self.node_id, node_id, OpEdge { arg: 0, view });
            Array {
                node_id,
                scope: self.scope,
            }
        })
    }

    fn broadcast(self, shape: Shape) -> Self {
        self.view(View::broadcast(self.shape(), shape))
    }

    fn unbroadcast(self, shape: Shape) -> Self {
        let mut output = self;

        while output.shape().len() > shape.len() {
            output = output.reduce_sum(0, false);
        }
        assert_eq!(output.shape().len(), shape.len());

        for (index, (source, target)) in output
            .shape()
            .iter()
            .copied()
            .zip(shape.iter().copied())
            .enumerate()
        {
            if source != target {
                assert_eq!(target, 1);
                output = output.reduce_sum(index as isize, true);
            }
        }
        output
    }

    fn unary_op(self, op: UnaryOp) -> Self {
        self.scope.with_state(|state| {
            let shape = state.ops[self.node_id].shape;
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::Unary(op),
                    &[self.node_id],
                ),
                scope: self.scope,
            }
        })
    }

    fn binary_op(self, rhs: impl IntoArray<'s>, op: BinaryOp) -> Self {
        let rhs = rhs.into_array(self.scope);
        let op_shape = self.scope.with_state(|state| {
            state.ops[self.node_id]
                .shape
                .broadcast_with(state.ops[rhs.node_id].shape)
        });

        let lhs = self.broadcast(op_shape).node_id;
        let rhs = rhs.broadcast(op_shape).node_id;

        self.scope.with_state(|state| Array {
            node_id: state
                .ops
                .new_node(state.next_colour, op_shape, Op::Binary(op), &[lhs, rhs]),
            scope: self.scope,
        })
    }

    fn compare_and_select(
        self,
        compare_mode: CompareMode,
        rhs: impl IntoArray<'s>,
        pass: impl IntoArray<'s>,
        fail: impl IntoArray<'s>,
    ) -> Self {
        let rhs = rhs.into_array(self.scope);
        let pass = pass.into_array(self.scope);
        let fail = fail.into_array(self.scope);

        let op_shape = self.scope.with_state(|state| {
            state.ops[self.node_id]
                .shape
                .broadcast_with(state.ops[rhs.node_id].shape)
                .broadcast_with(state.ops[pass.node_id].shape)
                .broadcast_with(state.ops[fail.node_id].shape)
        });

        let lhs = self.broadcast(op_shape).node_id;
        let rhs = rhs.broadcast(op_shape).node_id;
        let pass = pass.broadcast(op_shape).node_id;
        let fail = fail.broadcast(op_shape).node_id;

        self.scope.with_state(|state| Array {
            node_id: state.ops.new_node(
                state.next_colour,
                op_shape,
                Op::CompareAndSelect(compare_mode),
                &[lhs, rhs, pass, fail],
            ),
            scope: self.scope,
        })
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: Axis) -> Self {
        let shape = self.shape();
        if shape[axis] == 1 {
            self
        } else {
            self.scope.with_state(|state| {
                let shape = shape.reduce(axis);
                Array {
                    node_id: state.ops.new_node(
                        state.next_colour,
                        shape,
                        Op::Reduce { reduce_op, axis },
                        &[self.node_id],
                    ),
                    scope: self.scope,
                }
            })
        }
    }

    fn keep_axis(self, axis: Axis, keep_axis: bool) -> Self {
        if keep_axis {
            self
        } else {
            self.remove_axis(axis)
        }
    }

    pub fn one_hot(self, count: usize) -> Self {
        let shape = self.shape().one_hot(count);
        self.scope.coord(shape, -1).select_eq(self, 1.0, 0.0)
    }

    pub fn reduce_max(self, axis: isize, keep_axis: bool) -> Self {
        let axis = self.shape().axis(axis);
        self.reduce_op(ReduceOp::Max, axis)
            .keep_axis(axis, keep_axis)
    }
    pub fn reduce_sum(self, axis: isize, keep_axis: bool) -> Self {
        let axis = self.shape().axis(axis);
        self.reduce_op(ReduceOp::Sum, axis)
            .keep_axis(axis, keep_axis)
    }

    pub fn argmax(self, axis: isize, keep_axis: bool) -> Self {
        // implement with reduce_max for now
        let coord_or_zero = self.select_eq(self.reduce_max(axis, true), self.coord(axis), 0.0);
        coord_or_zero.reduce_max(axis, keep_axis)
    }

    pub fn coord(self, axis: isize) -> Self {
        self.scope.coord(self.shape(), axis)
    }

    pub fn select_eq(
        self,
        rhs: impl IntoArray<'s>,
        pass: impl IntoArray<'s>,
        fail: impl IntoArray<'s>,
    ) -> Self {
        self.compare_and_select(CompareMode::Eq, rhs, pass, fail)
    }
    pub fn select_gt(
        self,
        rhs: impl IntoArray<'s>,
        pass: impl IntoArray<'s>,
        fail: impl IntoArray<'s>,
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

    pub fn matmul(self, rhs: impl IntoArray<'s>) -> Self {
        let axis = Axis::from_index(0);
        let lhs = self.insert_axis(axis);
        let rhs = rhs.into_array(self.scope).insert_axis(axis);
        let result = lhs.batched_matmul(rhs, MatMulOutputMode::Batches);
        result.remove_axis(axis)
    }

    pub(crate) fn batched_matmul(self, rhs: Array, output_mode: MatMulOutputMode) -> Self {
        let chunks = self.scope.with_state(|state| {
            let shape = state.ops[self.node_id]
                .shape
                .batched_matmul(state.ops[rhs.node_id].shape, output_mode);
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::MatMul { output_mode },
                    &[self.node_id, rhs.node_id],
                ),
                scope: self.scope,
            }
        });
        let output = chunks.reduce_sum(0, false);
        match output_mode {
            MatMulOutputMode::Batches => output,
            MatMulOutputMode::Rows => output.permute_axes(&[1, 0, 2]),
        }
    }

    pub(crate) fn insert_axis(self, axis: Axis) -> Self {
        self.view(self.shape().identity_view().insert_axis(axis))
    }

    pub(crate) fn remove_axis(self, axis: Axis) -> Self {
        self.view(self.shape().identity_view().remove_axis(axis))
    }

    pub(crate) fn permute_axes(self, perm: &[usize]) -> Self {
        self.view(self.shape().identity_view().permute_axes(perm))
    }

    pub(crate) fn pad(self, axis: isize, pad: usize) -> Self {
        if pad == 0 {
            return self;
        }
        let shape = self.shape();
        self.view(shape.padded_view(shape.axis(axis), pad))
    }

    pub(crate) fn unpad(self, axis: isize, pad: usize) -> Self {
        if pad == 0 {
            return self;
        }
        self.scope.with_state(|state| {
            let shape = state.ops[self.node_id].shape;
            let axis = shape.axis(axis);
            let shape = shape.unpad(axis, pad);
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::Unpad { axis, pad },
                    &[self.node_id],
                ),
                scope: self.scope,
            }
        })
    }

    pub(crate) fn pad_image(self, pad: usize) -> Self {
        self.pad(-3, pad).pad(-2, pad)
    }

    pub(crate) fn unpad_image(self, pad: usize) -> Self {
        self.unpad(-3, pad).unpad(-2, pad)
    }

    fn image_to_windows(
        self,
        filter: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> Self {
        let input_shape = self.shape();
        let in_y_axis = input_shape.axis(-3);
        let in_x_axis = input_shape.axis(-2);
        let in_c_axis = input_shape.axis(-1);

        let mut view = input_shape.identity_view();

        view.output_shape = input_shape.image_to_windows(filter, stride, groups);
        let group_nc = view.output_shape[SignedIndex(-1)];
        let (stride_w, stride_h) = stride;

        view.output_mapping.truncate(view.output_shape.len() - 6);
        view.output_mapping.push(
            input_shape
                .identity_mapping(in_y_axis)
                .stepped(stride_h as isize),
        );
        view.output_mapping.push(
            input_shape
                .identity_mapping(in_x_axis)
                .stepped(stride_w as isize),
        );
        view.output_mapping.push(
            input_shape
                .identity_mapping(in_c_axis)
                .stepped(group_nc as isize),
        );
        view.output_mapping
            .push(input_shape.identity_mapping(in_y_axis));
        view.output_mapping
            .push(input_shape.identity_mapping(in_x_axis));
        view.output_mapping
            .push(input_shape.identity_mapping(in_c_axis));

        self.view(view)
    }

    fn windows_to_image(self, stride: (usize, usize)) -> Self {
        self.scope.with_state(|state| {
            let shape = state.ops[self.node_id].shape.windows_to_image(stride);
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::WindowsToImage { stride },
                    &[self.node_id],
                ),
                scope: self.scope,
            }
        })
    }

    pub fn reshape(self, shape: impl Into<Shape>) -> Self {
        self.scope.with_state(|state| {
            let shape = shape.into();
            assert_eq!(
                state.ops[self.node_id].shape.element_count(),
                shape.element_count()
            );
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::Unary(UnaryOp::Mov),
                    &[self.node_id],
                ),
                scope: self.scope,
            }
        })
    }

    pub fn transpose(self) -> Self {
        self.view(self.shape().identity_view().transposed())
    }

    pub fn shape(&self) -> Shape {
        self.scope.with_state(|state| state.ops[self.node_id].shape)
    }

    pub fn accumulate(&self, src: Array) {
        self.scope.with_state(|state| {
            assert_eq!(state.ops[self.node_id].op, Op::Unary(UnaryOp::Mov));
            assert_eq!(state.ops[self.node_id].shape, state.ops[src.node_id].shape);
            let src_id =
                if let Some(edge_ref) = state.ops.edges_directed(self.node_id, Incoming).next() {
                    // remove the edge from the current source to this move
                    let prev_edge_id = edge_ref.id();
                    let prev_src_id = edge_ref.source();
                    state.ops.remove_edge(prev_edge_id);

                    // accumulate with the given array
                    state.ops.new_node(
                        state.next_colour,
                        state.ops[src.node_id].shape,
                        Op::Binary(BinaryOp::Add),
                        &[prev_src_id, src.node_id],
                    )
                } else {
                    src.node_id
                };

            // add the edge to the move
            state.ops.add_edge(
                src_id,
                self.node_id,
                OpEdge {
                    arg: 0,
                    view: state.ops[src.node_id].shape.identity_view(),
                },
            );
        })
    }

    fn set_loss_grad(&self) {
        let grad_shape = self.shape();
        let mini_batch_size = grad_shape[0];
        let mini_batch_scale = self
            .scope
            .literal(1.0 / (mini_batch_size as f32))
            .broadcast(grad_shape);
        self.scope.with_state(|state| {
            assert_eq!(state.ops[self.node_id].op, Op::Unary(UnaryOp::Mov));
            assert_eq!(state.ops.edges_directed(self.node_id, Incoming).count(), 0);
            state.ops.add_edge(
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

impl<'s, T> ops::Add<T> for Array<'s>
where
    T: IntoArray<'s>,
{
    type Output = Array<'s>;
    fn add(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Add)
    }
}
impl<'s> ops::Add<Array<'s>> for f32 {
    type Output = Array<'s>;
    fn add(self, rhs: Array<'s>) -> Self::Output {
        self.into_array(rhs.scope).binary_op(rhs, BinaryOp::Add)
    }
}

impl<'s, T> ops::Sub<T> for Array<'s>
where
    T: IntoArray<'s>,
{
    type Output = Array<'s>;
    fn sub(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Sub)
    }
}
impl<'s> ops::Sub<Array<'s>> for f32 {
    type Output = Array<'s>;
    fn sub(self, rhs: Array<'s>) -> Self::Output {
        self.into_array(rhs.scope).binary_op(rhs, BinaryOp::Sub)
    }
}

impl<'s, T> ops::Mul<T> for Array<'s>
where
    T: IntoArray<'s>,
{
    type Output = Array<'s>;
    fn mul(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Mul)
    }
}
impl<'s> ops::Mul<Array<'s>> for f32 {
    type Output = Array<'s>;
    fn mul(self, rhs: Array<'s>) -> Self::Output {
        self.into_array(rhs.scope).binary_op(rhs, BinaryOp::Mul)
    }
}

impl<'s, T> ops::Div<T> for Array<'s>
where
    T: IntoArray<'s>,
{
    type Output = Array<'s>;
    fn div(self, rhs: T) -> Self::Output {
        self.binary_op(rhs, BinaryOp::Div)
    }
}
impl<'s> ops::Div<Array<'s>> for f32 {
    type Output = Array<'s>;
    fn div(self, rhs: Array<'s>) -> Self::Output {
        self.into_array(rhs.scope).binary_op(rhs, BinaryOp::Div)
    }
}

impl<'s> ops::Neg for Array<'s> {
    type Output = Array<'s>;
    fn neg(self) -> Self::Output {
        self.unary_op(UnaryOp::Neg)
    }
}

impl<'s> DualArray<'s> {
    pub fn new(value: Array<'s>, grad: Array<'s>) -> Self {
        Self {
            value_node_id: value.node_id,
            grad_node_id: grad.node_id,
            scope: value.scope,
        }
    }

    pub fn new_from_value(value: Array<'s>) -> Self {
        Self::new(value, value.clone_as_accumulator())
    }

    pub fn value(self) -> Array<'s> {
        Array {
            node_id: self.value_node_id,
            scope: self.scope,
        }
    }

    pub fn grad(self) -> Array<'s> {
        Array {
            node_id: self.grad_node_id,
            scope: self.scope,
        }
    }

    pub fn into_inner(self) -> (Array<'s>, Array<'s>) {
        (self.value(), self.grad())
    }

    pub fn shape(&self) -> Shape {
        self.value().shape()
    }

    pub fn scope(&self) -> &'s Scope {
        self.scope
    }

    pub fn square(self) -> Self {
        self * self
    }

    pub fn leaky_relu(self, leakiness: f32) -> Self {
        let (a, da) = self.into_inner();

        let b = a.select_gt(0.0, a, a * leakiness);

        let db = b.clone_as_accumulator();
        da.accumulate(a.select_gt(0.0, db, db * leakiness));

        Self::new(b, db)
    }

    pub(crate) fn batched_matmul(self, rhs: DualArray, output_mode: MatMulOutputMode) -> Self {
        let (a, da) = self.into_inner();
        let (b, db) = rhs.into_inner();

        let c = a.batched_matmul(b, output_mode);

        let dc = c.clone_as_accumulator();
        da.accumulate(dc.batched_matmul(b.transpose(), MatMulOutputMode::Batches));
        db.accumulate(a.transpose().batched_matmul(dc, MatMulOutputMode::Batches));

        Self::new(c, dc)
    }

    pub fn matmul(self, rhs: impl IntoDualArray<'s>) -> Self {
        let axis = Axis::from_index(0);
        let lhs = self.insert_axis(axis);
        let rhs = rhs.into_dual_array(self.scope).insert_axis(axis);
        let result = lhs.batched_matmul(rhs, MatMulOutputMode::Batches);
        result.remove_axis(axis)
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

    pub(crate) fn pad_image(self, pad: usize) -> Self {
        let (a, da) = self.into_inner();

        let b = a.pad_image(pad);

        let db = b.clone_as_accumulator();
        da.accumulate(db.unpad_image(pad));

        Self::new(b, db)
    }

    pub(crate) fn image_to_windows(
        self,
        filter: (usize, usize),
        stride: (usize, usize),
        groups: usize,
    ) -> Self {
        let (a, da) = self.into_inner();

        let b = a.image_to_windows(filter, stride, groups);

        let db = b.clone_as_accumulator();
        da.accumulate(db.windows_to_image(stride));

        Self::new(b, db)
    }

    pub fn next_colour(self) -> Self {
        self.scope().next_colour();
        self
    }

    pub fn map<F>(self, f: F) -> Self
    where
        F: FnOnce(DualArray<'s>) -> DualArray<'s>,
    {
        f(self)
    }

    pub fn conv2d(
        self,
        filter: impl IntoDualArray<'s>,
        pad: usize,
        stride: (usize, usize),
    ) -> Self {
        let filter = filter.into_dual_array(self.scope);

        // pad the input
        let padded = self.pad_image(pad);

        // copy the input into windows that match the filter size
        let padded_shape = padded.shape();
        let filter_shape = filter.shape();
        let [input_m, _input_h, _input_w, input_nc]: [usize; 4] = padded_shape.try_into().unwrap();
        let [filter_g, filter_oc, filter_h, filter_w, filter_ic]: [usize; 5] =
            filter_shape.try_into().unwrap();
        assert_eq!(input_nc, filter_g * filter_ic);
        let windows = padded.image_to_windows((filter_w, filter_h), stride, filter_g);

        // apply the filter using a matrix multiplication
        let windows_shape = windows.shape();
        let [windows_m, output_h, output_w, windows_g, windows_fh, windows_fw, windows_nc]: [usize;
            7] = windows_shape.try_into().unwrap();
        assert_eq!(input_m, windows_m);
        assert_eq!(filter_g, windows_g);
        assert_eq!(filter_h, windows_fh);
        assert_eq!(filter_w, windows_fw);
        assert_eq!(filter_ic, windows_nc);
        let a = windows
            .reshape([
                input_m * output_h * output_w,
                filter_g,
                filter_h * filter_w * filter_ic,
            ])
            .permute_axes(&[1, 0, 2]);
        let b = filter.reshape([filter_g, filter_oc, filter_h * filter_w * filter_ic]);
        let c = a.batched_matmul(b.transpose(), MatMulOutputMode::Rows);

        // reshape output back to 4D
        c.permute_axes(&[1, 0, 2])
            .reshape([input_m, output_h, output_w, filter_g * filter_oc])
    }

    pub fn max_pool2d(self, filter: (usize, usize), stride: (usize, usize)) -> Self {
        let windows = self.image_to_windows(filter, stride, 1);

        let [m, output_h, output_w, groups, filter_h, filter_w, group_nc]: [usize; 7] =
            windows.shape().try_into().unwrap();

        windows
            .reshape([
                m * output_h * output_w * groups,
                filter_h * filter_w,
                group_nc,
            ])
            .reduce_max(1, true)
            .reshape([m, output_h, output_w, groups * group_nc])
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: Axis) -> Self {
        let (a, da) = self.into_inner();

        let b = a.reduce_op(reduce_op, axis);

        let db = b.clone_as_accumulator();
        match reduce_op {
            ReduceOp::Max => da.accumulate(a.select_eq(b, db, 0.0)),
            ReduceOp::Sum => da.accumulate(db.broadcast(da.shape())),
        }

        Self::new(b, db)
    }

    fn insert_axis(self, axis: Axis) -> Self {
        let (a, da) = self.into_inner();

        let b = a.insert_axis(axis);

        let db = b.clone_as_accumulator();
        da.accumulate(db.remove_axis(axis));

        Self::new(b, db)
    }

    fn remove_axis(self, axis: Axis) -> Self {
        let (a, da) = self.into_inner();

        let b = a.remove_axis(axis);

        let db = b.clone_as_accumulator();
        da.accumulate(db.insert_axis(axis));

        Self::new(b, db)
    }

    fn keep_axis(self, axis: Axis, keep_axis: bool) -> Self {
        if keep_axis {
            self
        } else {
            self.remove_axis(axis)
        }
    }

    pub fn reduce_sum(self, axis: isize, keep_axis: bool) -> Self {
        let axis = self.shape().axis(axis);
        self.reduce_op(ReduceOp::Sum, axis)
            .keep_axis(axis, keep_axis)
    }
    pub fn reduce_max(self, axis: isize, keep_axis: bool) -> Self {
        let axis = self.shape().axis(axis);
        self.reduce_op(ReduceOp::Max, axis)
            .keep_axis(axis, keep_axis)
    }

    pub fn flatten(self) -> Self {
        let shape = self.shape();
        let (first, suffix) = shape.split_first().unwrap();
        let m = *first;
        let count = suffix.iter().copied().product();
        self.reshape([m, count])
    }

    pub fn set_loss(self) -> Array<'s> {
        self.grad().set_loss_grad();
        self.value()
    }

    pub(crate) fn permute_axes(self, perm: &[usize]) -> Self {
        let mut inv_perm: TinyVec<[usize; MAX_DIM]> = TinyVec::new();
        inv_perm.set_len(perm.len());
        for (src, dst) in perm.iter().copied().enumerate() {
            inv_perm[dst] = src;
        }
        assert!(inv_perm
            .iter()
            .copied()
            .enumerate()
            .all(|(dst, src)| perm[src] == dst));

        let (a, da) = self.into_inner();

        let b = a.permute_axes(perm);

        let db = b.clone_as_accumulator();
        da.accumulate(db.permute_axes(&inv_perm));

        Self::new(b, db)
    }
}

impl<'s, T> ops::Add<T> for DualArray<'s>
where
    T: IntoDualArray<'s>,
{
    type Output = DualArray<'s>;
    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into_dual_array(self.scope);

        let (a, da) = self.into_inner();
        let (b, db) = rhs.into_inner();

        let c = a + b;

        let dc = c.clone_as_accumulator();
        da.accumulate(dc.unbroadcast(a.shape()));
        db.accumulate(dc.unbroadcast(b.shape()));

        Self::new(c, dc)
    }
}

impl<'s, T> ops::Sub<T> for DualArray<'s>
where
    T: IntoDualArray<'s>,
{
    type Output = DualArray<'s>;
    fn sub(self, rhs: T) -> Self::Output {
        let rhs = rhs.into_dual_array(self.scope);

        let (a, da) = self.into_inner();
        let (b, db) = rhs.into_inner();

        let c = a - b;

        let dc = c.clone_as_accumulator();
        da.accumulate(dc.unbroadcast(a.shape()));
        db.accumulate(-dc.unbroadcast(b.shape()));

        Self::new(c, dc)
    }
}

impl<'s, T> ops::Mul<T> for DualArray<'s>
where
    T: IntoDualArray<'s>,
{
    type Output = DualArray<'s>;
    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into_dual_array(self.scope);

        let (a, da) = self.into_inner();
        let (b, db) = rhs.into_inner();

        let c = a * b;

        let dc = c.clone_as_accumulator();
        da.accumulate((b * dc).unbroadcast(a.shape()));
        db.accumulate((a * dc).unbroadcast(b.shape()));

        Self::new(c, dc)
    }
}

#[derive(Clone, Copy)]
struct GraphInput {
    value_node_id: OpNodeId,
    grad_node_id: Option<OpNodeId>,
}

struct ScopeState {
    ops: OpGraph,
    next_colour: usize,
    next_rand_uid: usize,
    variables: SharedVariables,
    inputs: SparseSecondaryMap<VariableId, GraphInput>,
    outputs: SparseSecondaryMap<VariableId, OpNodeId>,
}

pub struct Scope {
    state: RefCell<ScopeState>,
}

impl Scope {
    pub(crate) fn new(variables: SharedVariables) -> Self {
        Self {
            state: RefCell::new(ScopeState {
                ops: Default::default(),
                next_colour: 0,
                next_rand_uid: 0,
                variables,
                inputs: SparseSecondaryMap::new(),
                outputs: SparseSecondaryMap::new(),
            }),
        }
    }

    fn with_state<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut ScopeState) -> T,
    {
        let mut data = self.state.borrow_mut();
        f(&mut data)
    }

    pub fn literal(&self, value: f32) -> Array {
        self.with_state(|state| Array {
            node_id: state.ops.new_node(
                state.next_colour,
                [1],
                Op::Literal(NotNan::new(value).unwrap()),
                &[],
            ),
            scope: self,
        })
    }

    pub fn coord(&self, shape: impl Into<Shape>, axis: isize) -> Array {
        self.with_state(|state| {
            let shape = shape.into();
            let axis = shape.axis(axis);
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::BuiltIn(BuiltInOp::Coord { axis }),
                    &[],
                ),
                scope: self,
            }
        })
    }

    pub fn rand(&self, shape: impl Into<Shape>) -> Array {
        self.with_state(|state| {
            let shape = shape.into();
            let uid = state.next_rand_uid;
            state.next_rand_uid += 1;
            Array {
                node_id: state.ops.new_node(
                    state.next_colour,
                    shape,
                    Op::BuiltIn(BuiltInOp::Rand { uid }),
                    &[],
                ),
                scope: self,
            }
        })
    }

    fn input(&self, variable: &Variable) -> GraphInput {
        self.with_state(|state| {
            let variable_id = variable.checked_id(&state.variables);
            let shape = state.variables.borrow().get(variable_id).unwrap().shape;
            let next_colour = state.next_colour;
            let ops = &mut state.ops;
            *state
                .inputs
                .entry(variable_id)
                .unwrap()
                .or_insert_with(|| GraphInput {
                    value_node_id: ops.new_node(next_colour, shape, Op::Input { variable_id }, &[]),
                    grad_node_id: Some(ops.new_node(
                        next_colour,
                        shape,
                        Op::Unary(UnaryOp::Mov),
                        &[],
                    )),
                })
        })
    }

    pub fn parameter(&self, variable: &Variable) -> DualArray {
        let input = self.input(variable);
        DualArray {
            value_node_id: input.value_node_id,
            grad_node_id: input.grad_node_id.unwrap(),
            scope: self,
        }
    }

    pub fn read_variable(&self, variable: &Variable) -> Array {
        let input = self.input(variable);
        Array {
            node_id: input.value_node_id,
            scope: self,
        }
    }

    pub fn write_variable(&self, variable: &Variable, rhs: Array) {
        self.with_state(|state| {
            let variable_id = variable.checked_id(&state.variables);
            let shape = state.ops[rhs.node_id].shape;
            assert_eq!(
                state.variables.borrow().get(variable_id).unwrap().shape,
                shape
            );

            // update the output node for this variable (remove any old one)
            let node_id = state.ops.new_node(
                state.next_colour,
                shape,
                Op::Output { variable_id },
                &[rhs.node_id],
            );
            if let Some(node_id) = state.outputs.insert(variable_id, node_id) {
                state.ops.remove_node(node_id);
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

    pub fn update_variable<'s>(
        &'s self,
        variable: &Variable,
        f: impl FnOnce(Array<'s>) -> Array<'s>,
    ) -> Array<'s> {
        let result = f(self.read_variable(variable));
        self.write_variable(variable, result);
        result
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> Array {
        self.with_state(|state| Array {
            node_id: state
                .ops
                .new_node(state.next_colour, shape, Op::Unary(UnaryOp::Mov), &[]),
            scope: self,
        })
    }

    pub fn next_colour(&self) {
        self.with_state(|state| {
            state.next_colour += 1;
        })
    }

    pub fn trainable_parameters(&self) -> Vec<Variable> {
        self.with_state(|state| {
            let mut v = Vec::new();
            for node in state.ops.node_weights() {
                if let Op::Input { variable_id } = node.op {
                    let variable = Variable::new(variable_id, &state.variables);
                    if variable.is_trainable() {
                        v.push(variable);
                    }
                }
            }
            v
        })
    }

    pub fn build_graph(self) -> Graph {
        self.with_state(|state| {
            Graph::new(SharedVariables::clone(&state.variables), state.ops.clone())
        })
    }
}
