use crate::{graph::*, prelude::*, store::*};
use std::{cell::UnsafeCell, ops};

#[derive(Clone, Copy)]
pub struct Array<'builder> {
    index: NodeIndex,
    builder: &'builder GraphBuilder,
}

impl<'builder> Array<'builder> {
    pub fn with_name(self, name: impl Into<String>) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        data.nodes[self.index].name = Some(name.into());
        self
    }

    fn per_element_unary_op(self, op: PerElementOp) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        let shape = data.nodes[self.index].shape.clone();
        Array {
            index: data.new_node(shape, Op::PerElement(op), &[self.index]),
            builder,
        }
    }

    fn per_element_binary_op(self, rhs: Array, op: PerElementOp) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        let shape = data.nodes[self.index]
            .shape
            .per_element(&data.nodes[rhs.index].shape);
        Array {
            index: data.new_node(shape, Op::PerElement(op), &[self.index, rhs.index]),
            builder,
        }
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: isize) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        let shape = data.nodes[self.index].shape.reduce(axis);
        Array {
            index: data.new_node(shape, Op::Reduce { reduce_op, axis }, &[self.index]),
            builder,
        }
    }

    pub fn one_hot(self, count: AxisLen) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        let shape = data.nodes[self.index].shape.one_hot(count);
        Array {
            index: data.new_node(shape, Op::PerElement(PerElementOp::OneHot), &[self.index]),
            builder,
        }
    }

    pub fn reduce_max(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Max, axis)
    }
    pub fn reduce_sum(self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Sum, axis)
    }

    pub fn exp(self) -> Self {
        self.per_element_unary_op(PerElementOp::Exp)
    }
    pub fn log(self) -> Self {
        self.per_element_unary_op(PerElementOp::Log)
    }

    pub fn matmul(self, rhs: Array) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        let shape = data.nodes[self.index]
            .shape
            .matrix_multiply(&data.nodes[rhs.index].shape);
        Array {
            index: data.new_node(shape, Op::MatMul, &[self.index, rhs.index]),
            builder,
        }
    }

    pub fn transpose(self) -> Self {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        let shape = data.nodes[self.index].shape.transpose();
        Array {
            index: data.new_node(shape, Op::Transpose, &[self.index]),
            builder,
        }
    }

    pub fn shape(&self) -> Shape {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        data.nodes[self.index].shape.clone()
    }

    pub fn accumulate(&mut self, rhs: Array) {
        let builder = self.builder;
        let data = unsafe { builder.data.get().as_mut().unwrap() };
        assert_eq!(data.nodes[self.index].op, Op::Accumulate);
        assert_eq!(data.nodes[self.index].shape, data.nodes[rhs.index].shape);
        let node = &mut data.nodes[self.index];
        node.inputs.push(Input::Node {
            index: rhs.index,
            transpose: false,
        });
    }
}

struct GraphBuilderData {
    nodes: Store<NodeIndex, Node>,
    colour: usize,
}

impl GraphBuilderData {
    fn new_node(&mut self, shape: impl Into<Shape>, op: Op, inputs: &[NodeIndex]) -> NodeIndex {
        self.nodes.add(Node {
            name: None,
            colour: self.colour,
            shape: shape.into(),
            op,
            inputs: inputs
                .iter()
                .map(|&index| Input::Node {
                    index,
                    transpose: false,
                })
                .collect(),
        })
    }
}

pub struct GraphBuilder {
    data: UnsafeCell<GraphBuilderData>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            data: UnsafeCell::new(GraphBuilderData {
                nodes: Store::new(),
                colour: 0,
            }),
        }
    }

    fn literal(&self, value: f32) -> Array {
        let data = unsafe { self.data.get().as_mut().unwrap() };
        Array {
            index: data.new_node([1], Op::Literal(value), &[]),
            builder: self,
        }
    }

    pub fn variable(&self, shape: impl Into<Shape>, name: impl Into<String>) -> Array {
        let data = unsafe { self.data.get().as_mut().unwrap() };
        Array {
            index: data.new_node(shape, Op::Variable, &[]),
            builder: self,
        }
        .with_name(name)
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> Array {
        let data = unsafe { self.data.get().as_mut().unwrap() };
        Array {
            index: data.new_node(shape, Op::Accumulate, &[]),
            builder: self,
        }
    }

    pub fn next_colour(&self) {
        let data = unsafe { self.data.get().as_mut().unwrap() };
        data.colour += 1;
    }

    pub fn build(&self, roots: &[Array]) -> Graph {
        let data = unsafe { self.data.get().as_mut().unwrap() };
        let roots: Vec<_> = roots.iter().map(|a| a.index).collect();
        Graph::new(data.nodes.clone(), roots)
    }
}

impl<'builder> ops::Add for Array<'builder> {
    type Output = Array<'builder>;
    fn add(self, rhs: Array) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Add)
    }
}
impl<'builder> ops::Sub for Array<'builder> {
    type Output = Array<'builder>;
    fn sub(self, rhs: Array) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Sub)
    }
}
impl<'builder> ops::Mul for Array<'builder> {
    type Output = Array<'builder>;
    fn mul(self, rhs: Array) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Mul)
    }
}
impl<'builder> ops::Div for Array<'builder> {
    type Output = Array<'builder>;
    fn div(self, rhs: Array) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Div)
    }
}
impl<'builder> ops::Neg for Array<'builder> {
    type Output = Array<'builder>;
    fn neg(self) -> Self::Output {
        self.per_element_unary_op(PerElementOp::Neg)
    }
}

impl<'builder> ops::Mul<Array<'builder>> for f32 {
    type Output = Array<'builder>;
    fn mul(self, rhs: Array<'builder>) -> Self::Output {
        let lhs = rhs.builder.literal(self);
        lhs.per_element_binary_op(rhs, PerElementOp::Mul)
    }
}
impl<'builder> ops::Div<f32> for Array<'builder> {
    type Output = Array<'builder>;
    fn div(self, rhs: f32) -> Self::Output {
        let rhs = self.builder.literal(rhs);
        self.per_element_binary_op(rhs, PerElementOp::Div)
    }
}
