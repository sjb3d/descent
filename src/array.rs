use crate::{graph::*, prelude::*};
use std::{
    cell::{Cell, UnsafeCell},
    ops,
};

#[derive(Clone, Copy)]
pub struct Array<'builder> {
    index: NodeIndex,
    builder: &'builder GraphBuilder,
}

impl<'builder> Array<'builder> {
    pub fn with_name(self, name: impl Into<String>) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        graph[self.index].name = Some(name.into());
        self
    }

    fn per_element_unary_op(self, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.clone();
        Array {
            index: graph.new_node(
                builder.colour.get(),
                shape,
                Op::PerElement(op),
                &[self.index],
            ),
            builder,
        }
    }

    fn per_element_binary_op(self, rhs: Array, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.per_element(&graph[rhs.index].shape);
        Array {
            index: graph.new_node(
                builder.colour.get(),
                shape,
                Op::PerElement(op),
                &[self.index, rhs.index],
            ),
            builder,
        }
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: isize) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.reduce(axis);
        Array {
            index: graph.new_node(
                builder.colour.get(),
                shape,
                Op::Reduce { reduce_op, axis },
                &[self.index],
            ),
            builder,
        }
    }

    pub fn one_hot(self, count: AxisLen) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.one_hot(count);
        Array {
            index: graph.new_node(
                builder.colour.get(),
                shape,
                Op::PerElement(PerElementOp::OneHot),
                &[self.index],
            ),
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
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index]
            .shape
            .matrix_multiply(&graph[rhs.index].shape);
        Array {
            index: graph.new_node(
                builder.colour.get(),
                shape,
                Op::MatMul,
                &[self.index, rhs.index],
            ),
            builder,
        }
    }

    pub fn transpose(self) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.transpose();
        Array {
            index: graph.new_node(builder.colour.get(), shape, Op::Transpose, &[self.index]),
            builder,
        }
    }

    pub fn shape(&self) -> Shape {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        graph[self.index].shape.clone()
    }

    pub fn accumulate(&mut self, rhs: Array) {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        assert_eq!(graph[self.index].op, Op::Accumulate);
        assert_eq!(graph[self.index].shape, graph[rhs.index].shape);
        let node = &mut graph[self.index];
        node.inputs.push(rhs.index);
    }
}

pub struct GraphBuilder {
    graph: UnsafeCell<Graph>,
    colour: Cell<usize>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: UnsafeCell::new(Graph { nodes: Vec::new() }),
            colour: Cell::new(0),
        }
    }

    fn literal(&self, value: f32) -> Array {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        Array {
            index: graph.new_node(self.colour.get(), [1], Op::Literal(value), &[]),
            builder: self,
        }
    }

    pub fn variable(&self, shape: impl Into<Shape>, name: impl Into<String>) -> Array {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        Array {
            index: graph.new_node(self.colour.get(), shape, Op::Variable, &[]),
            builder: self,
        }
        .with_name(name)
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> Array {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        Array {
            index: graph.new_node(self.colour.get(), shape, Op::Accumulate, &[]),
            builder: self,
        }
    }

    pub fn next_colour(&self) {
        self.colour.set(self.colour.get() + 1);
    }

    pub fn build(self) -> Graph {
        self.graph.into_inner()
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
