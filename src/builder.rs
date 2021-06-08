use crate::{prelude::*, schedule::*};
use petgraph::Incoming;
use std::{cell::UnsafeCell, ops};

#[derive(Clone, Copy)]
pub struct Array<'builder> {
    index: NodeIndex,
    builder: &'builder GraphBuilder,
}

impl<'builder> Array<'builder> {
    pub fn with_name(self, name: impl Into<String>) -> Self {
        self.builder.with_data(|data| {
            data.graph[self.index].name = Some(name.into());
        });
        self
    }

    fn per_element_unary_op(self, op: PerElementOp) -> Self {
        self.builder.with_data(|data| {
            let shape = data.graph[self.index].shape.clone();
            Array {
                index: data.new_node(shape, Op::PerElement(op), &[self.index]),
                builder: self.builder,
            }
        })
    }

    fn per_element_binary_op(self, rhs: Array, op: PerElementOp) -> Self {
        self.builder.with_data(|data| {
            let shape = data.graph[self.index]
                .shape
                .per_element(&data.graph[rhs.index].shape);
            Array {
                index: data.new_node(shape, Op::PerElement(op), &[self.index, rhs.index]),
                builder: self.builder,
            }
        })
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: isize) -> Self {
        self.builder.with_data(|data| {
            let shape = data.graph[self.index].shape.reduce(axis);
            Array {
                index: data.new_node(shape, Op::Reduce { reduce_op, axis }, &[self.index]),
                builder: self.builder,
            }
        })
    }

    pub fn one_hot(self, count: AxisLen) -> Self {
        self.builder.with_data(|data| {
            let shape = data.graph[self.index].shape.one_hot(count);
            Array {
                index: data.new_node(shape, Op::PerElement(PerElementOp::OneHot), &[self.index]),
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

    pub fn exp(self) -> Self {
        self.per_element_unary_op(PerElementOp::Exp)
    }
    pub fn log(self) -> Self {
        self.per_element_unary_op(PerElementOp::Log)
    }

    pub fn matmul(self, rhs: Array) -> Self {
        self.builder.with_data(|data| {
            let shape = data.graph[self.index]
                .shape
                .matrix_multiply(&data.graph[rhs.index].shape);
            Array {
                index: data.new_node(shape, Op::MatMul, &[self.index, rhs.index]),
                builder: self.builder,
            }
        })
    }

    pub fn transpose(self) -> Self {
        self.builder.with_data(|data| {
            let shape = data.graph[self.index].shape.transpose();
            Array {
                index: data.new_node(shape, Op::Transpose, &[self.index]),
                builder: self.builder,
            }
        })
    }

    pub fn shape(&self) -> Shape {
        self.builder
            .with_data(|data| data.graph[self.index].shape.clone())
    }

    pub fn accumulate(&mut self, src: Array) {
        self.builder.with_data(|data| {
            assert_eq!(data.graph[self.index].op, Op::Accumulate);
            assert_eq!(data.graph[self.index].shape, data.graph[src.index].shape);
            let arg = data.graph.edges_directed(self.index, Incoming).count();
            data.graph.add_edge(
                src.index,
                self.index,
                Edge {
                    arg,
                    transpose: false,
                },
            );
        })
    }
}

struct GraphBuilderData {
    graph: Graph,
    colour: usize,
}

impl GraphBuilderData {
    fn new_node(&mut self, shape: impl Into<Shape>, op: Op, inputs: &[NodeIndex]) -> NodeIndex {
        let node_index = self.graph.add_node(Node {
            name: None,
            colour: self.colour,
            shape: shape.into(),
            op,
            kernel_index: None,
        });
        for (index, input) in inputs.iter().cloned().enumerate() {
            self.graph.add_edge(
                input,
                node_index,
                Edge {
                    arg: index,
                    transpose: false,
                },
            );
        }
        node_index
    }
}

pub struct GraphBuilder {
    data: UnsafeCell<GraphBuilderData>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            data: UnsafeCell::new(GraphBuilderData {
                graph: Default::default(),
                colour: 0,
            }),
        }
    }

    fn with_data<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut GraphBuilderData) -> T,
    {
        let data = unsafe { self.data.get().as_mut().unwrap() };
        f(data)
    }

    fn literal(&self, value: f32) -> Array {
        self.with_data(|data| Array {
            index: data.new_node([1], Op::Literal(value), &[]),
            builder: self,
        })
    }

    pub fn input(&self, shape: impl Into<Shape>, name: impl Into<String>) -> Array {
        self.with_data(|data| {
            Array {
                index: data.new_node(shape, Op::Input, &[]),
                builder: self,
            }
            .with_name(name)
        })
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> Array {
        self.with_data(|data| Array {
            index: data.new_node(shape, Op::Accumulate, &[]),
            builder: self,
        })
    }

    pub fn next_colour(&self) {
        self.with_data(|data| {
            data.colour += 1;
        })
    }

    pub fn build(&self, roots: &[Array]) -> Schedule {
        self.with_data(|data| {
            let roots: Vec<_> = roots.iter().map(|a| a.index).collect();
            Schedule::new(data.graph.clone(), roots)
        })
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
