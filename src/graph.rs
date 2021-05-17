use arrayvec::ArrayVec;
use std::{cell::UnsafeCell, convert::TryInto, fmt, ops};

pub const MAX_DIMS: usize = 4;

#[derive(Debug, Clone)]
pub struct Shape(ArrayVec<isize, MAX_DIMS>);

impl Shape {
    fn from_per_element(lhs: &Shape, rhs: &Shape) -> Self {
        // broadcast axes from 1 => n where necessary
        assert_eq!(lhs.0.len(), rhs.0.len());
        Shape(
            lhs.0
                .iter()
                .cloned()
                .zip(rhs.0.iter().cloned())
                .map(|(a, b)| match (a, b) {
                    (1, n) => n,
                    (m, 1) => m,
                    (m, n) => {
                        if m != -1 && n != -1 {
                            assert_eq!(m, n);
                        }
                        m
                    }
                })
                .collect(),
        )
    }

    fn from_matrix_multiply(lhs: &Shape, rhs: &Shape) -> Self {
        assert_eq!(rhs.0.len(), 2);
        let (a_last, a_prefix) = lhs.0.split_last().unwrap();
        let (b_first, b_suffix) = rhs.0.split_first().unwrap();
        assert_eq!(a_last, b_first);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(a_prefix).unwrap();
        v.try_extend_from_slice(b_suffix).unwrap();
        Shape(v)
    }

    fn from_transpose(shape: &Shape) -> Self {
        assert_eq!(shape.0.len(), 2);
        Shape(shape.0.iter().cloned().rev().collect())
    }

    fn from_reduce(shape: &Shape) -> Self {
        let (_, prefix) = shape.0.split_last().unwrap();
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(prefix).unwrap();
        v.push(1);
        Shape(v)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_list(self.0.iter(), f)
    }
}

impl<const N: usize> From<[isize; N]> for Shape {
    fn from(s: [isize; N]) -> Self {
        Self(s.iter().cloned().collect())
    }
}

struct Array {
    shape: Shape,
    name: Option<String>,
}

const MAX_INPUTS: usize = 3;
const MAX_OUTPUTS: usize = 2;

#[derive(Debug, Clone, Copy)]
struct ArrayIndex(usize);

#[derive(Clone, Copy)]
pub struct BuilderArray<'builder> {
    index: ArrayIndex,
    builder: &'builder GraphBuilder,
}

impl<'builder> BuilderArray<'builder> {
    fn per_element_unary_op(&self, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let output_index = graph.array(Shape::from_reduce(&graph[self.index].shape), None);
        graph.ops.push(Op::new(
            OpType::PerElement(op),
            &[self.index],
            &[output_index],
        ));
        BuilderArray {
            index: output_index,
            builder,
        }
    }

    fn per_element_binary_op(&self, rhs: BuilderArray, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let output_index = graph.array(
            Shape::from_per_element(&graph[self.index].shape, &graph[rhs.index].shape),
            None,
        );
        graph.ops.push(Op::new(
            OpType::PerElement(op),
            &[self.index, rhs.index],
            &[output_index],
        ));
        BuilderArray {
            index: output_index,
            builder,
        }
    }

    fn reduce_op(&self, op: ReduceOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let output_index = graph.array(graph[self.index].shape.clone(), None);
        graph
            .ops
            .push(Op::new(OpType::Reduce(op), &[self.index], &[output_index]));
        BuilderArray {
            index: output_index,
            builder,
        }
    }

    pub fn reduce_max(&self) -> Self {
        self.reduce_op(ReduceOp::Max)
    }
    pub fn reduce_sum(&self) -> Self {
        self.reduce_op(ReduceOp::Sum)
    }

    pub fn exp(&self) -> Self {
        self.per_element_unary_op(PerElementOp::Exp)
    }
    pub fn log(&self) -> Self {
        self.per_element_unary_op(PerElementOp::Log)
    }

    pub fn matmul(&self, rhs: BuilderArray) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let output_index = graph.array(
            Shape::from_matrix_multiply(&graph[self.index].shape, &graph[rhs.index].shape),
            None,
        );
        graph.ops.push(Op::new(
            OpType::MatMul,
            &[self.index, rhs.index],
            &[output_index],
        ));
        BuilderArray {
            index: output_index,
            builder,
        }
    }

    pub fn transpose(&self) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let output_index = graph.array(Shape::from_transpose(&graph[self.index].shape), None);
        graph
            .ops
            .push(Op::new(OpType::Transpose, &[self.index], &[output_index]));
        BuilderArray {
            index: output_index,
            builder,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ReduceOp {
    Max,
    Sum,
}

#[derive(Debug, Clone, Copy)]
enum PerElementOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
}

#[derive(Debug, Clone, Copy)]
enum OpType {
    PerElement(PerElementOp),
    MatMul,
    Transpose,
    Reduce(ReduceOp),
}

#[derive(Debug)]
struct Op {
    ty: OpType,
    inputs: ArrayVec<ArrayIndex, MAX_INPUTS>,
    outputs: ArrayVec<ArrayIndex, MAX_OUTPUTS>,
}

impl Op {
    fn new(ty: OpType, inputs: &[ArrayIndex], outputs: &[ArrayIndex]) -> Self {
        Self {
            ty,
            inputs: inputs.try_into().unwrap(),
            outputs: outputs.try_into().unwrap(),
        }
    }
}

pub struct Graph {
    arrays: Vec<Array>,
    ops: Vec<Op>,
}

impl Graph {
    fn array(&mut self, shape: impl Into<Shape>, name: Option<String>) -> ArrayIndex {
        let index = ArrayIndex(self.arrays.len());
        self.arrays.push(Array {
            shape: shape.into(),
            name,
        });
        index
    }

    pub fn print_state(&self) {
        for a in self.arrays.iter() {
            println!(
                "{}: {}",
                a.name.as_ref().map(String::as_str).unwrap_or("?"),
                a.shape
            );
        }
        for op in self.ops.iter() {
            println!("{:?}", op);
        }
    }
}

impl ops::Index<ArrayIndex> for Graph {
    type Output = Array;
    fn index(&self, index: ArrayIndex) -> &Self::Output {
        self.arrays.index(index.0)
    }
}

pub struct GraphBuilder {
    graph: UnsafeCell<Graph>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: UnsafeCell::new(Graph {
                arrays: Vec::new(),
                ops: Vec::new(),
            }),
        }
    }

    pub fn variable(&self, shape: impl Into<Shape>, name: &str) -> BuilderArray {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        let index = graph.array(shape, Some(name.to_owned()));
        BuilderArray {
            index,
            builder: self,
        }
    }

    pub fn finish(self) -> Graph {
        self.graph.into_inner()
    }
}

impl<'builder> ops::Add for BuilderArray<'builder> {
    type Output = BuilderArray<'builder>;
    fn add(self, rhs: BuilderArray) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Add)
    }
}
impl<'builder> ops::Sub for BuilderArray<'builder> {
    type Output = BuilderArray<'builder>;
    fn sub(self, rhs: BuilderArray) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Sub)
    }
}
impl<'builder> ops::Mul for BuilderArray<'builder> {
    type Output = BuilderArray<'builder>;
    fn mul(self, rhs: BuilderArray) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Mul)
    }
}
impl<'builder> ops::Div for BuilderArray<'builder> {
    type Output = BuilderArray<'builder>;
    fn div(self, rhs: BuilderArray) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Div)
    }
}
impl<'builder> ops::Neg for BuilderArray<'builder> {
    type Output = BuilderArray<'builder>;
    fn neg(self) -> Self::Output {
        self.per_element_unary_op(PerElementOp::Neg)
    }
}

fn display_list<I>(iter: I, f: &mut fmt::Formatter<'_>) -> fmt::Result
where
    I: IntoIterator,
    I::Item: fmt::Display,
{
    let mut iter = iter.into_iter().peekable();
    f.write_str("[")?;
    while let Some(x) = iter.next() {
        f.write_fmt(format_args!("{}", x))?;
        if iter.peek().is_some() {
            f.write_str(", ")?;
        }
    }
    f.write_str("]")
}
