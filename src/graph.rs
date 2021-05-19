use arrayvec::ArrayVec;
use std::{
    cell::Cell,
    cell::UnsafeCell,
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
    io, iter, ops,
};

type AxisLen = usize;
type ShapeVec = ArrayVec<AxisLen, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(ShapeVec);

impl Shape {
    fn iter_rev_then_one(&self, len: usize) -> impl Iterator<Item = &AxisLen> {
        self.0.iter().rev().chain(iter::repeat(&1)).take(len)
    }

    fn per_element(&self, rhs: &Shape) -> Self {
        // broadcast axes from 1 => n where necessary
        let len = self.0.len().max(rhs.0.len());
        let rev: ShapeVec = self
            .iter_rev_then_one(len)
            .zip(rhs.iter_rev_then_one(len))
            .map(|(&a, &b)| match (a, b) {
                (1, n) => n,
                (m, 1) => m,
                (m, n) => {
                    assert_eq!(m, n);
                    m
                }
            })
            .collect();
        Shape(rev.iter().cloned().rev().collect())
    }

    fn matrix_multiply(&self, rhs: &Shape) -> Self {
        assert_eq!(rhs.0.len(), 2);
        let (a_last, a_prefix) = self.0.split_last().unwrap();
        let (b_first, b_suffix) = rhs.0.split_first().unwrap();
        assert_eq!(*a_last, *b_first);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(a_prefix).unwrap();
        v.try_extend_from_slice(b_suffix).unwrap();
        Shape(v)
    }

    fn transpose(&self) -> Self {
        assert_eq!(self.0.len(), 2);
        Shape(self.0.iter().cloned().rev().collect())
    }

    fn reduce(&self, axis: isize) -> Self {
        // address from end if negative
        let axis = (axis as usize).wrapping_add(if axis < 0 { self.0.len() } else { 0 });

        // strip outermost dimension if reduced, otherwise keep with length 1
        if axis == 0 {
            Shape(self.0.iter().cloned().skip(1).collect())
        } else {
            let mut v = self.0.clone();
            v[axis] = 1;
            Shape(v)
        }
    }

    fn one_hot(&self, count: AxisLen) -> Self {
        // expand last axis (innermost dimension) from 1 to n
        let (last, prefix) = self.0.split_last().unwrap();
        assert_eq!(*last, 1);
        let mut v = ArrayVec::new();
        v.try_extend_from_slice(prefix).unwrap();
        v.push(count);
        Shape(v)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display_list(self.0.iter(), f)
    }
}

impl<const N: usize> From<[AxisLen; N]> for Shape {
    fn from(s: [AxisLen; N]) -> Self {
        Self(s.iter().cloned().collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReduceOp {
    Max,
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PerElementOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Exp,
    Log,
    OneHot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Op {
    Variable, // TODO: reference to storage
    Literal(f32),
    PerElement(PerElementOp),
    MatMul,
    Transpose,
    Reduce { reduce_op: ReduceOp, axis: isize },
}

struct Node {
    name: Option<String>,
    colour: usize,
    shape: Shape,
    op: Op,
    inputs: Vec<NodeIndex>,
}

impl Node {
    fn new(colour: usize, shape: impl Into<Shape>, op: Op, inputs: &[NodeIndex]) -> Self {
        Self {
            name: None,
            colour,
            shape: shape.into(),
            op,
            inputs: inputs.to_vec(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeIndex(usize);

#[derive(Clone, Copy)]
pub struct ArrayId<'builder> {
    index: NodeIndex,
    builder: &'builder GraphBuilder,
}

impl<'builder> ArrayId<'builder> {
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
        ArrayId {
            index: graph.push_node(Node::new(
                builder.colour.get(),
                shape,
                Op::PerElement(op),
                &[self.index],
            )),
            builder,
        }
    }

    fn per_element_binary_op(self, rhs: ArrayId, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.per_element(&graph[rhs.index].shape);
        ArrayId {
            index: graph.push_node(Node::new(
                builder.colour.get(),
                shape,
                Op::PerElement(op),
                &[self.index, rhs.index],
            )),
            builder,
        }
    }

    fn reduce_op(self, reduce_op: ReduceOp, axis: isize) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.reduce(axis);
        ArrayId {
            index: graph.push_node(Node::new(
                builder.colour.get(),
                shape,
                Op::Reduce { reduce_op, axis },
                &[self.index],
            )),
            builder,
        }
    }

    pub fn one_hot(self, count: AxisLen) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.one_hot(count);
        ArrayId {
            index: graph.push_node(Node::new(
                builder.colour.get(),
                shape,
                Op::PerElement(PerElementOp::OneHot),
                &[self.index],
            )),
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

    pub fn matmul(self, rhs: ArrayId) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index]
            .shape
            .matrix_multiply(&graph[rhs.index].shape);
        ArrayId {
            index: graph.push_node(Node::new(
                builder.colour.get(),
                shape,
                Op::MatMul,
                &[self.index, rhs.index],
            )),
            builder,
        }
    }

    pub fn transpose(self) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.transpose();
        ArrayId {
            index: graph.push_node(Node::new(
                builder.colour.get(),
                shape,
                Op::Transpose,
                &[self.index],
            )),
            builder,
        }
    }

    pub fn shape(&self) -> Shape {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        graph[self.index].shape.clone()
    }

    pub fn accumulate(&mut self, rhs: ArrayId) {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        assert_eq!(graph[self.index].op, Op::PerElement(PerElementOp::Add));
        assert_eq!(graph[self.index].shape, graph[rhs.index].shape);
        let node = &mut graph[self.index];
        node.inputs.push(rhs.index);
    }
}

pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    fn push_node(&mut self, node: Node) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(node);
        NodeIndex(index)
    }

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for (index, node) in self.nodes.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            node.colour.hash(&mut hasher);
            let col = ((hasher.finish() >> 40) as u32) | 0x404040;
            write!(
                w,
                "n{} [shape=box,style=filled,color=\"#{:06X}\",label=\"{:?}\\n",
                index, col, node.op
            )?;
            if let Some(s) = node.name.as_ref() {
                write!(w, "{}", s)?;
            }
            writeln!(w, "{}\"];", node.shape)?;
        }
        for (index, node) in self.nodes.iter().enumerate() {
            for input in node.inputs.iter() {
                writeln!(w, "n{} -> n{};", input.0, index)?;
            }
        }
        writeln!(w, "}}")
    }
}

impl ops::Index<NodeIndex> for Graph {
    type Output = Node;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.nodes.index(index.0)
    }
}
impl ops::IndexMut<NodeIndex> for Graph {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.nodes.index_mut(index.0)
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

    fn literal(&self, value: f32) -> ArrayId {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        ArrayId {
            index: graph.push_node(Node::new(self.colour.get(), [1], Op::Literal(value), &[])),
            builder: self,
        }
    }

    pub fn variable(&self, shape: impl Into<Shape>, name: impl Into<String>) -> ArrayId {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        ArrayId {
            index: graph.push_node(Node::new(self.colour.get(), shape, Op::Variable, &[])),
            builder: self,
        }
        .with_name(name)
    }

    pub fn accumulator(&self, shape: impl Into<Shape>) -> ArrayId {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        ArrayId {
            index: graph.push_node(Node::new(
                self.colour.get(),
                shape,
                Op::PerElement(PerElementOp::Add),
                &[],
            )),
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

impl<'builder> ops::Add for ArrayId<'builder> {
    type Output = ArrayId<'builder>;
    fn add(self, rhs: ArrayId) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Add)
    }
}
impl<'builder> ops::Sub for ArrayId<'builder> {
    type Output = ArrayId<'builder>;
    fn sub(self, rhs: ArrayId) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Sub)
    }
}
impl<'builder> ops::Mul for ArrayId<'builder> {
    type Output = ArrayId<'builder>;
    fn mul(self, rhs: ArrayId) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Mul)
    }
}
impl<'builder> ops::Div for ArrayId<'builder> {
    type Output = ArrayId<'builder>;
    fn div(self, rhs: ArrayId) -> Self::Output {
        self.per_element_binary_op(rhs, PerElementOp::Div)
    }
}
impl<'builder> ops::Neg for ArrayId<'builder> {
    type Output = ArrayId<'builder>;
    fn neg(self) -> Self::Output {
        self.per_element_unary_op(PerElementOp::Neg)
    }
}

impl<'builder> ops::Mul<ArrayId<'builder>> for f32 {
    type Output = ArrayId<'builder>;
    fn mul(self, rhs: ArrayId<'builder>) -> Self::Output {
        let lhs = rhs.builder.literal(self);
        lhs.per_element_binary_op(rhs, PerElementOp::Mul)
    }
}
impl<'builder> ops::Div<f32> for ArrayId<'builder> {
    type Output = ArrayId<'builder>;
    fn div(self, rhs: f32) -> Self::Output {
        let rhs = self.builder.literal(rhs);
        self.per_element_binary_op(rhs, PerElementOp::Div)
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
