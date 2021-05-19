use arrayvec::ArrayVec;
use std::{cell::UnsafeCell, fmt, io, iter, ops};

type AxisLen = usize;
type ShapeVec = ArrayVec<AxisLen, 4>;

#[derive(Debug, Clone)]
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
    OneHot,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    None,
    Literal(f32),
    PerElement(PerElementOp),
    MatMul,
    Transpose,
    Reduce { reduce_op: ReduceOp, axis: isize },
}

struct Node {
    shape: Shape,
    name: Option<String>,
    op: Op,
    inputs: Vec<NodeIndex>,
}

#[derive(Debug, Clone, Copy)]
struct NodeIndex(usize);

struct NodeBuilder {
    node: Node,
}

impl NodeBuilder {
    fn new(shape: impl Into<Shape>) -> Self {
        Self {
            node: Node {
                shape: shape.into(),
                name: None,
                op: Op::None,
                inputs: Vec::new(),
            },
        }
    }

    fn with_name(mut self, name: String) -> Self {
        self.node.name = Some(name);
        self
    }

    fn with_op(mut self, op: Op, inputs: &[NodeIndex]) -> Self {
        self.node.op = op;
        self.node.inputs = inputs.to_vec();
        self
    }

    fn build(self) -> Node {
        self.node
    }
}

#[derive(Clone, Copy)]
pub struct ArrayId<'builder> {
    index: NodeIndex,
    builder: &'builder GraphBuilder,
}

impl<'builder> ArrayId<'builder> {
    fn per_element_unary_op(&self, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.clone();
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(Op::PerElement(op), &[self.index])
                    .build(),
            ),
            builder,
        }
    }

    fn per_element_binary_op(&self, rhs: ArrayId, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.per_element(&graph[rhs.index].shape);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(Op::PerElement(op), &[self.index, rhs.index])
                    .build(),
            ),
            builder,
        }
    }

    fn reduce_op(&self, reduce_op: ReduceOp, axis: isize) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.reduce(axis);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(Op::Reduce { reduce_op, axis }, &[self.index])
                    .build(),
            ),
            builder,
        }
    }

    pub fn one_hot(&self, count: AxisLen) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.one_hot(count);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(Op::PerElement(PerElementOp::OneHot), &[self.index])
                    .build(),
            ),
            builder,
        }
    }

    pub fn reduce_max(&self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Max, axis)
    }
    pub fn reduce_sum(&self, axis: isize) -> Self {
        self.reduce_op(ReduceOp::Sum, axis)
    }

    pub fn exp(&self) -> Self {
        self.per_element_unary_op(PerElementOp::Exp)
    }
    pub fn log(&self) -> Self {
        self.per_element_unary_op(PerElementOp::Log)
    }

    pub fn matmul(&self, rhs: ArrayId) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index]
            .shape
            .matrix_multiply(&graph[rhs.index].shape);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(Op::MatMul, &[self.index, rhs.index])
                    .build(),
            ),
            builder,
        }
    }

    pub fn transpose(&self) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = graph[self.index].shape.transpose();
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(Op::Transpose, &[self.index])
                    .build(),
            ),
            builder,
        }
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
            write!(w, "n{} [label=\"{:?}\\n", index, node.op)?;
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

pub struct GraphBuilder {
    graph: UnsafeCell<Graph>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: UnsafeCell::new(Graph { nodes: Vec::new() }),
        }
    }

    fn literal(&self, value: f32) -> ArrayId {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new([])
                    .with_op(Op::Literal(value), &[])
                    .build(),
            ),
            builder: self,
        }
    }
    pub fn variable(&self, shape: impl Into<Shape>, name: &str) -> ArrayId {
        let graph = unsafe { self.graph.get().as_mut().unwrap() };
        ArrayId {
            index: graph.push_node(NodeBuilder::new(shape).with_name(name.to_owned()).build()),
            builder: self,
        }
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
