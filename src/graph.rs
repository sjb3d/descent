use arrayvec::ArrayVec;
use std::{cell::UnsafeCell, fmt, ops};

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
        // reduce last axis (innermost dimension), keep dimension with size 1
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

struct Node {
    shape: Shape,
    name: Option<String>,
    op: OpType,
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
                op: OpType::None,
                inputs: Vec::new(),
            },
        }
    }

    fn with_name(mut self, name: String) -> Self {
        self.node.name = Some(name);
        self
    }

    fn with_op(mut self, op: OpType, inputs: &[NodeIndex]) -> Self {
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
                    .with_op(OpType::PerElement(op), &[self.index])
                    .build(),
            ),
            builder,
        }
    }

    fn per_element_binary_op(&self, rhs: ArrayId, op: PerElementOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = Shape::from_per_element(&graph[self.index].shape, &graph[rhs.index].shape);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(OpType::PerElement(op), &[self.index, rhs.index])
                    .build(),
            ),
            builder,
        }
    }

    fn reduce_op(&self, op: ReduceOp) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = Shape::from_reduce(&graph[self.index].shape);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(OpType::Reduce(op), &[self.index])
                    .build(),
            ),
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

    pub fn matmul(&self, rhs: ArrayId) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = Shape::from_matrix_multiply(&graph[self.index].shape, &graph[rhs.index].shape);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(OpType::MatMul, &[self.index, rhs.index])
                    .build(),
            ),
            builder,
        }
    }

    pub fn transpose(&self) -> Self {
        let builder = self.builder;
        let graph = unsafe { builder.graph.get().as_mut().unwrap() };
        let shape = Shape::from_transpose(&graph[self.index].shape);
        ArrayId {
            index: graph.push_node(
                NodeBuilder::new(shape)
                    .with_op(OpType::Transpose, &[self.index])
                    .build(),
            ),
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
    None,
    PerElement(PerElementOp),
    MatMul,
    Transpose,
    Reduce(ReduceOp),
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

    pub fn print_state(&self) {
        for node in self.nodes.iter() {
            println!(
                "{}: {}, {:?}, {:?}",
                node.name.as_ref().map(String::as_str).unwrap_or("?"),
                node.shape,
                node.op,
                node.inputs
            );
        }
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
