use crate::prelude::*;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io, ops,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReduceOp {
    Max,
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PerElementOp {
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
pub(crate) enum Op {
    Variable, // TODO: reference to storage
    Literal(f32),
    PerElement(PerElementOp),
    MatMul,
    Transpose,
    Reduce { reduce_op: ReduceOp, axis: isize },
    Accumulate, // accumulates grad from backprop
}

pub(crate) struct Node {
    pub(crate) name: Option<String>,
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) inputs: Vec<NodeIndex>,
}

impl Node {
    pub(crate) fn new(
        colour: usize,
        shape: impl Into<Shape>,
        op: Op,
        inputs: &[NodeIndex],
    ) -> Self {
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
pub(crate) struct NodeIndex(usize);

pub struct Graph {
    pub(crate) nodes: Vec<Node>,
}

impl Graph {
    pub(crate) fn push_node(&mut self, node: Node) -> NodeIndex {
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
