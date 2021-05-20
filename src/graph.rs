use crate::prelude::*;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io, ops,
};

/*
    Steps:

    1. Gather nodes from the roots (DCE)
    2. Eliminate tranpose nodes, literal nodes
    3. Build kernels of various types

    Kernel building needs to:
    * Reduce/MatMul can just follow outputs through PerElement nodes
    * Remaining PerElement nodes need to check for sets:
      * Traverse nodes in reverse order
      * All dimensions match
      * No successor of a node is a predecessor of another node in the set
      * Split islands into separate sets

*/

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

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) name: Option<String>,
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) inputs: Vec<NodeIndex>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NodeIndex(usize);

pub struct Graph {
    pub(crate) nodes: Vec<Option<Node>>,
}

impl Graph {
    pub(crate) fn new_node(
        &mut self,
        colour: usize,
        shape: impl Into<Shape>,
        op: Op,
        inputs: &[NodeIndex],
    ) -> NodeIndex {
        let index = self.nodes.len();
        self.nodes.push(Some(Node {
            name: None,
            colour,
            shape: shape.into(),
            op,
            inputs: inputs.to_vec(),
        }));
        NodeIndex(index)
    }

    fn enumerate_nodes(&self) -> impl Iterator<Item = (NodeIndex, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| node.as_ref().map(|node| (NodeIndex(index), node)))
    }

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for (index, node) in self.enumerate_nodes() {
            let mut hasher = DefaultHasher::new();
            node.colour.hash(&mut hasher);
            let col = ((hasher.finish() >> 40) as u32) | 0x404040;
            write!(
                w,
                "n{} [shape=box,style=filled,color=\"#{:06X}\",label=\"{:?}\\n",
                index.0, col, node.op
            )?;
            if let Some(s) = node.name.as_ref() {
                write!(w, "{}", s)?;
            }
            writeln!(w, "{}\"];", node.shape)?;
        }
        for (index, node) in self.enumerate_nodes() {
            for input in node.inputs.iter() {
                writeln!(w, "n{} -> n{};", input.0, index.0)?;
            }
        }
        writeln!(w, "}}")
    }
}

impl ops::Index<NodeIndex> for Graph {
    type Output = Node;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.nodes.index(index.0).as_ref().unwrap()
    }
}
impl ops::IndexMut<NodeIndex> for Graph {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.nodes.index_mut(index.0).as_mut().unwrap()
    }
}
