use crate::prelude::*;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io, ops,
};
use bitvec::prelude::*;

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

#[derive(Debug, Clone)]
pub(crate) enum Input {
    Node {
        index: NodeIndex,
        transpose: bool, // TODO: more general view description?
    },
    Literal(f32),
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) name: Option<String>,
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) inputs: Vec<Input>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NodeIndex(pub(crate) usize);

impl ops::Index<NodeIndex> for Vec<Option<Node>> {
    type Output = Node;
    fn index(&self, index: NodeIndex) -> &Self::Output {
        self.index(index.0).as_ref().unwrap()
    }
}
impl ops::IndexMut<NodeIndex> for Vec<Option<Node>> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self.index_mut(index.0).as_mut().unwrap()
    }
}

fn enumerate_valid_nodes(nodes: &Vec<Option<Node>>) -> impl Iterator<Item = (NodeIndex, &Node)> {
    nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| node.as_ref().map(|node| (NodeIndex(index), node)))
}

pub struct Graph {
    nodes: Vec<Option<Node>>,
    roots: Vec<NodeIndex>,
    ordering: Vec<NodeIndex>,
}

impl Graph {
    pub(crate) fn new(nodes: &[Node], roots: &[NodeIndex]) -> Self {
        let mut graph = Self {
            nodes: nodes.iter().cloned().map(Some).collect(),
            roots: roots.to_vec(),
            ordering: Vec::new(),
        };
        graph.rebuild_ordering();
        graph.eliminate_dead_code();
        graph.rebuild_ordering();
        graph.eliminate_literals();
        graph.rebuild_ordering();
        graph
    }

    fn rebuild_ordering_visit(index: NodeIndex, nodes: &Vec<Option<Node>>, ordering: &mut Vec<NodeIndex>, visited: &mut BitVec, visiting: &mut BitVec) {
        if visited[index.0] {
            return;
        }
        if visiting[index.0] {
            panic!("graph is acyclic");
        }
        visiting.get_mut(index.0).unwrap().set(true);
        for input in nodes[index].inputs.iter() {
            if let Input::Node { index, .. } = input {
                Self::rebuild_ordering_visit(*index, nodes, ordering, visited, visiting);
            }
        }
        visiting.get_mut(index.0).unwrap().set(false);
        visited.get_mut(index.0).unwrap().set(true);
        ordering.push(index);
    }

    fn rebuild_ordering(&mut self) {
        let mut visited = bitvec![0; self.nodes.len()];
        let mut visiting = bitvec![0; self.nodes.len()];
        self.ordering.clear();
        for (index, _node) in enumerate_valid_nodes(&self.nodes) {
            Self::rebuild_ordering_visit(index, &self.nodes, &mut self.ordering, &mut visited, &mut visiting);
        }
    }

    fn eliminate_dead_code(&mut self) {
        let mut live = bitvec![0; self.nodes.len()];
        for index in self.roots.iter().cloned() {
            live.get_mut(index.0).unwrap().set(true);
        }
        for index in self.ordering.iter().rev().cloned() {
            if live[index.0] {
                for input in self.nodes[index].inputs.iter() {
                    if let Input::Node { index, ..} = input {
                        live.get_mut(index.0).unwrap().set(true);
                    }
                }
            }
        }
        for (seen, node) in live.iter().zip(self.nodes.iter_mut()) {
            if !seen && node.is_some() {
                node.take();
            }
        }
    }

    fn eliminate_literals(&mut self) {
    }

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for (index, node) in enumerate_valid_nodes(&self.nodes) {
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
        for (output_index, output_node) in enumerate_valid_nodes(&self.nodes) {
            for input in output_node.inputs.iter() {
                if let Input::Node {
                    index: input_index, ..
                } = input
                {
                    writeln!(w, "n{} -> n{};", input_index.0, output_index.0)?;
                }
            }
        }
        writeln!(w, "}}")
    }
}
