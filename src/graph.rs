use crate::{prelude::*, store::*};
use bitvec::prelude::*;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NodeIndex(usize);

impl Idx for NodeIndex {
    fn new(index: usize) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0
    }
}

#[derive(Default)]
struct NodeAccel {
    uses: Vec<NodeIndex>,
}

pub struct Graph {
    nodes: Store<NodeIndex, Node>,
    roots: Vec<NodeIndex>,
    ordering: Vec<NodeIndex>,
    accel: Vec<NodeAccel>,
}

impl Graph {
    pub(crate) fn new(nodes: Store<NodeIndex, Node>, roots: Vec<NodeIndex>) -> Self {
        let mut graph = Self {
            nodes,
            roots,
            ordering: Vec::new(),
            accel: Vec::new(),
        };
        graph.rebuild_ordering();
        graph.rebuild_accel();

        graph.lower_literals_and_transpose();
        graph.eliminate_dead_code();
        graph.rebuild_ordering();
        graph.rebuild_accel();

        graph
    }

    fn rebuild_ordering_visit(
        index: NodeIndex,
        nodes: &Store<NodeIndex, Node>,
        ordering: &mut Vec<NodeIndex>,
        visited: &mut BitVec,
        visiting: &mut BitVec,
    ) {
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
        for (index, _node) in self.nodes.iter() {
            Self::rebuild_ordering_visit(
                index,
                &self.nodes,
                &mut self.ordering,
                &mut visited,
                &mut visiting,
            );
        }
    }

    fn rebuild_accel(&mut self) {
        for accel in self.accel.iter_mut() {
            accel.uses.clear();
        }
        self.accel.resize_with(self.nodes.len(), Default::default);

        for (output_index, node) in self.nodes.iter() {
            for input in node.inputs.iter() {
                if let Input::Node {
                    index: input_index, ..
                } = input
                {
                    self.accel[input_index.0].uses.push(output_index);
                }
            }
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
                    if let Input::Node { index, .. } = input {
                        live.get_mut(index.0).unwrap().set(true);
                    }
                }
            }
        }
        self.nodes.retain(|index| live[index.0])
    }

    fn lower_literals_and_transpose(&mut self) {
        for index in self.ordering.iter().rev().cloned() {
            match self.nodes[index].op {
                Op::Literal(value) => {
                    for use_index in self.accel[index.0].uses.iter().cloned() {
                        for input in self.nodes[use_index].inputs.iter_mut() {
                            match input {
                                Input::Node {
                                    index: match_index, ..
                                } => {
                                    if *match_index == index {
                                        *input = Input::Literal(value);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Op::Transpose => {
                    for use_index in self.accel[index.0].uses.iter().cloned() {
                        assert_eq!(self.nodes[index].inputs.len(), 1);
                        let src_index = match self.nodes[index].inputs.first().cloned().unwrap() {
                            Input::Node { index, transpose } => {
                                assert_eq!(transpose, false);
                                index
                            }
                            _ => panic!("expected tranpose op input to be a node"),
                        };
                        for input in self.nodes[use_index].inputs.iter_mut() {
                            match input {
                                Input::Node {
                                    index: match_index,
                                    transpose,
                                } => {
                                    if *match_index == index {
                                        *input = Input::Node {
                                            index: src_index,
                                            transpose: !*transpose,
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for (index, node) in self.nodes.iter() {
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
        for (output_index, output_node) in self.nodes.iter() {
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
