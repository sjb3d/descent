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
pub(crate) struct Input {
    pub(crate) node_index: NodeIndex,
    pub(crate) transpose: bool, // TODO: more general view description?
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

        graph.eliminate_accumulate();
        graph.eliminate_dead_code();
        graph.rebuild_ordering();
        graph.rebuild_accel();

        graph.eliminate_transpose();
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
            panic!("graph has a cycle");
        }
        visiting.get_mut(index.0).unwrap().set(true);
        for input in nodes[index].inputs.iter() {
            Self::rebuild_ordering_visit(input.node_index, nodes, ordering, visited, visiting);
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
                self.accel[input.node_index.0].uses.push(output_index);
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
                    live.get_mut(input.node_index.0).unwrap().set(true);
                }
            }
        }
        self.nodes.retain(|index| live[index.0])
    }

    fn eliminate_accumulate(&mut self) {
        for index in self.ordering.iter().cloned() {
            if matches!(self.nodes[index].op, Op::Accumulate) {
                assert_eq!(self.nodes[index].inputs.len(), 1); // TODO: generate adds
                let input = self.nodes[index].inputs.first().cloned().unwrap();
                for use_index in self.accel[index.0].uses.iter().cloned() {
                    for use_input in self.nodes[use_index].inputs.iter_mut() {
                        if use_input.node_index == index {
                            assert_eq!(use_input.transpose, false);
                            *use_input = Input {
                                node_index: input.node_index,
                                transpose: input.transpose,
                            }
                        }
                    }
                }
            }
        }
    }

    fn eliminate_transpose(&mut self) {
        for index in self.ordering.iter().cloned() {
            if matches!(self.nodes[index].op, Op::Transpose) {
                assert_eq!(self.nodes[index].inputs.len(), 1);
                let input = self.nodes[index].inputs.first().cloned().unwrap();
                for use_index in self.accel[index.0].uses.iter().cloned() {
                    for use_input in self.nodes[use_index].inputs.iter_mut() {
                        if use_input.node_index == index {
                            assert_eq!(use_input.transpose, false);
                            *use_input = Input {
                                node_index: input.node_index,
                                transpose: !input.transpose,
                            }
                        }
                    }
                }
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
                write!(w, "n{} -> n{}", input.node_index.0, output_index.0)?;
                if input.transpose {
                    write!(w, " [label=\"T\"]")?;
                }
                writeln!(w, ";")?;
            }
        }
        writeln!(w, "}}")
    }
}
