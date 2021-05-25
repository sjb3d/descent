use crate::prelude::*;
use petgraph::{
    prelude::{*, NodeIndex as NodeIndexBase},
    visit::{IntoEdgeReferences, IntoNodeReferences, NodeRef, VisitMap, Visitable},
};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io, iter,
};

pub(crate) type Graph = StableDiGraph<Node, Edge, usize>;
pub(crate) type NodeIndex = NodeIndexBase<usize>;

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
pub(crate) struct Node {
    pub(crate) name: Option<String>,
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) kernel: Option<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Edge {
    pub(crate) arg: usize,
    pub(crate) transpose: bool,
}

impl Edge {
    pub(crate) fn chain(&self, rhs: &Edge) -> Self {
        Self {
            arg: rhs.arg,
            transpose: self.transpose ^ rhs.transpose,
        }
    }

    pub(crate) fn transposed(&self) -> Self {
        Self {
            transpose: !self.transpose,
            ..*self
        }
    }
}

pub struct Schedule {
    graph: Graph,
    roots: Vec<NodeIndex>,
    ordering: Vec<NodeIndex>,
    kernel_count: usize,
}

impl Schedule {
    pub(crate) fn new(
        graph: Graph,
        roots: Vec<NodeIndex>,
    ) -> Self {
        let mut graph = Self {
            graph,
            roots,
            ordering: Vec::new(),
            kernel_count: 0,
        };

        graph.rebuild_ordering();
        graph.eliminate_dead_code();

        graph.rebuild_ordering();
        graph.eliminate_accumulate();

        graph.rebuild_ordering();
        graph.eliminate_transpose();

        graph.rebuild_ordering();
        graph.build_kernels();

        graph
    }

    fn rebuild_ordering(&mut self) {
        self.ordering.clear();
        let mut topo = petgraph::visit::Topo::new(&self.graph);
        while let Some(node_index) = topo.next(&self.graph) {
            self.ordering.push(node_index);
        }
    }

    fn eliminate_dead_code(&mut self) {
        let mut live = self.graph.visit_map();
        for index in self.roots.iter().cloned() {
            live.visit(index);
        }
        for index in self.ordering.iter().rev().cloned() {
            if live.is_visited(&index) {
                for input_index in self.graph.neighbors_directed(index, Incoming) {
                    live.visit(input_index);
                }
            }
        }
        self.graph.retain_nodes(|_, index| live.is_visited(&index));
    }

    fn eliminate_accumulate(&mut self) {
        for node_index in self.ordering.iter().cloned() {
            if matches!(self.graph[node_index].op, Op::Accumulate) {
                assert_eq!(self.graph.edges_directed(node_index, Incoming).count(), 1); // TODO: generate adds
                let mut in_edges = self.graph.neighbors_directed(node_index, Incoming).detach();
                let (in_edge_index, in_node_index) = in_edges.next(&self.graph).unwrap();
                let mut out_edges = self.graph.neighbors_directed(node_index, Outgoing).detach();
                while let Some((out_edge_index, out_node_index)) = out_edges.next(&self.graph) {
                    self.graph.add_edge(
                        in_node_index,
                        out_node_index,
                        self.graph[in_edge_index].chain(&self.graph[out_edge_index]),
                    );
                }
                self.graph.remove_node(node_index);
            }
        }
    }

    fn eliminate_transpose(&mut self) {
        for node_index in self.ordering.iter().cloned() {
            if matches!(self.graph[node_index].op, Op::Transpose) {
                assert_eq!(
                    self.graph.neighbors_directed(node_index, Incoming).count(),
                    1
                );
                let mut in_edges = self.graph.neighbors_directed(node_index, Incoming).detach();
                let (in_edge_index, in_node_index) = in_edges.next(&self.graph).unwrap();
                let mut out_edges = self.graph.neighbors_directed(node_index, Outgoing).detach();
                while let Some((out_edge_index, out_node_index)) = out_edges.next(&self.graph) {
                    self.graph.add_edge(
                        in_node_index,
                        out_node_index,
                        self.graph[in_edge_index]
                            .chain(&self.graph[out_edge_index])
                            .transposed(),
                    );
                }
                self.graph.remove_node(node_index);
            }
        }
    }

    fn any_predecessor(
        &self,
        roots: &[NodeIndex],
        mut f: impl FnMut(NodeIndex) -> bool,
    ) -> bool {
        let mut markers = self.graph.visit_map();
        for &node_index in roots {
            markers.visit(node_index);
        }
        for node_index in self.ordering.iter().cloned().rev() {
            if self
                .graph
                .neighbors_directed(node_index, Outgoing)
                .any(|output_node_index| markers.is_visited(&output_node_index))
            {
                markers.visit(node_index);
                if f(node_index) {
                    return true;
                }
            }
        }
        return false;
    }

    fn any_successor(
        &self,
        roots: &[NodeIndex],
        mut f: impl FnMut(NodeIndex) -> bool,
    ) -> bool {
        let mut markers = self.graph.visit_map();
        for &node_index in roots {
            markers.visit(node_index);
        }
        for node_index in self.ordering.iter().cloned().rev() {
            if self
                .graph
                .neighbors_directed(node_index, Incoming)
                .any(|input_node_index| markers.is_visited(&input_node_index))
            {
                markers.visit(node_index);
                if f(node_index) {
                    return true;
                }
            }
        }
        return false;
    }

    fn build_kernels(&mut self) {
        for first_node_index in self.ordering.iter().cloned() {
            let first_node = &self.graph[first_node_index];
            if matches!(first_node.op, Op::PerElement(_)) && first_node.kernel.is_none() {
                let shape = first_node.shape.clone();

                let kernel = Some(self.kernel_count);
                self.kernel_count += 1;
                self.graph[first_node_index].kernel = kernel;

                'outer: loop {
                    'inner: for other_node_index in self.ordering.iter().cloned() {
                        let other_node = &self.graph[other_node_index];

                        // check this node has no kernel and matches shape
                        let can_include = matches!(other_node.op, Op::PerElement(_))
                            && other_node.kernel.is_none()
                            && other_node.shape == shape;
                        if !can_include {
                            continue 'inner;
                        }

                        // check we have an edge into the kernel from here
                        let has_kernel_neighbor = self
                            .graph
                            .neighbors_undirected(other_node_index)
                            .any(|neighbor_node_index| {
                                self.graph[neighbor_node_index].kernel == kernel
                            });
                        if !has_kernel_neighbor {
                            continue 'inner;
                        }

                        // check uses of this node don't re-enter this kernel
                        if self.any_successor(&[other_node_index], |node_index| {
                            self.graph[node_index].kernel.is_none()
                                && self
                                    .graph
                                    .neighbors_directed(node_index, Outgoing)
                                    .any(|node_index| self.graph[node_index].kernel == kernel)
                        }) {
                            continue 'inner;
                        }

                        // check inputs of this node don't re-enter this kernel
                        if self.any_predecessor(&[other_node_index], |node_index| {
                            self.graph[node_index].kernel.is_none()
                                && self
                                    .graph
                                    .neighbors_directed(node_index, Incoming)
                                    .any(|node_index| self.graph[node_index].kernel == kernel)
                        }) {
                            continue 'inner;
                        }

                        // ok to merge, restart search with new kernel
                        self.graph[other_node_index].kernel = kernel;
                        continue 'outer;
                    }
                    break 'outer;
                }
            }
        }
    }

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for kernel in iter::once(None).chain((0..self.kernel_count).map(Some)) {
            if let Some(index) = kernel {
                writeln!(w, "subgraph cluster{} {{ style=filled;", index)?;
            }
            for node_ref in self
                .graph
                .node_references()
                .filter(|node_ref| node_ref.weight().kernel == kernel)
            {
                let node = node_ref.weight();
                let mut hasher = DefaultHasher::new();
                node.colour.hash(&mut hasher);
                let col = ((hasher.finish() >> 40) as u32) | 0x404040;
                write!(
                    w,
                    "n{} [shape=box,style=filled,color=\"#{:06X}\",label=\"{:?}\\n",
                    node_ref.id().index(),
                    col,
                    node.op
                )?;
                if let Some(s) = node.name.as_ref() {
                    write!(w, "{}", s)?;
                }
                writeln!(w, "{}\"];", node.shape)?;
            }
            if kernel.is_some() {
                writeln!(w, "}}")?;
            }
        }
        for edge_ref in self.graph.edge_references() {
            write!(
                w,
                "n{} -> n{}",
                edge_ref.source().index(),
                edge_ref.target().index()
            )?;
            if edge_ref.weight().transpose {
                write!(w, " [label=\"T\"]")?;
            }
            writeln!(w, ";")?;
        }
        writeln!(w, "}}")
    }
}
