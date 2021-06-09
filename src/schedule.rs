use crate::prelude::*;
use petgraph::{
    prelude::{NodeIndex as NodeIndexBase, *},
    visit::{
        IntoEdgeReferences, IntoEdgesDirected, IntoNeighborsDirected, IntoNodeReferences, NodeRef,
        VisitMap, Visitable,
    },
};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fmt,
    fmt::Write,
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
    Input { variable_id: VariableId },
    Literal(f32),
    PerElement(PerElementOp),
    MatMul,
    Transpose,
    Reduce { reduce_op: ReduceOp, axis: isize },
    Accumulate, // accumulates grad from backprop
}

pub(crate) enum KernelType {
    PerElement,
    MatMul,
    Reduce,
}

pub(crate) struct Kernel {
    ty: KernelType,
    shape: Shape,
    inputs: Vec<NodeIndex>,
    members: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) name: Option<String>,
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) kernel_index: Option<usize>,
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
    kernels: Vec<Kernel>,
}

impl Schedule {
    pub(crate) fn new(graph: Graph, roots: Vec<NodeIndex>) -> Self {
        let mut graph = Self {
            graph,
            roots,
            ordering: Vec::new(),
            kernels: Vec::new(),
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

    fn any_predecessor(&self, roots: &[NodeIndex], mut f: impl FnMut(NodeIndex) -> bool) -> bool {
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

    fn any_successor(&self, roots: &[NodeIndex], mut f: impl FnMut(NodeIndex) -> bool) -> bool {
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
            if matches!(first_node.op, Op::PerElement(_)) && first_node.kernel_index.is_none() {
                let kernel_index = Some(self.kernels.len());
                self.kernels.push(Kernel {
                    ty: KernelType::PerElement,
                    shape: first_node.shape.clone(),
                    inputs: Vec::new(),
                    members: Vec::new(),
                    outputs: Vec::new(),
                });
                let kernel = self.kernels.last().unwrap();
                self.graph[first_node_index].kernel_index = kernel_index;

                'outer: loop {
                    'inner: for other_node_index in self.ordering.iter().cloned() {
                        let other_node = &self.graph[other_node_index];

                        // check this node has no kernel and matches shape
                        let can_include = matches!(other_node.op, Op::PerElement(_))
                            && other_node.kernel_index.is_none()
                            && other_node.shape == kernel.shape;
                        if !can_include {
                            continue 'inner;
                        }

                        // skip this node if any edges with kernel nodes are transpose
                        let mut has_kernel_neighbor = false;
                        if self
                            .graph
                            .edges_directed(other_node_index, Incoming)
                            .filter(|edge_ref| {
                                assert_eq!(edge_ref.target(), other_node_index);
                                self.graph[edge_ref.source()].kernel_index == kernel_index
                            })
                            .inspect(|_| has_kernel_neighbor = true)
                            .any(|edge_ref| edge_ref.weight().transpose)
                        {
                            continue 'inner;
                        }
                        if self
                            .graph
                            .edges_directed(other_node_index, Outgoing)
                            .filter(|edge_ref| {
                                assert_eq!(edge_ref.source(), other_node_index);
                                self.graph[edge_ref.target()].kernel_index == kernel_index
                            })
                            .inspect(|_| has_kernel_neighbor = true)
                            .any(|edge_ref| edge_ref.weight().transpose)
                        {
                            continue 'inner;
                        }

                        // placing this node in the kernel needs to save a load
                        // TODO: also check for sibling nodes?
                        if !has_kernel_neighbor {
                            continue 'inner;
                        }

                        // check uses of this node don't re-enter this kernel
                        if self.any_successor(&[other_node_index], |node_index| {
                            self.graph[node_index].kernel_index.is_none()
                                && self.graph.neighbors_directed(node_index, Outgoing).any(
                                    |node_index| {
                                        self.graph[node_index].kernel_index == kernel_index
                                    },
                                )
                        }) {
                            continue 'inner;
                        }

                        // check inputs of this node don't re-enter this kernel
                        if self.any_predecessor(&[other_node_index], |node_index| {
                            self.graph[node_index].kernel_index.is_none()
                                && self.graph.neighbors_directed(node_index, Incoming).any(
                                    |node_index| {
                                        self.graph[node_index].kernel_index == kernel_index
                                    },
                                )
                        }) {
                            continue 'inner;
                        }

                        // ok to merge, restart search with new kernel
                        self.graph[other_node_index].kernel_index = kernel_index;
                        continue 'outer;
                    }
                    break 'outer;
                }
            }
        }
        for node_index in self.ordering.iter().cloned() {
            if let Some(kernel_index) = self.graph[node_index].kernel_index {
                self.kernels[kernel_index].members.push(node_index);
            }
        }
        for (kernel_index, kernel) in self.kernels.iter_mut().enumerate() {
            let mut markers = self.graph.visit_map();
            let graph = &self.graph;
            for node_index in kernel.members.iter().cloned() {
                for input_node_index in
                    graph
                        .neighbors_directed(node_index, Incoming)
                        .filter(|&other_index| {
                            let other_node = &graph[other_index];
                            !matches!(other_node.op, Op::Literal(_))
                                && other_node.kernel_index != Some(kernel_index)
                        })
                {
                    if markers.visit(input_node_index) {
                        kernel.inputs.push(input_node_index);
                    }
                }
                if graph
                    .neighbors_directed(node_index, Outgoing)
                    .any(|other_index| graph[other_index].kernel_index != Some(kernel_index))
                {
                    kernel.outputs.push(node_index);
                }
            }
        }
    }

    fn generate_kernel_source(&self, kernel_index: usize) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        writeln!(w, "#version 460 core")?;

        let mut binding_index = 0;

        let kernel = &self.kernels[kernel_index];
        for node_index in kernel.inputs.iter().cloned() {
            writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
            writeln!(
                w,
                "readonly restrict buffer layout{0} {{ float input{0}[]; }};",
                node_index.index()
            )?;
            binding_index += 1;
        }
        for node_index in kernel.outputs.iter().cloned() {
            writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
            writeln!(
                w,
                "writeonly restrict buffer layout{0} {{ float output{0}[]; }};",
                node_index.index()
            )?;
            binding_index += 1;
        }

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        for node_index in kernel.members.iter().cloned() {
            let mut args = [String::new(), String::new()];
            for edge_ref in self.graph.edges_directed(node_index, Incoming) {
                let source_node_index = edge_ref.source();
                let source_node = &self.graph[source_node_index];
                let edge = edge_ref.weight();
                assert!(args[edge.arg].is_empty());
                args[edge.arg] = if source_node.kernel_index == Some(kernel_index) {
                    format!("tmp{}", source_node_index.index())
                } else if let Op::Literal(value) = source_node.op {
                    format!("{:#?}", value)
                } else {
                    assert!(!edge.transpose);
                    format!(
                        "input{}[gl_GlobalInvocationID.x]",
                        source_node_index.index()
                    )
                };
            }

            write!(w, "float tmp{} = ", node_index.index())?;
            if let Op::PerElement(per_element_op) = self.graph[node_index].op {
                match per_element_op {
                    PerElementOp::Add => write!(w, "{} + {}", args[0], args[1])?,
                    PerElementOp::Sub => write!(w, "{} - {}", args[0], args[1])?,
                    PerElementOp::Mul => write!(w, "{} * {}", args[0], args[1])?,
                    PerElementOp::Div => write!(w, "{} / {}", args[0], args[1])?,
                    PerElementOp::Neg => write!(w, "-{}", args[0])?,
                    PerElementOp::Exp => write!(w, "exp({})", args[0])?,
                    PerElementOp::Log => write!(w, "log({})", args[0])?,
                    PerElementOp::OneHot => write!(
                        w,
                        "(gl_GlobalInvocationID.x == uint({})) ? 1.0 : 0.0",
                        args[0]
                    )?,
                }
            } else {
                unreachable!()
            }
            writeln!(w, ";")?;
        }

        for node_index in kernel.outputs.iter().cloned() {
            writeln!(
                w,
                "output{0}[gl_GlobalInvocationID.x] = tmp{0};",
                node_index.index()
            )?;
        }

        writeln!(w, "}}")?;

        Ok(src)
    }

    pub fn compile_kernel_source(&self, kernel_index: usize) -> Option<String> {
        let source = self.generate_kernel_source(kernel_index).unwrap();
        println!("{}", source);

        let mut compiler = shaderc::Compiler::new().unwrap();
        match compiler.compile_into_spirv_assembly(
            &source,
            shaderc::ShaderKind::Compute,
            "kernel",
            "main",
            None,
        ) {
            Ok(artifact) => {
                if artifact.get_num_warnings() != 0 {
                    println!("{}", artifact.get_warning_messages());
                }
                let text = artifact.as_text();
                println!("{}", text);
                Some(text)
            }
            Err(err) => {
                println!("{}", err);
                None
            }
        }
    }

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for kernel in iter::once(None).chain((0..self.kernels.len()).map(Some)) {
            if let Some(index) = kernel {
                writeln!(w, "subgraph cluster{} {{ style=filled;", index)?;
            }
            for node_ref in self
                .graph
                .node_references()
                .filter(|node_ref| node_ref.weight().kernel_index == kernel)
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
