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
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Neg,
    Exp,
    Log,
    OneHot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Op {
    Input { variable_id: VariableId },
    Output { variable_id: VariableId },
    Literal(f32),
    Unary(UnaryOp),
    Binary(BinaryOp),
    MatMul,
    Transpose,
    Reduce { reduce_op: ReduceOp, axis: isize },
    Accumulate, // accumulates grad from backprop
}

#[derive(Debug)]
enum PerElementKernelOp {
    Load {
        input_index: usize,
    },
    Literal(f32),
    Unary {
        op: UnaryOp,
        src_index: usize,
    },
    Binary {
        op: BinaryOp,
        src1_index: usize,
        src2_index: usize,
    },
}

#[derive(Debug)]
struct PerElementKernelDesc {
    shape: Shape,
    inputs: Vec<Shape>,
    outputs: Vec<usize>,
    ops: Vec<PerElementKernelOp>,
}

#[derive(Debug)]
enum KernelDesc {
    PerElement(PerElementKernelDesc),
}

#[derive(Debug)]
pub(crate) struct Kernel {
    desc: KernelDesc,
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
    pub(crate) fn new(graph: Graph) -> Self {
        let roots = graph
            .node_indices()
            .filter(|&node_index| matches!(graph[node_index].op, Op::Output { .. }))
            .collect();

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
        // first gather per-element nodes into kernels
        for first_node_index in self.ordering.iter().cloned() {
            let first_node = &self.graph[first_node_index];
            if first_node.kernel_index.is_some() {
                continue;
            }
            if matches!(first_node.op, Op::Unary(_) | Op::Binary(_)) {
                let kernel_index = Some(self.kernels.len());
                let shape = first_node.shape.clone();
                self.kernels.push(Kernel {
                    desc: KernelDesc::PerElement(PerElementKernelDesc {
                        shape: shape.clone(),
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        ops: Vec::new(),
                    }),
                    inputs: Vec::new(),
                    members: Vec::new(),
                    outputs: Vec::new(),
                });
                self.graph[first_node_index].kernel_index = kernel_index;

                'outer: loop {
                    'inner: for other_node_index in self.ordering.iter().cloned() {
                        let other_node = &self.graph[other_node_index];

                        // check this node has no kernel and matches shape
                        let can_include = other_node.kernel_index.is_none()
                            && match other_node.op {
                                Op::Unary(_) | Op::Binary(_) => other_node.shape == shape,
                                Op::Literal(_) => true,
                                _ => false,
                            };
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

        // build per-element kernel members in usage order
        for node_index in self.ordering.iter().cloned() {
            if let Some(kernel_index) = self.graph[node_index].kernel_index {
                self.kernels[kernel_index].members.push(node_index);
            }
        }

        // finish the per-element kernel descriptions
        for (kernel_index, kernel) in self.kernels.iter_mut().enumerate() {
            let kernel_inputs = &mut kernel.inputs;
            let kernel_desc = match &mut kernel.desc {
                KernelDesc::PerElement(desc) => desc,
            };
            let mut node_op_index = HashMap::new();

            let graph = &self.graph;
            for node_index in kernel.members.iter().cloned() {
                // gather the arguments (loading as necessary)
                let mut args = [None, None];
                for edge_ref in graph.edges_directed(node_index, Incoming) {
                    let source_node_index = edge_ref.source();
                    let edge = edge_ref.weight();
                    assert_eq!(edge.transpose, false);
                    args[edge.arg] =
                        Some(*node_op_index.entry(source_node_index).or_insert_with(|| {
                            let source_node = &graph[source_node_index];
                            assert_ne!(source_node.kernel_index, Some(kernel_index));
                            let input_index = kernel_desc.inputs.len();
                            kernel_desc.inputs.push(source_node.shape.clone());
                            kernel_inputs.push(source_node_index);
                            let op_index = kernel_desc.ops.len();
                            kernel_desc
                                .ops
                                .push(PerElementKernelOp::Load { input_index });
                            op_index
                        }));
                }

                // emit the op
                let op_index = kernel_desc.ops.len();
                kernel_desc.ops.push(match graph[node_index].op {
                    Op::Unary(op) => PerElementKernelOp::Unary {
                        op,
                        src_index: args[0].unwrap(),
                    },
                    Op::Binary(op) => PerElementKernelOp::Binary {
                        op,
                        src1_index: args[0].unwrap(),
                        src2_index: args[1].unwrap(),
                    },
                    Op::Literal(value) => PerElementKernelOp::Literal(value),
                    _ => panic!("unexpected op type"),
                });
                node_op_index.insert(node_index, op_index);

                // store the result if necessary
                if graph
                    .neighbors_directed(node_index, Outgoing)
                    .any(|other_index| graph[other_index].kernel_index != Some(kernel_index))
                {
                    kernel_desc.outputs.push(op_index);
                    kernel.outputs.push(node_index);
                }
            }
        }
    }

    fn generate_kernel_source(&self, kernel_index: usize) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        writeln!(w, "#version 460 core")?;

        let desc = match &self.kernels[kernel_index].desc {
            KernelDesc::PerElement(desc) => desc,
        };

        let mut binding_index = 0;

        for input_index in 0..desc.inputs.len() {
            writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
            writeln!(
                w,
                "readonly restrict buffer input_layout{0} {{ float input{0}[]; }};",
                input_index
            )?;
            binding_index += 1;
        }
        for output_index in 0..desc.outputs.len() {
            writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
            writeln!(
                w,
                "writeonly restrict buffer output_layout{0} {{ float output{0}[]; }};",
                output_index
            )?;
            binding_index += 1;
        }

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        for (op_index, op) in desc.ops.iter().enumerate() {
            write!(w, "float tmp{} = ", op_index)?;
            match op {
                PerElementKernelOp::Load { input_index } => {
                    write!(w, "input{}[gl_GlobalInvocationID.x]", input_index)?
                }
                PerElementKernelOp::Literal(value) => write!(w, "{:#?}", value)?,
                PerElementKernelOp::Unary { op, src_index } => match op {
                    UnaryOp::Neg => write!(w, "-{}", src_index)?,
                    UnaryOp::Exp => write!(w, "exp({})", src_index)?,
                    UnaryOp::Log => write!(w, "log({})", src_index)?,
                    UnaryOp::OneHot => write!(
                        w,
                        "(gl_GlobalInvocationID.x == uint({})) ? 1.0 : 0.0",
                        src_index
                    )?,
                },
                PerElementKernelOp::Binary {
                    op,
                    src1_index,
                    src2_index,
                } => match op {
                    BinaryOp::Add => write!(w, "{} + {}", src1_index, src2_index)?,
                    BinaryOp::Sub => write!(w, "{} - {}", src1_index, src2_index)?,
                    BinaryOp::Mul => write!(w, "{} * {}", src1_index, src2_index)?,
                    BinaryOp::Div => write!(w, "{} / {}", src1_index, src2_index)?,
                },
            }
            writeln!(w, ";")?;
        }

        for (output_index, src_index) in desc.outputs.iter().enumerate() {
            writeln!(
                w,
                "output{}[gl_GlobalInvocationID.x] = tmp{};",
                output_index, src_index
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
