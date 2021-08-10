use crate::common::*;
use arrayvec::ArrayVec;
use petgraph::{
    prelude::*,
    visit::{
        IntoEdgeReferences, IntoNodeReferences, NodeIndexable, NodeRef, Topo, VisitMap, Visitable,
    },
};
use slotmap::{SecondaryMap, SlotMap};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    convert::TryInto,
    fs::File,
    hash::{Hash, Hasher},
    io, iter,
};
use tinyvec::ArrayVec as TinyVec;

fn get_arg_edge_ids(ops: &OpGraph, node_id: OpNodeId) -> TinyVec<[OpEdgeId; MAX_OP_ARGS]> {
    let mut v = [None; MAX_OP_ARGS];
    let mut n = 0;
    for edge_ref in ops.edges_directed(node_id, Incoming) {
        let edge = edge_ref.weight();
        assert!(v[edge.arg].is_none());
        v[edge.arg] = Some(edge_ref.id());
        n = n.max(edge.arg + 1);
    }
    v[..n].iter().copied().map(|id| id.unwrap()).collect()
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ArgSource {
    pub(crate) node_id: OpNodeId,
    pub(crate) view: View,
}

pub(crate) fn get_arg_sources(
    ops: &OpGraph,
    node_id: OpNodeId,
) -> TinyVec<[ArgSource; MAX_OP_ARGS]> {
    get_arg_edge_ids(ops, node_id)
        .iter()
        .copied()
        .map(|edge_id| ArgSource {
            node_id: ops.edge_endpoints(edge_id).unwrap().0,
            view: ops[edge_id].view,
        })
        .collect()
}

#[derive(Debug)]
pub(crate) struct Cluster {
    pub(crate) kernel: GenericKernel,
    pub(crate) inputs: Vec<OpNodeId>,
    pub(crate) members: Vec<OpNodeId>,
    pub(crate) outputs: Vec<OpNodeId>,
}

slotmap::new_key_type! {
    pub(crate) struct ClusterId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDotOutput {
    None,
    Cluster,
    Color,
}

pub struct Schedule {
    pub(crate) variables: SharedVariables,
    pub(crate) ops: OpGraph,
    pub(crate) ops_sorted: Vec<OpNodeId>,
    pub(crate) clusters: SlotMap<ClusterId, Cluster>,
    pub(crate) clusters_sorted: Vec<ClusterId>,
}

impl Schedule {
    pub(crate) fn new(variables: SharedVariables, ops: OpGraph) -> Self {
        let mut sched = Self {
            variables,
            ops,
            ops_sorted: Vec::new(),
            clusters: SlotMap::with_key(),
            clusters_sorted: Vec::new(),
        };

        sched.rebuild_ordering();
        sched.eliminate_dead_code();

        sched.rebuild_ordering();
        sched.eliminate_moves();

        sched.rebuild_ordering();
        sched.eliminate_common_subgraphs();

        sched.rebuild_ordering();
        sched.make_built_ins_and_literals_unique();

        sched.rebuild_ordering();
        sched.build_clusters();

        sched
    }

    fn rebuild_ordering(&mut self) {
        self.ops_sorted.clear();
        let mut topo = Topo::new(&self.ops);
        while let Some(node_id) = topo.next(&self.ops) {
            self.ops_sorted.push(node_id);
        }
        assert_eq!(self.ops.node_count(), self.ops_sorted.len());
    }

    fn eliminate_dead_code(&mut self) {
        let mut live = self.ops.visit_map();
        for node_ref in self.ops.node_references() {
            if matches!(node_ref.weight().op, Op::Output { .. }) {
                live.visit(node_ref.id());
            }
        }
        for index in self.ops_sorted.iter().rev().copied() {
            if live.is_visited(&index) {
                for input_index in self.ops.neighbors_directed(index, Incoming) {
                    live.visit(input_index);
                }
            }
        }
        self.ops.retain_nodes(|_, index| live.is_visited(&index));
    }

    fn eliminate_common_subgraphs(&mut self) {
        let mut hashes = vec![0u64; self.ops.node_bound()];
        let mut ids_from_hash = HashMap::new();
        for node_id in self.ops_sorted.iter().copied() {
            let node = &self.ops[node_id];
            let arg_sources = get_arg_sources(&self.ops, node_id);
            let hash = {
                let mut hasher = DefaultHasher::new();
                for arg_source in arg_sources.iter() {
                    arg_source.hash(&mut hasher);
                }
                node.shape.hash(&mut hasher);
                node.op.hash(&mut hasher);
                hasher.finish()
            };
            hashes[node_id.index()] = hash;
            if node.op.can_merge() {
                let ids = ids_from_hash.entry(hash).or_insert_with(Vec::new);
                if let Some(other_id) = ids.iter().copied().find(|&id| {
                    let other_node = &self.ops[id];
                    let other_arg_sources = get_arg_sources(&self.ops, id);
                    node.shape == other_node.shape
                        && node.op == other_node.op
                        && arg_sources == other_arg_sources
                }) {
                    let mut edges = self.ops.neighbors_directed(node_id, Outgoing).detach();
                    while let Some((edge_id, dst_id)) = edges.next(&self.ops) {
                        let edge = self.ops[edge_id].clone();
                        self.ops.add_edge(other_id, dst_id, edge);
                    }
                    self.ops.remove_node(node_id);
                } else {
                    ids.push(node_id);
                }
            }
        }
    }

    fn make_built_ins_and_literals_unique(&mut self) {
        for node_id in self.ops_sorted.iter().copied() {
            let node = &self.ops[node_id];
            if matches!(&node.op, Op::Literal(_) | Op::BuiltIn { .. }) {
                let orig_node = node.clone();
                let mut out_edges = self.ops.neighbors_directed(node_id, Outgoing).detach();
                while let Some((out_edge_id, out_node_id)) = out_edges.next(&self.ops) {
                    let new_node_id = self.ops.add_node(orig_node.clone());
                    let new_edge = self.ops[out_edge_id].clone();
                    self.ops.add_edge(new_node_id, out_node_id, new_edge);
                }
                self.ops.remove_node(node_id);
            }
        }
    }

    fn eliminate_moves(&mut self) {
        for node_id in self.ops_sorted.iter().copied() {
            if let Op::Unary(UnaryOp::Mov) = &self.ops[node_id].op {
                let in_edge_ref = self.ops.edges_directed(node_id, Incoming).only().unwrap();
                let in_edge_id = in_edge_ref.id();
                let in_node_id = in_edge_ref.source();
                let can_reshape = self.ops[in_node_id].op.can_reshape();
                let can_eliminate =
                    self.ops
                        .edges_directed(node_id, Outgoing)
                        .all(|out_edge_ref| {
                            self.ops[out_edge_ref.target()]
                                .op
                                .output_variable_id()
                                .is_none()
                                && self.ops[in_edge_id].view.can_view_through(
                                    &self.ops[out_edge_ref.id()].view,
                                    can_reshape,
                                )
                        });
                if can_eliminate {
                    let mut out_edges = self.ops.neighbors_directed(node_id, Outgoing).detach();
                    while let Some((out_edge_id, out_node_id)) = out_edges.next(&self.ops) {
                        let in_edge = &self.ops[in_edge_id];
                        let out_edge = &self.ops[out_edge_id];
                        assert_eq!(in_edge.arg, 0);
                        let new_edge = OpEdge {
                            arg: out_edge.arg,
                            view: in_edge.view.through(&out_edge.view, can_reshape),
                        };
                        self.ops.add_edge(in_node_id, out_node_id, new_edge);
                    }
                    self.ops.remove_node(node_id);
                }
            }
        }
    }

    fn any_predecessor(&self, roots: &[OpNodeId], mut f: impl FnMut(OpNodeId) -> bool) -> bool {
        let mut markers = self.ops.visit_map();
        for &node_id in roots {
            markers.visit(node_id);
        }
        for node_id in self.ops_sorted.iter().copied().rev() {
            if self
                .ops
                .neighbors_directed(node_id, Outgoing)
                .any(|output_node_id| markers.is_visited(&output_node_id))
            {
                markers.visit(node_id);
                if f(node_id) {
                    return true;
                }
            }
        }
        false
    }

    fn any_successor(&self, roots: &[OpNodeId], mut f: impl FnMut(OpNodeId) -> bool) -> bool {
        let mut markers = self.ops.visit_map();
        for &node_id in roots {
            markers.visit(node_id);
        }
        for node_id in self.ops_sorted.iter().copied().rev() {
            if self
                .ops
                .neighbors_directed(node_id, Incoming)
                .any(|input_node_id| markers.is_visited(&input_node_id))
            {
                markers.visit(node_id);
                if f(node_id) {
                    return true;
                }
            }
        }
        false
    }

    #[allow(clippy::blocks_in_if_conditions)]
    fn build_clusters(&mut self) {
        // first gather per-element nodes into kernels
        for first_node_id in self.ops_sorted.iter().copied() {
            let first_node = &self.ops[first_node_id];
            if first_node.cluster_id.is_some() {
                continue;
            }
            if first_node.op.is_per_element() {
                let element_count = first_node.shape.element_count();

                let cluster_id = Some(self.clusters.insert(Cluster {
                    kernel: GenericKernel::PerElement(PerElementKernel {
                        element_count,
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        ops: Vec::new(),
                    }),
                    inputs: Vec::new(),
                    members: Vec::new(),
                    outputs: Vec::new(),
                }));
                self.ops[first_node_id].cluster_id = cluster_id;

                'outer: loop {
                    'inner: for other_node_id in self.ops_sorted.iter().copied() {
                        let other_node = &self.ops[other_node_id];

                        // check this node has no cluster and matches element count
                        let can_include = other_node.cluster_id.is_none()
                            && other_node.op.is_per_element()
                            && other_node.shape.element_count() == element_count;
                        if !can_include {
                            continue 'inner;
                        }

                        // skip this node if any edges with cluster nodes have non-identity views
                        let mut has_kernel_neighbor = false;
                        for edge_ref in self
                            .ops
                            .edges_directed(other_node_id, Incoming)
                            .filter(|edge_ref| {
                                assert_eq!(edge_ref.target(), other_node_id);
                                self.ops[edge_ref.source()].cluster_id == cluster_id
                            })
                            .chain(self.ops.edges_directed(other_node_id, Outgoing).filter(
                                |edge_ref| {
                                    assert_eq!(edge_ref.source(), other_node_id);
                                    self.ops[edge_ref.target()].cluster_id == cluster_id
                                },
                            ))
                        {
                            has_kernel_neighbor = true;
                            if !edge_ref.weight().view.is_contiguous() {
                                continue 'inner;
                            }
                        }

                        // placing this node in the cluster needs to save a load
                        if !has_kernel_neighbor {
                            continue 'inner;
                        }

                        // check uses of this node don't re-enter this cluster
                        if self.any_successor(&[other_node_id], |node_id| {
                            self.ops[node_id].cluster_id.is_none()
                                && self
                                    .ops
                                    .neighbors_directed(node_id, Outgoing)
                                    .any(|node_id| self.ops[node_id].cluster_id == cluster_id)
                        }) {
                            continue 'inner;
                        }

                        // check inputs of this node don't re-enter this cluster
                        if self.any_predecessor(&[other_node_id], |node_id| {
                            self.ops[node_id].cluster_id.is_none()
                                && self
                                    .ops
                                    .neighbors_directed(node_id, Incoming)
                                    .any(|node_id| self.ops[node_id].cluster_id == cluster_id)
                        }) {
                            continue 'inner;
                        }

                        // ok to merge, restart search with new cluster
                        self.ops[other_node_id].cluster_id = cluster_id;
                        continue 'outer;
                    }
                    break 'outer;
                }
            }
        }

        // build per-element cluster members in usage order
        for node_id in self.ops_sorted.iter().copied() {
            if let Some(cluster_id) = self.ops[node_id].cluster_id {
                self.clusters[cluster_id].members.push(node_id);
            }
        }

        // finally build the per-element clusters and kernels
        for (cluster_id, cluster) in self.clusters.iter_mut() {
            let kernel = match &mut cluster.kernel {
                GenericKernel::PerElement(kernel) => kernel,
                _ => unreachable!(),
            };
            let members = &cluster.members;
            let inputs = &mut cluster.inputs;
            let outputs = &mut cluster.outputs;

            let mut arg_op_index = HashMap::new();
            let mut member_op_index = HashMap::new();

            let graph = &self.ops;
            for node_id in members.iter().copied() {
                // gather the arguments (loading as necessary)
                let arg_sources = get_arg_sources(graph, node_id);
                let args: TinyVec<[usize; MAX_OP_ARGS]> = arg_sources
                    .iter()
                    .map(|source| {
                        if let Some(op_index) = member_op_index.get(&source.node_id) {
                            *op_index
                        } else {
                            *arg_op_index.entry(*source).or_insert_with(|| {
                                let source_node = &graph[source.node_id];
                                assert_ne!(source_node.cluster_id, Some(cluster_id));
                                let op_index = kernel.ops.len();
                                match source_node.op {
                                    Op::Literal(value) => {
                                        kernel.ops.push(PerElementKernelOp::Literal(value));
                                    }
                                    Op::BuiltIn(op) => {
                                        kernel.ops.push(PerElementKernelOp::BuiltIn {
                                            op,
                                            view: source.view,
                                        });
                                    }
                                    _ => {
                                        let input_index = kernel.inputs.len();
                                        kernel.inputs.push(source.view);
                                        inputs.push(source.node_id);
                                        kernel.ops.push(PerElementKernelOp::Load { input_index });
                                    }
                                }
                                op_index
                            })
                        }
                    })
                    .collect();

                // emit the op
                let op = match graph[node_id].op {
                    Op::Unary(op) => PerElementKernelOp::Unary { op, args: args[0] },
                    Op::Binary(op) => PerElementKernelOp::Binary {
                        op,
                        args: args[..2].try_into().unwrap(),
                    },
                    Op::CompareAndSelect(compare_mode) => PerElementKernelOp::CompareAndSelect {
                        compare_mode,
                        args: args[..4].try_into().unwrap(),
                    },
                    _ => panic!("unexpected op type"),
                };
                let op_index = kernel.ops.len();
                kernel.ops.push(op);
                member_op_index.insert(node_id, op_index);

                // store the result if necessary
                if graph
                    .neighbors_directed(node_id, Outgoing)
                    .any(|other_id| graph[other_id].cluster_id != Some(cluster_id))
                {
                    kernel.outputs.push(op_index);
                    outputs.push(node_id);
                }
            }
        }

        // add reduction and matrix multiply kernels
        for node_id in self.ops_sorted.iter().copied() {
            let node = &self.ops[node_id];
            if node.cluster_id.is_none() {
                match node.op {
                    Op::Reduce { reduce_op, axis } => {
                        let arg_sources = get_arg_sources(&self.ops, node_id);
                        assert_eq!(arg_sources.len(), 1);
                        let src0 = &arg_sources[0];
                        self.ops[node_id].cluster_id = Some(self.clusters.insert(Cluster {
                            kernel: GenericKernel::Reduce(ReduceKernel {
                                shape: node.shape,
                                input: src0.view,
                                reduce_op,
                                axis,
                            }),
                            inputs: vec![src0.node_id],
                            members: vec![node_id],
                            outputs: vec![node_id],
                        }));
                    }
                    Op::MatMul => {
                        let arg_sources = get_arg_sources(&self.ops, node_id);
                        assert_eq!(arg_sources.len(), 2);
                        let kernel_inputs = arg_sources
                            .iter()
                            .map(|src| src.view)
                            .collect::<ArrayVec<_, 2>>()
                            .into_inner()
                            .unwrap();
                        self.ops[node_id].cluster_id = Some(self.clusters.insert(Cluster {
                            kernel: GenericKernel::MatMul(MatMulKernel {
                                shape: node.shape,
                                inputs: kernel_inputs,
                            }),
                            inputs: arg_sources.iter().map(|src| src.node_id).collect(),
                            members: vec![node_id],
                            outputs: vec![node_id],
                        }));
                    }
                    Op::Unpad { axis, pad } => {
                        let arg_sources = get_arg_sources(&self.ops, node_id);
                        assert_eq!(arg_sources.len(), 1);
                        let src0 = &arg_sources[0];
                        self.ops[node_id].cluster_id = Some(self.clusters.insert(Cluster {
                            kernel: GenericKernel::Unpad(UnpadKernel {
                                shape: node.shape,
                                input: src0.view,
                                axis,
                                pad,
                            }),
                            inputs: vec![src0.node_id],
                            members: vec![node_id],
                            outputs: vec![node_id],
                        }));
                    }
                    Op::WindowsToImage { stride } => {
                        let arg_sources = get_arg_sources(&self.ops, node_id);
                        assert_eq!(arg_sources.len(), 1);
                        let src0 = &arg_sources[0];
                        self.ops[node_id].cluster_id = Some(self.clusters.insert(Cluster {
                            kernel: GenericKernel::WindowsToImage(WindowsToImageKernel {
                                shape: node.shape,
                                input: src0.view,
                                stride,
                            }),
                            inputs: vec![src0.node_id],
                            members: vec![node_id],
                            outputs: vec![node_id],
                        }));
                    }
                    Op::Input { .. } | Op::Output { .. } | Op::Literal(_) | Op::BuiltIn(_) => {}
                    _ => panic!("unexpected op without a kernel"),
                }
            }
        }

        // make cluster ordering
        let mut cluster_graph = StableDiGraph::<ClusterId, (), usize>::default();
        let mut cluster_node_ids = SecondaryMap::new();
        for cluster_id in self.clusters.keys() {
            cluster_node_ids.insert(cluster_id, cluster_graph.add_node(cluster_id));
        }
        for (source_id, target_id) in self.ops.edge_references().filter_map(|edge_ref| {
            let source_id = self.ops[edge_ref.source()].cluster_id?;
            let target_id = self.ops[edge_ref.target()].cluster_id?;
            if source_id != target_id {
                Some((source_id, target_id))
            } else {
                None
            }
        }) {
            cluster_graph.add_edge(cluster_node_ids[source_id], cluster_node_ids[target_id], ());
        }
        self.clusters_sorted.clear();
        let mut topo = Topo::new(&cluster_graph);
        while let Some(cluster_node_id) = topo.next(&cluster_graph) {
            self.clusters_sorted.push(cluster_graph[cluster_node_id]);
        }
        assert_eq!(self.clusters_sorted.len(), self.clusters.len());
    }

    pub fn write_dot_file(&self, kernel_output: KernelDotOutput, path: &str) {
        let mut w = io::BufWriter::new(File::create(path).unwrap());
        self.write_dot(kernel_output, &mut w).unwrap();
    }

    fn write_dot(&self, kernel_output: KernelDotOutput, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for (index, cluster_id) in iter::once(None)
            .chain(self.clusters.keys().map(Some))
            .enumerate()
        {
            if kernel_output == KernelDotOutput::Cluster && cluster_id.is_some() {
                writeln!(w, "subgraph cluster{} {{ style=filled;", index)?;
            }
            for node_ref in self
                .ops
                .node_references()
                .filter(|node_ref| node_ref.weight().cluster_id == cluster_id)
            {
                let node = node_ref.weight();
                if let Op::Literal(value) = &node.op {
                    writeln!(
                        w,
                        "n{} [shape=none,label=\"{:E}\"];",
                        node_ref.id().index(),
                        value.into_inner()
                    )?;
                } else {
                    let hasher = if kernel_output == KernelDotOutput::Color {
                        cluster_id.map(|cluster_id| {
                            let mut hasher = DefaultHasher::new();
                            cluster_id.hash(&mut hasher);
                            hasher
                        })
                    } else {
                        let mut hasher = DefaultHasher::new();
                        node.colour.hash(&mut hasher);
                        Some(hasher)
                    };
                    let col = if let Some(hasher) = hasher {
                        let hash = hasher.finish();
                        ((((hash >> 48) ^ (hash >> 24) ^ hash) as u32) & 0xffffff) | 0x404040
                    } else {
                        0xffffff
                    };
                    write!(
                        w,
                        "n{} [shape=box,style=filled,color=\"#{:06X}\",label=\"{:?}\\n",
                        node_ref.id().index(),
                        col,
                        node.op
                    )?;
                    if let Op::Input { variable_id } | Op::Output { variable_id } = node.op {
                        write!(
                            w,
                            "{}",
                            self.variables
                                .as_ref()
                                .borrow()
                                .get(variable_id)
                                .unwrap()
                                .name
                        )?;
                    }
                    writeln!(w, "{}\"];", node.shape)?;
                }
            }
            if kernel_output == KernelDotOutput::Cluster && cluster_id.is_some() {
                writeln!(w, "}}")?;
            }
        }
        for edge_ref in self.ops.edge_references() {
            write!(
                w,
                "n{} -> n{}",
                edge_ref.source().index(),
                edge_ref.target().index()
            )?;
            if !edge_ref.weight().view.is_contiguous() {
                write!(w, " [label=\"V\"]")?;
            }
            writeln!(w, ";")?;
        }
        writeln!(w, "}}")
    }
}
