use crate::common::*;
use arrayvec::ArrayVec;
use petgraph::{
    prelude::*,
    visit::{IntoEdgeReferences, IntoNodeReferences, NodeRef, Topo, VisitMap, Visitable},
};
use slotmap::{SecondaryMap, SlotMap};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    io, iter,
};

pub(crate) fn get_arg_edge_ids<const N: usize>(ops: &OpGraph, node_id: OpNodeId) -> [OpEdgeId; N] {
    let mut edge_ids = [OpEdgeId::end(); N];
    for edge_ref in ops.edges_directed(node_id, Incoming) {
        let edge = edge_ref.weight();
        assert_eq!(edge_ids[edge.arg], OpEdgeId::end());
        edge_ids[edge.arg] = edge_ref.id();
    }
    edge_ids
}

#[derive(Debug)]
pub(crate) struct Cluster {
    pub(crate) kernel: Kernel,
    pub(crate) inputs: Vec<OpNodeId>,
    pub(crate) members: Vec<OpNodeId>,
    pub(crate) outputs: Vec<OpNodeId>,
}

slotmap::new_key_type! {
    pub(crate) struct ClusterId;
}

pub struct Graph {
    pub(crate) variables: SharedVariables,
    pub(crate) ops: OpGraph,
    pub(crate) ops_sorted: Vec<OpNodeId>,
    pub(crate) clusters: SlotMap<ClusterId, Cluster>,
    pub(crate) clusters_sorted: Vec<ClusterId>,
}

impl Graph {
    pub(crate) fn new(variables: SharedVariables, ops: OpGraph) -> Self {
        let mut graph = Self {
            variables,
            ops,
            ops_sorted: Vec::new(),
            clusters: SlotMap::with_key(),
            clusters_sorted: Vec::new(),
        };

        graph.rebuild_ordering();
        graph.eliminate_dead_code();

        graph.rebuild_ordering();
        graph.eliminate_accumulate_nodes();

        graph.rebuild_ordering();
        graph.eliminate_view_nodes();

        graph.rebuild_ordering();
        graph.build_clusters();

        graph
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

    fn eliminate_accumulate_nodes(&mut self) {
        for node_id in self.ops_sorted.iter().copied() {
            if matches!(self.ops[node_id].op, Op::Accumulate) {
                assert_eq!(self.ops.edges_directed(node_id, Incoming).count(), 1); // TODO: generate adds
                let mut in_edges = self.ops.neighbors_directed(node_id, Incoming).detach();
                let (in_edge_id, in_node_id) = in_edges.next(&self.ops).unwrap();
                let mut out_edges = self.ops.neighbors_directed(node_id, Outgoing).detach();
                while let Some((out_edge_id, out_node_id)) = out_edges.next(&self.ops) {
                    let in_edge = &self.ops[in_edge_id];
                    let out_edge = &self.ops[out_edge_id];
                    assert_eq!(in_edge.arg, 0);
                    let new_edge = OpEdge {
                        arg: out_edge.arg,
                        view: in_edge.view.through(&out_edge.view),
                    };
                    self.ops.add_edge(in_node_id, out_node_id, new_edge);
                }
                self.ops.remove_node(node_id);
            }
        }
    }

    fn eliminate_view_nodes(&mut self) {
        for node_id in self.ops_sorted.iter().copied() {
            if let Op::View(view) = &self.ops[node_id].op {
                let view = view.clone();
                assert_eq!(self.ops.neighbors_directed(node_id, Incoming).count(), 1);
                let mut in_edges = self.ops.neighbors_directed(node_id, Incoming).detach();
                let (in_edge_id, in_node_id) = in_edges.next(&self.ops).unwrap();
                let mut out_edges = self.ops.neighbors_directed(node_id, Outgoing).detach();
                while let Some((out_edge_id, out_node_id)) = out_edges.next(&self.ops) {
                    let in_edge = &self.ops[in_edge_id];
                    let out_edge = &self.ops[out_edge_id];
                    assert_eq!(in_edge.arg, 0);
                    let new_edge = OpEdge {
                        arg: out_edge.arg,
                        view: in_edge.view.through(&view).through(&out_edge.view),
                    };
                    self.ops.add_edge(in_node_id, out_node_id, new_edge);
                }
                self.ops.remove_node(node_id);
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
        return false;
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
        return false;
    }

    fn build_clusters(&mut self) {
        // first gather per-element nodes into kernels
        for first_node_id in self.ops_sorted.iter().copied() {
            let first_node = &self.ops[first_node_id];
            if !first_node.cluster_id.is_none() {
                continue;
            }
            if first_node.op.is_per_element() {
                let shape = first_node.shape.clone();

                let cluster_id = Some(self.clusters.insert(Cluster {
                    kernel: Kernel::PerElement(PerElementKernel {
                        shape: shape.clone(),
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

                        // check this node has no cluster and matches shape
                        let is_matching_shape =
                            other_node.op.is_per_element() && other_node.shape == shape;
                        let is_literal = matches!(other_node.op, Op::Literal(_));
                        let can_include =
                            other_node.cluster_id.is_none() && (is_matching_shape || is_literal);
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
                            if !is_literal && !edge_ref.weight().view.is_identity() {
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
                Kernel::PerElement(kernel) => kernel,
                _ => unreachable!(),
            };
            let members = &cluster.members;
            let inputs = &mut cluster.inputs;
            let outputs = &mut cluster.outputs;

            let mut node_op_index = HashMap::new();

            let graph = &self.ops;
            for node_id in members.iter().copied() {
                // gather the arguments (loading as necessary)
                let mut args = [None, None];
                for edge_ref in graph.edges_directed(node_id, Incoming) {
                    let source_node_id = edge_ref.source();
                    let edge = edge_ref.weight();
                    args[edge.arg] =
                        Some(*node_op_index.entry(source_node_id).or_insert_with(|| {
                            let source_node = &graph[source_node_id];
                            assert_ne!(source_node.cluster_id, Some(cluster_id));
                            let input_index = kernel.inputs.len();
                            kernel.inputs.push(KernelInput {
                                shape: source_node.shape.clone(),
                                view: edge.view.clone(),
                            });
                            inputs.push(source_node_id);
                            let op_index = kernel.ops.len();
                            kernel.ops.push(PerElementKernelOp::Load { input_index });
                            op_index
                        }));
                }

                // emit the op
                let op_index = kernel.ops.len();
                kernel.ops.push(match graph[node_id].op {
                    Op::BuiltIn(op) => PerElementKernelOp::BuiltIn(op),
                    Op::Unary(op) => PerElementKernelOp::Unary {
                        op,
                        arg0_index: args[0].unwrap(),
                    },
                    Op::Binary(op) => PerElementKernelOp::Binary {
                        op,
                        arg0_index: args[0].unwrap(),
                        arg1_index: args[1].unwrap(),
                    },
                    Op::Literal(value) => PerElementKernelOp::Literal(value),
                    _ => panic!("unexpected op type"),
                });
                node_op_index.insert(node_id, op_index);

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
                        let [edge_id] = get_arg_edge_ids(&self.ops, node_id);
                        let input_node_id = self.ops.edge_endpoints(edge_id).unwrap().0;
                        self.ops[node_id].cluster_id = Some(self.clusters.insert(Cluster {
                            kernel: Kernel::Reduce(ReduceKernel {
                                shape: node.shape.clone(),
                                input: KernelInput {
                                    shape: self.ops[input_node_id].shape.clone(),
                                    view: self.ops[edge_id].view.clone(),
                                },
                                reduce_op,
                                axis,
                            }),
                            inputs: vec![input_node_id],
                            members: vec![node_id],
                            outputs: vec![node_id],
                        }));
                    }
                    Op::MatMul => {
                        let edge_ids: [_; 2] = get_arg_edge_ids(&self.ops, node_id);
                        let input_node_ids = edge_ids
                            .iter()
                            .map(|&edge_id| self.ops.edge_endpoints(edge_id).unwrap().0)
                            .collect::<ArrayVec<_, 2>>()
                            .into_inner()
                            .unwrap();
                        let kernel_inputs = edge_ids
                            .iter()
                            .zip(input_node_ids.iter())
                            .map(|(&edge_id, &node_id)| KernelInput {
                                shape: self.ops[node_id].shape.clone(),
                                view: self.ops[edge_id].view.clone(),
                            })
                            .collect::<ArrayVec<_, 2>>()
                            .into_inner()
                            .unwrap();
                        self.ops[node_id].cluster_id = Some(self.clusters.insert(Cluster {
                            kernel: Kernel::MatMul(MatMulKernel {
                                shape: node.shape.clone(),
                                inputs: kernel_inputs,
                            }),
                            inputs: input_node_ids.iter().copied().collect(),
                            members: vec![node_id],
                            outputs: vec![node_id],
                        }));
                    }
                    Op::Input { .. } | Op::Output { .. } | Op::Literal(_) => {}
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

    pub fn write_dot(&self, w: &mut impl io::Write) -> io::Result<()> {
        writeln!(w, "digraph G {{")?;
        for (index, cluster_id) in iter::once(None)
            .chain(self.clusters.keys().map(|id| Some(id)))
            .enumerate()
        {
            if cluster_id.is_some() {
                writeln!(w, "subgraph cluster{} {{ style=filled;", index)?;
            }
            for node_ref in self
                .ops
                .node_references()
                .filter(|node_ref| node_ref.weight().cluster_id == cluster_id)
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
            if cluster_id.is_some() {
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
            if !edge_ref.weight().view.is_identity() {
                write!(w, " [label=\"V\"]")?;
            }
            writeln!(w, ";")?;
        }
        writeln!(w, "}}")
    }
}
