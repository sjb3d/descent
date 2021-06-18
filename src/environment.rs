use crate::{device::common::*, op::*, prelude::*};
use petgraph::visit::{IntoNodeReferences, NodeIndexable, NodeRef};
use slotmap::SlotMap;
use spark::vk;
use std::{cell::RefCell, collections::HashSet, io, rc::Rc};

slotmap::new_key_type! {
    pub struct Variable;
}

pub(crate) type SharedVariables = Rc<RefCell<SlotMap<Variable, VariableState>>>;

pub struct VariableWriter<'a>(StagingWriter<'a>);

impl<'a> io::Write for VariableWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(self.0.write_slice(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush_staging();
        Ok(())
    }
}

pub struct VariableReader<'a>(StagingReader<'a>);

impl<'a> io::Read for VariableReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Ok(self.0.read_slice(buf))
    }
}

pub(crate) struct VariableState {
    pub(crate) shape: Shape,
    pub(crate) name: String,
    pub(crate) buffer_id: Option<BufferId>,
}

struct Kernel {
    shader_module: vk::ShaderModule,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct KernelCache {
    context: SharedContext,
}

#[derive(Debug, Clone, Copy, Default)]
struct OpNodeRunState {
    usage_count: usize,
    buffer_id: Option<BufferId>,
}

pub struct Environment {
    context: SharedContext,
    fences: FenceSet,
    command_buffers: CommandBufferSet,
    buffer_heap: BufferHeap,
    staging_buffer: StagingBuffer,
    variables: SharedVariables,
}

impl Environment {
    pub fn new() -> Self {
        let context = Context::new();
        let fences = FenceSet::new(&context);
        let command_buffers = CommandBufferSet::new(&context, &fences);
        let buffer_heap = BufferHeap::new(&context);
        let staging_buffer = StagingBuffer::new(&context, &fences);
        Self {
            context,
            fences,
            command_buffers,
            buffer_heap,
            staging_buffer,
            variables: Rc::new(RefCell::new(SlotMap::with_key())),
        }
    }

    pub fn variable(&mut self, shape: impl Into<Shape>, name: impl Into<String>) -> Variable {
        let shape = shape.into();
        let name = name.into();
        let id = self.variables.borrow_mut().insert(VariableState {
            shape: shape.clone(),
            name,
            buffer_id: None,
        });
        id
    }

    pub fn writer(&mut self, var: Variable) -> VariableWriter {
        let mut variables = self.variables.borrow_mut();
        let state = &mut variables[var];
        if let Some(buffer_id) = state.buffer_id.take() {
            self.buffer_heap.free(buffer_id);
        }
        let buffer_id = self.buffer_heap.alloc(state.shape.buffer_size()).unwrap();
        state.buffer_id = Some(buffer_id);
        VariableWriter(StagingWriter::new(
            &mut self.staging_buffer,
            &mut self.command_buffers,
            &mut self.fences,
            self.buffer_heap.info(buffer_id),
        ))
    }

    pub fn reader(&mut self, var: Variable) -> VariableReader {
        let variables = self.variables.borrow();
        let state = &variables[var];
        let buffer_id = state.buffer_id.unwrap();
        VariableReader(StagingReader::new(
            &mut self.staging_buffer,
            &mut self.command_buffers,
            &mut self.fences,
            self.buffer_heap.info(buffer_id),
        ))
    }

    pub fn builder(&self) -> GraphBuilder {
        GraphBuilder::new(SharedVariables::clone(&self.variables))
    }

    pub fn run(&mut self, graph: &Graph) {
        let mut variables = self.variables.borrow_mut();

        // collect input and output variables
        let inputs: Vec<_> = graph
            .ops
            .node_references()
            .filter_map(|node_ref| {
                if matches!(node_ref.weight().op, Op::Input { .. }) {
                    Some(node_ref.id())
                } else {
                    None
                }
            })
            .collect();
        let outputs: Vec<_> = graph
            .ops
            .node_references()
            .filter_map(|node_ref| {
                if matches!(node_ref.weight().op, Op::Output { .. }) {
                    Some(node_ref.id())
                } else {
                    None
                }
            })
            .collect();
        let input_variables: HashSet<_> = inputs
            .iter()
            .map(|&node_index| graph.ops[node_index].op.input_variable().unwrap())
            .collect();
        let output_variables: HashSet<_> = outputs
            .iter()
            .map(|&node_index| graph.ops[node_index].op.output_variable().unwrap())
            .collect();

        // count up the number of times each node is used as an argument
        let mut node_states = vec![OpNodeRunState::default(); graph.ops.node_bound()];
        for node_index in graph
            .clusters
            .values()
            .flat_map(|cluster| cluster.inputs.iter())
        {
            node_states[node_index.index()].usage_count += 1;
        }

        // copy inputs to node, increment usage when variable is not an output, to preserve the buffer
        for node_index in inputs.iter().copied() {
            let variable = graph.ops[node_index].op.input_variable().unwrap();
            let var_state = &mut variables[variable];
            println!("input {:?}: {}", node_index, var_state.name);
            assert!(var_state.buffer_id.is_some());
            let node_state = &mut node_states[node_index.index()];
            if !output_variables.contains(&variable) {
                node_state.buffer_id = var_state.buffer_id.clone();
                node_state.usage_count += 1;
            } else {
                node_state.buffer_id = var_state.buffer_id.take();
            }
        }

        println!("{:?}", self.buffer_heap.heap_stats());

        // free buffers for variables only used as outputs
        for node_index in outputs.iter().copied() {
            let variable = graph.ops[node_index].op.output_variable().unwrap();
            if !input_variables.contains(&variable) {
                let var_state = &mut variables[variable];
                if let Some(buffer_id) = var_state.buffer_id.take() {
                    self.buffer_heap.free(buffer_id);
                }
            }
        }

        // run kernels in order, lazily allocate/free buffers
        for cluster_id in graph.clusters_sorted.iter().copied() {
            let cluster = &graph.clusters[cluster_id];
            println!("cluster {:?} {:?}", cluster_id, cluster.kernel);
            for node_index in cluster.outputs.iter().copied() {
                println!("allocate {:?}", node_index);
                let node_state = &mut node_states[node_index.index()];
                assert!(node_state.buffer_id.is_none());
                node_state.buffer_id = Some(
                    self.buffer_heap
                        .alloc(graph.ops[node_index].shape.buffer_size())
                        .unwrap(),
                );
            }
            // TODO: run kernel
            for node_index in cluster.inputs.iter().copied() {
                let node_state = &mut node_states[node_index.index()];
                node_state.usage_count -= 1;
                if node_state.usage_count == 0 {
                    // TODO: move literals into cluster, remove this check
                    if !matches!(graph.ops[node_index].op, Op::Literal(_)) {
                        println!("free {:?}", node_index);
                        self.buffer_heap.free(node_state.buffer_id.take().unwrap());
                    }
                }
            }
        }

        // assign buffers to outputs
        for node_index in outputs.iter().copied() {
            let variable = graph.ops[node_index].op.output_variable().unwrap();
            let var_state = &mut variables[variable];
            let [edge_index] = get_arg_edges(&graph.ops, node_index);
            let source_node_index = graph.ops.edge_endpoints(edge_index).unwrap().0;
            println!("output {:?}: {}", source_node_index, var_state.name);
            let source_node_state = &mut node_states[source_node_index.index()];
            assert!(source_node_state.buffer_id.is_some());
            var_state.buffer_id = source_node_state.buffer_id.take();
        }

        println!("{:?}", self.buffer_heap.heap_stats());
    }

    pub fn test(&mut self) {
        for _ in 0..4 {
            let cmd = self.command_buffers.acquire(&self.fences);
            cmd.submit(&mut self.fences);
        }
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe { device.device_wait_idle() }.unwrap();
    }
}
