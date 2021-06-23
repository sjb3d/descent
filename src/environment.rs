use crate::{common::*, device::common::*};
use petgraph::visit::{IntoNodeReferences, NodeIndexable, NodeRef};
use slotmap::SlotMap;
use spark::{vk, Builder};
use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    io,
    rc::Rc,
    slice,
};

slotmap::new_key_type! {
    pub struct VariableId;
}

pub(crate) type SharedVariables = Rc<RefCell<SlotMap<VariableId, Variable>>>;

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

pub(crate) struct Variable {
    pub(crate) shape: Shape,
    pub(crate) name: String,
    pub(crate) buffer_id: Option<BufferId>,
}

#[derive(Debug, Clone, Copy, Default)]
struct OpNodeStorage {
    usage_count: usize,
    buffer_id: Option<BufferId>,
}

struct DescriptorPoolSet {
    context: SharedContext,
    pools: VecDeque<Fenced<vk::DescriptorPool>>,
}

impl DescriptorPoolSet {
    const COUNT: usize = 2;

    const MAX_SETS: u32 = 512;
    const MAX_BUFFERS: u32 = 8 * Self::MAX_SETS;

    fn new(context: &SharedContext, fences: &FenceSet) -> Self {
        let device = &context.device;
        let descriptor_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: Self::MAX_BUFFERS,
        }];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(Self::MAX_SETS)
            .p_pool_sizes(&descriptor_pool_sizes);

        let mut pools = VecDeque::new();
        for _ in 0..Self::COUNT {
            let pool = unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
                .unwrap();
            pools.push_back(Fenced::new(pool, fences.old_id()));
        }
        Self {
            context: SharedContext::clone(context),
            pools,
        }
    }

    fn acquire(&mut self, fences: &FenceSet) -> ScopedDescriptorPool {
        let pool = self.pools.pop_front().unwrap().take_when_signaled(fences);
        ScopedDescriptorPool { pool, set: self }
    }
}

impl Drop for DescriptorPoolSet {
    fn drop(&mut self) {
        let device = &self.context.device;
        for pool in self.pools.iter() {
            unsafe {
                let pool = pool.get_unchecked();
                device.destroy_descriptor_pool(Some(*pool), None);
            }
        }
    }
}

struct ScopedDescriptorPool<'a> {
    pool: vk::DescriptorPool,
    set: &'a mut DescriptorPoolSet,
}

impl<'a> ScopedDescriptorPool<'a> {
    fn get(&self) -> vk::DescriptorPool {
        self.pool
    }

    fn recycle(self, fence: FenceId) {
        self.set.pools.push_back(Fenced::new(self.pool, fence));
    }
}

pub struct Environment {
    context: SharedContext,
    fences: FenceSet,
    command_buffers: CommandBufferSet,
    buffer_heap: BufferHeap,
    staging_buffer: StagingBuffer,
    variables: SharedVariables,
    kernel_cache: KernelCache,
    descriptor_pools: DescriptorPoolSet,
}

impl Environment {
    pub fn new() -> Self {
        let context = Context::new();
        let fences = FenceSet::new(&context);
        let command_buffers = CommandBufferSet::new(&context, &fences);
        let buffer_heap = BufferHeap::new(&context);
        let staging_buffer = StagingBuffer::new(&context, &fences);
        let kernel_cache = KernelCache::new(&context);
        let descriptor_pools = DescriptorPoolSet::new(&context, &fences);
        Self {
            context,
            fences,
            command_buffers,
            buffer_heap,
            staging_buffer,
            variables: Rc::new(RefCell::new(SlotMap::with_key())),
            kernel_cache,
            descriptor_pools,
        }
    }

    pub fn variable(&mut self, shape: impl Into<Shape>, name: impl Into<String>) -> VariableId {
        let shape = shape.into();
        let name = name.into();
        let variable_id = self.variables.borrow_mut().insert(Variable {
            shape: shape.clone(),
            name,
            buffer_id: None,
        });
        variable_id
    }

    pub fn writer(&mut self, variable_id: VariableId) -> VariableWriter {
        let mut variables = self.variables.borrow_mut();
        let var = &mut variables[variable_id];
        if let Some(buffer_id) = var.buffer_id.take() {
            self.buffer_heap.free(buffer_id);
        }
        let buffer_id = self.buffer_heap.alloc(var.shape.buffer_size()).unwrap();
        var.buffer_id = Some(buffer_id);
        VariableWriter(StagingWriter::new(
            &mut self.staging_buffer,
            &mut self.command_buffers,
            &mut self.fences,
            self.buffer_heap.info(buffer_id),
        ))
    }

    pub fn reader(&mut self, variable_id: VariableId) -> VariableReader {
        let variables = self.variables.borrow();
        let var = &variables[variable_id];
        let buffer_id = var.buffer_id.unwrap();
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
        let input_variable_ids: HashSet<_> = inputs
            .iter()
            .map(|&node_id| graph.ops[node_id].op.input_variable_id().unwrap())
            .collect();
        let output_variable_ids: HashSet<_> = outputs
            .iter()
            .map(|&node_id| graph.ops[node_id].op.output_variable_id().unwrap())
            .collect();

        // count up the number of times each node is used as an argument
        let mut node_storage = vec![OpNodeStorage::default(); graph.ops.node_bound()];
        for node_id in graph
            .clusters
            .values()
            .flat_map(|cluster| cluster.inputs.iter())
        {
            node_storage[node_id.index()].usage_count += 1;
        }

        // copy inputs to node, increment usage when variable is not an output, to preserve the buffer
        for node_id in inputs.iter().copied() {
            let variable_id = graph.ops[node_id].op.input_variable_id().unwrap();
            let var = &mut variables[variable_id];
            println!("input {:?}: {}", node_id, var.name);
            assert!(var.buffer_id.is_some());
            let storage = &mut node_storage[node_id.index()];
            if !output_variable_ids.contains(&variable_id) {
                storage.buffer_id = var.buffer_id.clone();
                storage.usage_count += 1;
            } else {
                storage.buffer_id = var.buffer_id.take();
            }
        }

        println!("{:?}", self.buffer_heap.heap_stats());

        // free buffers for variables only used as outputs
        for node_id in outputs.iter().copied() {
            let variable_id = graph.ops[node_id].op.output_variable_id().unwrap();
            if !input_variable_ids.contains(&variable_id) {
                let var = &mut variables[variable_id];
                if let Some(buffer_id) = var.buffer_id.take() {
                    self.buffer_heap.free(buffer_id);
                }
            }
        }

        // run kernels in order, lazily allocate/free buffers
        let device = &self.context.device;
        let cmd = self.command_buffers.acquire(&self.fences);
        let descriptor_pool = self.descriptor_pools.acquire(&self.fences);
        for cluster_id in graph.clusters_sorted.iter().copied() {
            let cluster = &graph.clusters[cluster_id];
            println!("cluster {:?} {:?}", cluster_id, cluster.kernel);
            for node_id in cluster.outputs.iter().copied() {
                println!("allocate {:?}", node_id);
                let node_state = &mut node_storage[node_id.index()];
                assert!(node_state.buffer_id.is_none());
                node_state.buffer_id = Some(
                    self.buffer_heap
                        .alloc(graph.ops[node_id].shape.buffer_size())
                        .unwrap(),
                );
            }

            let module = self.kernel_cache.module(&cluster.kernel);

            let descriptor_set = {
                let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool.get())
                    .p_set_layouts(slice::from_ref(&module.descriptor_set_layout));
                unsafe { device.allocate_descriptor_sets_single(&descriptor_set_allocate_info) }
                    .unwrap()
            };

            {
                let mut buffer_info = Vec::new();
                for node_id in cluster
                    .inputs
                    .iter()
                    .copied()
                    .chain(cluster.outputs.iter().copied())
                {
                    let node_state = &node_storage[node_id.index()];
                    let info = self.buffer_heap.info(node_state.buffer_id.unwrap());
                    buffer_info.push(vk::DescriptorBufferInfo {
                        buffer: Some(info.buffer),
                        offset: info.range.begin as vk::DeviceSize,
                        range: info.range.size() as vk::DeviceSize,
                    });
                }
                let mut writes = Vec::new();
                for (i, info) in buffer_info.iter().enumerate() {
                    writes.push(vk::WriteDescriptorSet {
                        dst_set: Some(descriptor_set),
                        dst_binding: i as u32,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: info,
                        ..Default::default()
                    });
                }
                unsafe { self.context.device.update_descriptor_sets(&writes, &[]) };
            }

            unsafe {
                device.cmd_bind_pipeline(
                    cmd.get(),
                    vk::PipelineBindPoint::COMPUTE,
                    module.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd.get(),
                    vk::PipelineBindPoint::COMPUTE,
                    module.pipeline_layout,
                    0,
                    slice::from_ref(&descriptor_set),
                    &[],
                );
                device.cmd_dispatch(cmd.get(), module.group_count, 1, 1);
            }

            {
                // ensure that compute results are visible for next kernel
                let memory_barrier = vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                };
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd.get(),
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        slice::from_ref(&memory_barrier),
                        &[],
                        &[],
                    )
                }
            }
            for node_id in cluster.inputs.iter().copied() {
                let node_state = &mut node_storage[node_id.index()];
                node_state.usage_count -= 1;
                if node_state.usage_count == 0 {
                    println!("free {:?}", node_id);
                    self.buffer_heap.free(node_state.buffer_id.take().unwrap());
                }
            }
        }
        let fence_id = cmd.submit(&mut self.fences);
        descriptor_pool.recycle(fence_id);

        // assign buffers to outputs
        for node_id in outputs.iter().copied() {
            let variable_id = graph.ops[node_id].op.output_variable_id().unwrap();
            let var = &mut variables[variable_id];
            let [edge_id] = get_arg_edge_ids(&graph.ops, node_id);
            let source_node_id = graph.ops.edge_endpoints(edge_id).unwrap().0;
            println!("output {:?}: {}", source_node_id, var.name);
            let source_storage = &mut node_storage[source_node_id.index()];
            assert!(source_storage.buffer_id.is_some());
            var.buffer_id = source_storage.buffer_id.take();
        }

        println!("{:?}", self.buffer_heap.heap_stats());
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe { device.device_wait_idle() }.unwrap();
    }
}
