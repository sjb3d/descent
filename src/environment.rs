use crate::{common::*, device::common::*};
use petgraph::visit::{IntoNodeReferences, NodeIndexable, NodeRef};
use rand::{distributions::Open01, Rng};
use slotmap::SlotMap;
use spark::{vk, Builder};
use std::{
    cell::RefCell, collections::HashSet, f32::consts::PI, ffi::CString, io, io::Write, rc::Rc,
    slice,
};

fn normal_from_uniform(u1: f32, u2: f32) -> f32 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn write_rand_normal(mut writer: impl Write, scale: f32, element_count: usize, rng: &mut impl Rng) {
    for _ in 0..element_count {
        let u1: f32 = rng.sample(Open01);
        let u2: f32 = rng.sample(Open01);
        let n: f32 = scale * normal_from_uniform(u1, u2);
        writer.write_all(bytemuck::bytes_of(&n)).unwrap();
    }
}

pub struct VariableWriter<'a>(StagingWriter<'a>);

impl<'a> VariableWriter<'a> {
    pub fn zero_fill(self) {
        // consume self, will zero to the end on drop
    }
}

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

impl<'a> io::BufRead for VariableReader<'a> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        Ok(self.0.peek().unwrap_or(&[]))
    }

    fn consume(&mut self, amt: usize) {
        self.0.advance(amt);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct OpNodeStorage {
    usage_count: usize,
    buffer_id: Option<BufferId>,
}

pub struct Environment {
    context: SharedContext,
    fences: FenceSet,
    command_buffers: CommandBuffers,
    buffer_heap: BufferHeap,
    staging_buffer: StagingBuffer,
    variables: SharedVariables,
    kernel_cache: KernelCache,
    descriptor_pools: DescriptorPools,
    timestamps: TimestampSets,
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment {
    pub fn new() -> Self {
        let context = Context::new();
        let fences = FenceSet::new(&context);
        let command_buffers = CommandBuffers::new(&context, &fences);
        let buffer_heap = BufferHeap::new(&context);
        let staging_buffer = StagingBuffer::new(&context, &fences);
        let kernel_cache = KernelCache::new(&context);
        let descriptor_pools = DescriptorPools::new(&context, &fences);
        let timestamps = TimestampSets::new(&context, &fences);
        Self {
            context,
            fences,
            command_buffers,
            buffer_heap,
            staging_buffer,
            variables: Rc::new(RefCell::new(SlotMap::with_key())),
            kernel_cache,
            descriptor_pools,
            timestamps,
        }
    }

    fn variable(
        &mut self,
        shape: impl Into<Shape>,
        name: impl Into<String>,
        reset_to: Option<VariableContents>,
    ) -> Variable {
        let shape = shape.into();
        let name = name.into();
        let variable_id = self.variables.borrow_mut().insert(VariableStorage {
            shape,
            name,
            reset_to,
            buffer_id: None,
        });
        Variable::new(variable_id, &self.variables)
    }

    pub fn static_parameter(
        &mut self,
        shape: impl Into<Shape>,
        name: impl Into<String>,
    ) -> Variable {
        self.variable(shape, name, None)
    }

    pub fn trainable_parameter(
        &mut self,
        shape: impl Into<Shape>,
        name: impl Into<String>,
        reset_to: VariableContents,
    ) -> Variable {
        self.variable(shape, name, Some(reset_to))
    }

    pub fn writer(&mut self, variable: &Variable) -> VariableWriter {
        let variable_id = variable.checked_id(&self.variables);
        let mut variables = self.variables.borrow_mut();
        let var = variables.get_mut(variable_id).unwrap();
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

    pub fn reader(&mut self, variable: &Variable) -> VariableReader {
        let variable_id = variable.checked_id(&self.variables);
        let variables = self.variables.borrow();
        let var = variables.get(variable_id).unwrap();
        let buffer_id = var.buffer_id.unwrap();
        VariableReader(StagingReader::new(
            &mut self.staging_buffer,
            &mut self.command_buffers,
            &mut self.fences,
            self.buffer_heap.info(buffer_id),
        ))
    }

    pub fn reset_variable(&mut self, variable: &Variable, rng: &mut impl Rng) {
        let shape = variable.shape();
        let writer = self.writer(variable);
        match variable.reset_to().unwrap() {
            VariableContents::Zero => writer.zero_fill(),
            VariableContents::RandNormal(scale) => {
                write_rand_normal(writer, scale, shape.element_count(), rng)
            }
        }
    }

    pub fn scope(&self) -> Scope {
        Scope::new(SharedVariables::clone(&self.variables))
    }

    pub fn run(&mut self, schedule: &Graph, rand_seed: u32) {
        let mut variables = self.variables.borrow_mut();

        // collect input and output variables
        let inputs: Vec<_> = schedule
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
        let outputs: Vec<_> = schedule
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
            .map(|&node_id| schedule.ops[node_id].op.input_variable_id().unwrap())
            .collect();
        let output_variable_ids: HashSet<_> = outputs
            .iter()
            .map(|&node_id| schedule.ops[node_id].op.output_variable_id().unwrap())
            .collect();

        // count up the number of times each node is used as an argument
        let mut node_storage = vec![OpNodeStorage::default(); schedule.ops.node_bound()];
        for node_id in schedule
            .clusters
            .values()
            .flat_map(|cluster| cluster.inputs.iter())
        {
            node_storage[node_id.index()].usage_count += 1;
        }

        // copy inputs to node, increment usage when variable is not an output, to preserve the buffer
        for node_id in inputs.iter().copied() {
            let variable_id = schedule.ops[node_id].op.input_variable_id().unwrap();
            let var = &mut variables[variable_id];
            assert!(var.buffer_id.is_some());
            let storage = &mut node_storage[node_id.index()];
            if !output_variable_ids.contains(&variable_id) {
                storage.buffer_id = var.buffer_id;
                storage.usage_count += 1;
            } else {
                storage.buffer_id = var.buffer_id.take();
            }
        }

        // free buffers for variables only used as outputs
        for node_id in outputs.iter().copied() {
            let variable_id = schedule.ops[node_id].op.output_variable_id().unwrap();
            if !input_variable_ids.contains(&variable_id) {
                let var = &mut variables[variable_id];
                if let Some(buffer_id) = var.buffer_id.take() {
                    self.buffer_heap.free(buffer_id);
                }
            }
        }

        // run kernels in order, lazily allocate/free buffers
        let instance = &self.context.instance;
        let device = &self.context.device;
        let cmd = self.command_buffers.acquire(&self.fences);
        let descriptor_pool = self.descriptor_pools.acquire(&self.fences);
        let mut timestamps = self.timestamps.acquire(cmd.get(), &self.fences);
        for cluster_id in schedule.clusters_sorted.iter().copied() {
            let cluster = &schedule.clusters[cluster_id];
            for node_id in cluster.outputs.iter().copied() {
                let node_state = &mut node_storage[node_id.index()];
                assert!(node_state.buffer_id.is_none());
                node_state.buffer_id = Some(
                    self.buffer_heap
                        .alloc(schedule.ops[node_id].shape.buffer_size())
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
                let label_name = cluster.kernel.label_name();
                timestamps.write_timestamp(cmd.get(), &label_name);
                if instance.extensions.ext_debug_utils {
                    let label_name = CString::new(label_name).unwrap();
                    let label = vk::DebugUtilsLabelEXT {
                        p_label_name: label_name.as_bytes_with_nul().as_ptr() as *const i8,
                        ..Default::default()
                    };
                    instance.cmd_begin_debug_utils_label_ext(cmd.get(), &label);
                }
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
                device.cmd_push_constants(
                    cmd.get(),
                    module.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    slice::from_ref(&rand_seed),
                );
                device.cmd_dispatch(cmd.get(), module.group_count as u32, 1, 1);
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
                    );
                    if instance.extensions.ext_debug_utils {
                        instance.cmd_end_debug_utils_label_ext(cmd.get());
                    }
                }
            }
            for node_id in cluster.inputs.iter().copied() {
                let node_state = &mut node_storage[node_id.index()];
                node_state.usage_count -= 1;
                if node_state.usage_count == 0 {
                    self.buffer_heap.free(node_state.buffer_id.take().unwrap());
                }
            }
        }
        timestamps.end(cmd.get());
        let fence_id = cmd.submit(&mut self.fences);
        descriptor_pool.recycle(fence_id);
        timestamps.recycle(fence_id);

        // assign buffers to outputs
        for node_id in outputs.iter().copied() {
            let variable_id = schedule.ops[node_id].op.output_variable_id().unwrap();
            let var = &mut variables[variable_id];
            let arg_sources = get_arg_sources(&schedule.ops, node_id);
            assert_eq!(arg_sources.len(), 1);
            let src0 = &arg_sources[0];
            let source_storage = &mut node_storage[src0.node_id.index()];
            assert!(source_storage.buffer_id.is_some());
            var.buffer_id = source_storage.buffer_id.take();
        }
    }

    pub fn print_timings(&mut self, label: &str) {
        self.timestamps.print_timings(label, &self.fences);
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe { device.device_wait_idle() }.unwrap();
    }
}
