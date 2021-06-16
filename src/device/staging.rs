use super::{common::*, heap::HeapRange};
use spark::vk;
use std::{collections::VecDeque, io, slice};

#[derive(Debug)]
struct StagingCursor {
    next: usize,
    limit: usize,
    range: HeapRange,
}

impl StagingCursor {
    fn new(range: HeapRange, max_size: usize) -> Self {
        let size = range.size().min(max_size);
        Self {
            next: range.begin,
            limit: range.begin + size,
            range,
        }
    }

    fn is_empty(&self) -> bool {
        self.next == self.range.begin
    }

    fn is_full(&self) -> bool {
        self.next == self.limit
    }
}

struct BufferCursor {
    next: usize,
    info: BufferInfo,
}

impl BufferCursor {
    fn new(info: BufferInfo) -> Self {
        Self {
            next: info.range.begin,
            info,
        }
    }

    fn is_empty(&self) -> bool {
        self.next == self.info.range.begin
    }

    fn remaining(&self) -> usize {
        self.info.range.end - self.next
    }

    fn is_full(&self) -> bool {
        self.next == self.info.range.end
    }
}

struct StagingWriter<'a> {
    staging: &'a mut StagingBuffer,
    command_buffers: &'a mut CommandBufferSet,
    fences: &'a mut FenceSet,
    src: Option<StagingCursor>,
    dest: BufferCursor,
}

impl<'a> StagingWriter<'a> {
    fn write_slice(&mut self, mut buf: &[u8]) -> usize {
        let mut written = 0;
        while !buf.is_empty() {
            if let Some(src) = self.src.as_mut() {
                let copy_dst = &mut self.staging.mapping(src);
                let copy_size = copy_dst.len().min(buf.len());
                copy_dst[..copy_size].copy_from_slice(&buf[..copy_size]);

                src.next += copy_size;
                buf = &buf[copy_size..];
                written += copy_size;

                if src.is_full() {
                    self.flush_src();
                }
            } else {
                self.next_src();
                if self.src.is_none() {
                    break;
                }
            }
        }
        written
    }

    fn write_zeros(&mut self, mut count: usize) -> usize {
        let mut written = 0;
        while count > 0 {
            if let Some(src) = self.src.as_mut() {
                let copy_dst = &mut self.staging.mapping(src);
                let copy_size = copy_dst.len().min(count);
                for b in copy_dst.iter_mut().take(copy_size) {
                    *b = 0;
                }

                src.next += copy_size;
                count -= copy_size;
                written += copy_size;

                if src.is_full() {
                    self.flush_src();
                }
            } else {
                self.next_src();
                if self.src.is_none() {
                    break;
                }
            }
        }
        written
    }

    fn next_src(&mut self) {
        if self.src.is_none() {
            let max_size = self.dest.remaining();
            if max_size > 0 {
                let range = self
                    .staging
                    .ranges
                    .pop_front()
                    .unwrap()
                    .take_when_signaled(self.fences);
                self.src = Some(StagingCursor::new(range, max_size));
            }
        }
    }

    fn flush_src(&mut self) {
        if let Some(src) = self.src.take() {
            if src.is_empty() {
                self.staging
                    .ranges
                    .push_front(Fenced::new(src.range, self.fences.old_id()));
            } else {
                let cmd = self.command_buffers.acquire(self.fences);

                if self.dest.is_empty() {
                    let buffer_memory_barrier = vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::empty(),
                        dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: Some(self.dest.info.buffer),
                        offset: self.dest.info.range.begin as vk::DeviceSize,
                        size: self.dest.info.range.size() as vk::DeviceSize,
                        ..Default::default()
                    };
                    unsafe {
                        self.staging.context.device.cmd_pipeline_barrier(
                            cmd.get(),
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &[],
                            slice::from_ref(&buffer_memory_barrier),
                            &[],
                        )
                    }
                }

                let transfer_size = src.next - src.range.begin;
                {
                    let region = vk::BufferCopy {
                        src_offset: src.range.begin as vk::DeviceSize,
                        dst_offset: self.dest.next as vk::DeviceSize,
                        size: transfer_size as vk::DeviceSize,
                    };

                    unsafe {
                        self.staging.context.device.cmd_copy_buffer(
                            cmd.get(),
                            self.staging.buffer,
                            self.dest.info.buffer,
                            slice::from_ref(&region),
                        )
                    };
                }
                self.dest.next += transfer_size;

                if self.dest.is_full() {
                    let buffer_memory_barrier = vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: Some(self.dest.info.buffer),
                        offset: self.dest.info.range.begin as vk::DeviceSize,
                        size: self.dest.info.range.size() as vk::DeviceSize,
                        ..Default::default()
                    };
                    unsafe {
                        self.staging.context.device.cmd_pipeline_barrier(
                            cmd.get(),
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            slice::from_ref(&buffer_memory_barrier),
                            &[],
                        )
                    }
                }

                let fence_id = cmd.submit(self.fences);
                self.staging
                    .ranges
                    .push_back(Fenced::new(src.range, fence_id));
            }
        }
    }
}

impl<'a> io::Write for StagingWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(self.write_slice(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_src();
        Ok(())
    }
}

impl<'a> Drop for StagingWriter<'a> {
    fn drop(&mut self) {
        self.write_zeros(self.dest.remaining());
        self.flush_src();
    }
}

pub(crate) struct StagingBuffer {
    context: SharedContext,
    device_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    mapping: *mut u8,
    ranges: VecDeque<Fenced<HeapRange>>,
}

impl StagingBuffer {
    const REGION_SIZE: usize = 4 * 1024 * 1024;
    const COUNT: usize = 2;

    pub(crate) fn new(context: &SharedContext, fences: &FenceSet) -> Self {
        let device = &context.device;
        let buffer = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: (Self::REGION_SIZE * Self::COUNT) as vk::DeviceSize,
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                ..Default::default()
            };
            unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap()
        };
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let device_memory = {
            let memory_type_index = context
                .get_memory_type_index(
                    mem_req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT
                        | vk::MemoryPropertyFlags::HOST_CACHED,
                )
                .unwrap();
            let memory_allocate_info = vk::MemoryAllocateInfo {
                allocation_size: mem_req.size,
                memory_type_index,
                ..Default::default()
            };
            unsafe { device.allocate_memory(&memory_allocate_info, None) }.unwrap()
        };
        unsafe { device.bind_buffer_memory(buffer, device_memory, 0) }.unwrap();
        let mapping = unsafe {
            context.device.map_memory(
                device_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap();

        let mut ranges = VecDeque::new();
        for i in 0..Self::COUNT {
            ranges.push_back(Fenced::new(
                HeapRange {
                    begin: i * Self::REGION_SIZE,
                    end: (i + 1) * Self::REGION_SIZE,
                },
                fences.old_id(),
            ));
        }

        Self {
            context: SharedContext::clone(context),
            device_memory,
            buffer,
            mapping: mapping as *mut _,
            ranges,
        }
    }

    fn mapping(&mut self, cursor: &StagingCursor) -> &mut [u8] {
        let full =
            unsafe { slice::from_raw_parts_mut(self.mapping, Self::REGION_SIZE * Self::COUNT) };
        &mut full[cursor.next..cursor.limit]
    }

    pub(crate) fn write_buffer(
        &mut self,
        buffer_info: BufferInfo,
        mut writer: impl FnMut(&mut dyn io::Write),
        command_buffers: &mut CommandBufferSet,
        fences: &mut FenceSet,
    ) {
        let mut w = StagingWriter {
            staging: self,
            command_buffers,
            fences,
            src: None,
            dest: BufferCursor::new(buffer_info),
        };
        writer(&mut w);
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe {
            device.destroy_buffer(Some(self.buffer), None);
            device.free_memory(Some(self.device_memory), None);
        }
    }
}
