use super::common::*;
use spark::vk;
use std::{collections::VecDeque, slice};

#[derive(Debug, Clone, Copy)]
struct StagingCursor {
    next: usize,
    end: usize,
    region: StagingBufferRegion,
}

impl StagingCursor {
    fn new(region: StagingBufferRegion, max_size: usize) -> Self {
        let begin = region.begin();
        let size = StagingBuffer::REGION_SIZE.min(max_size);
        Self {
            next: begin,
            end: begin + size,
            region,
        }
    }

    fn is_empty(&self) -> bool {
        self.next == self.region.begin()
    }

    fn is_full(&self) -> bool {
        self.next == self.end
    }

    fn remaining(&self) -> usize {
        self.end - self.next
    }
}

#[derive(Debug, Clone, Copy)]
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

    fn remaining(&self) -> usize {
        self.info.range.end - self.next
    }

    fn is_finished(&self) -> bool {
        self.next == self.info.range.end
    }
}

pub(crate) struct StagingWriter<'a> {
    owner: &'a mut StagingBuffer,
    command_buffers: &'a mut CommandBuffers,
    fences: &'a mut FenceSet,
    staging: Option<StagingCursor>,
    buffer: BufferCursor,
}

impl<'a> StagingWriter<'a> {
    pub(crate) fn new(
        owner: &'a mut StagingBuffer,
        command_buffers: &'a mut CommandBuffers,
        fences: &'a mut FenceSet,
        buffer_info: BufferInfo,
    ) -> Self {
        let mut writer = Self {
            owner,
            command_buffers,
            fences,
            staging: None,
            buffer: BufferCursor::new(buffer_info),
        };
        writer.next_staging();
        writer
    }

    pub(crate) fn write_slice(&mut self, mut buf: &[u8]) -> usize {
        let mut counter = 0;
        while let Some(staging) = self.staging.as_mut() {
            let copy_buf = self.owner.mapping(staging);
            let copy_size = copy_buf.len().min(buf.len());
            copy_buf[..copy_size].copy_from_slice(&buf[..copy_size]);

            staging.next += copy_size;
            buf = &buf[copy_size..];
            counter += copy_size;

            if staging.is_full() {
                self.flush_staging();
            }
            if buf.is_empty() {
                break;
            }
        }
        counter
    }

    pub(crate) fn write_zeros(&mut self, mut count: usize) -> usize {
        let mut counter = 0;
        while let Some(staging) = self.staging.as_mut() {
            let copy_buf = self.owner.mapping(staging);
            let copy_size = copy_buf.len().min(count);
            for b in copy_buf.iter_mut().take(copy_size) {
                *b = 0;
            }

            staging.next += copy_size;
            count -= copy_size;
            counter += copy_size;

            if staging.is_full() {
                self.flush_staging();
            }
            if count == 0 {
                break;
            }
        }
        counter
    }

    fn next_staging(&mut self) {
        assert!(self.staging.is_none());
        let max_size = self.buffer.remaining();
        if max_size > 0 {
            let range = self
                .owner
                .regions
                .pop_front()
                .unwrap()
                .take_when_signaled(self.fences);
            self.staging = Some(StagingCursor::new(range, max_size));
        }
    }

    pub(crate) fn flush_staging(&mut self) {
        if let Some(staging) = self.staging.take() {
            if staging.is_empty() {
                self.staging = Some(staging);
            } else {
                let cmd = self.command_buffers.acquire(self.fences);

                let staging_begin = staging.region.begin();
                let transfer_size = staging.next - staging_begin;
                {
                    let region = vk::BufferCopy {
                        src_offset: staging_begin as vk::DeviceSize,
                        dst_offset: self.buffer.next as vk::DeviceSize,
                        size: transfer_size as vk::DeviceSize,
                    };

                    unsafe {
                        self.owner.context.device.cmd_copy_buffer(
                            cmd.get(),
                            self.owner.buffer,
                            self.buffer.info.buffer,
                            slice::from_ref(&region),
                        )
                    };
                }
                self.buffer.next += transfer_size;

                let fence_id = cmd.submit(self.fences);
                self.owner
                    .regions
                    .push_back(Fenced::new(staging.region, fence_id));

                self.next_staging();
            }
        }
    }
}

impl<'a> Drop for StagingWriter<'a> {
    fn drop(&mut self) {
        self.write_zeros(self.buffer.remaining());
        self.flush_staging();
        assert!(self.staging.is_none());
    }
}

pub(crate) struct StagingReader<'a> {
    owner: &'a mut StagingBuffer,
    command_buffers: &'a mut CommandBuffers,
    fences: &'a mut FenceSet,
    buffer: BufferCursor,
    pending: VecDeque<Fenced<StagingCursor>>,
    staging: Option<StagingCursor>,
}

impl<'a> StagingReader<'a> {
    pub(crate) fn new(
        owner: &'a mut StagingBuffer,
        command_buffers: &'a mut CommandBuffers,
        fences: &'a mut FenceSet,
        buffer_info: BufferInfo,
    ) -> Self {
        let mut reader = Self {
            owner,
            command_buffers,
            fences,
            buffer: BufferCursor::new(buffer_info),
            pending: VecDeque::new(),
            staging: None,
        };
        while !reader.buffer.is_finished() {
            if let Some(region) = reader.owner.regions.pop_front() {
                let region = region.take_when_signaled(reader.fences);
                reader.add_pending(region);
            } else {
                break;
            }
        }
        reader.next_staging();
        reader
    }

    pub(crate) fn peek(&mut self) -> Option<&[u8]> {
        let owner = &mut self.owner;
        self.staging
            .as_ref()
            .map(move |staging| owner.mapping(staging) as &_)
    }

    pub(crate) fn advance(&mut self, amt: usize) {
        let staging = self.staging.as_mut().unwrap();
        assert!(amt <= staging.remaining());
        staging.next += amt;
        if staging.is_full() {
            let region = staging.region;
            self.staging = None;
            self.add_pending(region);
            self.next_staging();
        }
    }

    pub(crate) fn read_slice(&mut self, buf: &mut [u8]) -> usize {
        let limit = buf.len();
        let mut offset = 0;
        while let Some(copy_buf) = self.peek() {
            let copy_size = copy_buf.len().min(limit - offset);
            let next = offset + copy_size;
            buf[offset..next].copy_from_slice(&copy_buf[..copy_size]);

            offset = next;
            self.advance(copy_size);

            if offset == limit {
                break;
            }
        }
        offset
    }

    fn add_pending(&mut self, region: StagingBufferRegion) {
        if self.buffer.is_finished() {
            self.owner
                .regions
                .push_back(Fenced::new(region, self.fences.old_id()));
        } else {
            let cmd = self.command_buffers.acquire(self.fences);

            let staging = StagingCursor::new(region, self.buffer.remaining());
            let staging_begin = staging.region.begin();
            let transfer_size = staging.end - staging_begin;
            {
                let region = vk::BufferCopy {
                    src_offset: self.buffer.next as vk::DeviceSize,
                    dst_offset: staging_begin as vk::DeviceSize,
                    size: transfer_size as vk::DeviceSize,
                };

                unsafe {
                    self.owner.context.device.cmd_copy_buffer(
                        cmd.get(),
                        self.buffer.info.buffer,
                        self.owner.buffer,
                        slice::from_ref(&region),
                    )
                };
            }
            self.buffer.next += transfer_size;

            let fence_id = cmd.submit(self.fences);
            self.pending.push_back(Fenced::new(staging, fence_id));
        }
    }

    fn next_staging(&mut self) {
        assert!(self.staging.is_none());
        if let Some(pending) = self.pending.pop_front() {
            self.staging = Some(pending.take_when_signaled(self.fences));
        }
    }
}

impl<'a> Drop for StagingReader<'a> {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            self.owner
                .regions
                .push_back(Fenced::new(staging.region, self.fences.old_id()));
        }
        while let Some(pending) = self.pending.pop_front() {
            self.owner
                .regions
                .push_back(pending.map(|pending| pending.region));
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StagingBufferRegion(u8);

impl StagingBufferRegion {
    fn begin(&self) -> usize {
        (self.0 as usize) * StagingBuffer::REGION_SIZE
    }
}

pub(crate) struct StagingBuffer {
    context: SharedContext,
    device_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    mapping: *mut u8,
    regions: VecDeque<Fenced<StagingBufferRegion>>,
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
            ranges.push_back(Fenced::new(StagingBufferRegion(i as u8), fences.old_id()));
        }

        Self {
            context: SharedContext::clone(context),
            device_memory,
            buffer,
            mapping: mapping as *mut _,
            regions: ranges,
        }
    }

    fn mapping(&mut self, cursor: &StagingCursor) -> &mut [u8] {
        let full =
            unsafe { slice::from_raw_parts_mut(self.mapping, Self::REGION_SIZE * Self::COUNT) };
        &mut full[cursor.next..cursor.end]
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
