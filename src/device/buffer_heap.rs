use std::ops::Deref;

use crate::device::buffer_heap;
use spark::vk;

use super::{heap::*, prelude::*};
use slotmap::SlotMap;

slotmap::new_key_type! {
    struct HeapId;
    pub struct BufferId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkIndex(usize);

impl Tag for ChunkIndex {}

struct Chunk {
    device_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
}

struct Buffer {
    alloc: HeapAlloc<HeapId, ChunkIndex>,
}

pub struct BufferHeap {
    context: SharedContext,
    chunks: Vec<Chunk>,
    heap: Heap<HeapId, ChunkIndex>,
    buffers: SlotMap<BufferId, Buffer>,
}

impl BufferHeap {
    pub fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            chunks: Vec::new(),
            heap: Heap::default(),
            buffers: SlotMap::with_key(),
        }
    }

    fn extend_heap_by(&mut self, min_size: usize) {
        let chunk_size = (self
            .context
            .physical_device_properties
            .limits
            .max_storage_buffer_range as usize)
            .min(256 * 1024 * 1024)
            .max(min_size);
        let device = &self.context.device;
        let buffer = {
            let buffer_create_info = vk::BufferCreateInfo {
                size: chunk_size as vk::DeviceSize,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                ..Default::default()
            };
            unsafe { device.create_buffer(&buffer_create_info, None) }.unwrap()
        };
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let device_memory = {
            let memory_type_index = self
                .context
                .get_memory_type_index(
                    mem_req.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
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

        let chunk_index = ChunkIndex(self.chunks.len());
        self.chunks.push(Chunk {
            device_memory,
            buffer,
        });

        self.heap.extend_with(chunk_index, chunk_size);
    }

    fn alloc_from_heap(
        &mut self,
        size: usize,
        align: usize,
    ) -> Option<HeapAlloc<HeapId, ChunkIndex>> {
        match self.heap.alloc(size, align) {
            Some(alloc) => Some(alloc),
            None => {
                self.extend_heap_by(size);
                self.heap.alloc(size, align)
            }
        }
    }

    pub fn alloc(&mut self, size: usize) -> Option<BufferId> {
        let align = self
            .context
            .physical_device_properties
            .limits
            .non_coherent_atom_size as usize;
        let alloc = self.alloc_from_heap(size, align)?;
        Some(self.buffers.insert(Buffer { alloc }))
    }

    pub fn free(&mut self, id: BufferId) {
        if let Some(buffer) = self.buffers.remove(id) {
            self.heap.free(buffer.alloc.id);
        } else {
            panic!("tried to free invalid buffer id");
        }
    }
}

impl Drop for BufferHeap {
    fn drop(&mut self) {
        let device = &self.context.device;
        for chunk in self.chunks.drain(..) {
            unsafe {
                device.destroy_buffer(Some(chunk.buffer), None);
                device.free_memory(Some(chunk.device_memory), None);
            }
        }
    }
}
