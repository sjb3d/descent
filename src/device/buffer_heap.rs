use super::{heap::*, prelude::*};
use spark::vk;

slotmap::new_key_type! {
    pub struct BufferId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChunkIndex(usize);

struct Chunk {
    device_memory: vk::DeviceMemory,
    buffer: vk::Buffer,
}

pub struct BufferHeap {
    context: SharedContext,
    chunks: Vec<Chunk>,
    heap: Heap<BufferId, ChunkIndex>,
}

impl BufferHeap {
    const CHUNK_SIZE: usize = 256 * 1024 * 1024;

    pub fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            chunks: Vec::new(),
            heap: Heap::default(),
        }
    }

    fn extend_heap_by_at_least(&mut self, capacity: usize) {
        let chunk_size = Self::CHUNK_SIZE.max(capacity);
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

    pub fn alloc(&mut self, size: usize) -> Option<BufferId> {
        let align = self
            .context
            .physical_device_properties
            .limits
            .non_coherent_atom_size as usize;
        match self.heap.alloc(size, align) {
            Some(alloc) => Some(alloc),
            None => {
                self.extend_heap_by_at_least(size);
                self.heap.alloc(size, align)
            }
        }
    }

    pub fn free(&mut self, id: BufferId) {
        self.heap.free(id);
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
