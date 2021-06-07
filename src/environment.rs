use crate::device::prelude::*;

pub struct Environment {
    context: SharedContext,
    command_buffer_pool: CommandBufferPool,
    buffer_heap: BufferHeap,
}

impl Environment {
    pub fn new() -> Self {
        let context = Context::new();
        let command_buffer_pool = CommandBufferPool::new(&context);
        let buffer_heap = BufferHeap::new(&context);
        Self {
            context,
            command_buffer_pool,
            buffer_heap,
        }
    }

    pub fn test(&mut self) {
        let _cmd = self.command_buffer_pool.acquire();
        self.command_buffer_pool.submit();

        let a = self.buffer_heap.alloc(64 * 1024 * 1024).unwrap();
        let b = self.buffer_heap.alloc(64 * 1024 * 1024).unwrap();
        let c = self.buffer_heap.alloc(64 * 1024 * 1024).unwrap();
        let d = self.buffer_heap.alloc(64 * 1024 * 1024).unwrap();
        let e = self.buffer_heap.alloc(64 * 1024 * 1024).unwrap();

        self.buffer_heap.free(a);
        self.buffer_heap.free(b);
        self.buffer_heap.free(c);
        self.buffer_heap.free(d);
        self.buffer_heap.free(e);
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe { device.device_wait_idle() }.unwrap();
    }
}
