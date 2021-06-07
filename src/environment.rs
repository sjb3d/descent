use crate::device::prelude::*;

pub struct Environment {
    context: SharedContext,
    command_buffer_pool: CommandBufferPool,
}

impl Environment {
    pub fn new() -> Self {
        let context = Context::new();
        let command_buffer_pool = CommandBufferPool::new(&context);
        Self {
            context,
            command_buffer_pool,
        }
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let device = &self.context.device;
        unsafe { device.device_wait_idle() }.unwrap();
    }
}
