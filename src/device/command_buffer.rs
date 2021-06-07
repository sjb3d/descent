use super::prelude::*;
use arrayvec::ArrayVec;
use spark::{vk, Builder};
use std::slice;

#[derive(Debug)]
struct CommandBufferSet {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
}

impl CommandBufferSet {
    fn new(context: &SharedContext) -> Self {
        let device = &context.device;

        let pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo {
                queue_family_index: context.queue_family_index,
                ..Default::default()
            };
            unsafe { device.create_command_pool(&command_pool_create_info, None) }.unwrap()
        };

        let cmd = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: Some(pool),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            unsafe { device.allocate_command_buffers_single(&command_buffer_allocate_info) }
                .unwrap()
        };

        let fence = {
            let fence_create_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };
            unsafe { device.create_fence(&fence_create_info, None) }.unwrap()
        };

        Self { pool, cmd, fence }
    }

    fn acquire(&self, context: &Context) -> vk::CommandBuffer {
        self.wait_for_fence(context);

        let device = &context.device;
        unsafe {
            device.reset_fences(slice::from_ref(&self.fence)).unwrap();
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            device
                .begin_command_buffer(self.cmd, &command_buffer_begin_info)
                .unwrap();
        }
        self.cmd
    }

    fn submit(&self, context: &Context) {
        let device = &context.device;

        unsafe { device.end_command_buffer(self.cmd) }.unwrap();

        let submit_info = vk::SubmitInfo::builder().p_command_buffers(slice::from_ref(&self.cmd));
        unsafe {
            device
                .queue_submit(
                    context.queue,
                    slice::from_ref(&submit_info),
                    Some(self.fence),
                )
                .unwrap();
        }
    }

    fn wait_for_fence(&self, context: &Context) {
        let timeout_ns = 1000 * 1000 * 1000;
        loop {
            let res = unsafe {
                context
                    .device
                    .wait_for_fences(slice::from_ref(&self.fence), true, timeout_ns)
            };
            match res {
                Ok(_) => break,
                Err(vk::Result::TIMEOUT) => {}
                Err(err_code) => panic!("failed to wait for fence {}", err_code),
            }
        }
    }
}

pub struct CommandBufferPool {
    context: SharedContext,
    sets: [CommandBufferSet; Self::COUNT],
    index: usize,
}

impl CommandBufferPool {
    const COUNT: usize = 4;

    pub fn new(context: &SharedContext) -> Self {
        let mut sets = ArrayVec::new();
        for _ in 0..Self::COUNT {
            sets.push(CommandBufferSet::new(context));
        }
        Self {
            context: SharedContext::clone(context),
            sets: sets.into_inner().unwrap(),
            index: 0,
        }
    }

    pub fn acquire(&mut self) -> vk::CommandBuffer {
        self.index = (self.index + 1) % Self::COUNT;
        self.sets[self.index].acquire(&self.context)
    }

    pub fn submit(&self) {
        self.sets[self.index].submit(&self.context);
    }
}

impl Drop for CommandBufferPool {
    fn drop(&mut self) {
        let device = &self.context.device;
        for set in self.sets.iter() {
            unsafe {
                device.destroy_fence(Some(set.fence), None);
                device.free_command_buffers(set.pool, slice::from_ref(&set.cmd));
                device.destroy_command_pool(Some(set.pool), None);
            }
        }
    }
}
