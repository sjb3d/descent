use super::prelude::*;
use arrayvec::ArrayVec;
use spark::{vk, Device};
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
}

pub struct CommandBufferPool {
    context: SharedContext,
    sets: [CommandBufferSet; Self::COUNT],
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
        }
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
