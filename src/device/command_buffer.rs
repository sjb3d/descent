use super::common::*;
use spark::{vk, Builder};
use std::{collections::VecDeque, slice};

struct CommandBuffer {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl CommandBuffer {
    fn new(context: &SharedContext) -> Self {
        let pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo {
                queue_family_index: context.queue_family_index,
                ..Default::default()
            };
            unsafe {
                context
                    .device
                    .create_command_pool(&command_pool_create_info, None)
            }
            .unwrap()
        };

        let cmd = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: Some(pool),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            unsafe {
                context
                    .device
                    .allocate_command_buffers_single(&command_buffer_allocate_info)
            }
            .unwrap()
        };

        Self { pool, cmd }
    }
}

pub(crate) struct ScopedCommandBuffer<'a> {
    buffer: CommandBuffer,
    set: &'a mut CommandBufferSet,
}

impl<'a> ScopedCommandBuffer<'a> {
    pub(crate) fn get(&self) -> vk::CommandBuffer {
        self.buffer.cmd
    }

    pub(crate) fn submit(self, fences: &mut FenceSet) -> FenceId {
        self.set.submit(self.buffer, fences)
    }
}

pub(crate) struct CommandBufferSet {
    context: SharedContext,
    buffers: VecDeque<Fenced<CommandBuffer>>,
}

impl CommandBufferSet {
    const COUNT: usize = 2;

    pub(crate) fn new(context: &SharedContext, fences: &FenceSet) -> Self {
        let mut buffers = VecDeque::new();
        for _ in 0..Self::COUNT {
            buffers.push_back(Fenced::new(CommandBuffer::new(context), fences.old_id()));
        }
        Self {
            context: SharedContext::clone(&context),
            buffers,
        }
    }

    pub(crate) fn acquire(&mut self, fences: &FenceSet) -> ScopedCommandBuffer {
        let active = self.buffers.pop_front().unwrap().take_when_signaled(fences);

        unsafe {
            self.context
                .device
                .reset_command_pool(active.pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }

        let command_buffer_begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            self.context
                .device
                .begin_command_buffer(active.cmd, &command_buffer_begin_info)
                .unwrap();
        }

        ScopedCommandBuffer {
            buffer: active,
            set: self,
        }
    }

    fn submit(&mut self, active: CommandBuffer, fences: &mut FenceSet) -> FenceId {
        unsafe { self.context.device.end_command_buffer(active.cmd) }.unwrap();

        let (fence_id, fence) = fences.next_unsignaled();
        let submit_info = vk::SubmitInfo::builder().p_command_buffers(slice::from_ref(&active.cmd));
        unsafe {
            self.context
                .device
                .queue_submit(
                    self.context.queue,
                    slice::from_ref(&submit_info),
                    Some(fence),
                )
                .unwrap();
        }
        self.buffers.push_back(Fenced::new(active, fence_id));

        fence_id
    }
}

impl Drop for CommandBufferSet {
    fn drop(&mut self) {
        for buffer in self.buffers.iter() {
            unsafe {
                let buffer = buffer.get_unchecked();
                self.context
                    .device
                    .free_command_buffers(buffer.pool, slice::from_ref(&buffer.cmd));
                self.context
                    .device
                    .destroy_command_pool(Some(buffer.pool), None);
            }
        }
    }
}
