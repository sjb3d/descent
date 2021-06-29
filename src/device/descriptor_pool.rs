use super::common::*;
use spark::{vk, Builder};
use std::collections::VecDeque;

pub(crate) struct DescriptorPoolSet {
    context: SharedContext,
    pools: VecDeque<Fenced<vk::DescriptorPool>>,
}

impl DescriptorPoolSet {
    const COUNT: usize = 2;

    const MAX_SETS: u32 = 512;
    const MAX_BUFFERS: u32 = 8 * Self::MAX_SETS;

    pub(crate) fn new(context: &SharedContext, fences: &FenceSet) -> Self {
        let device = &context.device;
        let descriptor_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: Self::MAX_BUFFERS,
        }];
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(Self::MAX_SETS)
            .p_pool_sizes(&descriptor_pool_sizes);

        let mut pools = VecDeque::new();
        for _ in 0..Self::COUNT {
            let pool = unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
                .unwrap();
            pools.push_back(Fenced::new(pool, fences.old_id()));
        }
        Self {
            context: SharedContext::clone(context),
            pools,
        }
    }

    pub(crate) fn acquire(&mut self, fences: &FenceSet) -> ScopedDescriptorPool {
        let pool = self.pools.pop_front().unwrap().take_when_signaled(fences);
        unsafe {
            self.context
                .device
                .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                .unwrap();
        }
        ScopedDescriptorPool { pool, set: self }
    }
}

impl Drop for DescriptorPoolSet {
    fn drop(&mut self) {
        let device = &self.context.device;
        for pool in self.pools.iter() {
            unsafe {
                let pool = pool.get_unchecked();
                device.destroy_descriptor_pool(Some(*pool), None);
            }
        }
    }
}

pub(crate) struct ScopedDescriptorPool<'a> {
    pool: vk::DescriptorPool,
    set: &'a mut DescriptorPoolSet,
}

impl<'a> ScopedDescriptorPool<'a> {
    pub(crate) fn get(&self) -> vk::DescriptorPool {
        self.pool
    }

    pub(crate) fn recycle(self, fence: FenceId) {
        self.set.pools.push_back(Fenced::new(self.pool, fence));
    }
}
