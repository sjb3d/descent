use super::common::*;
use arrayvec::ArrayVec;
use spark::vk;
use std::slice;

#[derive(Clone, Copy)]
pub(crate) struct FenceId(usize);

impl FenceId {
    fn index(&self) -> usize {
        self.0 % FenceSet::COUNT
    }
}

pub(crate) struct FenceSet {
    context: SharedContext,
    fences: [vk::Fence; Self::COUNT],
    counter: usize,
}

impl FenceSet {
    const COUNT: usize = 2;

    pub(crate) fn new(context: &SharedContext) -> Self {
        let mut fences = ArrayVec::new();
        for _ in 0..Self::COUNT {
            let fence = {
                let fence_create_info = vk::FenceCreateInfo {
                    flags: vk::FenceCreateFlags::SIGNALED,
                    ..Default::default()
                };
                unsafe { context.device.create_fence(&fence_create_info, None) }.unwrap()
            };
            fences.push(fence);
        }
        Self {
            context: SharedContext::clone(context),
            fences: fences.into_inner().unwrap(),
            counter: 1,
        }
    }

    pub(crate) fn old_id(&self) -> FenceId {
        FenceId(self.counter.wrapping_sub(1))
    }

    fn id_needs_wait(&self, id: FenceId) -> bool {
        id.0.wrapping_sub(self.counter) < Self::COUNT
    }

    pub(crate) fn next_unsignaled(&mut self) -> (FenceId, vk::Fence) {
        self.wait_for_signal(FenceId(self.counter));

        let id = FenceId(self.counter.wrapping_add(Self::COUNT));
        self.counter = self.counter.wrapping_add(1);

        let fence = self.fences[id.index()];
        unsafe {
            self.context
                .device
                .reset_fences(slice::from_ref(&fence))
                .unwrap();
        }
        (id, fence)
    }

    pub(crate) fn wait_for_signal(&self, id: FenceId) {
        if !self.id_needs_wait(id) {
            return;
        }

        let fence = self.fences[id.index()];
        let timeout_ns = 1000 * 1000 * 1000;
        loop {
            let res = unsafe {
                self.context
                    .device
                    .wait_for_fences(slice::from_ref(&fence), true, timeout_ns)
            };
            match res {
                Ok(_) => break,
                Err(vk::Result::TIMEOUT) => {}
                Err(err_code) => panic!("failed to wait for fence {}", err_code),
            }
        }
    }
}

impl Drop for FenceSet {
    fn drop(&mut self) {
        for fence in self.fences.iter().copied() {
            unsafe {
                self.context.device.destroy_fence(Some(fence), None);
            }
        }
    }
}

pub(crate) struct Fenced<T> {
    value: T,
    fence_id: FenceId,
}

impl<T> Fenced<T> {
    pub(crate) fn new(value: T, fence_id: FenceId) -> Self {
        Self { value, fence_id }
    }

    pub(crate) fn take_when_signaled(self, set: &FenceSet) -> T {
        set.wait_for_signal(self.fence_id);
        self.value
    }

    pub(crate) fn map<U, F>(self, f: F) -> Fenced<U>
    where
        F: FnOnce(T) -> U,
    {
        Fenced {
            value: f(self.value),
            fence_id: self.fence_id,
        }
    }

    pub(crate) unsafe fn get_unchecked(&self) -> &T {
        &self.value
    }
}
