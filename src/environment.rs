use crate::{device::common::*, prelude::*};
use slotmap::SlotMap;
use std::{cell::RefCell, rc::Rc};

slotmap::new_key_type! {
    pub struct VariableId;
}

type SharedVariables = Rc<RefCell<SlotMap<VariableId, VariableData>>>;

pub struct Variable {
    pub(crate) id: VariableId,
    variables: SharedVariables,
}

impl Variable {
    pub(crate) fn name(&self) -> String {
        self.variables.borrow().get(self.id).unwrap().name.clone()
    }

    pub(crate) fn shape(&self) -> Shape {
        self.variables.borrow().get(self.id).unwrap().shape.clone()
    }
}

struct VariableData {
    shape: Shape,
    name: String,
    buffer_id: Option<BufferId>,
}

pub struct Environment {
    context: SharedContext,
    command_buffer_pool: CommandBufferPool,
    buffer_heap: BufferHeap,
    variables: SharedVariables,
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
            variables: Rc::new(RefCell::new(SlotMap::with_key())),
        }
    }

    pub fn variable(&mut self, shape: impl Into<Shape>, name: impl Into<String>) -> Variable {
        let shape = shape.into();
        let name = name.into();
        let id = self.variables.borrow_mut().insert(VariableData {
            shape: shape.clone(),
            name,
            buffer_id: None,
        });
        Variable {
            id,
            variables: SharedVariables::clone(&self.variables),
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
