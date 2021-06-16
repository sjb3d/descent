use crate::{device::common::*, prelude::*};
use slotmap::SlotMap;
use std::{cell::RefCell, io, mem, rc::Rc};

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
        self.variables
            .as_ref()
            .borrow()
            .get(self.id)
            .unwrap()
            .name
            .clone()
    }

    pub(crate) fn shape(&self) -> Shape {
        self.variables
            .as_ref()
            .borrow()
            .get(self.id)
            .unwrap()
            .shape
            .clone()
    }
}

pub struct VariableWriter<'a>(StagingWriter<'a>);

impl<'a> io::Write for VariableWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(self.0.write_slice(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush_staging();
        Ok(())
    }
}

pub struct VariableReader<'a>(StagingReader<'a>);

impl<'a> io::Read for VariableReader<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        Ok(self.0.read_slice(buf))
    }
}

struct VariableData {
    shape: Shape,
    name: String,
    buffer_id: Option<BufferId>,
}

pub struct Environment {
    context: SharedContext,
    fences: FenceSet,
    command_buffers: CommandBufferSet,
    buffer_heap: BufferHeap,
    staging_buffer: StagingBuffer,
    variables: SharedVariables,
}

impl Environment {
    pub fn new() -> Self {
        let context = Context::new();
        let fences = FenceSet::new(&context);
        let command_buffers = CommandBufferSet::new(&context, &fences);
        let buffer_heap = BufferHeap::new(&context);
        let staging_buffer = StagingBuffer::new(&context, &fences);
        Self {
            context,
            fences,
            command_buffers,
            buffer_heap,
            staging_buffer,
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

    pub fn writer(&mut self, var: &Variable) -> VariableWriter {
        let mut variables = self.variables.borrow_mut();
        let variable_data = &mut variables[var.id];
        if let Some(buffer_id) = variable_data.buffer_id.take() {
            self.buffer_heap.free(buffer_id);
        }
        let buffer_id = self
            .buffer_heap
            .alloc(variable_data.shape.dim_product() * mem::size_of::<f32>())
            .unwrap();
        variable_data.buffer_id = Some(buffer_id);
        VariableWriter(StagingWriter::new(
            &mut self.staging_buffer,
            &mut self.command_buffers,
            &mut self.fences,
            self.buffer_heap.info(buffer_id),
        ))
    }

    pub fn reader(&mut self, var: &Variable) -> VariableReader {
        let variables = self.variables.borrow();
        let variable_data = &variables[var.id];
        let buffer_id = variable_data.buffer_id.unwrap();
        VariableReader(StagingReader::new(
            &mut self.staging_buffer,
            &mut self.command_buffers,
            &mut self.fences,
            self.buffer_heap.info(buffer_id),
        ))
    }

    pub fn test(&mut self) {
        for _ in 0..4 {
            let cmd = self.command_buffers.acquire(&self.fences);
            cmd.submit(&mut self.fences);
        }

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
