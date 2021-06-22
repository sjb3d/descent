use crate::{common::*, device::common::*};
use ordered_float::NotNan;
use shaderc::{Compiler, ShaderKind};
use spark::{vk, Builder};
use std::{collections::HashMap, ffi::CStr, fmt, fmt::Write, mem, slice};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum PerElementKernelOp {
    Load {
        input_index: usize,
    },
    Literal(NotNan<f32>),
    Unary {
        op: UnaryOp,
        arg0_index: usize,
    },
    Binary {
        op: BinaryOp,
        arg0_index: usize,
        arg1_index: usize,
    },
}

fn generate_input_buffer(
    binding_index: usize,
    input_index: usize,
    w: &mut impl Write,
) -> fmt::Result {
    writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
    writeln!(
        w,
        "readonly restrict buffer input_layout{0} {{ float input{0}[]; }};",
        input_index
    )?;
    Ok(())
}

fn generate_output_buffer(
    binding_index: usize,
    output_index: usize,
    w: &mut impl Write,
) -> fmt::Result {
    writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
    writeln!(
        w,
        "writeonly restrict buffer output_layout{0} {{ float output{0}[]; }};",
        output_index
    )?;
    Ok(())
}

fn generate_coord(shape: &Shape, w: &mut impl Write) -> fmt::Result {
    writeln!(w, "uint coord[{}];", shape.len())?;
    write!(w, "if (!compute_grid_coord(coord")?;
    for &n in shape.iter() {
        write!(w, ", {}", n)?;
    }
    writeln!(w, ")) {{ return; }}")?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelInput {
    pub(crate) shape: Shape,
    pub(crate) view: View,
}

impl KernelInput {
    fn indexer(&self) -> ViewIndexer {
        ViewIndexer::new(&self.shape, &self.view)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PerElementKernel {
    pub(crate) shape: Shape,
    pub(crate) inputs: Vec<KernelInput>,
    pub(crate) outputs: Vec<usize>,
    pub(crate) ops: Vec<PerElementKernelOp>,
}

impl PerElementKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        let mut binding_index = 0;
        for input_index in 0..self.inputs.len() {
            generate_input_buffer(binding_index, input_index, w)?;
            binding_index += 1;
        }
        for output_index in 0..self.outputs.len() {
            generate_output_buffer(binding_index, output_index, w)?;
            binding_index += 1;
        }

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        generate_coord(&self.shape, w)?;

        for (op_index, op) in self.ops.iter().enumerate() {
            write!(w, "float tmp{} = ", op_index)?;
            match op {
                PerElementKernelOp::Load { input_index } => {
                    let input = &self.inputs[*input_index];
                    if input.view.is_identity() {
                        write!(w, "input{}[gl_GlobalInvocationID.x]", input_index)?
                    } else {
                        let indexer = input.indexer();
                        write!(w, "input{}[{}", input_index, indexer.offset)?;
                        for (index, scale) in indexer.scales.iter().copied().enumerate() {
                            if scale != 0 {
                                write!(w, " + {}*coord[{}]", scale, index)?;
                            }
                        }
                        write!(w, "]")?;
                    }
                }
                PerElementKernelOp::Literal(value) => write!(w, "{:#?}", value.into_inner())?,
                PerElementKernelOp::Unary { op, arg0_index } => match op {
                    UnaryOp::Neg => write!(w, "-tmp{}", arg0_index)?,
                    UnaryOp::Exp => write!(w, "exp(tmp{})", arg0_index)?,
                    UnaryOp::Log => write!(w, "log(tmp{})", arg0_index)?,
                    UnaryOp::OneHot => write!(
                        w,
                        "(coord[{}] == uint(tmp{})) ? 1.0 : 0.0",
                        self.shape.len() - 1,
                        arg0_index
                    )?,
                },
                PerElementKernelOp::Binary {
                    op,
                    arg0_index,
                    arg1_index,
                } => match op {
                    BinaryOp::Add => write!(w, "tmp{} + tmp{}", arg0_index, arg1_index)?,
                    BinaryOp::Sub => write!(w, "tmp{} - tmp{}", arg0_index, arg1_index)?,
                    BinaryOp::Mul => write!(w, "tmp{} * tmp{}", arg0_index, arg1_index)?,
                    BinaryOp::Div => write!(w, "tmp{} / tmp{}", arg0_index, arg1_index)?,
                },
            }
            writeln!(w, ";")?;
        }

        for (output_index, src_index) in self.outputs.iter().enumerate() {
            writeln!(
                w,
                "output{}[gl_GlobalInvocationID.x] = tmp{};",
                output_index, src_index
            )?;
        }

        writeln!(w, "}}")?;

        Ok(src)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MatMulKernel {
    pub(crate) shape: Shape,
    pub(crate) k: isize,
    pub(crate) inputs: [KernelInput; 2],
}

impl MatMulKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        let mut binding_index = 0;
        for input_index in 0..2 {
            generate_input_buffer(binding_index, input_index, w)?;
            binding_index += 1;
        }
        for output_index in 0..1 {
            generate_output_buffer(binding_index, output_index, w)?;
            binding_index += 1;
        }

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        generate_coord(&self.shape, w)?;

        let indexer0 = self.inputs[0].indexer();
        let (stride0, scales0) = indexer0.scales.split_last().unwrap();
        write!(w, "uint base0 = {}", indexer0.offset)?;
        for (index, scale) in scales0.iter().copied().enumerate() {
            if scale != 0 {
                write!(w, " + {}*coord[{}]", scale, index)?;
            }
        }
        writeln!(w, ";")?;
        writeln!(w, "uint stride0 = {};", stride0)?;

        let indexer1 = self.inputs[1].indexer();
        let (stride1, scales1) = indexer1.scales.split_first().unwrap();
        assert_eq!(indexer1.scales.len(), 2);
        writeln!(
            w,
            "uint base1 = {} + {}*coord[{}];",
            indexer1.offset,
            scales1[0],
            scales0.len()
        )?;
        writeln!(w, "uint stride1 = {};", stride1)?;

        writeln!(w, "float sum = 0.f;")?;
        writeln!(w, "for (uint k = 0; k < {}; ++k) {{", self.k)?;
        writeln!(w, "float tmp0 = input0[base0 + k*stride0]")?;
        writeln!(w, "float tmp1 = input1[base1 + k*stride1]")?;
        writeln!(w, "sum += tmp0 * tmp1;")?;
        writeln!(w, "}}")?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = sum;",)?;

        writeln!(w, "}}")?;

        Ok(src)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ReduceKernel {
    pub(crate) input_shape: Shape,
    pub(crate) reduce_op: ReduceOp,
    pub(crate) axis: Axis,
}

impl ReduceKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{ }}")?;

        Ok(src)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    PerElement(PerElementKernel),
    Reduce(ReduceKernel),
    MatMul(MatMulKernel),
}

impl Kernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        match self {
            Kernel::PerElement(kernel) => kernel.generate_source(),
            Kernel::MatMul(kernel) => kernel.generate_source(),
            Kernel::Reduce(kernel) => kernel.generate_source(),
        }
    }

    fn buffer_count(&self) -> usize {
        match self {
            Kernel::PerElement(kernel) => kernel.inputs.len() + kernel.outputs.len(),
            Kernel::MatMul(..) => 3,
            Kernel::Reduce(..) => 2,
        }
    }

    fn group_count(&self) -> u32 {
        let shape = match self {
            Kernel::PerElement(kernel) => &kernel.shape,
            Kernel::MatMul(kernel) => &kernel.shape,
            Kernel::Reduce(kernel) => &kernel.input_shape, // HACK!
        };
        ((shape.element_count() as u32) + 63) / 64
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KernelModule {
    pub(crate) shader_module: vk::ShaderModule,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) group_count: u32,
}

struct KernelCacheWorker {
    context: SharedContext,
    compiler: Compiler,
}

impl KernelCacheWorker {
    fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            compiler: Compiler::new().unwrap(),
        }
    }

    fn create_module(&mut self, kernel: &Kernel) -> KernelModule {
        let device = &self.context.device;

        let source = kernel.generate_source().unwrap();
        println!("{}", source);

        let shader_module = match self.compiler.compile_into_spirv(
            &source,
            ShaderKind::Compute,
            "kernel",
            "main",
            None,
        ) {
            Ok(artifact) => {
                if artifact.get_num_warnings() != 0 {
                    println!("{}", artifact.get_warning_messages());
                }
                let words = artifact.as_binary();
                let shader_module_create_info = vk::ShaderModuleCreateInfo {
                    code_size: words.len() * mem::size_of::<u32>(),
                    p_code: words.as_ptr(),
                    ..Default::default()
                };
                unsafe { device.create_shader_module(&shader_module_create_info, None) }.unwrap()
            }
            Err(err) => {
                panic!("failed to compile shader {}", err);
            }
        };

        let descriptor_set_layout = {
            let mut bindings = Vec::new();
            for _ in 0..kernel.buffer_count() {
                let i = bindings.len();
                bindings.push(vk::DescriptorSetLayoutBinding {
                    binding: i as u32,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                });
            }
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().p_bindings(&bindings);
            unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap()
        };

        let pipeline_layout = {
            let create_info = vk::PipelineLayoutCreateInfo::builder()
                .p_set_layouts(slice::from_ref(&descriptor_set_layout));
            unsafe { device.create_pipeline_layout(&create_info, None) }.unwrap()
        };

        let pipeline = {
            let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
            let pipeline_create_info = vk::ComputePipelineCreateInfo {
                stage: vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::COMPUTE,
                    module: Some(shader_module),
                    p_name: shader_entry_name.as_ptr(),
                    ..Default::default()
                },
                layout: Some(pipeline_layout),
                ..Default::default()
            };
            unsafe { device.create_compute_pipelines_single(None, &pipeline_create_info, None) }
                .unwrap()
        };

        KernelModule {
            shader_module,
            descriptor_set_layout,
            pipeline_layout,
            pipeline,
            group_count: kernel.group_count(),
        }
    }
}

pub(crate) struct KernelCache {
    worker: KernelCacheWorker,
    modules: HashMap<Kernel, KernelModule>,
}

impl KernelCache {
    pub(crate) fn new(context: &SharedContext) -> Self {
        Self {
            worker: KernelCacheWorker::new(context),
            modules: HashMap::new(),
        }
    }

    pub(crate) fn module(&mut self, kernel: &Kernel) -> KernelModule {
        *self.modules.entry(kernel.clone()).or_insert_with({
            let worker = &mut self.worker;
            move || worker.create_module(kernel)
        })
    }
}

impl Drop for KernelCache {
    fn drop(&mut self) {
        let device = &self.worker.context.device;
        for (_, module) in self.modules.drain() {
            unsafe {
                device.destroy_pipeline(Some(module.pipeline), None);
                device.destroy_pipeline_layout(Some(module.pipeline_layout), None);
                device.destroy_descriptor_set_layout(Some(module.descriptor_set_layout), None);
                device.destroy_shader_module(Some(module.shader_module), None);
            }
        }
    }
}
