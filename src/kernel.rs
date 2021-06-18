use crate::{common::*, device::common::*};
use ordered_float::NotNan;
use spark::vk;
use std::{collections::HashMap, fmt, fmt::Write, mem};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelInput {
    pub(crate) view: View,
    pub(crate) shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PerElementKernel {
    pub(crate) shape: Shape,
    pub(crate) inputs: Vec<KernelInput>,
    pub(crate) outputs: Vec<usize>,
    pub(crate) ops: Vec<PerElementKernelOp>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ReduceKernel {
    pub(crate) input_shape: Shape,
    pub(crate) reduce_op: ReduceOp,
    pub(crate) axis: Axis,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MatMulKernel {
    pub(crate) inputs: [KernelInput; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    PerElement(PerElementKernel),
    Reduce(ReduceKernel),
    MatMul(MatMulKernel),
}

impl PerElementKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        let mut binding_index = 0;

        for input_index in 0..self.inputs.len() {
            writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
            writeln!(
                w,
                "readonly restrict buffer input_layout{0} {{ float input{0}[]; }};",
                input_index
            )?;
            binding_index += 1;
        }
        for output_index in 0..self.outputs.len() {
            writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
            writeln!(
                w,
                "writeonly restrict buffer output_layout{0} {{ float output{0}[]; }};",
                output_index
            )?;
            binding_index += 1;
        }

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(w, "uint coord[{}];", self.shape.len())?;
        write!(w, "if (!compute_grid_coord(coord")?;
        for &n in self.shape.iter() {
            write!(w, ", {}", n)?;
        }
        writeln!(w, ")) {{ return; }}")?;

        for (op_index, op) in self.ops.iter().enumerate() {
            write!(w, "float tmp{} = ", op_index)?;
            match op {
                PerElementKernelOp::Load { input_index } => {
                    let input = &self.inputs[*input_index];
                    if input.view.is_identity() {
                        write!(w, "input{}[gl_GlobalInvocationID.x]", input_index)?
                    } else {
                        let params = FlatIndexParams::new(&input.shape, &input.view);
                        write!(w, "input{}[{}", input_index, params.offset)?;
                        for (index, scale) in params.scale.iter().copied().enumerate() {
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

impl MatMulKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{ }}")?;

        Ok(src)
    }
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

impl Kernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        match self {
            Kernel::PerElement(kernel) => kernel.generate_source(),
            Kernel::MatMul(kernel) => kernel.generate_source(),
            Kernel::Reduce(kernel) => kernel.generate_source(),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KernelModule {
    pub(crate) shader_module: vk::ShaderModule,
    pub(crate) descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pub(crate) pipeline_layout: Option<vk::PipelineLayout>,
    pub(crate) pipeline: Option<vk::Pipeline>,
}

pub(crate) struct KernelCache {
    context: SharedContext,
    modules: HashMap<Kernel, KernelModule>,
}

impl KernelCache {
    pub(crate) fn new(context: &SharedContext) -> Self {
        Self {
            context: SharedContext::clone(context),
            modules: HashMap::new(),
        }
    }

    pub(crate) fn module(&mut self, kernel: &Kernel) -> KernelModule {
        *self.modules.entry(kernel.clone()).or_insert_with({
            let device = &self.context.device;
            move || {
                let source = kernel.generate_source().unwrap();

                let mut compiler = shaderc::Compiler::new().unwrap();
                let shader_module = match compiler.compile_into_spirv(
                    &source,
                    shaderc::ShaderKind::Compute,
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
                        unsafe { device.create_shader_module(&shader_module_create_info, None) }
                            .unwrap()
                    }
                    Err(err) => {
                        panic!("failed to compile shader {}", err);
                    }
                };

                KernelModule {
                    shader_module,
                    descriptor_set_layout: None,
                    pipeline_layout: None,
                    pipeline: None,
                }
            }
        })
    }
}
