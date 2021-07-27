use crate::{common::*, device::common::*};
use ordered_float::NotNan;
use shaderc::{Compiler, ShaderKind};
use spark::{vk, Builder};
use std::{collections::HashMap, convert::TryInto, ffi::CStr, fmt, fmt::Write, mem, slice};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PerElementKernelOp {
    Load {
        input_index: usize,
    },
    Literal(NotNan<f32>),
    BuiltIn {
        op: BuiltInOp,
        view: View,
    },
    Unary {
        op: UnaryOp,
        args: usize,
    },
    Binary {
        op: BinaryOp,
        args: [usize; 2],
    },
    CompareAndSelect {
        compare_mode: CompareMode,
        args: [usize; 4],
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

fn generate_coord(name: &str, shape: &Shape, w: &mut impl Write) -> fmt::Result {
    writeln!(w, "int {}[{}];", name, shape.len())?;
    write!(w, "compute_grid_coord(gl_GlobalInvocationID.x, {}", name)?;
    for &n in shape.iter() {
        write!(w, ", {}", n)?;
    }
    writeln!(w, ");")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PerElementKernel {
    pub(crate) element_count: usize,
    pub(crate) inputs: Vec<View>,
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

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.element_count
        )?;

        let mut coord_sets = HashMap::new();
        for (op_index, op) in self.ops.iter().enumerate() {
            match op {
                PerElementKernelOp::Load { input_index } => {
                    let input = &self.inputs[*input_index];
                    let coord_shape = &input.output_shape;
                    let next_coord_index = coord_sets.len();
                    let (coord_name, coord_indexer) =
                        coord_sets.entry(coord_shape).or_insert_with(|| {
                            let coord_indexer = coord_shape.identity_view().indexer();
                            let coord_name = format!("coord{}", next_coord_index);
                            generate_coord(&coord_name, coord_shape, w).unwrap();
                            (coord_name, coord_indexer)
                        });
                    let input_indexer = input.indexer();
                    write!(w, "float tmp{} = input{}[", op_index, input_index)?;
                    if &input_indexer == coord_indexer {
                        write!(w, "gl_GlobalInvocationID.x")?
                    } else {
                        write!(w, "{}", input_indexer.offset)?;
                        for (index, scale) in input_indexer.scales.iter().copied().enumerate() {
                            write!(w, " + ({})*{}[{}]", scale, coord_name, index)?;
                        }
                    }
                    writeln!(w, "];")?;
                }
                PerElementKernelOp::Literal(value) => {
                    writeln!(w, "float tmp{} = {:#?};", op_index, value.into_inner())?
                }
                PerElementKernelOp::BuiltIn { op, view } => {
                    let coord_shape = &view.output_shape;
                    let next_coord_index = coord_sets.len();
                    let (coord_name, coord_indexer) =
                        coord_sets.entry(coord_shape).or_insert_with(|| {
                            let coord_indexer = coord_shape.identity_view().indexer();
                            let coord_name = format!("coord{}", next_coord_index);
                            generate_coord(&coord_name, coord_shape, w).unwrap();
                            (coord_name, coord_indexer)
                        });
                    let input_indexer = view.indexer();
                    match op {
                        BuiltInOp::Coord { axis } => {
                            write!(w, "float tmp{} = float(", op_index)?;
                            if &input_indexer == coord_indexer {
                                write!(w, "{}[{}]", coord_name, axis.index())?;
                            } else {
                                write!(w, "{}", view.input_offsets[axis.index()])?;
                                for (index, mapping) in
                                    view.output_mapping.iter().copied().enumerate()
                                {
                                    match mapping {
                                        AxisMapping::Source {
                                            axis: source_axis,
                                            step,
                                        } if &source_axis == axis => {
                                            write!(w, " + ({})*{}[{}]", step, coord_name, index)?;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            writeln!(w, ");")?;
                        }
                        BuiltInOp::Rand { uid } => {
                            write!(w, "float tmp{} = rand_from_index({}, ", op_index, uid)?;
                            if &input_indexer == coord_indexer {
                                write!(w, "int(gl_GlobalInvocationID.x)")?
                            } else {
                                write!(w, "{}", input_indexer.offset)?;
                                for (index, scale) in
                                    input_indexer.scales.iter().copied().enumerate()
                                {
                                    write!(w, " + ({})*{}[{}]", scale, coord_name, index)?;
                                }
                            }
                            writeln!(w, ");")?;
                        }
                    }
                }
                PerElementKernelOp::Unary { op, args } => {
                    write!(w, "float tmp{} = ", op_index)?;
                    match op {
                        UnaryOp::Mov => write!(w, "tmp{}", args)?,
                        UnaryOp::Neg => write!(w, "-tmp{}", args)?,
                        UnaryOp::Sqrt => write!(w, "sqrt(tmp{})", args)?,
                        UnaryOp::Exp => write!(w, "exp(tmp{})", args)?,
                        UnaryOp::Log => write!(w, "log(tmp{})", args)?,
                    }
                    writeln!(w, ";")?;
                }
                PerElementKernelOp::Binary { op, args } => {
                    write!(w, "float tmp{} = ", op_index)?;
                    match op {
                        BinaryOp::Add => write!(w, "tmp{} + tmp{}", args[0], args[1])?,
                        BinaryOp::Sub => write!(w, "tmp{} - tmp{}", args[0], args[1])?,
                        BinaryOp::Mul => write!(w, "tmp{} * tmp{}", args[0], args[1])?,
                        BinaryOp::Div => write!(w, "tmp{} / tmp{}", args[0], args[1])?,
                    }
                    writeln!(w, ";")?;
                }
                PerElementKernelOp::CompareAndSelect { compare_mode, args } => {
                    write!(w, "float tmp{} = ", op_index)?;
                    match compare_mode {
                        CompareMode::Eq => write!(
                            w,
                            "(tmp{} == tmp{}) ? tmp{} : tmp{}",
                            args[0], args[1], args[2], args[3]
                        )?,
                        CompareMode::Gt => write!(
                            w,
                            "(tmp{} > tmp{}) ? tmp{} : tmp{}",
                            args[0], args[1], args[2], args[3]
                        )?,
                    }
                    writeln!(w, ";")?;
                }
            }
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
    pub(crate) inputs: [View; 2],
}

impl MatMulKernel {
    const TILE_M: usize = 16;
    const TILE_N: usize = 16;
    const TILE_K: usize = 16;
    const GROUP_SIZE: usize = 64;

    fn k(&self) -> usize {
        self.inputs[0].output_shape[1]
    }

    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        generate_input_buffer(0, 0, w)?;
        generate_input_buffer(1, 1, w)?;
        generate_output_buffer(2, 0, w)?;

        assert_eq!(self.inputs[0].output_shape.len(), 2);
        assert_eq!(self.inputs[1].output_shape.len(), 2);
        let indexer0 = self.inputs[0].indexer();
        let indexer1 = self.inputs[1].indexer();

        let [r, m, n]: [usize; 3] = self.shape.as_slice().try_into().unwrap();
        let k = self.k();
        let k_chunk_size_in_tiles = k.div_round_up(Self::TILE_K).div_round_up(r);

        writeln!(
            w,
            "\
            float load_a(uvec2 coord) {{
                float tmp = 0.f;
                if (coord.x < {} && coord.y < {}) {{
                    tmp = input0[{} + ({})*coord.x + ({})*coord.y];
                }}
                return tmp;
            }}",
            k, m, indexer0.offset, indexer0.scales[1], indexer0.scales[0],
        )?;
        writeln!(
            w,
            "\
            float load_b(uvec2 coord) {{
                float tmp = 0.f;
                if (coord.x < {} && coord.y < {}) {{
                    tmp = input1[{} + ({})*coord.x + ({})*coord.y];
                }}
                return tmp;
            }}",
            n, k, indexer1.offset, indexer1.scales[1], indexer1.scales[0],
        )?;
        writeln!(
            w,
            "\
            void store_c(uint k_chunk_index, uvec2 coord, float value) {{
                if (coord.x < {} && coord.y < {}) {{
                    output0[k_chunk_index*{} + coord.y*{} + coord.x] = value;
                }}
            }}",
            n,
            m,
            m * n,
            n
        )?;

        writeln!(w, "const uint M = {};", m)?;
        writeln!(w, "const uint N = {};", n)?;
        writeln!(w, "const uint K = {};", k)?;
        writeln!(w, "const uint TILE_M = {};", Self::TILE_M)?;
        writeln!(w, "const uint TILE_N = {};", Self::TILE_N)?;
        writeln!(w, "const uint TILE_K = {};", Self::TILE_K)?;
        writeln!(w, "const uint GROUP_SIZE = {};", Self::GROUP_SIZE)?;
        writeln!(
            w,
            "const uint K_CHUNK_SIZE_IN_TILES = {};",
            k_chunk_size_in_tiles
        )?;

        let load_a_column_major = indexer0.scales[0].abs() < indexer0.scales[1].abs();
        let load_b_column_major = indexer1.scales[0].abs() < indexer1.scales[1].abs();
        let bool_value = |b| if b { "true" } else { "false" };
        writeln!(
            w,
            "const bool LOAD_A_COLUMN_MAJOR = {};",
            bool_value(load_a_column_major)
        )?;
        writeln!(
            w,
            "const bool LOAD_B_COLUMN_MAJOR = {};",
            bool_value(load_b_column_major)
        )?;

        assert_eq!((Self::TILE_M * Self::TILE_N) % Self::GROUP_SIZE, 0);
        write!(w, "{}", include_str!("kernel_matmul.glsl"))?;

        Ok(src)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ReduceKernel {
    pub(crate) shape: Shape,
    pub(crate) input: View,
    pub(crate) reduce_op: ReduceOp,
    pub(crate) axis: Axis,
}

impl ReduceKernel {
    fn k(&self) -> usize {
        self.input.output_shape[self.axis]
    }

    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        generate_input_buffer(0, 0, w)?;
        generate_output_buffer(1, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("coord", &self.shape, w)?;

        let k = self.k();

        let indexer = self.input.indexer();
        write!(w, "int base = {}", indexer.offset)?;
        for (index, scale) in indexer.scales.iter().copied().enumerate() {
            if index != self.axis.index() {
                let offset = if self.axis.index() == 0 { 1 } else { 0 };
                write!(w, " + ({})*coord[{}]", scale, index - offset)?;
            }
        }
        writeln!(w, ";")?;
        writeln!(w, "int stride = {};", indexer.scales[self.axis.index()])?;

        writeln!(
            w,
            "float result = {};",
            match self.reduce_op {
                ReduceOp::Max => "uintBitsToFloat(0xff800000)",
                ReduceOp::Sum => "0.f",
            }
        )?;
        writeln!(w, "for (int k = 0; k < {}; ++k) {{", k)?;
        writeln!(w, "float tmp = input0[base + k*stride];")?;
        writeln!(
            w,
            "{};",
            match self.reduce_op {
                ReduceOp::Max => "result = max(result, tmp)",
                ReduceOp::Sum => "result += tmp",
            }
        )?;
        writeln!(w, "}}")?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = result;")?;

        writeln!(w, "}}")?;

        Ok(src)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ImageToWindowsKernel {
    pub(crate) shape: Shape,
    pub(crate) input: View,
    pub(crate) pad: usize,
    pub(crate) stride: (usize, usize),
}

impl ImageToWindowsKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        generate_input_buffer(0, 0, w)?;
        generate_output_buffer(1, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("coord", &self.shape, w)?;

        let batch_dims = self.shape.len() - 6;
        let group_nc = self.shape[-1];

        writeln!(w, "int out_y = coord[{}];", batch_dims)?;
        writeln!(w, "int out_x = coord[{}];", batch_dims + 1)?;
        writeln!(w, "int group_index = coord[{}];", batch_dims + 2)?;
        writeln!(w, "int filter_y = coord[{}];", batch_dims + 3)?;
        writeln!(w, "int filter_x = coord[{}];", batch_dims + 4)?;
        writeln!(w, "int group_c = coord[{}];", batch_dims + 5)?;

        writeln!(w, "int in_c = group_index*{} + group_c;", group_nc)?;

        writeln!(w, "uint input_w = {};", self.input.output_shape[-2])?;
        writeln!(w, "uint input_h = {};", self.input.output_shape[-3])?;

        let (stride_w, stride_h) = self.stride;
        writeln!(
            w,
            "int in_x = out_x*{} + filter_x - {};",
            stride_w, self.pad
        )?;
        writeln!(
            w,
            "int in_y = out_y*{} + filter_y - {};",
            stride_h, self.pad
        )?;

        writeln!(w, "float tmp = 0.f;")?;
        writeln!(w, "if (uint(in_x) < input_w && uint(in_y) < input_h) {{")?;
        let indexer = self.input.indexer();
        write!(w, "tmp = input0[{}", indexer.offset)?;
        for (index, scale) in indexer.scales.iter().copied().enumerate() {
            if index < batch_dims {
                write!(w, " + ({})*coord[{}]", scale, index)?;
            } else {
                write!(
                    w,
                    " + ({})*{}",
                    scale,
                    ["in_y", "in_x", "in_c"][index - batch_dims]
                )?;
            }
        }
        writeln!(w, "];")?;
        writeln!(w, "}}")?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = tmp;")?;

        writeln!(w, "}}")?;

        Ok(src)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct WindowsToImageKernel {
    pub(crate) shape: Shape,
    pub(crate) input: View,
    pub(crate) pad: usize,
    pub(crate) stride: (usize, usize),
}

impl WindowsToImageKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        write!(w, "{}", include_str!("kernel_common.glsl"))?;

        generate_input_buffer(0, 0, w)?;
        generate_output_buffer(1, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("coord", &self.shape, w)?;

        let (_, suffix) = self.input.output_shape.rsplit_at(6);
        let [out_h, out_w, _groups, filter_h, filter_w, group_nc]: [usize; 6] =
            suffix.try_into().unwrap();
        let (stride_w, stride_h) = self.stride;

        let batch_dims = self.shape.len() - 3;
        writeln!(w, "int in_y = coord[{}];", batch_dims)?;
        writeln!(w, "int in_x = coord[{}];", batch_dims + 1)?;
        writeln!(w, "int in_c = coord[{}];", batch_dims + 2)?;

        writeln!(w, "int group_index = in_c/{};", group_nc)?;
        writeln!(w, "int group_c = in_c - group_index*{};", group_nc)?;

        writeln!(w, "uint out_w = {};", out_w)?;
        writeln!(w, "uint out_h = {};", out_h)?;

        writeln!(w, "uint in_x_padded = uint(in_x) + {};", self.pad)?;
        writeln!(w, "uint in_y_padded = uint(in_y) + {};", self.pad)?;
        writeln!(w, "int filter_base_x = int(in_x_padded % {});", stride_w)?;
        writeln!(w, "int filter_base_y = int(in_y_padded % {});", stride_h)?;
        writeln!(w, "int count_x = {};", filter_w.div_round_up(stride_w))?;
        writeln!(w, "int count_y = {};", filter_h.div_round_up(stride_h))?;
        writeln!(w, "int out_x_base = int(in_x_padded/{});", stride_w)?;
        writeln!(w, "int out_y_base = int(in_y_padded/{});", stride_h)?;

        writeln!(w, "float tmp = 0.f;")?;
        writeln!(w, "for (int index_y = 0; index_y < count_y; ++index_y)",)?;
        writeln!(w, "for (int index_x = 0; index_x < count_x; ++index_x) {{",)?;
        writeln!(w, "int filter_x = filter_base_x + {}*index_x;", stride_w)?;
        writeln!(w, "int filter_y = filter_base_y + {}*index_y;", stride_h)?;
        writeln!(w, "int out_x = out_x_base - index_x;")?;
        writeln!(w, "int out_y = out_y_base - index_y;")?;
        writeln!(
            w,
            "if (filter_x < {} && filter_y < {} && uint(out_x) < out_w && uint(out_y) < out_w) {{",
            filter_w, filter_h
        )?;

        let indexer = self.input.indexer();
        writeln!(w, "tmp += input0[{}", indexer.offset)?;
        for (index, scale) in indexer.scales.iter().copied().enumerate() {
            if index < batch_dims {
                write!(w, " + ({})*coord[{}]", scale, index)?;
            } else {
                write!(
                    w,
                    " + ({})*{}",
                    scale,
                    [
                        "out_y",
                        "out_x",
                        "group_index",
                        "filter_y",
                        "filter_x",
                        "group_c"
                    ][index - batch_dims]
                )?;
            }
        }
        writeln!(w, "];")?;

        writeln!(w, "")?;
        writeln!(w, "}}")?;
        writeln!(w, "}}")?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = tmp;")?;

        writeln!(w, "}}")?;

        Ok(src)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    PerElement(PerElementKernel),
    Reduce(ReduceKernel),
    MatMul(MatMulKernel),
    ImageToWindows(ImageToWindowsKernel),
    WindowsToImage(WindowsToImageKernel),
}

impl Kernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        match self {
            Kernel::PerElement(kernel) => kernel.generate_source(),
            Kernel::MatMul(kernel) => kernel.generate_source(),
            Kernel::Reduce(kernel) => kernel.generate_source(),
            Kernel::ImageToWindows(kernel) => kernel.generate_source(),
            Kernel::WindowsToImage(kernel) => kernel.generate_source(),
        }
    }

    fn buffer_count(&self) -> usize {
        match self {
            Kernel::PerElement(kernel) => kernel.inputs.len() + kernel.outputs.len(),
            Kernel::MatMul(..) => 3,
            Kernel::Reduce(..) => 2,
            Kernel::ImageToWindows(..) => 2,
            Kernel::WindowsToImage(..) => 2,
        }
    }

    fn group_count(&self) -> usize {
        match self {
            Kernel::PerElement(kernel) => kernel.element_count.div_round_up(64),
            Kernel::MatMul(kernel) => {
                let [r, m, n]: [usize; 3] = kernel.shape.as_slice().try_into().unwrap();
                r * m.div_round_up(MatMulKernel::TILE_M) * n.div_round_up(MatMulKernel::TILE_N)
            }
            Kernel::Reduce(kernel) => kernel.shape.element_count().div_round_up(64),
            Kernel::ImageToWindows(kernel) => kernel.shape.element_count().div_round_up(64),
            Kernel::WindowsToImage(kernel) => kernel.shape.element_count().div_round_up(64),
        }
    }

    pub(crate) fn label_name(&self) -> String {
        match self {
            Kernel::PerElement(kernel) => format!(
                "PerElement ({} ops) [{}]",
                kernel.ops.len(),
                kernel.element_count
            ),
            Kernel::MatMul(kernel) => format!("MatMul (k={}) {}", kernel.k(), kernel.shape),
            Kernel::Reduce(kernel) => format!("Reduce (k={}) {}", kernel.k(), kernel.shape),
            Kernel::ImageToWindows(kernel) => format!("ImageToWindows {}", kernel.shape),
            Kernel::WindowsToImage(kernel) => format!("WindowsToImage {}", kernel.shape),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KernelModule {
    pub(crate) shader_module: vk::ShaderModule,
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) group_count: usize,
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
        //println!("{}", source);

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
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: 4,
            };
            let create_info = vk::PipelineLayoutCreateInfo::builder()
                .p_set_layouts(slice::from_ref(&descriptor_set_layout))
                .p_push_constant_ranges(slice::from_ref(&push_constant_range));
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
