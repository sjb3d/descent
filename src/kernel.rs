use crate::{common::*, device::common::*};
use shaderc::{Compiler, ShaderKind};
use spark::{vk, Builder};
use std::{collections::HashMap, convert::TryInto, ffi::CStr, fmt, fmt::Write, mem, slice};

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PerElementKernelOp {
    Load {
        input_index: usize,
    },
    Literal(Literal),
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

fn generate_atomic_buffer(
    binding_index: usize,
    output_index: usize,
    w: &mut impl Write,
) -> fmt::Result {
    writeln!(w, "layout(std430, set = 0, binding = {})", binding_index)?;
    writeln!(
        w,
        "restrict buffer output_layout{0} {{ float output{0}[]; }};",
        output_index
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

fn generate_coord(name: &str, shape: Shape, w: &mut impl Write) -> fmt::Result {
    writeln!(w, "int {}[{}];", name, shape.len())?;
    write!(w, "compute_grid_coord(gl_GlobalInvocationID.x, {}", name)?;
    for &n in shape.iter() {
        write!(w, ", {}", n)?;
    }
    writeln!(w, ");")?;
    Ok(())
}

fn generate_load_coord(
    view: &View,
    input_axis: Axis,
    coord_name: &str,
    w: &mut impl Write,
) -> fmt::Result {
    let needs_clamp = view.input_needs_clamp(input_axis);
    let offset = view.input_offsets[input_axis.index()];
    if needs_clamp {
        write!(w, "clamp({}", offset)?;
    } else {
        write!(w, "({}", offset)?;
    }

    for (coord_index, mapping) in view.output_mapping.iter().copied().enumerate() {
        match mapping {
            AxisMapping::Source { axis, step } => {
                if axis == input_axis {
                    write!(w, " + ({})*{}[{}]", step, coord_name, coord_index)?;
                }
            }
            AxisMapping::Broadcast => {}
        }
    }

    if needs_clamp {
        write!(w, ", 0, {})", view.input_shape[input_axis] - 1)?;
    } else {
        write!(w, ")")?;
    }

    Ok(())
}

fn generate_load_index(view: &View, coord_name: &str, w: &mut impl Write) -> fmt::Result {
    let input_strides = view.input_shape.strides();
    for i in 0..view.input_shape.len() {
        if i > 0 {
            write!(w, " + ")?;
        }
        generate_load_coord(view, Axis::from_index(i), coord_name, w)?;
        write!(w, "*({})", input_strides[i])?;
    }
    Ok(())
}

pub(crate) trait Kernel {
    fn generate_source(&self) -> Result<String, fmt::Error>;
    fn buffer_count(&self) -> usize;
    fn group_count(&self) -> usize;
    fn label_name(&self) -> String;
    fn requires_atomic_float(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FillKernel {
    pub(crate) value: Literal,
    pub(crate) element_count: usize,
}

impl Kernel for FillKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_output_buffer(0, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.element_count
        )?;
        match self.value {
            Literal::F32(value) => writeln!(
                w,
                "output0[gl_GlobalInvocationID.x] = {:#?};",
                value.into_inner()
            )?,
            Literal::U32(value) => {
                writeln!(w, "output0[gl_GlobalInvocationID.x] = U2F({});", value)?
            }
        };

        writeln!(w, "}}")?;

        Ok(src)
    }

    fn buffer_count(&self) -> usize {
        1
    }

    fn group_count(&self) -> usize {
        self.element_count.div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!("Fill({:?}) {}", self.value, self.element_count)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PerElementKernel {
    pub(crate) element_count: usize,
    pub(crate) inputs: Vec<View>,
    pub(crate) outputs: Vec<usize>,
    pub(crate) ops: Vec<PerElementKernelOp>,
}

impl Kernel for PerElementKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

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

        let mut coord_set_names = HashMap::new();
        let get_coord_set_name =
            |names: &mut HashMap<Shape, String>, shape: Shape, w: &mut String| {
                let next_index = names.len();
                names
                    .entry(shape)
                    .or_insert_with(|| {
                        let name = format!("coord{}", next_index);
                        generate_coord(&name, shape, w).unwrap();
                        name
                    })
                    .clone()
            };

        for (op_index, op) in self.ops.iter().enumerate() {
            match op {
                PerElementKernelOp::Load { input_index } => {
                    let view = &self.inputs[*input_index];
                    let coord_shape = view.output_shape;
                    let coord_name = get_coord_set_name(&mut coord_set_names, coord_shape, w);

                    write!(w, "float tmp{} = input{}[", op_index, input_index)?;
                    if *view == coord_shape.identity_view() {
                        write!(w, "gl_GlobalInvocationID.x")?
                    } else {
                        generate_load_index(view, &coord_name, w)?;
                    }
                    writeln!(w, "];")?;
                }
                PerElementKernelOp::Literal(value) => match value {
                    Literal::F32(value) => {
                        writeln!(w, "float tmp{} = {:#?};", op_index, value.into_inner())?
                    }
                    Literal::U32(value) => writeln!(w, "float tmp{} = U2F({});", op_index, value)?,
                },
                PerElementKernelOp::BuiltIn { op, view } => {
                    let coord_shape = view.output_shape;
                    let coord_name = get_coord_set_name(&mut coord_set_names, coord_shape, w);
                    match op {
                        BuiltInOp::Coord => {
                            write!(w, "float tmp{} = float(", op_index)?;
                            generate_load_index(view, &coord_name, w)?;
                            writeln!(w, ");")?;
                        }
                        BuiltInOp::Rand { uid } => {
                            write!(w, "float tmp{} = rand_from_index({}, ", op_index, uid)?;
                            if *view == coord_shape.identity_view() {
                                write!(w, "int(gl_GlobalInvocationID.x)")?
                            } else {
                                generate_load_index(view, &coord_name, w)?;
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
                        UnaryOp::Sin => write!(w, "sin(tmp{})", args)?,
                        UnaryOp::Cos => write!(w, "cos(tmp{})", args)?,
                        UnaryOp::UintToFloat => write!(w, "float(F2U(tmp{}))", args)?,
                        UnaryOp::FloatToUint => write!(w, "U2F(uint(tmp{}))", args)?,
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
                        BinaryOp::Pow => write!(w, "pow(tmp{}, tmp{})", args[0], args[1])?,
                        BinaryOp::UAdd => {
                            write!(w, "U2F(F2U(tmp{}) + F2U(tmp{}))", args[0], args[1])?
                        }
                        BinaryOp::UMul => {
                            write!(w, "U2F(F2U(tmp{}) * F2U(tmp{}))", args[0], args[1])?
                        }
                        BinaryOp::URem => {
                            write!(w, "U2F(F2U(tmp{}) % F2U(tmp{}))", args[0], args[1])?
                        }
                        BinaryOp::UBitXor => {
                            write!(w, "U2F(F2U(tmp{}) ^ F2U(tmp{}))", args[0], args[1])?
                        }
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

    fn buffer_count(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }

    fn group_count(&self) -> usize {
        self.element_count.div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!(
            "PerElement ({} ops) [{}]",
            self.ops.len(),
            self.element_count
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MatMulKernel {
    pub(crate) shape: Shape,
    pub(crate) output_mode: MatMulOutputMode,
    pub(crate) a: View,
    pub(crate) b: View,
}

impl MatMulKernel {
    const TILE_M: usize = 16;
    const TILE_N: usize = 16;
    const TILE_K: usize = 16;
    const GROUP_SIZE: usize = 64;

    fn m(&self) -> usize {
        self.a.output_shape[SignedIndex(-2)]
    }
    fn n(&self) -> usize {
        self.b.output_shape[SignedIndex(-1)]
    }
    fn k(&self) -> usize {
        self.a.output_shape[SignedIndex(-1)]
    }

    fn k_chunk_count(&self) -> usize {
        self.shape[0]
    }
    fn batch_count(&self) -> usize {
        self.shape[match self.output_mode {
            MatMulOutputMode::Batches => 1,
            MatMulOutputMode::Rows => 2,
        }]
    }
}

impl Kernel for MatMulKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_input_buffer(0, 0, w)?;
        generate_input_buffer(1, 1, w)?;
        generate_output_buffer(2, 0, w)?;

        assert_eq!(self.a.output_shape.len(), 3);
        assert_eq!(self.b.output_shape.len(), 3);

        let m = self.m();
        let n = self.n();
        let k = self.k();
        let k_chunk_count = self.k_chunk_count();
        let k_chunk_size_in_tiles = k.div_round_up(Self::TILE_K).div_round_up(k_chunk_count);
        let batch_count = self.batch_count();

        for i in 0..2 {
            writeln!(w, "int load_index{}(uint batch_index, uvec2 coord) {{", i)?;
            writeln!(
                w,
                "int icoord[3];
                icoord[0] = int(batch_index);
                icoord[1] = int(coord.y);
                icoord[2] = int(coord.x);"
            )?;
            write!(w, "return ")?;
            generate_load_index(
                match i {
                    0 => &self.a,
                    1 => &self.b,
                    _ => unreachable!(),
                },
                "icoord",
                w,
            )?;
            writeln!(w, "; }}")?;
        }

        writeln!(
            w,
            "\
            float load_a(uint batch_index, uvec2 coord) {{
                float tmp = 0.f;
                if (coord.x < {} && coord.y < {}) {{
                    int icoord[2];
                    icoord[0] = int(coord.y);
                    icoord[1] = int(coord.x);
                    tmp = input0[load_index0(batch_index, coord)];
                }}
                return tmp;
            }}",
            k, m,
        )?;
        writeln!(
            w,
            "\
            float load_b(uint batch_index, uvec2 coord) {{
                float tmp = 0.f;
                if (coord.x < {} && coord.y < {}) {{
                    tmp = input1[load_index1(batch_index, coord)];
                }}
                return tmp;
            }}",
            n, k,
        )?;

        let (batch_stride, row_stride) = match self.output_mode {
            MatMulOutputMode::Batches => (m * n, n),
            MatMulOutputMode::Rows => (n, batch_count * n),
        };
        writeln!(
            w,
            "\
            void store_c(uint k_chunk_index, uint batch_index, uvec2 coord, float value) {{
                if (coord.x < {} && coord.y < {}) {{
                    output0[k_chunk_index*{} + batch_index*{} + coord.y*{} + coord.x] = value;
                }}
            }}",
            n,
            m,
            batch_count * m * n,
            batch_stride,
            row_stride
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
        writeln!(w, "const uint K_CHUNK_COUNT = {};", k_chunk_count)?;
        writeln!(w, "const uint BATCH_COUNT = {};", batch_count)?;

        let load_a_in_columns = self.a.load_in_columns_hint();
        let load_b_in_columns = self.b.load_in_columns_hint();
        let bool_value = |b| if b { "true" } else { "false" };
        writeln!(
            w,
            "const bool LOAD_A_IN_COLUMNS = {};",
            bool_value(load_a_in_columns)
        )?;
        writeln!(
            w,
            "const bool LOAD_B_IN_COLUMNS = {};",
            bool_value(load_b_in_columns)
        )?;

        assert_eq!((Self::TILE_M * Self::TILE_N) % Self::GROUP_SIZE, 0);
        write!(w, "{}", include_str!("kernel_matmul.glsl"))?;

        Ok(src)
    }

    fn buffer_count(&self) -> usize {
        3
    }

    fn group_count(&self) -> usize {
        self.batch_count()
            * self.k_chunk_count()
            * self.m().div_round_up(MatMulKernel::TILE_M)
            * self.n().div_round_up(MatMulKernel::TILE_N)
    }

    fn label_name(&self) -> String {
        format!("MatMul (k={}) {}", self.k(), self.shape)
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
}

impl Kernel for ReduceKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_input_buffer(0, 0, w)?;
        generate_output_buffer(1, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("out_coord", self.shape, w)?;

        let k = self.k();

        writeln!(w, "int in_coord[{}];", self.input.output_shape.len())?;

        for index in 0..self.input.output_shape.len() {
            if index != self.axis.index() {
                writeln!(w, "in_coord[{0}] = out_coord[{0}];", index)?;
            }
        }

        writeln!(
            w,
            "float result = {};",
            match self.reduce_op {
                ReduceOp::Max => "U2F(0xff800000)",
                ReduceOp::Sum => "0.f",
            }
        )?;
        writeln!(w, "for (int k = 0; k < {}; ++k) {{", k)?;
        writeln!(w, "in_coord[{}] = k;", self.axis.index())?;
        write!(w, "float tmp = input0[")?;
        generate_load_index(&self.input, "in_coord", w)?;
        writeln!(w, "];")?;
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

    fn buffer_count(&self) -> usize {
        2
    }

    fn group_count(&self) -> usize {
        self.shape.element_count().div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!("Reduce (k={}) {}", self.k(), self.shape)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct UnpadKernel {
    pub(crate) shape: Shape,
    pub(crate) input: View,
    pub(crate) axis: Axis,
    pub(crate) pad: usize,
}

impl Kernel for UnpadKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_input_buffer(0, 0, w)?;
        generate_output_buffer(1, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("coord", self.shape, w)?;

        writeln!(w, "int out_coord = coord[{}];", self.axis.index())?;
        writeln!(w, "int in_coord = out_coord + {};", self.pad)?;
        writeln!(
            w,
            "int k_min = in_coord - ((out_coord == 0) ? {} : 0);",
            self.pad
        )?;
        writeln!(
            w,
            "int k_max = in_coord + ((out_coord == {}) ? {} : 0);",
            self.shape[self.axis] - 1,
            self.pad
        )?;

        writeln!(w, "float sum = 0.f;")?;
        writeln!(w, "for (int k = k_min; k <= k_max; ++k) {{")?;
        writeln!(w, "coord[{}] = k;", self.axis.index())?;
        write!(w, "sum += input0[")?;
        generate_load_index(&self.input, "coord", w)?;
        writeln!(w, "];")?;
        writeln!(w, "}}")?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = sum;")?;

        writeln!(w, "}}")?;

        Ok(src)
    }

    fn buffer_count(&self) -> usize {
        2
    }

    fn group_count(&self) -> usize {
        self.shape.element_count().div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!("Unpad {}", self.shape)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct WindowsToImageKernel {
    pub(crate) shape: Shape,
    pub(crate) input: View,
    pub(crate) stride: (usize, usize),
}

impl Kernel for WindowsToImageKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_input_buffer(0, 0, w)?;
        generate_output_buffer(1, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("coord", self.shape, w)?;

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

        writeln!(w, "int filter_base_x = int(uint(in_x) % {});", stride_w)?;
        writeln!(w, "int filter_base_y = int(uint(in_y) % {});", stride_h)?;
        writeln!(w, "int count_x = {};", filter_w.div_round_up(stride_w))?;
        writeln!(w, "int count_y = {};", filter_h.div_round_up(stride_h))?;
        writeln!(w, "int out_x_base = int(uint(in_x)/{});", stride_w)?;
        writeln!(w, "int out_y_base = int(uint(in_y)/{});", stride_h)?;

        writeln!(w, "int in_coord[{}];", batch_dims + 6)?;
        for i in 0..batch_dims {
            writeln!(w, "in_coord[{}] = coord[{}];", i, i)?;
        }

        writeln!(w, "float tmp = 0.f;")?;
        writeln!(w, "for (int index_y = 0; index_y < count_y; ++index_y)",)?;
        writeln!(w, "for (int index_x = 0; index_x < count_x; ++index_x) {{",)?;
        writeln!(w, "int filter_x = filter_base_x + {}*index_x;", stride_w)?;
        writeln!(w, "int filter_y = filter_base_y + {}*index_y;", stride_h)?;
        writeln!(
            w,
            "if (filter_x < {} && filter_y < {}) {{",
            filter_w, filter_h
        )?;
        writeln!(w, "int out_x = out_x_base - index_x;")?;
        writeln!(w, "int out_y = out_y_base - index_y;")?;

        writeln!(w, "in_coord[{}] = out_y;", batch_dims)?;
        writeln!(w, "in_coord[{}] = out_x;", batch_dims + 1)?;
        writeln!(w, "in_coord[{}] = group_index;", batch_dims + 2)?;
        writeln!(w, "in_coord[{}] = filter_y;", batch_dims + 3)?;
        writeln!(w, "in_coord[{}] = filter_x;", batch_dims + 4)?;
        writeln!(w, "in_coord[{}] = group_c;", batch_dims + 5)?;

        write!(w, "tmp += input0[")?;
        generate_load_index(&self.input, "in_coord", w)?;
        writeln!(w, "];")?;

        writeln!(w, "}}")?;
        writeln!(w, "}}")?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = tmp;")?;

        writeln!(w, "}}")?;

        Ok(src)
    }

    fn buffer_count(&self) -> usize {
        2
    }

    fn group_count(&self) -> usize {
        self.shape.element_count().div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!("WindowsToImage {}", self.shape)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GatherKernel {
    pub(crate) shape: Shape,
    pub(crate) values: View,
    pub(crate) axis: Axis,
    pub(crate) indices: View,
}

impl Kernel for GatherKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_input_buffer(0, 0, w)?;
        generate_input_buffer(1, 1, w)?;
        generate_output_buffer(2, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.shape.element_count()
        )?;
        generate_coord("tmp_coord", self.shape, w)?;

        writeln!(w, "int indices_coord[1];")?;
        writeln!(w, "indices_coord[0] = tmp_coord[{}];", self.axis.index())?;
        writeln!(w, "int gather_index = F2I(input1[")?;
        generate_load_index(&self.indices, "indices_coord", w)?;
        writeln!(w, "]);")?;
        writeln!(w, "tmp_coord[{}] = gather_index;", self.axis.index())?;

        writeln!(w, "output0[gl_GlobalInvocationID.x] = input0[")?;
        generate_load_index(&self.values, "tmp_coord", w)?;
        writeln!(w, "];")?;

        writeln!(w, "}}")?;

        Ok(src)
    }

    fn buffer_count(&self) -> usize {
        3
    }

    fn group_count(&self) -> usize {
        self.shape.element_count().div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!("Gather {}", self.shape)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ScatterAddKernel {
    pub(crate) shape: Shape,
    pub(crate) values: View,
    pub(crate) axis: Axis,
    pub(crate) indices: View,
}

impl Kernel for ScatterAddKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        let mut src = String::new();
        let w = &mut src;

        generate_input_buffer(0, 0, w)?;
        generate_input_buffer(1, 1, w)?;
        generate_atomic_buffer(2, 0, w)?;

        writeln!(w, "layout(local_size_x = 64) in;")?;
        writeln!(w, "void main() {{")?;

        writeln!(
            w,
            "if (gl_GlobalInvocationID.x >= {}) {{ return; }}",
            self.values.output_shape.element_count()
        )?;

        generate_coord("tmp_coord", self.values.output_shape, w)?;
        writeln!(w, "float value = input0[")?;
        generate_load_index(&self.values, "tmp_coord", w)?;
        writeln!(w, "];")?;

        writeln!(w, "int in_coord1[1];")?;
        writeln!(w, "in_coord1[0] = tmp_coord[{}];", self.axis.index())?;
        writeln!(w, "int scatter_index = F2I(input1[")?;
        generate_load_index(&self.indices, "in_coord1", w)?;
        writeln!(w, "]);")?;
        writeln!(w, "tmp_coord[{}] = scatter_index;", self.axis.index())?;

        writeln!(w, "atomicAdd(output0[")?;
        generate_load_index(&self.shape.identity_view(), "tmp_coord", w)?;
        writeln!(w, "], value);")?;

        writeln!(w, "}}")?;

        Ok(src)
    }

    fn buffer_count(&self) -> usize {
        3
    }

    fn group_count(&self) -> usize {
        self.values.output_shape.element_count().div_round_up(64)
    }

    fn label_name(&self) -> String {
        format!("ScatterAdd {}", self.values.output_shape)
    }

    fn requires_atomic_float(&self) -> bool {
        true
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum GenericKernel {
    Fill(FillKernel),
    PerElement(PerElementKernel),
    Reduce(ReduceKernel),
    MatMul(MatMulKernel),
    Unpad(UnpadKernel),
    WindowsToImage(WindowsToImageKernel),
    Gather(GatherKernel),
    ScatterAdd(ScatterAddKernel),
}

impl GenericKernel {
    fn as_kernel(&self) -> &dyn Kernel {
        match self {
            GenericKernel::Fill(kernel) => kernel,
            GenericKernel::PerElement(kernel) => kernel,
            GenericKernel::MatMul(kernel) => kernel,
            GenericKernel::Reduce(kernel) => kernel,
            GenericKernel::Unpad(kernel) => kernel,
            GenericKernel::WindowsToImage(kernel) => kernel,
            GenericKernel::Gather(kernel) => kernel,
            GenericKernel::ScatterAdd(kernel) => kernel,
        }
    }
}

impl Kernel for GenericKernel {
    fn generate_source(&self) -> Result<String, fmt::Error> {
        self.as_kernel().generate_source()
    }

    fn buffer_count(&self) -> usize {
        self.as_kernel().buffer_count()
    }

    fn group_count(&self) -> usize {
        self.as_kernel().group_count()
    }

    fn label_name(&self) -> String {
        self.as_kernel().label_name()
    }

    fn requires_atomic_float(&self) -> bool {
        self.as_kernel().requires_atomic_float()
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

    fn create_module(&mut self, kernel: &GenericKernel) -> KernelModule {
        let device = &self.context.device;

        let mut source = kernel.generate_source().unwrap();
        //println!("{}", source);

        source.insert_str(0, include_str!("kernel_common.glsl"));
        if kernel.requires_atomic_float() {
            assert!(self.context.has_shader_atomic_float_add);
            source.insert_str(0, "#extension GL_EXT_shader_atomic_float : require\n");
        }
        source.insert_str(0, "#version 460 core\n");

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
    modules: HashMap<GenericKernel, KernelModule>,
}

impl KernelCache {
    pub(crate) fn new(context: &SharedContext) -> Self {
        Self {
            worker: KernelCacheWorker::new(context),
            modules: HashMap::new(),
        }
    }

    pub(crate) fn module(&mut self, kernel: &GenericKernel) -> KernelModule {
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
