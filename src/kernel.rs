use crate::{graph::*, prelude::*};

#[derive(Debug)]
pub(crate) enum PerElementKernelOp {
    Load {
        input_index: usize,
    },
    Literal(f32),
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

#[derive(Debug)]
pub(crate) struct KernelInput {
    pub(crate) view: View,
    pub(crate) shape: Shape,
}

#[derive(Debug)]
pub(crate) struct PerElementKernel {
    pub(crate) shape: Shape,
    pub(crate) inputs: Vec<KernelInput>,
    pub(crate) outputs: Vec<usize>,
    pub(crate) ops: Vec<PerElementKernelOp>,
}

#[derive(Debug)]
pub(crate) struct ReduceKernel {
    pub(crate) input_shape: Shape,
    pub(crate) reduce_op: ReduceOp,
    pub(crate) axis: isize,
}

#[derive(Debug)]
pub(crate) struct MatMulKernel {
    pub(crate) inputs: [KernelInput; 2],
}

#[derive(Debug)]
pub(crate) enum Kernel {
    PerElement(PerElementKernel),
    Reduce(ReduceKernel),
    MatMul(MatMulKernel),
}
