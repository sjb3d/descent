use crate::common::*;
use ordered_float::NotNan;
use petgraph::prelude::*;

pub(crate) type OpGraph = StableDiGraph<OpNode, OpEdge, usize>;
pub(crate) type OpNodeId = NodeIndex<usize>;
pub(crate) type OpEdgeId = EdgeIndex<usize>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ReduceOp {
    Max,
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BuiltInOp {
    Coord { shape: Shape, axis: Axis },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CompareMode {
    Eq,
    Gt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum UnaryOp {
    Mov,
    Neg,
    Sqrt,
    Exp,
    Log,
}

pub(crate) const MAX_OP_ARGS: usize = 4;

pub(crate) const MATMUL_MAX_K_SIZE: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    Input { variable_id: VariableId },
    Output { variable_id: VariableId },
    Literal(NotNan<f32>),
    BuiltIn(BuiltInOp),
    Unary(UnaryOp),
    Binary(BinaryOp),
    CompareAndSelect(CompareMode),
    Accumulate, // accumulates grad from backprop
    MatMul,
    Reduce { reduce_op: ReduceOp, axis: Axis },
    ImageToWindows { pad: usize },
    WindowsToImage { pad: usize },
}

impl Op {
    pub(crate) fn input_variable_id(&self) -> Option<VariableId> {
        match self {
            Self::Input { variable_id } => Some(*variable_id),
            _ => None,
        }
    }

    pub(crate) fn output_variable_id(&self) -> Option<VariableId> {
        match self {
            Self::Output { variable_id } => Some(*variable_id),
            _ => None,
        }
    }

    pub(crate) fn is_per_element(&self) -> bool {
        matches!(
            self,
            Self::BuiltIn(..) | Self::Unary(..) | Self::Binary(..) | Self::CompareAndSelect(..)
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct OpNode {
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) cluster_id: Option<ClusterId>,
}

#[derive(Debug, Clone)]
pub(crate) struct OpEdge {
    pub(crate) arg: usize,
    pub(crate) view: View,
}
