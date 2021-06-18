use crate::common::*;
use petgraph::prelude::*;

pub(crate) type OpGraph = StableDiGraph<OpNode, OpEdge, usize>;
pub(crate) type OpNodeId = NodeIndex<usize>;
pub(crate) type OpEdgeId = EdgeIndex<usize>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReduceOp {
    Max,
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Neg,
    Exp,
    Log,
    OneHot,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Op {
    Input { variable_id: VariableId },
    Output { variable_id: VariableId },
    Literal(f32),
    View(View),
    Unary(UnaryOp),
    Binary(BinaryOp),
    MatMul,
    Reduce { reduce_op: ReduceOp, axis: isize },
    Accumulate, // accumulates grad from backprop
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
