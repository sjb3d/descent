use crate::common::*;
use petgraph::prelude::*;
use slotmap::Key;

pub(crate) type OpGraph = StableDiGraph<OpNode, OpEdge, usize>;
pub(crate) type OpNodeIndex = NodeIndex<usize>;
pub(crate) type OpEdgeIndex = EdgeIndex<usize>;

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

#[derive(Debug, Clone)]
pub(crate) struct OpNode {
    pub(crate) name: Option<String>,
    pub(crate) colour: usize,
    pub(crate) shape: Shape,
    pub(crate) op: Op,
    pub(crate) cluster_id: ClusterId,
}

impl OpNode {
    pub(crate) fn new(colour: usize, shape: Shape, op: Op) -> Self {
        Self {
            name: None,
            colour,
            shape,
            op,
            cluster_id: ClusterId::null(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct OpEdge {
    pub(crate) arg: usize,
    pub(crate) view: View,
}
