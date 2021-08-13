use crate::common::*;
use ordered_float::NotNan;
use petgraph::prelude::*;
use slotmap::Key;
use std::fmt;

pub(crate) trait Only: Iterator {
    fn only(&mut self) -> Option<Self::Item>;
}

impl<I: Iterator> Only for I {
    fn only(&mut self) -> Option<Self::Item> {
        let first = self.next();
        first.filter(|_| self.next().is_none())
    }
}

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
    Coord,
    Rand { uid: usize },
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
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum UnaryOp {
    Mov,
    Neg,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
}

pub(crate) const MAX_OP_ARGS: usize = 4;

pub(crate) const MATMUL_MAX_K_SIZE: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MatMulOutputMode {
    Batches,
    Rows,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    Input { variable_id: VariableId },
    Output { variable_id: VariableId },
    Literal(NotNan<f32>),
    BuiltIn(BuiltInOp),
    Unary(UnaryOp),
    Binary(BinaryOp),
    CompareAndSelect(CompareMode),
    MatMul { output_mode: MatMulOutputMode },
    Reduce { reduce_op: ReduceOp, axis: Axis }, // TODO: 2D version?
    Unpad { axis: Axis, pad: usize },           // TODO: 2D version?
    WindowsToImage { stride: (usize, usize) },
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
            Self::Unary(_) | Self::Binary(_) | Self::CompareAndSelect(_)
        )
    }

    pub(crate) fn can_reshape(&self) -> bool {
        !matches!(self, Self::BuiltIn(_))
    }

    pub(crate) fn can_merge(&self) -> bool {
        !matches!(self, Self::Input { .. } | Self::Output { .. })
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { variable_id } => write!(f, "Input({:?})", variable_id.data()),
            Self::Output { variable_id } => write!(f, "Output({:?})", variable_id.data()),
            Self::Literal(value) => write!(f, "{:?}", value),
            Self::BuiltIn(built_in_op) => match built_in_op {
                BuiltInOp::Coord => write!(f, "Coord"),
                BuiltInOp::Rand { .. } => write!(f, "Rand"),
            },
            Self::Unary(unary_op) => write!(f, "{:?}", unary_op),
            Self::Binary(binary_op) => write!(f, "{:?}", binary_op),
            Self::CompareAndSelect(compare_mode) => write!(f, "Select{:?}", compare_mode),
            Self::MatMul { .. } => write!(f, "MatMul"),
            Self::Reduce { reduce_op, axis } => {
                write!(f, "Reduce{:?}({})", reduce_op, axis.index())
            }
            Self::Unpad { axis, pad } => write!(f, "Unpad{}({})", pad, axis.index()),
            Self::WindowsToImage { .. } => write!(f, "WindowsToImage"),
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

pub(crate) trait OpGraphExt {
    fn new_node(
        &mut self,
        colour: usize,
        shape: impl Into<Shape>,
        op: Op,
        inputs: &[OpNodeId],
    ) -> OpNodeId;
}

impl OpGraphExt for OpGraph {
    fn new_node(
        &mut self,
        colour: usize,
        shape: impl Into<Shape>,
        op: Op,
        inputs: &[OpNodeId],
    ) -> OpNodeId {
        let shape = shape.into();
        let node_id = self.add_node(OpNode {
            colour,
            shape,
            op,
            cluster_id: None,
        });
        for (index, input_id) in inputs.iter().copied().enumerate() {
            self.add_edge(
                input_id,
                node_id,
                OpEdge {
                    arg: index,
                    view: self[input_id].shape.identity_view(),
                },
            );
        }
        node_id
    }
}
