pub mod array;
pub mod builder;
mod device;
pub mod environment;
pub mod prelude {
    pub use crate::{builder::*, environment::*, graph::*, shape::*};
}
mod common {
    pub(crate) use crate::{op::*, kernel::*, prelude::*};
}
mod op;
mod kernel;
pub mod graph;
pub mod shape;
