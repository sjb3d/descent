pub mod array;
pub mod builder;
mod device;
pub mod environment;
pub mod prelude {
    pub use crate::{builder::*, environment::*, graph::*, shape::*};
}
mod common {
    pub(crate) use crate::{kernel::*, op::*, prelude::*};
}
pub mod graph;
mod kernel;
mod op;
pub mod shape;
