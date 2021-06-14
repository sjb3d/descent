pub mod array;
pub mod builder;
mod device;
pub mod environment;
pub mod prelude {
    pub use crate::{builder::*, environment::*, schedule::*, shape::*};
}
mod common {
    pub(crate) use crate::{graph::*, kernel::*, prelude::*};
}
mod graph;
mod kernel;
pub mod schedule;
pub mod shape;
