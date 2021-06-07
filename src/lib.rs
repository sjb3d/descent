pub mod array;
pub mod builder;
mod device;
pub mod environment;
pub mod prelude {
    pub use crate::{builder::*, environment::*, shape::*};
}
pub mod schedule;
pub mod shape;
