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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use std::io::{Read, Write};

    #[test]
    fn reduce() {
        let mut env = Environment::new();

        let a_data: Vec<_> = (0..100).map(|i| i as f32).collect();
        let b_data: Vec<_> = a_data.chunks(10).map(|v| v.iter().sum::<f32>()).collect();

        let a_var = env.variable([10, 10], "a");
        let b_var = env.variable([10, 1], "b");

        env.writer(a_var)
            .write_all(bytemuck::cast_slice(&a_data))
            .unwrap();

        let g = env.builder();
        g.output(b_var, g.input(a_var).value().reduce_sum(-1));

        let g = g.build();
        env.run(&g);

        let mut b_result = vec![0f32; 10];
        env.reader(b_var)
            .read_exact(bytemuck::cast_slice_mut(&mut b_result))
            .unwrap();
        assert_eq!(b_result, b_data);
    }
}
