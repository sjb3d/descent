pub mod builder;
mod device;
pub mod environment;
pub mod prelude {
    pub use crate::{builder::*, environment::*, schedule::*, shape::*, variable::*};
}
mod common {
    pub(crate) use crate::{kernel::*, op::*, prelude::*};
}
mod kernel;
mod op;
pub mod schedule;
pub mod shape;
pub mod variable;

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use bytemuck::{cast_slice, cast_slice_mut};
    use std::io::prelude::*;

    #[test]
    fn variables() {
        let mut env = Environment::new();

        let a_var = env.variable([10], "a");

        let a_data = [0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        env.writer(a_var).write_all(cast_slice(&a_data)).unwrap();

        let mut a_result = [0f32; 10];
        env.reader(a_var)
            .read_exact(cast_slice_mut(&mut a_result))
            .unwrap();

        assert_eq!(a_data, a_result);
    }

    #[test]
    fn reduce() {
        let mut env = Environment::new();

        let a_data: Vec<_> = (0..100).map(|i| i as f32).collect();
        let b_data: Vec<_> = a_data.chunks(10).map(|v| v.iter().sum::<f32>()).collect();

        let a_var = env.variable([10, 10], "a");
        let b_var = env.variable([10, 1], "b");

        env.writer(a_var).write_all(cast_slice(&a_data)).unwrap();

        let g = env.graph();
        g.write_variable(b_var, g.parameter(a_var).value().reduce_sum(-1));

        let g = g.build_schedule();
        env.run(&g);

        let mut b_result = vec![0f32; 10];
        env.reader(b_var)
            .read_exact(cast_slice_mut(&mut b_result))
            .unwrap();
        assert_eq!(b_result, b_data);
    }
}
