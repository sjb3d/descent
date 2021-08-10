mod device;
pub mod environment;
pub mod graph;
pub mod prelude {
    pub use crate::{environment::*, graph::*, schedule::*, shape::*, variable::*};
}
mod common {
    pub(crate) use crate::{kernel::*, op::*, prelude::*};
}
mod kernel;
pub mod layer;
pub mod loss;
mod op;
pub mod optimizer;
pub mod schedule;
pub mod shape;
pub mod variable;

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use bytemuck::{cast_slice, cast_slice_mut};
    use std::{io::prelude::*, iter};

    const TEST_RAND_SEED: u32 = 0x5EED5EED;

    #[test]
    fn variables() {
        let mut env = Environment::new();

        let a_var = env.static_parameter([10], "a");

        let a_data = [0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        env.writer(&a_var).write_all(cast_slice(&a_data)).unwrap();

        let mut a_result = [0f32; 10];
        env.reader(&a_var)
            .read_exact(cast_slice_mut(&mut a_result))
            .unwrap();

        assert_eq!(a_data, a_result);
    }

    #[test]
    fn reduce() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b_data: Vec<f32> = a_data.chunks(10).map(|v| v.iter().sum::<f32>()).collect();

        let a_var = env.static_parameter([10, 10], "a");
        let b_var = env.static_parameter([10, 1], "b");

        env.writer(&a_var).write_all(cast_slice(&a_data)).unwrap();

        let g = env.graph();
        g.write_variable(&b_var, g.read_variable(&a_var).reduce_sum(-1, true));

        let g = g.build_schedule();
        env.run(&g, TEST_RAND_SEED);

        let mut b_result = vec![0f32; b_data.len()];
        env.reader(&b_var)
            .read_exact(cast_slice_mut(&mut b_result))
            .unwrap();
        assert_eq!(b_result, b_data);
    }

    #[test]
    fn pad_image() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = iter::repeat(1.0).take(64).collect();
        let b_data: Vec<f32> = iter::repeat(1.0).take(100).collect();

        let a_var = env.static_parameter([1, 8, 8, 1], "a");
        let b_var = env.static_parameter([1, 10, 10, 1], "b");

        env.writer(&a_var).write_all(cast_slice(&a_data)).unwrap();

        let g = env.graph();
        g.write_variable(&b_var, g.read_variable(&a_var).pad_image(1));

        let g = g.build_schedule();
        env.run(&g, TEST_RAND_SEED);

        let mut b_result = vec![0f32; b_data.len()];
        env.reader(&b_var)
            .read_exact(cast_slice_mut(&mut b_result))
            .unwrap();
        assert_eq!(b_result, b_data);
    }

    #[test]
    fn unpad_image() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = iter::repeat(1.0).take(100).collect();

        let unpad = |a| if a == 0 || a == 7 { 2.0 } else { 1.0 };
        let b_data: Vec<f32> = (0..8)
            .flat_map(move |y| {
                let ny = unpad(y);
                (0..8).map(move |x| ny * unpad(x))
            })
            .collect();

        let a_var = env.static_parameter([1, 10, 10, 1], "a");
        let b_var = env.static_parameter([1, 8, 8, 1], "b");

        env.writer(&a_var).write_all(cast_slice(&a_data)).unwrap();

        let g = env.graph();
        g.write_variable(&b_var, g.read_variable(&a_var).unpad_image(1));

        let g = g.build_schedule();
        env.run(&g, TEST_RAND_SEED);

        let mut b_result = vec![0f32; b_data.len()];
        env.reader(&b_var)
            .read_exact(cast_slice_mut(&mut b_result))
            .unwrap();
        assert_eq!(b_result, b_data);
    }

    #[test]
    fn conv2d() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = iter::repeat(1.0).take(100).collect();
        let b_data: Vec<f32> = iter::repeat(1.0).take(9).collect();
        let c_data: Vec<f32> = iter::repeat(9.0).take(64).collect();

        let a_var = env.static_parameter([1, 10, 10, 1], "a");
        let b_var = env.static_parameter([1, 3, 3, 1], "b");
        let c_var = env.static_parameter([1, 8, 8, 1], "c");

        env.writer(&a_var).write_all(cast_slice(&a_data)).unwrap();
        env.writer(&b_var).write_all(cast_slice(&b_data)).unwrap();

        let g = env.graph();
        g.write_variable(
            &c_var,
            g.parameter(&a_var).conv2d(&b_var, 0, (1, 1), 1).value(),
        );

        let g = g.build_schedule();
        env.run(&g, TEST_RAND_SEED);

        let mut c_result = vec![0f32; c_data.len()];
        env.reader(&c_var)
            .read_exact(cast_slice_mut(&mut c_result))
            .unwrap();
        assert_eq!(c_result, c_data);
    }

    #[test]
    fn max_pool2d() {
        let mut env = Environment::new();

        let a_data: Vec<_> = (0..100).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..25)
            .map(|i| (11 + 2 * (i % 5) + 20 * (i / 5)) as f32)
            .collect();

        let a_var = env.static_parameter([1, 10, 10, 1], "a");
        let b_var = env.static_parameter([1, 5, 5, 1], "b");

        env.writer(&a_var).write_all(cast_slice(&a_data)).unwrap();

        let g = env.graph();
        g.write_variable(
            &b_var,
            g.parameter(&a_var).max_pool2d((2, 2), (2, 2)).value(),
        );

        let g = g.build_schedule();
        env.run(&g, TEST_RAND_SEED);

        let mut b_result = vec![0f32; b_data.len()];
        env.reader(&b_var)
            .read_exact(cast_slice_mut(&mut b_result))
            .unwrap();
        assert_eq!(b_result, b_data);
    }
}
