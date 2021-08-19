pub mod array;
mod device;
pub mod environment;
pub mod prelude {
    pub use crate::{array::*, environment::*, graph::*, shape::*, variable::*};
}
mod common {
    pub(crate) use crate::{kernel::*, op::*, prelude::*};
}
pub mod graph;
mod kernel;
pub mod layer;
pub mod loss;
mod op;
pub mod optimizer;
pub mod shape;
pub mod variable;

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use std::iter;

    const TEST_RAND_SEED: u32 = 0x5EED5EED;

    #[test]
    fn parameters() {
        let mut env = Environment::new();

        let a_data = vec![0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let a_param = env.static_parameter_with_data([10], "a", &a_data);

        assert_eq!(env.read_parameter_to_vec(&a_param), a_data);
    }

    #[test]
    fn reduce() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b_data: Vec<f32> = a_data.chunks(10).map(|v| v.iter().sum::<f32>()).collect();

        let a_param = env.static_parameter_with_data([10, 10], "a", &a_data);
        let b_param = env.static_parameter([10, 1], "b");

        let g = env.build_graph(|scope| {
            scope.write_parameter_value(
                &b_param,
                scope.parameter_value(&a_param).reduce_sum(-1, true),
            );
        });
        env.run(&g, TEST_RAND_SEED);

        assert_eq!(env.read_parameter_to_vec(&b_param), b_data);
    }

    #[test]
    fn pad_image() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = iter::repeat(1.0).take(64).collect();
        let b_data: Vec<f32> = iter::repeat(1.0).take(100).collect();

        let a_param = env.static_parameter_with_data([1, 8, 8, 1], "a", &a_data);
        let b_param = env.static_parameter([1, 10, 10, 1], "b");

        let g = env.build_graph(|scope| {
            scope.write_parameter_value(&b_param, scope.parameter_value(&a_param).pad_image(1));
        });
        env.run(&g, TEST_RAND_SEED);

        assert_eq!(env.read_parameter_to_vec(&b_param), b_data);
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

        let a_param = env.static_parameter_with_data([1, 10, 10, 1], "a", &a_data);
        let b_param = env.static_parameter([1, 8, 8, 1], "b");

        let g = env.build_graph(|scope| {
            scope.write_parameter_value(&b_param, scope.parameter_value(&a_param).unpad_image(1));
        });
        env.run(&g, TEST_RAND_SEED);

        assert_eq!(env.read_parameter_to_vec(&b_param), b_data);
    }

    #[test]
    fn conv2d() {
        let mut env = Environment::new();

        let a_data: Vec<f32> = iter::repeat(1.0).take(100).collect();
        let b_data: Vec<f32> = iter::repeat(1.0).take(9).collect();
        let c_data: Vec<f32> = iter::repeat(9.0).take(64).collect();

        let a_param = env.static_parameter_with_data([1, 10, 10, 1], "a", &a_data);
        let b_param = env.static_parameter_with_data([1, 1, 3, 3, 1], "b", &b_data);
        let c_param = env.static_parameter([1, 8, 8, 1], "c");

        let g = env.build_graph(|scope| {
            scope.write_parameter_value(
                &c_param,
                scope
                    .parameter(&a_param)
                    .conv2d(&b_param, 0, (1, 1))
                    .value(),
            );
        });
        env.run(&g, TEST_RAND_SEED);

        assert_eq!(env.read_parameter_to_vec(&c_param), c_data);
    }

    #[test]
    fn max_pool2d() {
        let mut env = Environment::new();

        let a_data: Vec<_> = (0..100).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..25)
            .map(|i| (11 + 2 * (i % 5) + 20 * (i / 5)) as f32)
            .collect();

        let a_param = env.static_parameter_with_data([1, 10, 10, 1], "a", &a_data);
        let b_param = env.static_parameter([1, 5, 5, 1], "b");

        let g = env.build_graph(|scope| {
            scope.write_parameter_value(
                &b_param,
                scope.parameter(&a_param).max_pool2d((2, 2), (2, 2)).value(),
            );
        });
        env.run(&g, TEST_RAND_SEED);

        assert_eq!(env.read_parameter_to_vec(&b_param), b_data);
    }
}
