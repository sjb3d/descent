use descent::prelude::*;

fn main() {
    let random_seed = 0x5EED5EED;

    let mut env = Environment::new();

    let m_param =
        env.static_parameter_with_data([3, 3], "m", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let x_param = env.static_parameter_with_data([3, 1], "x", &[4.0, 5.0, 6.0]);
    let y_param = env.static_parameter_with_data([3, 1], "y", &[1.0, 2.0, 3.0]);
    let z_param = env.static_parameter([3, 1], "z");

    let graph = env.build_graph(|scope| {
        let m = scope.parameter_value(&m_param);
        let x = scope.parameter_value(&x_param);
        let y = scope.parameter_value(&y_param);
        let z = 2.0 * m.matmul(x) + y * y + 1.0;
        scope.write_parameter_value(&z_param, z);
    });
    graph.write_dot_file(KernelDotOutput::Cluster, "array_values.dot");

    env.run(&graph, random_seed);
    assert_eq!(&env.read_parameter_to_vec(&z_param), &[10.0, 15.0, 22.0]);

    let x_param = env.trainable_parameter([1], "x", Initializer::Zero);

    let graph = env.build_graph(|scope| {
        let x = scope.parameter(&x_param);
        let y = x.sin();
        let _loss = (y.square() + y * 3.0).set_loss();
        scope.write_parameter_value(&x_param, x.value() - 0.1 * x.loss_grad());
    });
    graph.write_dot_file(KernelDotOutput::Cluster, "array_grad.dot");
}
