use descent::prelude::*;

fn main() {
    let mut env = Environment::new();

    let m_var =
        env.static_parameter_with_data([3, 3], "m", &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let x_var = env.static_parameter_with_data([3, 1], "x", &[4.0, 5.0, 6.0]);
    let y_var = env.static_parameter_with_data([3, 1], "y", &[1.0, 2.0, 3.0]);
    let z_var = env.static_parameter([3, 1], "z");

    let graph = env.build_graph(|scope| {
        let m = scope.read_variable(&m_var);
        let x = scope.read_variable(&x_var);
        let y = scope.read_variable(&y_var);
        scope.next_colour();

        let z = 2.0 * m.matmul(x) + y * y + 1.0;

        scope.next_colour();
        scope.write_variable(&z_var, z);
    });
    graph.write_dot_file(KernelDotOutput::Cluster, "array.dot");
    env.run(&graph, 0x5EED5EED);

    assert_eq!(&env.read_variable_to_vec(&z_var), &[10.0, 15.0, 22.0]);
}
