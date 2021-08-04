use crate::common::*;

pub fn add_weight_decay_to_grad(graph: &Graph, variables: &[Variable], weight_decay: f32) {
    if weight_decay == 0.0 {
        return;
    }

    graph.next_colour();
    for var in variables.iter() {
        let (w, g) = graph.parameter(var).into_inner();
        g.accumulate(w * weight_decay);
    }
}

pub fn stochastic_gradient_descent_step<'g>(
    graph: &'g Graph,
    variables: &[Variable],
    learning_rate: impl IntoArray<'g>,
) {
    graph.next_colour();

    let learning_rate = learning_rate.into_array(graph);
    for var in variables.iter() {
        let g = graph.parameter(var).grad();
        graph.update_variable(var, |theta| theta - learning_rate * g);
    }
}

pub fn adam_step<'g>(
    env: &mut Environment,
    graph: &'g Graph,
    variables: &[Variable],
    learning_rate: impl IntoArray<'g>,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    graph.next_colour();

    let t_var = env.static_parameter([1], "t");
    env.writer(&t_var).zero_fill();

    let t = graph.update_variable(&t_var, |t| t + 1.0);
    let alpha = learning_rate.into_array(graph) * (1.0 - (beta2.ln() * t).exp()).sqrt()
        / (1.0 - (beta1.ln() * t).exp());

    for var in variables.iter() {
        let shape = var.shape();
        let m_var = env.static_parameter(shape.clone(), "m");
        let v_var = env.static_parameter(shape.clone(), "v");
        env.writer(&m_var).zero_fill();
        env.writer(&v_var).zero_fill();

        let g = graph.parameter(var).grad();
        let m = graph.update_variable(&m_var, |m| m * beta1 + g * (1.0 - beta1));
        let v = graph.update_variable(&v_var, |v| v * beta2 + g * g * (1.0 - beta2));

        graph.update_variable(var, |theta| theta - alpha * m / (v.sqrt() + epsilon));
    }
}
