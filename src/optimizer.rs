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

pub trait Optimizer {
    fn reset_state(&self, env: &mut Environment);
}

pub struct StochasticGradientDescent {}

impl StochasticGradientDescent {
    pub fn new<'g>(
        graph: &'g Graph,
        variables: &[Variable],
        learning_rate: impl IntoArray<'g>,
    ) -> Self {
        graph.next_colour();

        let learning_rate = learning_rate.into_array(graph);
        for var in variables.iter() {
            let g = graph.parameter(var).grad();
            graph.update_variable(var, |theta| theta - learning_rate * g);
        }

        Self {}
    }
}

impl Optimizer for StochasticGradientDescent {
    fn reset_state(&self, _env: &mut Environment) {}
}

pub struct Adam {
    state: Vec<Variable>,
}

impl Adam {
    pub fn new<'g>(
        env: &mut Environment,
        graph: &'g Graph,
        variables: &[Variable],
        learning_rate: impl IntoArray<'g>,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        graph.next_colour();
        let mut state = Vec::new();

        let t_var = env.static_parameter([1], "t");
        let t = graph.update_variable(&t_var, |t| t + 1.0);
        state.push(t_var);

        let alpha = learning_rate.into_array(graph) * (1.0 - (beta2.ln() * t).exp()).sqrt()
            / (1.0 - (beta1.ln() * t).exp());

        for var in variables.iter() {
            let shape = var.shape();
            let m_var = env.static_parameter(shape, "m");
            let v_var = env.static_parameter(shape, "v");

            let g = graph.parameter(var).grad();
            let m = graph.update_variable(&m_var, |m| m * beta1 + g * (1.0 - beta1));
            let v = graph.update_variable(&v_var, |v| v * beta2 + g * g * (1.0 - beta2));
            state.push(m_var);
            state.push(v_var);

            graph.update_variable(var, |theta| theta - alpha * m / (v.sqrt() + epsilon));
        }

        Self { state }
    }
}

impl Optimizer for Adam {
    fn reset_state(&self, env: &mut Environment) {
        for var in self.state.iter() {
            env.writer(var).zero_fill()
        }
    }
}
