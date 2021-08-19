use crate::common::*;

pub fn add_weight_decay_to_grad(scope: &Scope, parameters: &[Parameter], weight_decay: f32) {
    if weight_decay == 0.0 {
        return;
    }

    scope.next_colour();
    for param in parameters.iter() {
        let (w, g) = scope.parameter(param).into_inner();
        g.accumulate(w * weight_decay);
    }
}

pub trait Optimizer {
    fn reset_state(&self, env: &mut Environment);
}

pub struct StochasticGradientDescent {
    state: Vec<Parameter>,
}

impl StochasticGradientDescent {
    pub fn new<'s>(
        env: &mut Environment,
        scope: &'s Scope,
        parameters: &[Parameter],
        learning_rate: impl IntoArray<'s>,
        momentum: f32,
    ) -> Self {
        scope.next_colour();
        let mut state = Vec::new();

        let learning_rate = learning_rate.into_array(scope);
        for param in parameters.iter() {
            let g = scope.parameter(param).loss_grad();
            if momentum == 0.0 {
                scope.update_parameter_value(param, |theta| theta - learning_rate * g);
            } else {
                let shape = param.shape();
                let v_param = env.static_parameter(shape, "v");
                let v = scope.update_parameter_value(&v_param, |v| v * momentum + g);
                scope.update_parameter_value(param, |theta| theta - learning_rate * v);
                state.push(v_param);
            }
        }

        Self { state }
    }
}

impl Optimizer for StochasticGradientDescent {
    fn reset_state(&self, env: &mut Environment) {
        for param in self.state.iter() {
            env.writer(param).zero_fill()
        }
    }
}

pub struct Adam {
    state: Vec<Parameter>,
}

impl Adam {
    pub fn new<'s>(
        env: &mut Environment,
        scope: &'s Scope,
        parameters: &[Parameter],
        learning_rate: impl IntoArray<'s>,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        scope.next_colour();
        let mut state = Vec::new();

        let t_param = env.static_parameter([1], "t");
        let t = scope.update_parameter_value(&t_param, |t| t + 1.0);
        state.push(t_param);

        let alpha = learning_rate.into_array(scope) * (1.0 - (beta2.ln() * t).exp()).sqrt()
            / (1.0 - (beta1.ln() * t).exp());

        for param in parameters.iter() {
            let shape = param.shape();
            let m_param = env.static_parameter(shape, "m");
            let v_param = env.static_parameter(shape, "v");

            let g = scope.parameter(param).loss_grad();
            let m = scope.update_parameter_value(&m_param, |m| m * beta1 + g * (1.0 - beta1));
            let v = scope.update_parameter_value(&v_param, |v| v * beta2 + g * g * (1.0 - beta2));
            state.push(m_param);
            state.push(v_param);

            scope.update_parameter_value(param, |theta| theta - alpha * m / (v.sqrt() + epsilon));
        }

        let tmp = Self { state };
        tmp.reset_state(env);
        tmp
    }
}

impl Optimizer for Adam {
    fn reset_state(&self, env: &mut Environment) {
        for param in self.state.iter() {
            env.writer(param).zero_fill()
        }
    }
}
