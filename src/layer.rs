use crate::common::*;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::io::Write;

fn write_xavier_uniform(env: &mut Environment, variable: &Variable, rng: &mut impl Rng) {
    let shape = variable.shape();

    let mut writer = env.writer(&variable);
    let a = (6.0 / (shape[0] as f32)).sqrt();
    let dist = Uniform::new(-a, a);
    for _ in 0..shape.element_count() {
        let x: f32 = dist.sample(rng);
        writer.write_all(bytemuck::bytes_of(&x)).unwrap();
    }
}

pub trait Layer {
    #[must_use]
    fn forward_pass<'g>(&self, graph: &'g Graph, input: DualArray<'g>) -> DualArray<'g>;

    fn collect_parameters(&self, _parameters: &mut Vec<Variable>) {}
}

pub struct Linear {
    w: Variable,
    b: Variable, // TODO: optional bias?
}

impl Linear {
    pub fn new(env: &mut Environment, input: usize, output: usize, rng: &mut impl Rng) -> Self {
        let w = env.variable([input, output], "w");
        let b = env.variable([output], "b");

        write_xavier_uniform(env, &w, rng);
        env.writer(&b).zero_fill();

        Self { w, b }
    }
}

impl Layer for Linear {
    fn forward_pass<'g>(&self, graph: &'g Graph, input: DualArray<'g>) -> DualArray<'g> {
        let w = graph.parameter(&self.w);
        let b = graph.parameter(&self.b);
        input.matmul(w) + b
    }

    fn collect_parameters(&self, parameters: &mut Vec<Variable>) {
        parameters.push(self.w.clone());
        parameters.push(self.b.clone());
    }
}

pub struct LeakyRelu {
    amount: f32,
}

impl LeakyRelu {
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl Layer for LeakyRelu {
    fn forward_pass<'g>(&self, _graph: &'g Graph, input: DualArray<'g>) -> DualArray<'g> {
        input.leaky_relu(self.amount)
    }
}

pub struct LayeredNetwork {
    layers: Vec<Box<dyn Layer>>,
}

impl LayeredNetwork {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: impl Layer + 'static) {
        self.layers.push(Box::new(layer));
    }
}

impl Default for LayeredNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for LayeredNetwork {
    fn forward_pass<'g>(&self, graph: &'g Graph, input: DualArray<'g>) -> DualArray<'g> {
        let mut x = input;
        for layer in self.layers.iter() {
            graph.next_colour();
            x = layer.forward_pass(graph, x);
        }
        graph.next_colour();
        x
    }

    fn collect_parameters(&self, parameters: &mut Vec<Variable>) {
        for layer in self.layers.iter() {
            layer.collect_parameters(parameters);
        }
    }
}
