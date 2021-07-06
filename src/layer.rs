use crate::common::*;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::{convert::TryInto, io::Write};

fn write_rand_uniform(
    mut writer: impl Write,
    fan_in: usize,
    element_count: usize,
    rng: &mut impl Rng,
) {
    let a = (6.0 / (fan_in as f32)).sqrt();
    let dist = Uniform::new(-a, a);
    for _ in 0..element_count {
        let x: f32 = dist.sample(rng);
        writer.write_all(bytemuck::bytes_of(&x)).unwrap();
    }
}

pub struct Linear {
    pub hidden_units: usize,
}

impl Linear {
    pub fn new(hidden_units: usize) -> Self {
        Self { hidden_units }
    }
}

pub struct Conv2D {
    pub num_filters: usize,
    pub filter_w: usize,
    pub filter_h: usize,
    pub pad: usize,
}

impl Conv2D {
    pub fn new(num_filters: usize, filter_w: usize, filter_h: usize, pad: usize) -> Self {
        Self {
            num_filters,
            filter_w,
            filter_h,
            pad,
        }
    }
}

pub struct MaxPool2D {
    pub pool_w: usize,
    pub pool_h: usize,
}

impl MaxPool2D {
    pub fn new(pool_w: usize, pool_h: usize) -> Self {
        Self { pool_w, pool_h }
    }
}

pub enum Layer {
    Linear(Linear),
    LeakyRelu(f32),
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    Flatten,
}

pub struct NetworkBuilder {
    layers: Vec<Layer>,
}

impl NetworkBuilder {
    pub fn with_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn finish(
        mut self,
        input_shape: impl Into<Shape>,
        env: &mut Environment,
        rng: &mut impl Rng,
    ) -> Network {
        let mut ctx = NetworkContext {
            env,
            rng,
            parameters: Vec::new(),
            shape: input_shape.into(),
        };
        let mut layers: Vec<Box<dyn LayerInstance>> = Vec::new();
        for layer in self.layers.drain(..) {
            layers.push(match layer {
                Layer::Linear(params) => Box::new(LinearInstance::new(&mut ctx, params)),
                Layer::LeakyRelu(amount) => Box::new(LeakyReluInstance::new(amount)),
                Layer::Conv2D(params) => Box::new(Conv2DInstance::new(&mut ctx, params)),
                Layer::MaxPool2D(params) => Box::new(MaxPool2DInstance::new(&mut ctx, params)),
                Layer::Flatten => Box::new(FlattenInstance::new(&mut ctx)),
            });
        }
        Network {
            layers,
            parameters: ctx.parameters,
        }
    }
}

pub struct Network {
    layers: Vec<Box<dyn LayerInstance>>,
    parameters: Vec<Variable>,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder { layers: Vec::new() }
    }

    pub fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        let mut x = input;
        for layer in self.layers.iter() {
            x.graph().next_colour();
            x = layer.forward_pass(x);
        }
        x.graph().next_colour();
        x
    }

    pub fn parameters(&self) -> &[Variable] {
        self.parameters.as_slice()
    }
}

struct NetworkContext<'c, R: Rng> {
    env: &'c mut Environment,
    rng: &'c mut R,
    parameters: Vec<Variable>,
    shape: Shape,
}

trait LayerInstance {
    fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g>;
}

struct LinearInstance {
    w: Variable,
    b: Variable, // TODO: optional?
}

impl LinearInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>, params: Linear) -> Self {
        let [input]: [usize; 1] = ctx.shape.as_slice().try_into().unwrap();
        let output = params.hidden_units;

        let w = ctx.env.variable([input, output], "w");
        let b = ctx.env.variable([output], "b");

        write_rand_uniform(ctx.env.writer(&w), input, input * output, ctx.rng);
        ctx.env.writer(&b).zero_fill();

        ctx.parameters.push(w.clone());
        ctx.parameters.push(b.clone());
        ctx.shape = Shape::from([output]);

        Self { w, b }
    }
}

impl LayerInstance for LinearInstance {
    fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        input.matmul(&self.w) + &self.b
    }
}

struct LeakyReluInstance {
    amount: f32,
}

impl LeakyReluInstance {
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl LayerInstance for LeakyReluInstance {
    fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        input.leaky_relu(self.amount)
    }
}

struct Conv2DInstance {
    f: Variable,
    pad: usize,
    b: Variable, // TODO: optional?
}

impl Conv2DInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>, params: Conv2D) -> Self {
        let [in_h, in_w, in_nc]: [usize; 3] = ctx.shape.as_slice().try_into().unwrap();
        let Conv2D {
            num_filters: out_nc,
            filter_w,
            filter_h,
            pad,
        } = params;

        let out_w = 1 + in_w + 2 * pad - filter_w;
        let out_h = 1 + in_h + 2 * pad - filter_h;

        let f = ctx.env.variable([out_nc, filter_h, filter_w, in_nc], "f");
        let b = ctx.env.variable([out_nc], "b");

        let fan_in = filter_h * filter_w * in_nc;
        write_rand_uniform(ctx.env.writer(&f), fan_in, fan_in * out_nc, ctx.rng);
        ctx.env.writer(&b).zero_fill();

        ctx.parameters.push(f.clone());
        ctx.parameters.push(b.clone());
        ctx.shape = Shape::from([out_h, out_w, out_nc]);

        Self { f, pad, b }
    }
}

impl LayerInstance for Conv2DInstance {
    fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        input.conv2d(&self.f, self.pad) + &self.b
    }
}

struct MaxPool2DInstance {
    pool_w: usize,
    pool_h: usize,
}

impl MaxPool2DInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>, params: MaxPool2D) -> Self {
        let [in_h, in_w, in_nc]: [usize; 3] = ctx.shape.as_slice().try_into().unwrap();
        let MaxPool2D { pool_w, pool_h } = params;

        ctx.shape = Shape::from([in_h / pool_h, in_w / pool_w, in_nc]);

        Self { pool_w, pool_h }
    }
}

impl LayerInstance for MaxPool2DInstance {
    fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        input.max_pool(-3, self.pool_h).max_pool(-2, self.pool_w)
    }
}

struct FlattenInstance {}

impl FlattenInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>) -> Self {
        ctx.shape = Shape::from([ctx.shape.element_count()]);

        Self {}
    }
}

impl LayerInstance for FlattenInstance {
    fn forward_pass<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        let input_shape = input.shape();

        let (first, remain) = input_shape.split_first().unwrap();
        let m = *first;
        let element_count = remain.iter().copied().product();
        input.reshape([m, element_count])
    }
}
