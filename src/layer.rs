use crate::common::*;
use rand::{distributions::Open01, Rng};
use std::{convert::TryInto, f32::consts::PI, io::Write};

fn normal_from_uniform(u1: f32, u2: f32) -> f32 {
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn write_rand_uniform(
    mut writer: impl Write,
    fan_in: usize,
    element_count: usize,
    rng: &mut impl Rng,
) {
    let a = (2.0 / (fan_in as f32)).sqrt();
    for _ in 0..element_count {
        let u1: f32 = rng.sample(Open01);
        let u2: f32 = rng.sample(Open01);
        let n: f32 = a * normal_from_uniform(u1, u2);
        writer.write_all(bytemuck::bytes_of(&n)).unwrap();
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
    pub filter: (usize, usize),
    pub pad: usize,
    pub stride: (usize, usize),
    pub groups: usize,
    pub is_blur: bool,
}

impl Conv2D {
    pub fn new(num_filters: usize, filter_w: usize, filter_h: usize) -> Self {
        Self {
            num_filters,
            filter: (filter_w, filter_h),
            pad: 0,
            stride: (1, 1),
            groups: 1,
            is_blur: false,
        }
    }

    pub fn with_pad(mut self, pad: usize) -> Self {
        self.pad = pad;
        self
    }

    pub fn with_stride(mut self, stride_w: usize, stride_h: usize) -> Self {
        self.stride = (stride_w, stride_h);
        self
    }

    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    pub fn with_blur(mut self) -> Self {
        self.is_blur = true;
        self
    }
}

pub struct MaxPool2D {
    pub filter: (usize, usize),
    pub stride: (usize, usize),
}

impl MaxPool2D {
    pub fn new(filter_w: usize, filter_h: usize) -> Self {
        Self {
            filter: (filter_w, filter_h),
            stride: (filter_w, filter_h),
        }
    }

    pub fn with_stride(mut self, stride_w: usize, stride_h: usize) -> Self {
        self.stride = (stride_w, stride_h);
        self
    }
}

pub enum Layer {
    Linear(Linear),
    LeakyRelu(f32),
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    MaxBlurPool2D,
    Flatten,
    Dropout(f32),
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
            match layer {
                Layer::Linear(params) => {
                    layers.push(Box::new(LinearInstance::new(&mut ctx, params)))
                }
                Layer::LeakyRelu(amount) => layers.push(Box::new(LeakyReluInstance::new(amount))),
                Layer::Conv2D(params) => {
                    layers.push(Box::new(Conv2DInstance::new(&mut ctx, params)))
                }
                Layer::MaxPool2D(params) => {
                    layers.push(Box::new(MaxPool2DInstance::new(&mut ctx, params)))
                }
                Layer::MaxBlurPool2D => {
                    let num_channels = ctx.shape[SignedIndex(-1)];
                    layers.push(Box::new(MaxPool2DInstance::new(
                        &mut ctx,
                        MaxPool2D::new(2, 2).with_stride(1, 1),
                    )));
                    layers.push(Box::new(Conv2DInstance::new(
                        &mut ctx,
                        Conv2D::new(1, 3, 3)
                            .with_pad(1)
                            .with_stride(2, 2)
                            .with_groups(num_channels)
                            .with_blur(),
                    )));
                }
                Layer::Flatten => layers.push(Box::new(FlattenInstance::new(&mut ctx))),
                Layer::Dropout(amount) => layers.push(Box::new(DropoutInstance::new(amount))),
            }
        }
        Network {
            layers,
            parameters: ctx.parameters,
        }
    }
}

struct EvalContext {
    is_training: bool,
}

pub struct Network {
    layers: Vec<Box<dyn LayerInstance>>,
    parameters: Vec<Variable>,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder { layers: Vec::new() }
    }

    fn eval<'g>(&self, ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
        let mut x = input;
        for layer in self.layers.iter() {
            x.graph().next_colour();
            x = layer.eval(ctx, x);
        }
        x.graph().next_colour();
        x
    }

    pub fn train<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        self.eval(&EvalContext { is_training: true }, input)
    }

    pub fn test<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        self.eval(&EvalContext { is_training: false }, input)
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
    fn eval<'g>(&self, ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g>;
}

struct LinearInstance {
    w: Variable,
    b: Variable, // TODO: optional?
}

impl LinearInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>, params: Linear) -> Self {
        let [input]: [usize; 1] = ctx.shape.try_into().unwrap();
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
    fn eval<'g>(&self, _ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
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
    fn eval<'g>(&self, _ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
        input.leaky_relu(self.amount)
    }
}

struct Conv2DInstance {
    pad: usize,
    stride: (usize, usize),
    groups: usize,
    f: Variable,
    b: Variable, // TODO: optional?
}

impl Conv2DInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>, params: Conv2D) -> Self {
        let Conv2D {
            num_filters: filter_oc,
            filter,
            pad,
            stride,
            groups,
            is_blur,
        } = params;

        let padded_shape = ctx
            .shape
            .pad(ctx.shape.axis(-3), pad)
            .pad(ctx.shape.axis(-2), pad);

        let window_shape = padded_shape.image_to_windows(filter, stride, groups);
        let [out_h, out_w, _groups, filter_h, filter_w, filter_ic]: [usize; 6] =
            window_shape.try_into().unwrap();

        let f = ctx
            .env
            .variable([filter_oc, filter_h, filter_w, filter_ic], "f");
        let b = ctx.env.variable([filter_oc], "b");
        ctx.env.writer(&b).zero_fill();

        if is_blur {
            assert_eq!([filter_oc, filter_h, filter_w, filter_ic], [1, 3, 3, 1]);
            let f_data: [f32; 9] = [
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                4.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
                2.0 / 16.0,
                1.0 / 16.0,
            ];
            ctx.env
                .writer(&f)
                .write_all(bytemuck::bytes_of(&f_data))
                .unwrap();
        } else {
            let fan_in = filter_h * filter_w * filter_ic;
            write_rand_uniform(ctx.env.writer(&f), fan_in, fan_in * filter_oc, ctx.rng);

            ctx.parameters.push(f.clone());
            ctx.parameters.push(b.clone());
        }

        ctx.shape = Shape::from([out_h, out_w, groups * filter_oc]);

        Self {
            pad,
            stride,
            groups,
            f,
            b,
        }
    }
}

impl LayerInstance for Conv2DInstance {
    fn eval<'g>(&self, _ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
        let conv = input.conv2d(&self.f, self.pad, self.stride, self.groups);

        let out_shape = conv.shape();
        let (out_nc, prefix) = out_shape.split_last().unwrap();

        let mut v = ShapeVec::new();
        v.extend_from_slice(prefix);
        v.push(self.groups);
        v.push(out_nc / self.groups);
        let bias_shape = Shape::new(v);

        (conv.reshape(bias_shape) + &self.b).reshape(out_shape)
    }
}

struct MaxPool2DInstance {
    filter: (usize, usize),
    stride: (usize, usize),
}

impl MaxPool2DInstance {
    pub fn new<'c, R: Rng>(ctx: &mut NetworkContext<'c, R>, params: MaxPool2D) -> Self {
        let [in_h, in_w, in_nc]: [usize; 3] = ctx.shape.try_into().unwrap();
        let MaxPool2D { filter, stride } = params;

        let (filter_w, filter_h) = filter;
        let (stride_w, stride_h) = stride;

        let out_w = (in_w - filter_w) / stride_w + 1;
        let out_h = (in_h - filter_h) / stride_h + 1;

        ctx.shape = Shape::from([out_h, out_w, in_nc]);

        Self { filter, stride }
    }
}

impl LayerInstance for MaxPool2DInstance {
    fn eval<'g>(&self, _ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
        input.max_pool2d(self.filter, self.stride)
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
    fn eval<'g>(&self, _ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
        let input_shape = input.shape();

        let (first, remain) = input_shape.split_first().unwrap();
        let m = *first;
        let element_count = remain.iter().copied().product();
        input.reshape([m, element_count])
    }
}

struct DropoutInstance {
    amount: f32,
}

impl DropoutInstance {
    fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl LayerInstance for DropoutInstance {
    fn eval<'g>(&self, ctx: &EvalContext, input: DualArray<'g>) -> DualArray<'g> {
        if !ctx.is_training {
            return input;
        }

        let graph = input.graph();
        let shape = input.shape();

        let rv = graph.rand(shape);

        let (a, da) = input.into_inner();

        let survivor_scale = 1.0 / (1.0 - self.amount);
        let b = rv.select_gt(self.amount, survivor_scale * a, 0.0);
        let db = b.clone_as_accumulator();
        da.accumulate(rv.select_gt(self.amount, survivor_scale * db, 0.0));

        DualArray::new(b, db)
    }
}
