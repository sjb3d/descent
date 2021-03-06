use crate::common::*;
use std::{io::Write, mem};

pub struct EvalContext {
    is_training: bool,
}

pub trait Module {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s>;
}

pub trait ModuleExt: Module {
    fn train<'s>(&self, input: DualArray<'s>) -> DualArray<'s> {
        self.eval(input, &EvalContext { is_training: true })
    }

    fn test<'s>(&self, input: DualArray<'s>) -> DualArray<'s> {
        self.eval(input, &EvalContext { is_training: false })
    }
}

impl<T> ModuleExt for T where T: Module + ?Sized {}

pub trait ApplyModule<M: Module + ?Sized> {
    fn apply(self, module: &M, ctx: &EvalContext) -> Self;
}

impl<'s, M> ApplyModule<M> for DualArray<'s>
where
    M: Module + ?Sized,
{
    fn apply(self, module: &M, ctx: &EvalContext) -> Self {
        self.map(|x| module.eval(x, ctx))
    }
}

pub struct DenseBuilder {
    input: usize,
    output: usize,
    w_initializer: Initializer,
    b_initializer: Initializer,
}

impl DenseBuilder {
    pub fn with_w_initializer(mut self, w_initializer: Initializer) -> Self {
        self.w_initializer = w_initializer;
        self
    }

    pub fn with_b_initializer(mut self, b_initializer: Initializer) -> Self {
        self.b_initializer = b_initializer;
        self
    }

    pub fn build(self, env: &mut Environment) -> Dense {
        let DenseBuilder {
            input,
            output,
            w_initializer,
            b_initializer,
        } = self;

        let w = env.trainable_parameter([input, output], "w", w_initializer);
        let b = env.trainable_parameter([output], "b", b_initializer);

        Dense { w, b }
    }
}

pub struct Dense {
    w: Parameter,
    b: Parameter, // TODO: optional?
}

impl Dense {
    pub fn builder(input: usize, output: usize) -> DenseBuilder {
        DenseBuilder {
            input,
            output,
            w_initializer: Initializer::for_relu(input),
            b_initializer: Initializer::Zero,
        }
    }
}

impl Module for Dense {
    fn eval<'s>(&self, input: DualArray<'s>, _ctx: &EvalContext) -> DualArray<'s> {
        input.next_colour().matmul(&self.w) + &self.b
    }
}

pub struct Conv2DBuilder {
    input_channels: usize,
    output_channels: usize,
    filter: (usize, usize),
    pad: usize,
    stride: (usize, usize),
    groups: usize,
    is_blur: bool,
}

impl Conv2DBuilder {
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

    pub fn build(self, env: &mut Environment) -> Conv2D {
        let Self {
            input_channels,
            output_channels,
            filter,
            pad,
            stride,
            groups,
            is_blur,
        } = self;
        let filter_ic = input_channels / groups;
        let filter_oc = output_channels / groups;
        assert_eq!(filter_ic * groups, input_channels);
        assert_eq!(filter_oc * groups, output_channels);
        let (filter_w, filter_h) = filter;

        let (f, b) = if is_blur {
            let f = env.static_parameter([groups, filter_oc, filter_h, filter_w, filter_ic], "f");
            let b = env.static_parameter([output_channels], "b");

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

            let mut w = env.writer(&f);
            for _ in 0..groups {
                w.write_all(bytemuck::bytes_of(&f_data)).unwrap();
            }
            mem::drop(w);
            env.writer(&b).zero_fill();

            (f, b)
        } else {
            let f = env.trainable_parameter(
                [groups, filter_oc, filter_h, filter_w, filter_ic],
                "f",
                Initializer::for_relu(filter_h * filter_w * filter_ic),
            );
            let b = env.trainable_parameter([output_channels], "b", Initializer::Zero);
            (f, b)
        };

        Conv2D { f, b, pad, stride }
    }
}

pub struct Conv2D {
    f: Parameter,
    b: Parameter, // TODO: optional?
    pad: usize,
    stride: (usize, usize),
}

impl Conv2D {
    pub fn builder(
        input_channels: usize,
        output_channels: usize,
        filter_w: usize,
        filter_h: usize,
    ) -> Conv2DBuilder {
        Conv2DBuilder {
            input_channels,
            output_channels,
            filter: (filter_w, filter_h),
            pad: 0,
            stride: (1, 1),
            groups: 1,
            is_blur: false,
        }
    }
}

impl Module for Conv2D {
    fn eval<'s>(&self, input: DualArray<'s>, _ctx: &EvalContext) -> DualArray<'s> {
        let conv = input.next_colour().conv2d(&self.f, self.pad, self.stride);

        conv + &self.b
    }
}

#[derive(Default)]
pub struct MaxPool2D {}

impl Module for MaxPool2D {
    fn eval<'s>(&self, input: DualArray<'s>, _ctx: &EvalContext) -> DualArray<'s> {
        input.next_colour().max_pool2d((2, 2), (2, 2))
    }
}

pub struct MaxBlurPool2D {
    blur: Conv2D,
}

impl MaxBlurPool2D {
    pub fn new(env: &mut Environment, channels: usize) -> Self {
        Self {
            blur: Conv2D::builder(channels, channels, 3, 3)
                .with_pad(1)
                .with_stride(2, 2)
                .with_groups(channels)
                .with_blur()
                .build(env),
        }
    }
}

impl Module for MaxBlurPool2D {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        input
            .next_colour()
            .max_pool2d((2, 2), (1, 1))
            .map(|x| self.blur.eval(x, ctx))
    }
}

pub struct Dropout {
    amount: f32,
}

impl Dropout {
    pub fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl Module for Dropout {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        if !ctx.is_training {
            return input;
        }

        let scope = input.scope();
        let shape = input.shape();

        scope.next_colour();
        let rv = scope.rand(shape).value();

        let (a, da) = input.into_inner();

        let survivor_scale = 1.0 / (1.0 - self.amount);
        let (b, db) = rv
            .select_gt(self.amount, survivor_scale * a, 0.0)
            .with_empty_grad();
        da.accumulate(rv.select_gt(self.amount, survivor_scale * db, 0.0));

        (b, db).into()
    }
}

struct LSTMWeight {
    input: Parameter,
    hidden: Parameter,
    bias: Parameter,
}

impl LSTMWeight {
    fn new(env: &mut Environment, prefix: &str, input: usize, output: usize) -> Self {
        let input = env.trainable_parameter(
            [input, output],
            &format!("{}_wi", prefix),
            Initializer::RandNormal(0.01),
        );
        let hidden = env.trainable_parameter(
            [output, output],
            &format!("{}_wh", prefix),
            Initializer::RandNormal(0.01),
        );
        let bias = env.trainable_parameter([output], &format!("{}_b", prefix), Initializer::Zero);
        Self {
            input,
            hidden,
            bias,
        }
    }

    fn eval<'s>(&self, input: DualArray<'s>, hidden: Option<DualArray<'s>>) -> DualArray<'s> {
        let mut x = input.matmul(&self.input);
        if let Some(hidden) = hidden {
            x += hidden.matmul(&self.hidden);
        }
        x + &self.bias
    }
}

pub struct LSTMCell {
    forget_gate: LSTMWeight,
    input_gate: LSTMWeight,
    output_gate: LSTMWeight,
    cell_input: LSTMWeight,
}

impl LSTMCell {
    pub fn new(env: &mut Environment, input: usize, output: usize) -> Self {
        Self {
            forget_gate: LSTMWeight::new(env, "forget", input, output),
            input_gate: LSTMWeight::new(env, "input", input, output),
            output_gate: LSTMWeight::new(env, "output", input, output),
            cell_input: LSTMWeight::new(env, "cell", input, output),
        }
    }
}

impl Module for LSTMCell {
    fn eval<'s>(&self, input: DualArray<'s>, _ctx: &EvalContext) -> DualArray<'s> {
        let time_axis = -2;
        let timestep_count = input.shape()[SignedIndex(time_axis)];
        let mut prev_cell = None;
        let mut prev_hidden = None;
        for i in 0..timestep_count {
            let input = input.next_colour().lock_axis(time_axis, i, false);

            let input_gate = self.input_gate.eval(input, prev_hidden).sigmoid();
            let output_gate = self.output_gate.eval(input, prev_hidden).sigmoid();
            let cell_input = self.cell_input.eval(input, prev_hidden).tanh();

            let mut cell = input_gate * cell_input;
            if let Some(prev_cell) = prev_cell {
                // TODO: fix dead code elimination for gradients
                // (disconnect accumulates that are not from the chosen loss)
                // then we can move the forget gate code out of this "if let"
                let forget_gate = self.forget_gate.eval(input, prev_hidden).sigmoid();
                cell += forget_gate * prev_cell;
            }

            let hidden = output_gate * cell.tanh();

            prev_cell = Some(cell);
            prev_hidden = Some(hidden);
        }
        prev_hidden.unwrap()
    }
}
