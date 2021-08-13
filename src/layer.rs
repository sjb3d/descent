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
    initializer: Initializer,
}

impl DenseBuilder {
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    pub fn build(self, env: &mut Environment) -> Dense {
        let DenseBuilder {
            input,
            output,
            initializer,
        } = self;

        let w = env.trainable_parameter([input, output], "w", initializer);
        let b = env.trainable_parameter([output], "b", Initializer::Zero);

        Dense { w, b }
    }
}

pub struct Dense {
    w: Variable,
    b: Variable, // TODO: optional?
}

impl Dense {
    pub fn builder(input: usize, output: usize) -> DenseBuilder {
        DenseBuilder {
            input,
            output,
            initializer: Initializer::for_relu(input),
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
    f: Variable,
    b: Variable, // TODO: optional?
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
            .with_grad();
        da.accumulate(rv.select_gt(self.amount, survivor_scale * db, 0.0));

        (b, db).into()
    }
}
