use crate::common::*;
use std::io::Write;

pub struct EvalContext {
    is_training: bool,
}

pub trait Module {
    fn eval<'g>(&self, input: DualArray<'g>, ctx: &EvalContext) -> DualArray<'g>;
}

pub trait ModuleExt: Module {
    fn train<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        self.eval(input, &EvalContext { is_training: true })
    }

    fn test<'g>(&self, input: DualArray<'g>) -> DualArray<'g> {
        self.eval(input, &EvalContext { is_training: false })
    }
}

impl<T> ModuleExt for T where T: Module + ?Sized {}

pub trait ApplyModule<M: Module + ?Sized> {
    fn apply(self, module: &M, ctx: &EvalContext) -> Self;
}

impl<'g, M> ApplyModule<M> for DualArray<'g>
where
    M: Module + ?Sized,
{
    fn apply(self, module: &M, ctx: &EvalContext) -> Self {
        self.map(|x| module.eval(x, ctx))
    }
}

pub struct Dense {
    w: Variable,
    b: Variable, // TODO: optional?
}

impl Dense {
    pub fn new(env: &mut Environment, input: usize, output: usize) -> Self {
        let w = env.trainable_parameter(
            [input, output],
            "w",
            VariableContents::normal_from_fan_in(input * output),
        );
        let b = env.trainable_parameter([output], "b", VariableContents::Zero);

        Self { w, b }
    }
}

impl Module for Dense {
    fn eval<'g>(&self, input: DualArray<'g>, _ctx: &EvalContext) -> DualArray<'g> {
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
            input_channels: filter_ic,
            output_channels: filter_oc,
            filter,
            pad,
            stride,
            groups,
            is_blur,
        } = self;
        let (filter_w, filter_h) = filter;

        let (f, b) = if is_blur {
            let f = env.static_parameter([filter_oc, filter_h, filter_w, filter_ic], "f");
            let b = env.static_parameter([filter_oc], "b");

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

            env.writer(&f)
                .write_all(bytemuck::bytes_of(&f_data))
                .unwrap();
            env.writer(&b).zero_fill();

            (f, b)
        } else {
            let f = env.trainable_parameter(
                [filter_oc, filter_h, filter_w, filter_ic],
                "f",
                VariableContents::normal_from_fan_in(filter_h * filter_w * filter_ic),
            );
            let b = env.trainable_parameter([filter_oc], "b", VariableContents::Zero);
            (f, b)
        };

        Conv2D {
            f,
            b,
            pad,
            stride,
            groups,
        }
    }
}

pub struct Conv2D {
    f: Variable,
    b: Variable, // TODO: optional?
    pad: usize,
    stride: (usize, usize),
    groups: usize,
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
    fn eval<'g>(&self, input: DualArray<'g>, _ctx: &EvalContext) -> DualArray<'g> {
        let conv = input
            .next_colour()
            .conv2d(&self.f, self.pad, self.stride, self.groups);

        let out_shape = conv.shape();
        let bias_shape = {
            let (out_nc, prefix) = out_shape.split_last().unwrap();
            let filter_oc = self.f.shape()[0];
            assert_eq!(*out_nc, filter_oc * self.groups);

            let mut v = ShapeVec::new();
            v.extend_from_slice(prefix);
            v.push(self.groups);
            v.push(filter_oc);
            Shape::new(v)
        };
        (conv.reshape(bias_shape) + &self.b).reshape(out_shape)
    }
}

#[derive(Default)]
pub struct MaxPool2D {}

impl Module for MaxPool2D {
    fn eval<'g>(&self, input: DualArray<'g>, _ctx: &EvalContext) -> DualArray<'g> {
        input.next_colour().max_pool2d((2, 2), (2, 2))
    }
}

pub struct MaxBlurPool2D {
    blur: Conv2D,
}

impl MaxBlurPool2D {
    pub fn new(env: &mut Environment, channels: usize) -> Self {
        Self {
            blur: Conv2D::builder(1, 1, 3, 3)
                .with_pad(1)
                .with_stride(2, 2)
                .with_groups(channels)
                .with_blur()
                .build(env),
        }
    }
}

impl Module for MaxBlurPool2D {
    fn eval<'g>(&self, input: DualArray<'g>, ctx: &EvalContext) -> DualArray<'g> {
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
    fn eval<'g>(&self, input: DualArray<'g>, ctx: &EvalContext) -> DualArray<'g> {
        if !ctx.is_training {
            return input;
        }

        let graph = input.graph();
        let shape = input.shape();

        graph.next_colour();
        let rv = graph.rand(shape);

        let (a, da) = input.into_inner();

        let survivor_scale = 1.0 / (1.0 - self.amount);
        let b = rv.select_gt(self.amount, survivor_scale * a, 0.0);
        let db = b.clone_as_accumulator();
        da.accumulate(rv.select_gt(self.amount, survivor_scale * db, 0.0));

        DualArray::new(b, db)
    }
}
