use descent::{layer::*, optimizer::*, prelude::*};
use rand::{Rng, RngCore, SeedableRng};
use stb::image;
use std::{f32::consts::PI, ffi::CString, fs, io::Write, mem};

struct TestRelu {
    freq_count: usize,
    hidden_layers: Vec<Dense>,
    final_layer: Dense,
}

impl TestRelu {
    fn new(env: &mut Environment, freq_count: usize, hidden_units: &[usize]) -> Self {
        let mut hidden_layers = Vec::new();
        let mut prev_units = 4*freq_count;
        for hidden_units in hidden_units.iter().copied() {
            hidden_layers.push(Dense::new(env, prev_units, hidden_units));
            prev_units = hidden_units;
        }
        Self {
            freq_count,
            hidden_layers,
            final_layer: Dense::new(env, prev_units, 3),
        }
    }
}

impl Module for TestRelu {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        let mut x = position_encoding(input, self.freq_count);
        for layer in self.hidden_layers.iter() {
            x = x.apply(layer, ctx).leaky_relu(0.01);
        }
        x.apply(&self.final_layer, ctx)
    }
}

fn position_encoding<'s>(x: DualArray<'s>, freq_count: usize) -> DualArray<'s> {
    let scope = x.scope();

    let freq = scope.literal(2.0).pow(scope.coord(freq_count)) * 2.0 * PI;
    let phase = scope.coord(2).reshape([2, 1]) * 0.5 * PI;

    let shape = x.shape();
    let calc_shape = shape + Shape::from([1, 1]);
    let output_shape = {
        let mut tmp = shape;
        tmp[SignedIndex(-1)] *= 2 * freq_count;
        tmp
    };
    (x.reshape(calc_shape) * freq + phase)
        .sin()
        .reshape(output_shape)
}

fn main() {
    let (info, data) = image::stbi_load_from_reader(
        &mut fs::File::open("data/images/cat.png").unwrap(),
        stb::image::Channels::Rgb,
    )
    .unwrap();
    let width = info.width as usize;
    let height = info.height as usize;

    let mut env = Environment::new();

    let freq_count = 6;
    let module = TestRelu::new(&mut env, freq_count, &[256, 64, 64, 64]);

    let m = 4096;
    let x_var = env.static_parameter([m, 2], "x");
    let y_var = env.static_parameter([m, 3], "y");
    let loss_sum_var = env.static_parameter([1], "loss");
    let (train_graph, parameters, _optimizer) = {
        let scope = env.scope();

        let x = module.train(scope.parameter(&x_var));
        let loss = (x - &y_var).square().reduce_sum(-1, true).set_loss();
        scope.update_variable(&loss_sum_var, |loss_sum| {
            loss_sum + loss.reduce_sum(0, false)
        });

        let parameters = scope.trainable_parameters();
        let optimizer = Adam::new(&mut env, &scope, &parameters, 0.005, 0.9, 0.999, 1.0E-8);

        (scope.build_graph(), parameters, optimizer)
    };
    println!(
        "trainable parameters: {}",
        parameters
            .iter()
            .map(|var| var.shape().element_count())
            .sum::<usize>()
    );

    let pixel_count = height * width;
    let image_var = env.static_parameter([pixel_count, 3], "image");
    let test_graph = env.build_graph(|scope| {
        let u = (scope.coord(width) + 0.5) * (1.0 / (width as f32));
        let v = (scope.coord(height) + 0.5) * (1.0 / (height as f32));
        let x = scope
            .coord(2)
            .select_eq(0.0, u.reshape([1, width, 1]), v.reshape([height, 1, 1]))
            .reshape([pixel_count, 2]);
        let x = module.test(x);
        scope.write_variable(&image_var, x.value());
    });
    test_graph.write_dot_file(KernelDotOutput::Cluster, "test.dot");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    for var in parameters.iter() {
        env.reset_variable(var, &mut rng);
    }

    for epoch_index in 0..10000 {
        // generate batch from a random set of pixels
        let mut y_data: Vec<f32> = Vec::new();
        let mut w = env.writer(&x_var);
        for _ in 0..m {
            let x0 = rng.gen_range(0..width);
            let x1 = rng.gen_range(0..height);
            let x_data: [f32; 2] = [
                ((x0 as f32) + 0.5) * (1.0 / (width as f32)),
                ((x1 as f32) + 0.5) * (1.0 / (height as f32)),
            ];
            w.write_all(bytemuck::cast_slice(&x_data)).unwrap();
            let pixel_index = x1 * width + x0;
            for y in &data.as_slice()[3 * pixel_index..3 * (pixel_index + 1)] {
                y_data.push((*y as f32) / 255.0);
            }
        }
        mem::drop(w);
        env.writer(&y_var)
            .write_all(bytemuck::cast_slice(&y_data))
            .unwrap();

        // run training
        env.writer(&loss_sum_var).zero_fill();
        env.run(&train_graph, rng.next_u32());
        let done_counter = epoch_index + 1;
        if done_counter % 100 == 0 {
            let train_loss = env.read_variable_scalar(&loss_sum_var) / (m as f32);
            println!("epoch: {}, loss: {}", epoch_index, train_loss);

            env.run(&test_graph, rng.next_u32());
            let pixels: Vec<u8> = env
                .read_variable_to_vec(&image_var)
                .iter()
                .map(|&x| (x * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
                .collect();
            let name = format!("temp/image_{}.png", done_counter);
            stb::image_write::stbi_write_png(
                CString::new(name).unwrap().as_c_str(),
                info.width,
                info.height,
                3,
                &pixels,
                3 * info.width,
            )
            .unwrap();
        }
    }
}
