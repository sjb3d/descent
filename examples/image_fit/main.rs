use descent::{module::*, optimizer::*, prelude::*};
use rand::{Rng, RngCore, SeedableRng};
use stb::image;
use std::{
    f32::consts::PI,
    ffi::CString,
    fs::File,
    io::{BufWriter, Write},
    mem,
    path::PathBuf,
};
use structopt::StructOpt;
use strum::{EnumString, EnumVariantNames, VariantNames};

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
enum NetworkType {
    Relu,
    ReluPE,
    Siren,
    MultiHash,
}

#[derive(Debug, StructOpt)]
#[structopt(
    no_version,
    name = "image_fit",
    about = "Example networks to fit a single image."
)]
struct AppParams {
    #[structopt(possible_values=&NetworkType::VARIANTS, default_value="siren")]
    network: NetworkType,

    #[structopt(long)]
    show_timings: bool,

    #[structopt(long)]
    quiet: bool,

    #[structopt(long)]
    csv_file_name: Option<PathBuf>,

    #[structopt(long)]
    image_prefix: Option<String>,

    #[structopt(long)]
    output_all_images: bool,
}

struct Relu {
    freq_count: usize,
    hidden_layers: Vec<Dense>,
    final_layer: Dense,
}

impl Relu {
    fn new(env: &mut Environment, freq_count: usize, hidden_units: &[usize]) -> Self {
        let mut hidden_layers = Vec::new();
        let mut prev_units = if freq_count == 0 { 2 } else { 4 * freq_count };
        for hidden_units in hidden_units.iter().copied() {
            hidden_layers.push(Dense::builder(prev_units, hidden_units).build(env));
            prev_units = hidden_units;
        }
        Self {
            freq_count,
            hidden_layers,
            final_layer: Dense::builder(prev_units, 3).build(env),
        }
    }
}

impl Module for Relu {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        let mut x = input;
        if self.freq_count != 0 {
            x = positional_encoding(input, self.freq_count);
        }
        for layer in self.hidden_layers.iter() {
            x = x.apply(layer, ctx).leaky_relu(0.01);
        }
        x.apply(&self.final_layer, ctx)
    }
}

struct Siren {
    hidden_layers: Vec<Dense>,
    final_layer: Dense,
}

impl Siren {
    fn new(env: &mut Environment, hidden_units: &[usize]) -> Self {
        let mut hidden_layers = Vec::new();
        let mut prev_units = 2;
        for (index, hidden_units) in hidden_units.iter().copied().enumerate() {
            hidden_layers.push(
                Dense::builder(prev_units, hidden_units)
                    .with_w_initializer(Initializer::for_siren(prev_units, index == 0))
                    .with_b_initializer(Initializer::RandUniform(1.0))
                    .build(env),
            );
            prev_units = hidden_units;
        }
        Self {
            hidden_layers,
            final_layer: Dense::builder(prev_units, 3).build(env),
        }
    }
}

impl Module for Siren {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        let mut x = input;
        for layer in self.hidden_layers.iter() {
            x = x.apply(layer, ctx).sin();
        }
        x.apply(&self.final_layer, ctx)
    }
}

struct HashGrid {
    grid_size: usize,
    stride: usize,
    t: Parameter,
}

impl HashGrid {
    fn new(
        env: &mut Environment,
        grid_size: usize,
        entry_count: usize,
        values_per_entry: usize,
    ) -> Self {
        let grid_point_count = grid_size + 1;
        let max_entry_count = grid_point_count * grid_point_count;
        let entry_count = entry_count.min(max_entry_count);
        let stride = if entry_count == max_entry_count {
            grid_point_count
        } else {
            1526263 // large prime
        };
        let t = env.trainable_parameter(
            [entry_count, values_per_entry],
            "t",
            Initializer::RandUniform(1.0E-4),
        );
        Self {
            grid_size,
            stride,
            t,
        }
    }
}

impl Module for HashGrid {
    fn eval<'s>(&self, input: DualArray<'s>, _ctx: &EvalContext) -> DualArray<'s> {
        let scope = input.scope();
        let (x, _dx) = input.into_inner();

        let (t, dt) = scope.parameter(&self.t).into_inner();
        let entry_count = t.shape()[0];
        let stride = self.stride as u32;

        let cf = (x * 0.5 + 0.5) * (self.grid_size as f32);
        let c = cf.into_u32();
        let f = cf - c.into_f32();

        let c0 = c.lock_axis(-1, 0, false);
        let c1 = c.lock_axis(-1, 1, false);
        let f0 = f.lock_axis(-1, 0, true);
        let f1 = f.lock_axis(-1, 1, true);

        let ia = ((c0 + 0) ^ (c1 * stride + 0)) % (entry_count as u32);
        let ib = ((c0 + 1) ^ (c1 * stride + 0)) % (entry_count as u32);
        let ic = ((c0 + 0) ^ (c1 * stride + stride)) % (entry_count as u32);
        let id = ((c0 + 1) ^ (c1 * stride + stride)) % (entry_count as u32);

        let ta = t.gather(-2, ia);
        let tb = t.gather(-2, ib);
        let tc = t.gather(-2, ic);
        let td = t.gather(-2, id);
        let g0 = 1.0 - f0;
        let g1 = 1.0 - f1;
        let wa = g0 * g1;
        let wb = f0 * g1;
        let wc = g0 * f1;
        let wd = f0 * f1;

        let (y, dy) = (ta * wa + tb * wb + tc * wc + td * wd).with_empty_grad();

        dt.accumulate(
            scope
                .literal(0.0)
                .value()
                .broadcast(dt.shape())
                .scatter_add(dy * wa, -2, ia)
                .scatter_add(dy * wb, -2, ib)
                .scatter_add(dy * wc, -2, ic)
                .scatter_add(dy * wd, -2, id),
        );

        (y, dy).into()
    }
}

struct MultiHashGrid {
    grids: Vec<HashGrid>,
    hidden_layers: Vec<Dense>,
    final_layer: Dense,
}

impl MultiHashGrid {
    fn new(
        env: &mut Environment,
        min_grid_size: usize,
        max_grid_size: usize,
        level_count: usize,
        entry_count: usize,
        hidden_units: &[usize],
    ) -> Self {
        let values_per_entry = 2;
        let mut grids = Vec::new();
        let b = (((max_grid_size as f32).ln() - (min_grid_size as f32).ln())
            / ((level_count - 1) as f32))
            .exp();
        println!("b = {}", b);
        for level_index in 0..level_count {
            let grid_size = ((min_grid_size as f32) * b.powi(level_index as i32)) as usize;
            grids.push(HashGrid::new(env, grid_size, entry_count, values_per_entry));
        }
        let mut hidden_layers = Vec::new();
        let mut prev_units = grids.len() * values_per_entry;
        for hidden_units in hidden_units.iter().copied() {
            hidden_layers.push(Dense::builder(prev_units, hidden_units).build(env));
            prev_units = hidden_units;
        }
        Self {
            grids,
            hidden_layers,
            final_layer: Dense::builder(prev_units, 3).build(env),
        }
    }
}

impl Module for MultiHashGrid {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        let mut x = self
            .grids
            .iter()
            .map(|grid| grid.eval(input, ctx))
            .reduce(|a, b| a.concat(b, -1))
            .unwrap();
        for layer in self.hidden_layers.iter() {
            x = layer.eval(x, ctx).leaky_relu(0.01);
        }
        self.final_layer.eval(x, ctx)
    }
}

fn positional_encoding<'s>(x: DualArray<'s>, freq_count: usize) -> DualArray<'s> {
    let scope = x.scope();

    let freq = scope.literal(2.0).pow(scope.coord(freq_count)) * PI;
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
        &mut File::open("data/images/cat.jpg").unwrap(),
        stb::image::Channels::Rgb,
    )
    .unwrap();
    let width = info.width as usize;
    let height = info.height as usize;

    let mut env = Environment::new();

    let app_params = AppParams::from_args();
    let pe_freq_count = 8;
    let module: Box<dyn Module> = {
        let env = &mut env;
        let hidden_units = &[256, 128, 64, 32];
        match app_params.network {
            NetworkType::Relu => Box::new(Relu::new(env, 0, hidden_units)),
            NetworkType::ReluPE => Box::new(Relu::new(env, pe_freq_count, hidden_units)),
            NetworkType::Siren => Box::new(Siren::new(env, hidden_units)),
            NetworkType::MultiHash => {
                Box::new(MultiHashGrid::new(env, 2, 512, 10, 4096, &[64, 64]))
            }
        }
    };

    let m = 1 << 14;
    let x_param = env.static_parameter([m, 2], "x");
    let y_param = env.static_parameter([m, 3], "y");
    let learning_rate_scale_param = env.static_parameter([1], "lr_scale");
    let loss_sum_param = env.static_parameter([1], "loss");
    let (train_graph, parameters, _optimizer) = {
        let scope = env.scope();

        let x = module.train(scope.parameter(&x_param));
        let loss = (x - &y_param).square().reduce_sum(-1, true).set_loss();
        scope.update_parameter_value(&loss_sum_param, |loss_sum| {
            loss_sum + loss.reduce_sum(0, false)
        });

        let learning_rate_scale = scope.parameter_value(&learning_rate_scale_param);
        let parameters = scope.trainable_parameters();
        let optimizer = Adam::new(
            &mut env,
            &scope,
            &parameters,
            0.02 * learning_rate_scale,
            0.9,
            0.99,
            1.0E-8,
        );

        (scope.build_graph(), parameters, optimizer)
    };
    println!(
        "trainable parameters: {}",
        parameters
            .iter()
            .map(|param| param.shape().element_count())
            .sum::<usize>()
    );

    let pixel_count = height * width;
    let image_param = env.static_parameter([pixel_count, 3], "image");
    let test_graph = env.build_graph(|scope| {
        let u = (scope.coord(width) + 0.5) * (2.0 / (width as f32)) - 1.0;
        let v = (scope.coord(height) + 0.5) * (2.0 / (height as f32)) - 1.0;
        let x = scope
            .coord(2)
            .select_eq(0.0, u.reshape([1, width, 1]), v.reshape([height, 1, 1]))
            .reshape([pixel_count, 2]);
        let x = module.test(x);
        scope.write_parameter_value(&image_param, x.value());
    });

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    for param in parameters.iter() {
        env.reset_parameter(param, &mut rng);
    }

    let mut stats_w = app_params
        .csv_file_name
        .map(|path| BufWriter::new(File::create(path).unwrap()));

    let epoch_count = 200;
    for epoch_index in 0..epoch_count {
        let epoch_t = (epoch_index as f32) + 0.5;
        let learning_rate_scale = (epoch_t / 10.0).min(1.0) * 0.5f32.powf(epoch_t / 40.0);
        env.writer(&learning_rate_scale_param)
            .write_all(bytemuck::bytes_of(&learning_rate_scale))
            .unwrap();

        // loop over batches to roughly cover the whole image
        env.writer(&loss_sum_param).zero_fill();
        let mini_batch_count = (width * height) / m;
        for _ in 0..mini_batch_count {
            // generate batch from a random set of pixels
            let mut y_data: Vec<f32> = Vec::new();
            let mut w = env.writer(&x_param);
            for _ in 0..m {
                let x0 = rng.gen_range(0..width);
                let x1 = rng.gen_range(0..height);
                let x_data: [f32; 2] = [
                    ((x0 as f32) + 0.5) * (2.0 / (width as f32)) - 1.0,
                    ((x1 as f32) + 0.5) * (2.0 / (height as f32)) - 1.0,
                ];
                w.write_all(bytemuck::cast_slice(&x_data)).unwrap();
                let pixel_index = x1 * width + x0;
                for y in &data.as_slice()[3 * pixel_index..3 * (pixel_index + 1)] {
                    y_data.push((*y as f32) / 255.0);
                }
            }
            mem::drop(w);
            env.writer(&y_param)
                .write_all(bytemuck::cast_slice(&y_data))
                .unwrap();

            // run training
            env.run(&train_graph, rng.next_u32());
        }
        if app_params.show_timings && epoch_index < 2 {
            env.print_timings("training")
        }

        let done_counter = epoch_index + 1;
        let train_loss = env.read_parameter_scalar(&loss_sum_param) / (m as f32);
        if !app_params.quiet {
            println!(
                "epoch: {}, lr_scale: {}, loss: {}",
                done_counter, learning_rate_scale, train_loss
            );
        }
        if let Some(w) = stats_w.as_mut() {
            if epoch_index == 0 {
                writeln!(w, "# epoch, loss").unwrap();
            }
            writeln!(w, "{}, {}", done_counter, train_loss).unwrap();
            if done_counter == epoch_count {
                writeln!(w).unwrap();
            }
        }
        if let Some(image_prefix) = app_params.image_prefix.as_ref() {
            if app_params.output_all_images || done_counter == epoch_count {
                env.run(&test_graph, rng.next_u32());
                let pixels: Vec<u8> = env
                    .read_parameter_to_vec(&image_param)
                    .iter()
                    .map(|&x| (x * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
                    .collect();
                let name = format!("{}_{}.jpg", image_prefix, done_counter);
                stb::image_write::stbi_write_jpg(
                    CString::new(name).unwrap().as_c_str(),
                    info.width,
                    info.height,
                    3,
                    &pixels,
                    90,
                )
                .unwrap();
            }
        }
    }
}
