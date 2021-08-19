use descent::{layer::*, loss::*, optimizer::*, prelude::*};
use flate2::bufread::GzDecoder;
use rand::{prelude::SliceRandom, RngCore, SeedableRng};
use std::{
    convert::TryInto,
    fs::File,
    io::{self, prelude::*, BufReader, BufWriter},
    path::{Path, PathBuf},
};
use structopt::StructOpt;
use strum::{EnumString, EnumVariantNames, VariantNames};

fn load_gz_bytes(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let reader = BufReader::new(File::open(path).unwrap());
    let mut decoder = GzDecoder::new(reader);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn read_be_u32(bytes: &[u8]) -> (u32, &[u8]) {
    let (prefix, suffix) = bytes.split_at(4);
    (u32::from_be_bytes(prefix.try_into().unwrap()), suffix)
}

fn read_images_info(bytes: &[u8]) -> ((usize, usize, usize), &[u8]) {
    let (magic, bytes) = read_be_u32(bytes);
    assert_eq!(magic, 2051);
    let (images, bytes) = read_be_u32(bytes);
    let (rows, bytes) = read_be_u32(bytes);
    let (cols, bytes) = read_be_u32(bytes);
    ((images as usize, rows as usize, cols as usize), bytes)
}

fn read_labels_info(bytes: &[u8]) -> (usize, &[u8]) {
    let (magic, bytes) = read_be_u32(bytes);
    assert_eq!(magic, 2049);
    let (items, bytes) = read_be_u32(bytes);
    ((items as usize), bytes)
}

fn unpack_images(
    env: &mut Environment,
    variable: &Variable,
    bytes: &[u8],
    indices: &[usize],
) -> io::Result<()> {
    let ((_, rows, cols), bytes) = read_images_info(bytes);
    let pixel_count = rows * cols;
    let mut w = env.writer(variable);
    let mut image = Vec::<f32>::with_capacity(pixel_count);
    for index in indices.iter().copied() {
        let begin = index * pixel_count;
        let end = begin + pixel_count;
        image.clear();
        image.extend(bytes[begin..end].iter().map(|&c| (c as f32) / 255.0));
        w.write_all(bytemuck::cast_slice(&image))?;
    }
    Ok(())
}

fn unpack_labels(
    env: &mut Environment,
    variable: &Variable,
    bytes: &[u8],
    indices: &[usize],
) -> io::Result<()> {
    let (_, bytes) = read_labels_info(bytes);
    let labels: Vec<f32> = indices.iter().map(|&index| bytes[index] as f32).collect();
    let mut w = env.writer(variable);
    w.write_all(bytemuck::cast_slice(&labels))
}

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
enum NetworkType {
    Linear,
    SingleLayer,
    ConvNet,
    ConvBlurNet,
}

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
enum OptimizerType {
    Descent,
    Adam,
}

#[derive(Debug, StructOpt)]
#[structopt(
    no_version,
    name = "fashion_mnist",
    about = "Example networks to train using the Fashion MNIST dataset."
)]
struct AppParams {
    #[structopt(possible_values=&NetworkType::VARIANTS, default_value="single-layer")]
    network: NetworkType,

    #[structopt(short, long, possible_values=&OptimizerType::VARIANTS, default_value="adam")]
    optimizer: OptimizerType,

    #[structopt(short, long, default_value = "1.0E-8")]
    weight_decay: f32,

    #[structopt(short, long, default_value = "1000")]
    mini_batch_size: usize,

    #[structopt(short, long, default_value = "40")]
    epoch_count: usize,

    #[structopt(short, long, default_value = "1")]
    trial_count: usize,

    #[structopt(long)]
    output_dot_files: bool,

    #[structopt(long)]
    show_timings: bool,

    #[structopt(long)]
    quiet: bool,

    #[structopt(long)]
    csv_file_name: Option<PathBuf>,
}

struct Linear {
    fc: Dense,
}

impl Linear {
    fn new(env: &mut Environment) -> Self {
        Self {
            fc: Dense::builder(28 * 28, 10).build(env),
        }
    }
}

impl Module for Linear {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        input.flatten().apply(&self.fc, ctx)
    }
}

struct SingleLayer {
    fc1: Dense,
    fc2: Dense,
}

impl SingleLayer {
    fn new(env: &mut Environment) -> Self {
        let hidden_units = 300;
        Self {
            fc1: Dense::builder(28 * 28, hidden_units).build(env),
            fc2: Dense::builder(hidden_units, 10).build(env),
        }
    }
}

impl Module for SingleLayer {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        input
            .flatten()
            .apply(&self.fc1, ctx)
            .leaky_relu(0.01)
            .apply(&self.fc2, ctx)
    }
}

struct ConvNet {
    conv1: Conv2D,
    pool1: Box<dyn Module>,
    conv2: Conv2D,
    pool2: Box<dyn Module>,
    fc1: Dense,
    fc2: Dense,
}

impl ConvNet {
    fn new(env: &mut Environment, use_blur_pool: bool) -> Self {
        let c1 = 16;
        let c2 = 32;
        let hidden = 128;
        Self {
            conv1: Conv2D::builder(1, c1, 3, 3).with_pad(1).build(env),
            pool1: if use_blur_pool {
                Box::new(MaxBlurPool2D::new(env, c1))
            } else {
                Box::new(MaxPool2D::default())
            },
            conv2: Conv2D::builder(c1, c2, 3, 3)
                .with_pad(1)
                .with_groups(2)
                .build(env),
            pool2: if use_blur_pool {
                Box::new(MaxBlurPool2D::new(env, c2))
            } else {
                Box::new(MaxPool2D::default())
            },
            fc1: Dense::builder(7 * 7 * c2, hidden).build(env),
            fc2: Dense::builder(hidden, 10).build(env),
        }
    }
}

impl Module for ConvNet {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        input
            .apply(&self.conv1, ctx)
            .leaky_relu(0.01)
            .apply(self.pool1.as_ref(), ctx)
            .apply(&self.conv2, ctx)
            .leaky_relu(0.01)
            .apply(self.pool2.as_ref(), ctx)
            .flatten()
            .apply(&Dropout::new(0.5), ctx)
            .apply(&self.fc1, ctx)
            .leaky_relu(0.01)
            .apply(&self.fc2, ctx)
    }
}

fn main() {
    let app_params = AppParams::from_args();

    let mut env = Environment::new();
    let module: Box<dyn Module> = {
        let env = &mut env;
        match app_params.network {
            NetworkType::Linear => Box::new(Linear::new(env)),
            NetworkType::SingleLayer => Box::new(SingleLayer::new(env)),
            NetworkType::ConvNet => Box::new(ConvNet::new(env, false)),
            NetworkType::ConvBlurNet => Box::new(ConvNet::new(env, true)),
        }
    };

    let m = app_params.mini_batch_size;
    let x_var = env.static_parameter([m, 28, 28, 1], "x");
    let y_var = env.static_parameter([m, 1], "y");

    let learning_rate_scale_var = env.static_parameter([1], "lr_scale");
    let loss_sum_var = env.static_parameter([1], "loss");
    let accuracy_sum_var = env.static_parameter([1], "accuracy");

    // build a graph for training, collect the trainable parameters
    let (train_graph, parameters, optimizer) = {
        let scope = env.scope();

        // emit the ops for the network
        let x = module.train(scope.parameter(&x_var));
        let loss = softmax_cross_entropy_loss(x, &y_var).set_loss();
        let accuracy = softmax_cross_entropy_accuracy(x, &y_var);

        // update sum of loss and accuracy
        scope.update_variable(&loss_sum_var, |loss_sum| {
            loss_sum + loss.reduce_sum(0, false)
        });
        scope.update_variable(&accuracy_sum_var, |accuracy_sum| {
            accuracy_sum + accuracy.reduce_sum(0, false)
        });

        // train using gradient of the loss (scaled for size of mini batch)
        let learning_rate_scale = scope.read_variable(&learning_rate_scale_var);
        let parameters = scope.trainable_parameters();
        add_weight_decay_to_grad(&scope, &parameters, app_params.weight_decay);
        let optimizer: Box<dyn Optimizer> = match app_params.optimizer {
            OptimizerType::Descent => Box::new(StochasticGradientDescent::new(
                &mut env,
                &scope,
                &parameters,
                0.1 * learning_rate_scale,
                0.9,
            )),
            OptimizerType::Adam => Box::new(Adam::new(
                &mut env,
                &scope,
                &parameters,
                0.005 * learning_rate_scale,
                0.9,
                0.999,
                1.0E-8,
            )),
        };

        (scope.build_graph(), parameters, optimizer)
    };
    println!(
        "trainable parameters: {}",
        parameters
            .iter()
            .map(|var| var.shape().element_count())
            .sum::<usize>()
    );

    // build a graph to evaluate the test set (keeps parameters unchanged)
    let test_graph = env.build_graph(|scope| {
        // emit the ops for the network
        let x = module.test(scope.parameter(&x_var));
        let loss = softmax_cross_entropy_loss(x, &y_var).set_loss();
        let accuracy = softmax_cross_entropy_accuracy(x, &y_var);

        // update sum of loss and accuracy
        scope.update_variable(&loss_sum_var, |loss_sum| {
            loss_sum + loss.reduce_sum(0, false)
        });
        scope.update_variable(&accuracy_sum_var, |accuracy_sum| {
            accuracy_sum + accuracy.reduce_sum(0, false)
        });
    });

    // build a graph to evaluate the L2 norm of training parameters (to check weight decay)
    let norm_var = env.static_parameter([1], "norm");
    let norm_graph = env.build_graph(|scope| {
        let mut sum = scope.literal(0.0).value();
        for var in parameters.iter() {
            let x = scope.read_variable(&var);
            let x = x.reshape([x.shape().element_count()]);
            let x = x * x * 0.5;
            sum = sum + x.reduce_sum(0, true);
        }
        scope.write_variable(&norm_var, sum);
    });

    // write graphs out to disk if necessary
    if app_params.output_dot_files {
        train_graph.write_dot_file(KernelDotOutput::Cluster, "train.dot");
        train_graph.write_dot_file(KernelDotOutput::None, "train_s.dot");
        train_graph.write_dot_file(KernelDotOutput::Color, "train_k.dot");
        test_graph.write_dot_file(KernelDotOutput::Cluster, "test.dot");
    }

    // load training data
    let train_images = load_gz_bytes("data/fashion_mnist/train-images-idx3-ubyte.gz").unwrap();
    let train_labels = load_gz_bytes("data/fashion_mnist/train-labels-idx1-ubyte.gz").unwrap();
    let ((train_image_count, train_image_rows, train_image_cols), _) =
        read_images_info(&train_images);
    let (train_label_count, _) = read_labels_info(&train_labels);
    assert_eq!(train_image_count, train_label_count);
    assert_eq!(train_image_count % m, 0);
    assert_eq!(train_image_rows, 28);
    assert_eq!(train_image_cols, 28);

    // load test data
    let test_images = load_gz_bytes("data/fashion_mnist/t10k-images-idx3-ubyte.gz").unwrap();
    let test_labels = load_gz_bytes("data/fashion_mnist/t10k-labels-idx1-ubyte.gz").unwrap();
    let ((test_image_count, test_image_rows, test_image_cols), _) = read_images_info(&test_images);
    let (test_label_count, _) = read_labels_info(&test_labels);
    assert_eq!(test_image_count, test_label_count);
    assert_eq!(test_image_count % m, 0);
    assert_eq!(test_image_rows, 28);
    assert_eq!(test_image_cols, 28);

    // maybe writing stats to file
    let mut stats_w = app_params
        .csv_file_name
        .map(|path| BufWriter::new(File::create(path).unwrap()));

    // attempt to train 5 times with different random seeds
    for trial_index in 0..app_params.trial_count {
        // reset all trainable variables and optimizer state
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(trial_index as u64);
        for var in parameters.iter() {
            env.reset_variable(var, &mut rng);
        }
        optimizer.reset_state(&mut env);

        // run epochs
        let mut indices = Vec::new();
        for epoch_index in 0..app_params.epoch_count {
            // update learning for this epoch (halve every 40 epochs)
            let learning_rate_scale = 0.5f32.powf((epoch_index as f32) / 40.0);
            env.writer(&learning_rate_scale_var)
                .write_all(bytemuck::bytes_of(&learning_rate_scale))
                .unwrap();

            // loop over training mini-batches
            env.writer(&loss_sum_var).zero_fill();
            env.writer(&accuracy_sum_var).zero_fill();
            indices.clear();
            indices.extend(0..train_image_count);
            indices.shuffle(&mut rng);
            for batch_indices in indices.chunks(m) {
                unpack_images(&mut env, &x_var, &train_images, batch_indices).unwrap();
                unpack_labels(&mut env, &y_var, &train_labels, batch_indices).unwrap();
                env.run(&train_graph, rng.next_u32());
            }
            if app_params.show_timings && epoch_index < 2 {
                env.print_timings("training");
            }
            let train_loss = env.read_variable_scalar(&loss_sum_var) / (train_image_count as f32);
            let train_accuracy =
                env.read_variable_scalar(&accuracy_sum_var) / (train_image_count as f32);

            // loop over test mini-batches to evaluate loss and accuracy
            env.writer(&loss_sum_var).zero_fill();
            env.writer(&accuracy_sum_var).zero_fill();
            indices.clear();
            indices.extend(0..test_image_count);
            for batch_indices in indices.chunks(m) {
                unpack_images(&mut env, &x_var, &test_images, batch_indices).unwrap();
                unpack_labels(&mut env, &y_var, &test_labels, batch_indices).unwrap();
                env.run(&test_graph, rng.next_u32());
            }
            if app_params.show_timings && epoch_index < 2 {
                env.print_timings("testing");
            }
            let test_loss = env.read_variable_scalar(&loss_sum_var) / (test_image_count as f32);
            let test_accuracy =
                env.read_variable_scalar(&accuracy_sum_var) / (test_image_count as f32);

            // compute the norm of all the parameters
            env.run(&norm_graph, rng.next_u32());
            let norm = env.read_variable_scalar(&norm_var);

            let done_counter = epoch_index + 1;
            if !app_params.quiet {
                println!(
                    "epoch: {}, loss: {}/{}, accuracy: {}/{}, w_norm: {}",
                    done_counter,
                    train_loss,
                    test_loss,
                    train_accuracy,
                    test_accuracy,
                    norm.sqrt()
                );
            }
            if let Some(w) = stats_w.as_mut() {
                if epoch_index == 0 {
                    writeln!(
                        w,
                        "# epoch, train_loss, test_loss, train_accuracy, test_accuracy"
                    )
                    .unwrap();
                }
                writeln!(
                    w,
                    "{}, {}, {}, {}, {}",
                    done_counter, train_loss, test_loss, train_accuracy, test_accuracy
                )
                .unwrap();
                if done_counter == app_params.epoch_count {
                    writeln!(w).unwrap();
                }
            }
        }
    }
}
