use bytemuck::{Pod, Zeroable};
use descent::{layer::*, loss::*, optimizer::*, prelude::*};
use flate2::bufread::GzDecoder;
use rand::{prelude::SliceRandom, Rng, SeedableRng};
use std::{
    convert::TryInto,
    fs::File,
    io::{self, prelude::*, BufReader, BufWriter},
    path::Path,
};
use structopt::StructOpt;
use strum::{EnumString, EnumVariantNames, VariantNames};

trait ReadValue<T> {
    fn read_value(&mut self) -> io::Result<T>;
}

impl<R, T> ReadValue<T> for R
where
    R: io::Read,
    T: Pod + Zeroable,
{
    fn read_value(&mut self) -> io::Result<T> {
        let mut res = T::zeroed();
        self.read_exact(bytemuck::bytes_of_mut(&mut res))?;
        Ok(res)
    }
}

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
enum TestNetwork {
    Linear,
    Hidden300,
    ConvNet,
    ConvBlurNet,
}

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
enum Optimizer {
    Descent,
    Adam,
}

#[derive(Debug, StructOpt)]
#[structopt(no_version)]
struct AppParams {
    #[structopt(possible_values=&TestNetwork::VARIANTS, default_value="hidden300")]
    network: TestNetwork,

    #[structopt(short, long, possible_values=&Optimizer::VARIANTS, default_value="adam")]
    optimizer: Optimizer,

    #[structopt(short, long, default_value = "1000")]
    mini_batch_size: usize,

    #[structopt(short, long, default_value = "40")]
    epoch_count: usize,

    #[structopt(short, long, default_value = "0.0005")]
    weight_decay: f32,
}

struct TestLinear {
    fc: Dense,
}

impl TestLinear {
    fn new(env: &mut Environment, rng: &mut impl Rng) -> Self {
        Self {
            fc: Dense::new(env, rng, 28 * 28, 10),
        }
    }
}

impl Module for TestLinear {
    fn eval<'g>(&self, input: DualArray<'g>, ctx: &EvalContext) -> DualArray<'g> {
        input.flatten().eval_with(&self.fc, ctx)
    }
}

struct TestHidden300 {
    fc1: Dense,
    fc2: Dense,
}

impl TestHidden300 {
    fn new(env: &mut Environment, rng: &mut impl Rng) -> Self {
        Self {
            fc1: Dense::new(env, rng, 28 * 28, 300),
            fc2: Dense::new(env, rng, 300, 10),
        }
    }
}

impl Module for TestHidden300 {
    fn eval<'g>(&self, input: DualArray<'g>, ctx: &EvalContext) -> DualArray<'g> {
        input
            .flatten()
            .eval_with(&self.fc1, ctx)
            .leaky_relu(0.01)
            .eval_with(&self.fc2, ctx)
    }
}

struct TestConvNet {
    conv1: Conv2D,
    pool1: Box<dyn Module>,
    conv2: Conv2D,
    pool2: Box<dyn Module>,
    fc1: Dense,
    fc2: Dense,
}

impl TestConvNet {
    fn new(env: &mut Environment, rng: &mut impl Rng, use_blur_pool: bool) -> Self {
        Self {
            conv1: Conv2D::builder(1, 32, 3, 3).with_pad(1).build(env, rng),
            pool1: if use_blur_pool {
                Box::new(MaxBlurPool2D::new(env, rng, 32))
            } else {
                Box::new(MaxPool2D::new())
            },
            conv2: Conv2D::builder(32, 64, 3, 3).with_pad(1).build(env, rng),
            pool2: if use_blur_pool {
                Box::new(MaxBlurPool2D::new(env, rng, 64))
            } else {
                Box::new(MaxPool2D::new())
            },
            fc1: Dense::new(env, rng, 7 * 7 * 64, 128),
            fc2: Dense::new(env, rng, 128, 10),
        }
    }
}

impl Module for TestConvNet {
    fn eval<'g>(&self, input: DualArray<'g>, ctx: &EvalContext) -> DualArray<'g> {
        input
            .eval_with(&self.conv1, ctx)
            .leaky_relu(0.01)
            .eval_with(self.pool1.as_ref(), ctx)
            .eval_with(&self.conv2, ctx)
            .leaky_relu(0.01)
            .eval_with(self.pool2.as_ref(), ctx)
            .flatten()
            .eval_with(&Dropout::new(0.5), ctx)
            .eval_with(&self.fc1, ctx)
            .leaky_relu(0.01)
            .eval_with(&self.fc2, ctx)
    }
}

fn main() {
    let app_params = AppParams::from_args();

    let mut env = Environment::new();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let module: Box<dyn Module> = {
        let env = &mut env;
        let rng = &mut rng;
        match app_params.network {
            TestNetwork::Linear => Box::new(TestLinear::new(env, rng)),
            TestNetwork::Hidden300 => Box::new(TestHidden300::new(env, rng)),
            TestNetwork::ConvNet => Box::new(TestConvNet::new(env, rng, false)),
            TestNetwork::ConvBlurNet => Box::new(TestConvNet::new(env, rng, true)),
        }
    };

    let m = app_params.mini_batch_size;
    let x_var = env.static_parameter([m, 28, 28, 1], "x");
    let y_var = env.static_parameter([m, 1], "y");

    let loss_sum_var = env.static_parameter([1], "loss");
    let accuracy_sum_var = env.static_parameter([1], "accuracy");

    // build a graph for training, collect the trainable parameters
    let (train_graph, parameters) = {
        let graph = env.graph();

        // emit the graph for the network
        let x = module.train(graph.parameter(&x_var));
        let loss = softmax_cross_entropy_loss(x, &y_var);

        // accumulate loss (into variable)
        graph.update_variable(&loss_sum_var, |loss_sum| {
            loss_sum + loss.value().reduce_sum(0)
        });

        // accumulate accuracy (into variable)
        let accuracy = softmax_cross_entropy_accuracy(x, &y_var);
        graph.update_variable(&accuracy_sum_var, |accuracy_sum| {
            accuracy_sum + accuracy.reduce_sum(0)
        });

        // train using gradient of the loss (scaled for size of mini batch)
        loss.set_loss();
        let parameters = graph.trainable_parameters();
        add_weight_decay_to_grad(&graph, &parameters, app_params.weight_decay);
        match app_params.optimizer {
            Optimizer::Descent => stochastic_gradient_descent_step(&graph, &parameters, 0.1),
            Optimizer::Adam => adam_step(&mut env, &graph, &parameters, 0.002, 0.9, 0.999, 1.0E-8),
        }

        (graph.build_schedule(), parameters)
    };
    train_graph
        .write_dot(
            KernelDotOutput::None,
            &mut BufWriter::new(File::create("train_s.dot").unwrap()),
        )
        .unwrap();
    train_graph
        .write_dot(
            KernelDotOutput::Color,
            &mut BufWriter::new(File::create("train_k.dot").unwrap()),
        )
        .unwrap();

    // build a graph to evaluate the test set (keeps parameters unchanged)
    let test_graph = {
        let graph = env.graph();

        // emit the graph for the network
        let x = module.test(graph.parameter(&x_var));
        let loss = softmax_cross_entropy_loss(x, &y_var);

        // accumulate loss (into variable)
        graph.update_variable(&loss_sum_var, |loss_sum| {
            loss_sum + loss.value().reduce_sum(0)
        });

        // accumulate accuracy (into variable)
        let accuracy = softmax_cross_entropy_accuracy(x, &y_var);
        graph.update_variable(&accuracy_sum_var, |accuracy_sum| {
            accuracy_sum + accuracy.reduce_sum(0)
        });

        graph.build_schedule()
    };
    test_graph
        .write_dot(
            KernelDotOutput::Cluster,
            &mut BufWriter::new(File::create("test.dot").unwrap()),
        )
        .unwrap();

    // build a graph to evaluate the L2 norm of training parameters (to check weight decay)
    let norm_var = env.static_parameter([1], "norm");
    let norm_graph = {
        let graph = env.graph();

        let mut sum = graph.literal(0.0);
        for var in parameters.iter() {
            let x = graph.read_variable(&var);
            let x = x.reshape([x.shape().element_count()]);
            let x = x * x * 0.5;
            sum = sum + x.reduce_sum(0);
        }
        graph.write_variable(&norm_var, sum);

        graph.build_schedule()
    };

    // load training data
    let train_images = load_gz_bytes("data/fashion/train-images-idx3-ubyte.gz").unwrap();
    let train_labels = load_gz_bytes("data/fashion/train-labels-idx1-ubyte.gz").unwrap();
    let ((train_image_count, train_image_rows, train_image_cols), _) =
        read_images_info(&train_images);
    let (train_label_count, _) = read_labels_info(&train_labels);
    assert_eq!(train_image_count, train_label_count);
    assert_eq!(train_image_count % m, 0);
    assert_eq!(train_image_rows, 28);
    assert_eq!(train_image_cols, 28);

    // load test data
    let test_images = load_gz_bytes("data/fashion/t10k-images-idx3-ubyte.gz").unwrap();
    let test_labels = load_gz_bytes("data/fashion/t10k-labels-idx1-ubyte.gz").unwrap();
    let ((test_image_count, test_image_rows, test_image_cols), _) = read_images_info(&test_images);
    let (test_label_count, _) = read_labels_info(&test_labels);
    assert_eq!(test_image_count, test_label_count);
    assert_eq!(test_image_count % m, 0);
    assert_eq!(test_image_rows, 28);
    assert_eq!(test_image_cols, 28);

    // run epochs
    let mut indices = Vec::new();
    for epoch_index in 0..app_params.epoch_count {
        // loop over training mini-batches
        indices.clear();
        indices.extend(0..train_image_count);
        indices.shuffle(&mut rng);
        env.writer(&loss_sum_var).zero_fill();
        env.writer(&accuracy_sum_var).zero_fill();
        for batch_indices in indices.chunks(m) {
            unpack_images(&mut env, &x_var, &train_images, batch_indices).unwrap();
            unpack_labels(&mut env, &y_var, &train_labels, batch_indices).unwrap();
            env.run(&train_graph);
        }
        if epoch_index < 2 {
            env.print_timings();
        }
        let train_loss: f32 = env.reader(&loss_sum_var).read_value().unwrap();
        let train_accuracy: f32 = env.reader(&accuracy_sum_var).read_value().unwrap();

        // loop over test mini-batches to evaluate loss and accuracy
        env.writer(&loss_sum_var).zero_fill();
        env.writer(&accuracy_sum_var).zero_fill();
        indices.clear();
        indices.extend(0..test_image_count);
        for batch_indices in indices.chunks(m) {
            unpack_images(&mut env, &x_var, &test_images, batch_indices).unwrap();
            unpack_labels(&mut env, &y_var, &test_labels, batch_indices).unwrap();
            env.run(&test_graph);
        }
        if epoch_index < 2 {
            env.print_timings();
        }
        let test_loss: f32 = env.reader(&loss_sum_var).read_value().unwrap();
        let test_accuracy: f32 = env.reader(&accuracy_sum_var).read_value().unwrap();

        // compute the norm of all the parameters
        env.run(&norm_graph);
        let norm: f32 = env.reader(&norm_var).read_value().unwrap();

        println!(
            "epoch: {}, loss: {}/{}, accuracy: {}/{}, w_norm: {}",
            epoch_index,
            train_loss / (train_image_count as f32),
            test_loss / (test_image_count as f32),
            train_accuracy / (train_image_count as f32),
            test_accuracy / (test_image_count as f32),
            norm.sqrt()
        );
    }
}
