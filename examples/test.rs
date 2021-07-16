use bytemuck::{Pod, Zeroable};
use descent::{layer::*, prelude::*};
use flate2::bufread::GzDecoder;
use rand::{prelude::SliceRandom, SeedableRng};
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
    image_base: usize,
    image_count: usize,
) -> io::Result<()> {
    let ((image_end, rows, cols), bytes) = read_images_info(bytes);
    assert!(image_base + image_count <= image_end);
    let pixel_count = rows * cols;
    let mut w = env.writer(variable);
    let mut image = Vec::<f32>::with_capacity(pixel_count);
    for image_index in 0..image_count {
        let begin = (image_base + image_index) * pixel_count;
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
    label_base: usize,
    label_count: usize,
) -> io::Result<()> {
    let (label_end, bytes) = read_labels_info(bytes);
    assert!(label_base + label_count <= label_end);
    let begin = label_base;
    let end = begin + label_count;
    let labels: Vec<f32> = bytes[begin..end].iter().map(|&n| n as f32).collect();
    let mut w = env.writer(variable);
    w.write_all(bytemuck::cast_slice(&labels))
}

fn softmax_cross_entropy_loss<'g>(z: DualArray<'g>, y: impl IntoArray<'g>) -> DualArray<'g> {
    let (z, dz) = z.into_inner();
    let y = y.into_array(z.graph());

    // softmax
    let t = (z - z.reduce_max(-1)).exp();
    let p = t / t.reduce_sum(-1);

    // cross entropy loss
    let h = y.one_hot(10);
    let loss = -(h * p.log()).reduce_sum(-1); // TODO: pick element of p using value of y
    let dloss = loss.graph().accumulator(loss.shape());

    // backprop (softmax with cross entropy directly)
    dz.accumulate((p - h) * dloss);

    DualArray::new(loss, dloss)
}

fn softmax_cross_entropy_accuracy<'g>(z: DualArray<'g>, y: impl IntoArray<'g>) -> Array<'g> {
    let z = z.value();
    let y = y.into_array(z.graph());

    // index of most likely choice
    let pred = z.argmax(-1);

    // set to 1 when correct, 0 when incorrect
    pred.select_eq(y, 1.0, 0.0)
}

fn stochastic_gradient_descent_step(
    graph: &Graph,
    variables: &[Variable],
    mini_batch_size: usize,
    learning_rate: f32,
) {
    graph.next_colour();

    let alpha = learning_rate / (mini_batch_size as f32);
    for var in variables.iter() {
        let g = graph.parameter(var).grad();
        graph.update_variable(var, |theta| theta - alpha * g);
    }
}

fn adam_step(
    env: &mut Environment,
    graph: &Graph,
    variables: &[Variable],
    mini_batch_size: usize,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    graph.next_colour();

    let t_var = env.variable([1], "t");
    env.writer(&t_var).zero_fill();

    let t = graph.update_variable(&t_var, |t| t + 1.0);
    let alpha =
        learning_rate * (1.0 - (beta2.ln() * t).exp()).sqrt() / (1.0 - (beta1.ln() * t).exp());

    for var in variables.iter() {
        let shape = var.shape();
        let m_var = env.variable(shape.clone(), "m");
        let v_var = env.variable(shape.clone(), "v");
        env.writer(&m_var).zero_fill();
        env.writer(&v_var).zero_fill();

        let g = graph.parameter(var).grad();
        let rcp = 1.0 / (mini_batch_size as f32);

        let m = graph.update_variable(&m_var, |m| m * beta1 + g * ((1.0 - beta1) * rcp));
        let v = graph.update_variable(&v_var, |v| v * beta2 + g * g * ((1.0 - beta2) * rcp * rcp));

        graph.update_variable(var, |theta| theta - alpha * m / (v.sqrt() + epsilon));
    }
}

#[derive(Debug, EnumString, EnumVariantNames)]
#[strum(serialize_all = "kebab_case")]
enum TestNetwork {
    Linear,
    Hidden300,
    ConvNet,
}

#[derive(Debug, StructOpt)]
#[structopt(no_version)]
struct AppParams {
    #[structopt(possible_values=&TestNetwork::VARIANTS, default_value="hidden300")]
    network: TestNetwork,

    #[structopt(short, long, global = true, default_value = "1000")]
    mini_batch_size: usize,

    #[structopt(short, long, global = true, default_value = "40")]
    epoch_count: usize,
}

fn main() {
    let app_params = AppParams::from_args();

    let mut env = Environment::new();

    let network = Network::builder();
    let network = match app_params.network {
        TestNetwork::Linear => network
            .with_layer(Layer::Flatten)
            .with_layer(Layer::Linear(Linear::new(10))),
        TestNetwork::Hidden300 => network
            .with_layer(Layer::Flatten)
            .with_layer(Layer::Linear(Linear::new(300)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::Linear(Linear::new(10))),
        TestNetwork::ConvNet => network
            .with_layer(Layer::Conv2D(Conv2D::new(32, 3, 3, 1)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::MaxPool2D(MaxPool2D::new(2, 2)))
            .with_layer(Layer::Conv2D(Conv2D::new(64, 3, 3, 1)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::MaxPool2D(MaxPool2D::new(2, 2)))
            .with_layer(Layer::Flatten)
            .with_layer(Layer::Dropout(0.5))
            .with_layer(Layer::Linear(Linear::new(128)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::Linear(Linear::new(10))),
    };
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let network = network.finish([28, 28, 1], &mut env, &mut rng);

    let m = app_params.mini_batch_size;
    let x_var = env.variable([m, 28, 28, 1], "x");
    let y_var = env.variable([m, 1], "y");

    let loss_sum_var = env.variable([1], "loss");
    let accuracy_sum_var = env.variable([1], "accuracy");

    let train_graph = {
        let graph = env.graph();

        // emit the graph for the network
        let x = network.train(graph.parameter(&x_var));
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

        // train using gradient of the loss
        loss.set_loss();
        let parameters = network.parameters();
        if false {
            stochastic_gradient_descent_step(&graph, parameters, m, 0.1);
        } else {
            adam_step(&mut env, &graph, parameters, m, 0.001, 0.9, 0.999, 1.0E-8);
        }

        graph.build_schedule()
    };
    train_graph
        .write_dot(&mut BufWriter::new(File::create("train.dot").unwrap()))
        .unwrap();

    let test_graph = {
        let graph = env.graph();

        // emit the graph for the network
        let x = network.test(graph.parameter(&x_var));
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
        .write_dot(&mut BufWriter::new(File::create("test.dot").unwrap()))
        .unwrap();

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
    for epoch_index in 0..app_params.epoch_count {
        // loop over training mini-batches
        let mut batch_indices: Vec<_> = (0..(train_image_count / m)).collect();
        batch_indices.shuffle(&mut rng);
        env.writer(&loss_sum_var).zero_fill();
        env.writer(&accuracy_sum_var).zero_fill();
        for batch_index in batch_indices.iter().copied() {
            let first_index = batch_index * m;
            unpack_images(&mut env, &x_var, &train_images, first_index, m).unwrap();
            unpack_labels(&mut env, &y_var, &train_labels, first_index, m).unwrap();
            env.run(&train_graph);
        }
        env.print_timings();
        let train_loss: f32 = env.reader(&loss_sum_var).read_value().unwrap();
        let train_accuracy: f32 = env.reader(&accuracy_sum_var).read_value().unwrap();

        // loop over test mini-batches to evaluate loss and accuracy
        env.writer(&loss_sum_var).zero_fill();
        env.writer(&accuracy_sum_var).zero_fill();
        for batch_index in 0..(test_image_count / m) {
            let first_index = batch_index * m;
            unpack_images(&mut env, &x_var, &test_images, first_index, m).unwrap();
            unpack_labels(&mut env, &y_var, &test_labels, first_index, m).unwrap();
            env.run(&test_graph);
        }
        env.print_timings();
        let test_loss: f32 = env.reader(&loss_sum_var).read_value().unwrap();
        let test_accuracy: f32 = env.reader(&accuracy_sum_var).read_value().unwrap();

        println!(
            "epoch: {}, loss: {}/{}, accuracy: {}/{}",
            epoch_index,
            train_loss / (train_image_count as f32),
            test_loss / (test_image_count as f32),
            train_accuracy / (train_image_count as f32),
            test_accuracy / (test_image_count as f32)
        );
    }
}
