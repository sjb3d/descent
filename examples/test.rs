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

fn softmax_cross_entropy_loss<'g>(z: DualArray<'g>, y: impl IntoArray<'g>) -> DualArray<'g> {
    let (z, dz) = z.into_inner();
    let y = y.into_array(z.graph());

    // softmax
    let t = (z - z.reduce_max(-1)).exp();
    let p = t / t.reduce_sum(-1);

    // cross entropy loss
    let loss = y.select_eq(p.coord(-1), -p.log(), 0.0).reduce_sum(-1); // TODO: pick element of p using value of y
    let dloss = loss.clone_as_accumulator();

    // backprop (softmax with cross entropy directly)
    let n = p.shape()[SignedIndex(-1)];
    dz.accumulate((p - y.one_hot(n)) * dloss);

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

fn add_weight_decay_to_grad(graph: &Graph, variables: &[Variable], weight_decay: f32) {
    if weight_decay == 0.0 {
        return;
    }

    for var in variables.iter() {
        let (w, g) = graph.parameter(var).into_inner();
        g.accumulate(w * weight_decay);
    }
}

fn stochastic_gradient_descent_step(graph: &Graph, variables: &[Variable], learning_rate: f32) {
    graph.next_colour();

    for var in variables.iter() {
        let g = graph.parameter(var).grad();
        graph.update_variable(var, |theta| theta - learning_rate * g);
    }
}

fn adam_step(
    env: &mut Environment,
    graph: &Graph,
    variables: &[Variable],
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
        let m = graph.update_variable(&m_var, |m| m * beta1 + g * (1.0 - beta1));
        let v = graph.update_variable(&v_var, |v| v * beta2 + g * g * (1.0 - beta2));

        graph.update_variable(var, |theta| theta - alpha * m / (v.sqrt() + epsilon));
    }
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
            .with_layer(Layer::Conv2D(Conv2D::new(32, 3, 3).with_pad(1)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::MaxPool2D(MaxPool2D::new(2, 2)))
            .with_layer(Layer::Conv2D(Conv2D::new(64, 3, 3).with_pad(1)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::MaxPool2D(MaxPool2D::new(2, 2)))
            .with_layer(Layer::Flatten)
            .with_layer(Layer::Dropout(0.5))
            .with_layer(Layer::Linear(Linear::new(128)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::Linear(Linear::new(10))),
        TestNetwork::ConvBlurNet => network
            .with_layer(Layer::Conv2D(Conv2D::new(32, 3, 3).with_pad(1)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::MaxPool2D(MaxPool2D::new(2, 2).with_stride(1, 1)))
            .with_layer(Layer::Conv2D(
                Conv2D::new(1, 3, 3)
                    .with_pad(1)
                    .with_stride(2, 2)
                    .with_groups(32)
                    .with_blur(),
            ))
            .with_layer(Layer::Conv2D(Conv2D::new(64, 3, 3).with_pad(1)))
            .with_layer(Layer::LeakyRelu(0.01))
            .with_layer(Layer::MaxPool2D(MaxPool2D::new(2, 2).with_stride(1, 1)))
            .with_layer(Layer::Conv2D(
                Conv2D::new(1, 3, 3)
                    .with_pad(1)
                    .with_stride(2, 2)
                    .with_groups(64)
                    .with_blur(),
            ))
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

        // train using gradient of the loss (scaled for size of mini batch)
        loss.set_loss();
        let parameters = network.parameters();
        add_weight_decay_to_grad(&graph, parameters, app_params.weight_decay);
        match app_params.optimizer {
            Optimizer::Descent => stochastic_gradient_descent_step(&graph, parameters, 0.1),
            Optimizer::Adam => adam_step(&mut env, &graph, parameters, 0.002, 0.9, 0.999, 1.0E-8),
        }

        graph.build_schedule()
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
        .write_dot(
            KernelDotOutput::Cluster,
            &mut BufWriter::new(File::create("test.dot").unwrap()),
        )
        .unwrap();

    let norm_var = env.variable([1], "norm");
    let norm_graph = {
        let graph = env.graph();

        let mut sum = graph.literal(0.0);
        for var in network.parameters().iter() {
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

        env.run(&norm_graph);
        let norm: f32 = env.reader(&norm_var).read_value().unwrap();

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
