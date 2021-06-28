use descent::prelude::*;
use flate2::bufread::GzDecoder;
use rand::{
    distributions::{Distribution, Uniform},
    prelude::SliceRandom,
    Rng, SeedableRng,
};
use std::{
    convert::TryInto,
    fs::File,
    io::{self, prelude::*, BufReader, BufWriter},
    path::Path,
};

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
    variable_id: VariableId,
    bytes: &[u8],
    image_base: usize,
    image_count: usize,
) -> io::Result<()> {
    let ((image_end, rows, cols), bytes) = read_images_info(bytes);
    assert!(image_base + image_count <= image_end);
    let pixel_count = rows * cols;
    let mut w = env.writer(variable_id);
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
    variable_id: VariableId,
    bytes: &[u8],
    label_base: usize,
    label_count: usize,
) -> io::Result<()> {
    let (label_end, bytes) = read_labels_info(bytes);
    assert!(label_base + label_count <= label_end);
    let begin = label_base;
    let end = begin + label_count;
    let labels: Vec<f32> = bytes[begin..end].iter().map(|&n| n as f32).collect();
    let mut w = env.writer(variable_id);
    w.write_all(bytemuck::cast_slice(&labels))
}

fn xavier_uniform(
    env: &mut Environment,
    shape: impl Into<Shape>,
    name: &str,
    rng: &mut impl Rng,
) -> VariableId {
    let shape = shape.into();
    let variable_id = env.variable(shape.clone(), name);

    let mut writer = env.writer(variable_id);
    let a = (6.0 / (shape[0] as f32)).sqrt();
    let dist = Uniform::new(-a, a);
    for _ in 0..shape.element_count() {
        let x: f32 = dist.sample(rng);
        writer.write_all(bytemuck::bytes_of(&x)).unwrap();
    }

    variable_id
}

fn zeros(env: &mut Environment, shape: impl Into<Shape>, name: &str) -> VariableId {
    let shape = shape.into();
    let variable_id = env.variable(shape, name);

    let writer = env.writer(variable_id);
    writer.zero_fill();

    variable_id
}

fn softmax_cross_entropy_loss<'builder>(
    z: DualArray<'builder>,
    y: Array<'builder>,
) -> Array<'builder> {
    let (z, dz) = (z.value(), z.grad());

    // softmax
    let t = (z - z.reduce_max(-1)).exp();
    let p = t / t.reduce_sum(-1);

    // cross entropy loss
    let h = y.one_hot(10);
    let loss = -(h * p.log()).reduce_sum(-1); // TODO: pick element of p using value of y

    // backprop (softmax with cross entropy directly)
    dz.accumulate(p - h);

    loss
}

fn stochastic_gradient_descent_step(
    graph: &GraphBuilder,
    variable_ids: &[VariableId],
    mini_batch_size: usize,
    learning_rate: f32,
) {
    let alpha = learning_rate / (mini_batch_size as f32);
    for id in variable_ids.iter().copied() {
        let p = graph.input(id);
        graph.output(id, p.value() - alpha * p.grad());
    }
}

fn adam_step(
    env: &mut Environment,
    graph: &GraphBuilder,
    variable_ids: &[VariableId],
    mini_batch_size: usize,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    let t_id = env.variable([1], "t");
    env.writer(t_id).zero_fill();

    let t = graph.input(t_id).value() + 1.0;
    graph.output(t_id, t);

    let alpha = learning_rate * (1.0 - (graph.literal(beta2).log() * t).exp()).sqrt()
        / (1.0 - (graph.literal(beta1).log() * t).exp());

    for id in variable_ids.iter().copied() {
        let p = graph.input(id);
        let shape = p.shape();

        let m_id = env.variable(shape.clone(), "m");
        let v_id = env.variable(shape.clone(), "v");
        env.writer(m_id).zero_fill();
        env.writer(v_id).zero_fill();

        let m = graph.input(m_id).value();
        let v = graph.input(v_id).value();
        let g = p.grad();

        let rcp = 1.0 / (mini_batch_size as f32);
        let m = m * beta1 + g * ((1.0 - beta1) * rcp);
        let v = v * beta2 + g * g * ((1.0 - beta2) * rcp * rcp);
        graph.output(m_id, m);
        graph.output(v_id, v);

        graph.output(id, p.value() - alpha * m / (v.sqrt() + epsilon));
    }
}

fn main() {
    let epoch_count = 40;
    let m = 1000; // mini-batch size

    let mut env = Environment::new();

    let x_id = env.variable([m, 28 * 28], "x");
    let y_id = env.variable([m, 1], "y");

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let w1_id = xavier_uniform(&mut env, [28 * 28, 300], "w1", &mut rng);
    let b1_id = zeros(&mut env, [300], "b1");
    let leakiness = 0.01;

    let w2_id = xavier_uniform(&mut env, [300, 10], "w2", &mut rng);
    let b2_id = zeros(&mut env, [10], "b2");

    let loss_sum_id = env.variable([1], "loss");
    let accuracy_sum_id = env.variable([1], "accuracy");

    let train_graph = {
        let graph = env.builder();
        let x = graph.input(x_id);
        let y = graph.input(y_id).value();

        // linear layer (leaky relu)
        graph.next_colour();
        let w1 = graph.input(w1_id);
        let b1 = graph.input(b1_id);
        let z1 = x.matmul(w1) + b1;
        let a1 = z1.leaky_relu(leakiness);

        // linear layer (no activation)
        graph.next_colour();
        let w2 = graph.input(w2_id);
        let b2 = graph.input(b2_id);
        let z2 = a1.matmul(w2) + b2;

        // loss function
        graph.next_colour();
        let _loss = softmax_cross_entropy_loss(z2, y);

        // update parameters from gradients
        graph.next_colour();
        let variable_ids = [w1_id, b1_id, w2_id, b2_id];
        if false {
            stochastic_gradient_descent_step(&graph, &variable_ids, m, 0.1);
        } else {
            adam_step(
                &mut env,
                &graph,
                &variable_ids,
                m,
                0.001,
                0.9,
                0.999,
                1.0E-8,
            );
        }

        graph.build()
    };
    let mut f = BufWriter::new(File::create("train.dot").unwrap());
    train_graph.write_dot(&mut f).unwrap();

    let test_graph = {
        let graph = env.builder();
        let x = graph.input(x_id);
        let y = graph.input(y_id).value();

        // linear layer (leaky relu)
        graph.next_colour();
        let w1 = graph.input(w1_id);
        let b1 = graph.input(b1_id);
        let z1 = x.matmul(w1) + b1;
        let a1 = z1.leaky_relu(leakiness);

        // linear layer (no activation)
        graph.next_colour();
        let w2 = graph.input(w2_id);
        let b2 = graph.input(b2_id);
        let z2 = a1.matmul(w2) + b2;

        // accumulate loss  (into variable)
        graph.next_colour();
        let loss = softmax_cross_entropy_loss(z2, y);
        let loss_sum = graph.input(loss_sum_id).value() + loss.reduce_sum(0);
        graph.output(loss_sum_id, loss_sum);

        // accumulate accuracy (into variable)
        graph.next_colour();
        let pred = z2.value().argmax(-1);
        let accuracy = pred.select_eq(y, graph.literal(1.0), graph.literal(0.0));
        let accuracy_sum = graph.input(accuracy_sum_id).value() + accuracy.reduce_sum(0);
        graph.output(accuracy_sum_id, accuracy_sum);

        graph.build()
    };
    let mut f = BufWriter::new(File::create("test.dot").unwrap());
    test_graph.write_dot(&mut f).unwrap();

    // load training data
    let train_images = load_gz_bytes("data/train-images-idx3-ubyte.gz").unwrap();
    let train_labels = load_gz_bytes("data/train-labels-idx1-ubyte.gz").unwrap();
    let ((train_image_count, train_image_rows, train_image_cols), _) =
        read_images_info(&train_images);
    let (train_label_count, _) = read_labels_info(&train_labels);
    assert_eq!(train_image_count, train_label_count);
    assert_eq!(train_image_count % m, 0);
    assert_eq!(train_image_rows, 28);
    assert_eq!(train_image_cols, 28);

    // load test data
    let test_images = load_gz_bytes("data/t10k-images-idx3-ubyte.gz").unwrap();
    let test_labels = load_gz_bytes("data/t10k-labels-idx1-ubyte.gz").unwrap();
    let ((test_image_count, test_image_rows, test_image_cols), _) = read_images_info(&test_images);
    let (test_label_count, _) = read_labels_info(&test_labels);
    assert_eq!(test_image_count, test_label_count);
    assert_eq!(test_image_count % m, 0);
    assert_eq!(test_image_rows, 28);
    assert_eq!(test_image_cols, 28);

    // run epochs
    for epoch_index in 0..epoch_count {
        // loop over training mini-batches
        let mut batch_indices: Vec<_> = (0..(train_image_count / m)).collect();
        batch_indices.shuffle(&mut rng);
        for batch_index in batch_indices.iter().copied() {
            let first_index = batch_index * m;
            unpack_images(&mut env, x_id, &train_images, first_index, m).unwrap();
            unpack_labels(&mut env, y_id, &train_labels, first_index, m).unwrap();
            env.run(&train_graph);
        }

        // loop over test mini-batches to evaluate loss and accuracy
        env.writer(loss_sum_id).zero_fill();
        env.writer(accuracy_sum_id).zero_fill();
        for batch_index in 0..(test_image_count / m) {
            let first_index = batch_index * m;
            unpack_images(&mut env, x_id, &test_images, first_index, m).unwrap();
            unpack_labels(&mut env, y_id, &test_labels, first_index, m).unwrap();
            env.run(&test_graph);
        }
        let mut total_loss = 0f32;
        let mut total_accuracy = 0f32;
        env.reader(loss_sum_id)
            .read_exact(bytemuck::bytes_of_mut(&mut total_loss))
            .unwrap();
        env.reader(accuracy_sum_id)
            .read_exact(bytemuck::bytes_of_mut(&mut total_accuracy))
            .unwrap();
        println!(
            "epoch: {}, loss: {}, accuracy: {}",
            epoch_index,
            total_loss / (test_image_count as f32),
            total_accuracy / (test_image_count as f32)
        );
    }
}
