use descent::prelude::*;
use flate2::bufread::GzDecoder;
use rand::{
    distributions::{Distribution, Uniform},
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
    y: DualArray<'builder>,
) -> Array<'builder> {
    let (z, dz) = (z.value(), z.grad());
    let y = y.value();

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

fn main() {
    let mut env = Environment::new();

    let m = 1000;
    let x_id = env.variable([m, 28 * 28], "x");
    let y_id = env.variable([m, 1], "y");

    let g = env.builder();
    let x = g.input(x_id);
    let y = g.input(y_id);

    // linear layer (no activation)
    g.next_colour();
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    let w_id = xavier_uniform(&mut env, [28 * 28, 10], "w", &mut rng);
    let b_id = zeros(&mut env, [10], "b");

    let w = g.input(w_id);
    let b = g.input(b_id);
    let z = x.matmul(w) + b;

    // loss function
    g.next_colour();
    let loss = softmax_cross_entropy_loss(z, y);

    // keep track of mean loss
    let mean_loss_id = env.variable([1], "loss");
    g.output(mean_loss_id, loss.reduce_sum(0) / (m as f32));

    // gradient descent step
    g.next_colour();
    let alpha = 0.1 / (m as f32);
    g.output(w_id, w.value() - alpha * w.grad());
    g.output(b_id, b.value() - alpha * b.grad());

    // build a graph that will write the outputs
    let graph = g.build();

    let mut f = BufWriter::new(File::create("debug.dot").unwrap());
    graph.write_dot(&mut f).unwrap();

    // load training data
    let train_images = load_gz_bytes("data/train-images-idx3-ubyte.gz").unwrap();
    let train_labels = load_gz_bytes("data/train-labels-idx1-ubyte.gz").unwrap();
    let ((train_image_count, train_image_rows, train_image_cols), _) =
        read_images_info(&train_images);
    let (train_label_count, _) = read_labels_info(&train_labels);
    assert_eq!(train_image_count, train_label_count);
    assert_eq!(train_image_rows, 28);
    assert_eq!(train_image_cols, 28);

    // loop over batches
    let batch_size = m as usize;
    for batch_index in 0..(train_image_count / batch_size) {
        let first_index = batch_index * batch_size;
        unpack_images(&mut env, x_id, &train_images, first_index, batch_size).unwrap();
        unpack_labels(&mut env, y_id, &train_labels, first_index, batch_size).unwrap();

        env.run(&graph);

        let mut mean_loss = 0f32;
        env.reader(mean_loss_id)
            .read_exact(bytemuck::bytes_of_mut(&mut mean_loss))
            .unwrap();
        println!("mean loss: {}", mean_loss);
    }
}
