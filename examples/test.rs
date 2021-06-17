use descent::{
    array::{Array as ArrayOld, Size as SizeOld},
    prelude::*,
};
use rand::SeedableRng;
use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

fn read_be_u32(reader: &mut impl Read) -> u32 {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).unwrap();
    u32::from_be_bytes(bytes)
}

fn load_images(path: impl AsRef<Path>) -> ArrayOld {
    let mut reader = BufReader::new(File::open(path).unwrap());
    let magic = read_be_u32(&mut reader);
    assert_eq!(magic, 2051);
    let image_count = read_be_u32(&mut reader) as usize;
    let rows = read_be_u32(&mut reader) as usize;
    let cols = read_be_u32(&mut reader) as usize;
    let mut elements = Vec::new();
    let mut image = vec![0u8; rows * cols];
    for _ in 0..image_count {
        reader.read_exact(&mut image).unwrap();
        elements.extend(image.iter().map(|&c| (c as f32) / 255.0));
    }
    ArrayOld::from_elements(elements, [image_count, rows * cols])
}

fn load_labels(path: impl AsRef<Path>) -> ArrayOld {
    let mut reader = BufReader::new(File::open(path).unwrap());
    let magic = read_be_u32(&mut reader);
    assert_eq!(magic, 2049);
    let label_count = read_be_u32(&mut reader) as usize;
    let mut labels = vec![0u8; label_count];
    reader.read_exact(&mut labels).unwrap();
    let elements = labels.into_iter().map(|n| n as f32).collect();
    ArrayOld::from_elements(elements, [label_count, 1])
}

fn softmax_cross_entropy_loss<'builder>(
    z: Tensor<'builder>,
    y: Tensor<'builder>,
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
    env.test();

    let m = 1000;
    let x_var = env.variable([m, 28 * 28], "x");
    let y_var = env.variable([m, 1], "y");

    let g = GraphBuilder::new();
    let x = g.input(&x_var);
    let y = g.input(&y_var);

    // linear layer (no activation)
    g.next_colour();
    let w_var = env.variable([28 * 28, 10], "w");
    let b_var = env.variable([10], "b");

    let b_write: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    env.writer(&b_var)
        .write_all(bytemuck::cast_slice(&b_write))
        .unwrap();

    let mut b_read = [0f32; 10];
    env.reader(&b_var)
        .read_exact(bytemuck::cast_slice_mut(&mut b_read))
        .unwrap();

    assert_eq!(b_write, b_read);

    let w = g.input(&w_var);
    let b = g.input(&b_var);
    let z = x.matmul(w) + b;

    // loss function
    g.next_colour();
    let loss = softmax_cross_entropy_loss(z, y);

    // keep track of mean loss
    let mean_loss_var = env.variable([1], "loss");
    g.output(&mean_loss_var, loss.reduce_sum(0) / (m as f32));

    // gradient descent step
    g.next_colour();
    let alpha = 0.1 / (m as f32);
    g.output(&w_var, w.value() - alpha * w.grad());
    g.output(&b_var, b.value() - alpha * b.grad());

    // build a graph that will write the outputs
    let graph = g.build();

    let mut f = BufWriter::new(File::create("debug.dot").unwrap());
    graph.write_dot(&mut f).unwrap();

    graph.compile_kernel_source(0);

    if env::args().len() > 1 {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

        // load all training data
        let train_images = load_images("data/train-images-idx3-ubyte");
        let train_labels = load_labels("data/train-labels-idx1-ubyte");
        println!("{}, {}", train_images.size(), train_labels.size());

        // manually implement forward pass
        let x = &train_images;
        let w = ArrayOld::xavier_uniform([28 * 28, 10], &mut rng);
        let b = ArrayOld::zeros([1, 10]);

        let z = x * &w + &b;
        println!("{}", z.size());

        // compute loss

        // propagate backwards

        // tests
        let s: SizeOld = [4, 2, 3].into();
        let m = ArrayOld::from_elements((0..s.elements()).map(|n| n as f32).collect(), s);
        println!("{}", m);

        let t = ArrayOld::from_elements(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [3, 2]);
        println!("{}", t);
        println!("{}", &m * &t);
    }
}
