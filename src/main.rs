mod array;
mod graph;

use array::*;
use graph::*;
use rand::SeedableRng;
use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Read},
    path::Path,
};

fn read_be_u32(reader: &mut impl Read) -> u32 {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes).unwrap();
    u32::from_be_bytes(bytes)
}

fn load_images(path: impl AsRef<Path>) -> Array {
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
    Array::from_elements(elements, [image_count, rows * cols])
}

fn load_labels(path: impl AsRef<Path>) -> Array {
    let mut reader = BufReader::new(File::open(path).unwrap());
    let magic = read_be_u32(&mut reader);
    assert_eq!(magic, 2049);
    let label_count = read_be_u32(&mut reader) as usize;
    let mut labels = vec![0u8; label_count];
    reader.read_exact(&mut labels).unwrap();
    let elements = labels.into_iter().map(|n| n as f32).collect();
    Array::from_elements(elements, [label_count, 1])
}

fn main() {
    let g = GraphBuilder::new();

    let m = 1000;
    let x = g.variable([m, 28 * 28], "x");
    let y = g.variable([m, 1], "y");

    let w = g.variable([28 * 28, 10], "w");
    let b = g.variable([10], "b");

    // linear layer (no activation)
    let z = x.matmul(w) + b;

    // softmax
    let t = (z - z.reduce_max(-1)).exp();
    let p = t / t.reduce_sum(-1);

    // cross entropy loss (mean over batch)
    let h = y.one_hot(10);
    let loss = -(h * p.log()).reduce_sum(-1); // TODO: pick element of p using value of y
    let _mean_loss = loss.reduce_sum(0) / (m as f32);

    // backprop
    let dz = (p - h) / (m as f32); // softmax with cross entropy
    let dw = x.transpose().matmul(dz);
    let _dx = dz.matmul(w.transpose());
    let db = dz.reduce_sum(0);

    // gradient descent step
    let alpha = 0.1;
    let _w = w - alpha * dw;
    let _b = b - alpha * db;

    // codegen steps:
    // * lower into compute graph:
    //   * subgraphs for per-element ops (with multiple outputs)
    //   * merge single-input per-element ops onto previous pass?
    //   * transpose as argument modifier
    //   * literal as argument modifier

    let mut f = BufWriter::new(File::create("debug.dot").unwrap());
    g.build().write_dot(&mut f).unwrap();

    if env::args().len() > 1 {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

        // load all training data
        let train_images = load_images("data/train-images-idx3-ubyte");
        let train_labels = load_labels("data/train-labels-idx1-ubyte");
        println!("{}, {}", train_images.size(), train_labels.size());

        // manually implement forward pass
        let x = &train_images;
        let w = Array::xavier_uniform([28 * 28, 10], &mut rng);
        let b = Array::zeros([1, 10]);

        let z = x * &w + &b;
        println!("{}", z.size());

        // compute loss

        // propagate backwards

        // tests
        let s: Size = [4, 2, 3].into();
        let m = Array::from_elements((0..s.elements()).map(|n| n as f32).collect(), s);
        println!("{}", m);

        let t = Array::from_elements(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [3, 2]);
        println!("{}", t);
        println!("{}", &m * &t);
    }
}
