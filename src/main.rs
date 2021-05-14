mod array;

use array::*;
use rand::SeedableRng;
use std::{
    fs::File,
    io::{BufReader, Read},
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
