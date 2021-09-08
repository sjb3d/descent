use descent::{loss::*, module::*, optimizer::*, prelude::*};
use rand::{prelude::SliceRandom, RngCore, SeedableRng};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::{prelude::*, BufReader},
    iter,
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum JsonLabel {
    Positive,
    Negative,
    Neutral,
    Mixed,
}

#[derive(Debug, Deserialize)]
struct JsonRecord {
    sentence: String,
    gold_label: Option<JsonLabel>,
}

struct Record {
    sentence: Vec<u16>,
    label: u8,
}

fn vocab_iter<'s>(s: &'s str) -> impl Iterator<Item = String> + 's {
    s.split_whitespace().filter_map(|word| {
        let entry: String = word
            .chars()
            .filter(|c| c.is_alphabetic())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if entry.is_empty() {
            None
        } else {
            Some(entry)
        }
    })
}

fn main() {
    const MAX_WORD_COUNT: usize = 32;
    let f =
        File::open("data/dynasent/dynasent-v1.1/dynasent-v1.1-round01-yelp-train.jsonl").unwrap();
    let records: Vec<_> = BufReader::new(f)
        .lines()
        .filter_map(|line| {
            let record: JsonRecord = serde_json::from_str(&line.unwrap()).unwrap();
            if matches!(record.gold_label, None | Some(JsonLabel::Mixed))
                || vocab_iter(&record.sentence).count() > MAX_WORD_COUNT
            {
                None
            } else {
                Some(record)
            }
        })
        .collect();

    let mut word_map = HashMap::new();
    for record in records.iter() {
        for word in vocab_iter(&record.sentence) {
            *word_map.entry(word).or_insert(0) += 1;
        }
    }
    let retain_limit = records.len() / 5;
    let mut next_index = 1;
    word_map.retain(|_key, value| {
        if *value < retain_limit {
            *value = next_index;
            next_index += 1;
            true
        } else {
            false
        }
    });
    let vocab_size = next_index;
    println!("vocab size: {}", vocab_size);

    let records: Vec<_> = records
        .iter()
        .map(|record| {
            let sentence: Vec<_> = vocab_iter(&record.sentence)
                .filter_map(|word| word_map.get(&word).map(|index| *index as u16))
                .collect();
            let label = match record.gold_label {
                Some(JsonLabel::Negative) => 0,
                Some(JsonLabel::Neutral) => 1,
                Some(JsonLabel::Positive) => 2,
                _ => unreachable!(),
            };
            Record { sentence, label }
        })
        .collect();
    println!("records: {}", records.len());
    println!(
        "sentence length max: {}, avg: {}",
        records
            .iter()
            .map(|record| record.sentence.len())
            .max()
            .unwrap_or(0),
        records
            .iter()
            .map(|record| record.sentence.len())
            .sum::<usize>()
            / records.len()
    );

    let mut env = Environment::new();

    let embedding_size = 128;
    let lstm_size = 64;
    let lstm = LSTMCell::new(&mut env, embedding_size, lstm_size);
    let fc = Dense::builder(lstm_size, 3).build(&mut env);

    let m = 256;
    let x_param = env.static_parameter([m, MAX_WORD_COUNT, 1], "x");
    let y_param = env.static_parameter([m, 1], "y");

    let embedding = env.trainable_parameter(
        [vocab_size, embedding_size],
        "em",
        Initializer::RandUniform(1.0),
    );
    let loss_sum_param = env.static_parameter([1], "loss");
    let accuracy_sum_param = env.static_parameter([1], "accuracy");

    let (train_graph, parameters, _optimizer) = {
        let scope = env.scope();

        let x: DualArray = scope
            .parameter(&x_param)
            .value()
            .one_hot(vocab_size)
            .with_empty_grad()
            .into();
        let x = x
            .reshape([m * MAX_WORD_COUNT, vocab_size])
            .matmul(&embedding)
            .reshape([m, MAX_WORD_COUNT, embedding_size]);
        let x = lstm.train(x);
        let x = fc.train(x);
        let loss = softmax_cross_entropy_loss(x, &y_param).set_loss();
        let accuracy = softmax_cross_entropy_accuracy(x, &y_param);

        scope.update_parameter_value(&loss_sum_param, |loss_sum| {
            loss_sum + loss.reduce_sum(0, false)
        });
        scope.update_parameter_value(&accuracy_sum_param, |accuracy_sum| {
            accuracy_sum + accuracy.reduce_sum(0, false)
        });

        let parameters = scope.trainable_parameters();
        let optimizer = Adam::new(&mut env, &scope, &parameters, 0.002, 0.9, 0.999, 1.0E-8);

        (scope.build_graph(), parameters, optimizer)
    };

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);
    for param in parameters.iter() {
        env.reset_parameter(param, &mut rng);
    }

    let mini_batch_per_epoch = records.len() / m;
    for epoch_index in 0..40 {
        env.writer(&loss_sum_param).zero_fill();
        env.writer(&accuracy_sum_param).zero_fill();

        for _ in 0..mini_batch_per_epoch {
            let mut xw = env.writer(&x_param);
            let mut labels = Vec::new();
            for record in records.choose_multiple(&mut rng, m) {
                let sentence: Vec<_> = record
                    .sentence
                    .iter()
                    .copied()
                    .chain(iter::repeat(0))
                    .take(MAX_WORD_COUNT)
                    .map(|w| w as f32)
                    .collect();
                xw.write(bytemuck::cast_slice(&sentence)).unwrap();
                labels.push(record.label as f32);
            }
            xw.zero_fill();
            let mut yw = env.writer(&y_param);
            yw.write(bytemuck::cast_slice(&labels)).unwrap();
            yw.zero_fill();

            env.run(&train_graph, rng.next_u32());
        }

        if epoch_index < 2 {
            env.print_timings("training");
        }

        let train_count = m * mini_batch_per_epoch;
        let train_loss = env.read_parameter_scalar(&loss_sum_param) / (train_count as f32);
        let train_accuracy = env.read_parameter_scalar(&accuracy_sum_param) / (train_count as f32);

        let done_counter = epoch_index + 1;
        println!(
            "epoch: {}, loss: {}, accuracy: {}",
            done_counter, train_loss, train_accuracy
        );
    }
}
