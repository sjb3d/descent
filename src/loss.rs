use crate::common::*;

#[allow(clippy::many_single_char_names)]
pub fn softmax_cross_entropy_loss<'s>(z: DualArray<'s>, y: impl IntoArray<'s>) -> DualArray<'s> {
    let (z, dz) = z.next_colour().into_inner();
    let y = y.into_array(z.scope());

    // softmax
    let t = (z - z.reduce_max(-1, true)).exp();
    let p = t / t.reduce_sum(-1, true);

    // cross entropy loss
    let (loss, dloss) = y
        .select_eq(p.coord(-1), -p.log(), 0.0)
        .reduce_sum(-1, true)
        .with_empty_grad(); // TODO: pick element of p using value of y

    // backprop (softmax with cross entropy directly)
    let n = p.shape()[SignedIndex(-1)];
    dz.accumulate((p - y.one_hot(n)) * dloss);

    (loss, dloss).into()
}

pub fn softmax_cross_entropy_accuracy<'s>(z: DualArray<'s>, y: impl IntoArray<'s>) -> Array<'s> {
    let z = z.value();
    let y = y.into_array(z.scope());

    // index of most likely choice
    let pred = z.argmax(-1, true);

    // set to 1 when correct, 0 when incorrect
    pred.select_eq(y, 1.0, 0.0)
}
