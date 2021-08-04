use crate::common::*;

pub fn softmax_cross_entropy_loss<'g>(z: DualArray<'g>, y: impl IntoArray<'g>) -> DualArray<'g> {
    let (z, dz) = z.next_colour().into_inner();
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

pub fn softmax_cross_entropy_accuracy<'g>(z: DualArray<'g>, y: impl IntoArray<'g>) -> Array<'g> {
    let z = z.value();
    let y = y.into_array(z.graph());

    // index of most likely choice
    let pred = z.argmax(-1);

    // set to 1 when correct, 0 when incorrect
    pred.select_eq(y, 1.0, 0.0)
}
