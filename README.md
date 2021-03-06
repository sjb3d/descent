# descent

Toy library for neural networks in Rust using Vulkan compute shaders.

## Features

- Multi-dimensional arrays backed by Vulkan device memory
- Use Rust syntax to build a computation graph, run as Vulkan compute shaders
  - Supports vector arithmetic and per-element sin/cos/exp/log/etc
  - 1D reduction, 2D matrix multiply, 2D convolutions and 2D max pool supported
  - Concatenation, gather loads and scatter adds
  - Softmax cross entropy loss
  - Ops are fused into larger compute shaders where possible (to reduce bandwidth cost)
  - Implements broadcasts/padding/windowing/reshapes as views (zero copy) where possible
- Supports one level of automatic derivatives for back-propagation
- Some example optimisers:
  - Stochastic gradient descent (with momentum)
  - Adam
- Optional higher-level API of neural network building blocks
  - Can generate different code for train vs test (e.g. dropout only affects training)
- Deterministic results (except for scatter add which currently uses float atomics...)

## Example Network

The top-level API of neural network building blocks can be used to compactly describe multi-layer networks.  Here is a small convolutional neural network with dropout and (leaky) ReLU activation using this API:

```rust
struct ConvNet {
    conv1: Conv2D,
    conv2: Conv2D,
    fc1: Dense,
    fc2: Dense,
}

impl ConvNet {
    fn new(env: &mut Environment) -> Self {
        // create and store parameters for layers that require them
        let c1 = 16;
        let c2 = 32;
        let hidden = 128;
        Self {
            conv1: Conv2D::builder(1, c1, 3, 3).with_pad(1).build(env),
            conv2: Conv2D::builder(c1, c2, 3, 3)
                .with_pad(1)
                .with_groups(2)
                .build(env),
            fc1: Dense::builder(7 * 7 * c2, hidden).build(env),
            fc2: Dense::builder(hidden, 10).build(env),
        }
    }
}

impl Module for ConvNet {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        // generates ops for the value (forwards) and gradient (backwards) through the layers
        input
            .apply(&self.conv1, ctx)
            .leaky_relu(0.01)
            .max_pool2d((2, 2), (2, 2))
            .apply(&self.conv2, ctx)
            .leaky_relu(0.01)
            .max_pool2d((2, 2), (2, 2))
            .flatten()
            .apply(&Dropout::new(0.5), ctx)
            .apply(&self.fc1, ctx)
            .leaky_relu(0.01)
            .apply(&self.fc2, ctx)
    }
}
```

See the [fashion_mnist example](examples/fashion_mnist) for more networks using this API.

## Examples

Please follow the link in the name of each example to show a more detailed description of each one.

Name | Description
--- | ---
[array_api](examples/array_api) | Demonstrates the low-level `Array` API for building computation graphs.  See the README for more details.
[fashion_mnist](examples/fashion_mnist) | Trains a few different network types on the Fashion-MNIST dataset.  Demonstrates the use of anti-aliasing during max pooling for improved accuracy.  See the README for a comparison of network performance.
[image_fit](examples/image_fit) | Overfits a few different network types to a single RGB image.  Compares ReLU with positional encoding to a SIREN network. _Update: now also compares to a multi-level hash encoding._

## Dependencies

The following crates have been very useful to develop this project:

- [petgraph](https://github.com/petgraph/petgraph): used for all graph data structures
- [slotmap](https://github.com/orlp/slotmap): storage with stable keys
- [shaderc](https://github.com/google/shaderc-rs): interface to GLSL compiler to generate SPIR-V for shaders

## Potential Future Work

- [ ] Lookahead optimiser?
- [ ] Recurrent network
- [ ] SDF fitting
- [x] Multi-level hash encoding
- [ ] Make concat zero-copy (writeable views)
