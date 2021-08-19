# descent

Tensor operations in Rust using Vulkan compute shaders.

## Tensor Variables

A _variable_ is a tensor with known shape, backed by Vulkan device memory, optionally initialized with some data.

Variables are created via an _environment_, which wraps a Vulkan device.

```rust
let mut env = Environment::new();

let m_var = env.static_parameter_with_data(
    [3, 3],
    "m",
    &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
let x_var = env.static_parameter_with_data([3, 1], "x", &[4.0, 5.0, 6.0]);
let y_var = env.static_parameter_with_data([3, 1], "y", &[1.0, 2.0, 3.0]);
let z_var = env.static_parameter([3, 1], "z");
```

## Tensor Graphs

A _scope_ can be created to keep track of expressions that read or write to variables.

These expressions do not directly affect any of the variables, they simply keep track of their inputs and outputs in a graph data structure.

```rust
let graph = env.build_graph(|scope| {
    let m = scope.read_variable(&m_var);
    let x = scope.read_variable(&x_var);
    let y = scope.read_variable(&y_var);

    // capture the following as the effect of the graph
    let z = 2.0 * m.matmul(x) + y * y + 1.0;
    
    scope.write_variable(&z_var, z);
});
```

After all operations have been added, the scope can be built into a _graph_ that can be run on the Vulkan device.  The graph for the example above is as follows:

![array graph](docs/array.svg)

This graph has been built as 2 GPU compute shaders (the grey boxes above): a matrix multiply and a fused per-element shader for the remaining ops.

## Derivatives

Variables can also be tracked with gradients, with expressions in rust essentially tracking a (value, grad) pair instead of only the value.

In this slightly higher-level API, expressions in rust generate graph operations for the value (forwards) and gradient (backwards) at the same time.

This lets us build the full graph for back-propagation in small (composable) chunks of forward-like code, with the final loss function connecting the value and grad ops into a single graph.  See the examples for more details.

TODO: extend array example, add diagram 

## Neural Network Modules

Finally there is an optional top-level API that provides neural network building blocks, such as fully-connected layers or 2D convolutions.

This API makes it easier to build larger networks, with each layer keeping track of its own variables.  For example, here is a network with a single hidden layer with leaky ReLU activation:

```rust
struct SingleLayer {
    fc1: Dense,
    fc2: Dense,
}

impl SingleLayer {
    fn new(env: &mut Environment) -> Self {
        let hidden_units = 300;
        Self {
            fc1: Dense::builder(28 * 28, hidden_units).build(env),
            fc2: Dense::builder(hidden_units, 10).build(env),
        }
    }
}

impl Module for SingleLayer {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        // generates ops for the value (forwards) and gradient (backwards) through the layers
        input
            .flatten()
            .apply(&self.fc1, ctx)
            .leaky_relu(0.01)
            .apply(&self.fc2, ctx)
    }
}
```

## Examples

Please the follow the link in the title of each example for a more detailed README of each one.

### [array](examples/array)

Demonstrates the lowest-level API reading and writing (value-only) arrays.

### [fashion_mnist](examples/fashion_mnist)

Trains a few different network types on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  Demonstrates the use of anti-aliasing during max pooling for improved accuracy.

### [image_fit](examples/image_fit)

Overfits a few different network types to a single RGB image.  Compares ReLU with position encoding to a SIREN network.

## Dependencies

The following crates have been very useful to develop this project:

- [petgraph](https://github.com/petgraph/petgraph): used for all graph data structures
- [slotmap](https://github.com/orlp/slotmap): storage with stable keys
- [shaderc](https://github.com/google/shaderc-rs): interface to GLSL compiler to generate SPIR-V for shaders
