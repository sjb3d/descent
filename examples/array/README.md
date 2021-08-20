# array example

This example creates some small graphs directly using the array API.

## Parameters

A _parameter_ is a handle to a persistant array with some fixed shape.  The memory for the array is provided by Vulkan device memory when necessary.

Parameters are created via an _environment_, which wraps a Vulkan device.

```rust
let mut env = Environment::new();

let z_param = env.static_parameter([3, 1], "z");
```

## Graphs

A _graph_ is a set of expressions that read and write to _parameters_, that can be run on the Vulkan device.

At the lowest level, graphs are built by using the `Array` type.  This type does not directly do any operations on parameters, but can be used to build up a computation graph using standard Rust syntax.

A _scope_ is a temporary data structure used to track expressions involving the `Array` type.  Many parameter updates can be added to a single scope.  Once the scope is complete, it can be compiled into a graph.

Here is some code from this example that builds a graph using the `Array` type:

```rust
let graph = env.build_graph(|scope| {
    let m = scope.parameter_value(&m_param);
    let x = scope.parameter_value(&x_param);
    let y = scope.parameter_value(&y_param);

    // build an expression involving m, x and y
    let z = 2.0 * m.matmul(x) + y * y + 1.0;

    scope.write_parameter_value(&z_param, z);
});
```

In order to actually perform the operations on the parameters, the graph can be run on the Vulkan device using the environment.

```rust
env.run(&graph, random_seed);
```

The graph is run as a set compute shaders that run on the Vulkan device.  To avoid needlessly wasting bandwidth, operations are fused into a single kernel where possible.  A visualisation of the graph for the example above is as follows:

![array graph](../../docs/array_values.svg)

The grey boxes above are individual compute shaders: one matrix multiply and one fused per-element shader.  (In future this may become a single fused shader.)

Temporary memory to pass data between shaders is allocated/freed automatically as the graph is run.

## Derivatives

To simplify generating code for back-propagation, there is a higher-level API that manipulates `DualArray` values.

A `DualArray` value is a pair of `Array` values: one for the (forward) value and one for the (backward) gradient.
It is expected that the gradient that is an expression that computes the derivative of the loss function w.r.t. this variable, i.e. exactly what is required for gradient descent.

API functions for `DualArray` values compute expressions for both the forward and backward passes.  This lets us build the full graph for a gradient descent step in small (composable) chunks of forward-like code, with the final loss function connecting the value and grad ops into a single graph.

Here is an example that directly constructs a toy loss function, then adds code for a step of gradient descent:

```rust
let graph = env.build_graph(|scope| {
    let x = scope.parameter(&x_param);
    let y = x.sin();
    let _loss = (y.square() + y * 3.0).set_loss();
    scope.write_parameter_value(&x_param, x.value() - 0.1 * x.loss_grad());
});
```

Since x is a `DualArray` we did not have to explicitly write code for back-propagation.  The graph for this example is as follows:

![](../../docs/array_grad.svg)

Inspecting this graph, x is updated proportional to `(2*sin(x) + 3)*cos(x)`, which matches what we expect for a loss function of `sin^2(x) + 3*sin(x)`.

The `DualArray` API is usually implemented in terms of the `Array` API, and can easily be extended with new functions where they have known derivative.
For example, here is the implemention of `sin()` with some additional comments:

```rust
impl<'s> DualArray<'s> {
    pub fn sin(self) -> Self {
        // get Array for value, loss_grad ("dx" means dL/dx)
        let (a, da) = self.into_inner();
        let (b, db) = a.sin().with_empty_grad();

        // add back-propagation step to compute da from db
        // using dL/da = dL/db * db/da, db/da = cos(a)
        da.accumulate(db * a.cos());

        // into DualArray
        (b, db).into()
    }
}
```