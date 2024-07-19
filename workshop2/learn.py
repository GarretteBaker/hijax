"""
Teacher-student perceptron learning, vanilla JAX.

Notes:

* did anyone try the challenge from last week?
* new dependency `pip install plotille` for easy-ish plotting
* workshop 2 demo: stochastic gradient descent
* challenge 2: multi-layer perceptron!
"""

import time

import plotille
import tqdm
import tyro

import jax
import jax.numpy as jnp


# # # 
# Training loop


def main(
    num_steps: int = 200,
    learning_rate: float = 0.001,
    seed: int = 0,
):
    rng = jax.random.key(seed)
    rng, w_star_rng, w_rng = jax.random.split(rng, 3)
    w_star = initialize(w_star_rng)
    w = initialize(w_rng)
    val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    print(vis(student=w, teacher=w_star))
    for i in range(num_steps):
        learning_rate_used = learning_rate
        rng, x_rng = jax.random.split(rng)
        x = jax.random.uniform(x_rng, minval=-4.0, maxval=4.0, shape=(1,))
        loss, grad = val_and_grad(w, w_star, x)
        w = (
            w[0] - learning_rate_used * grad[0], 
            w[1] - learning_rate_used * grad[1], 
            w[2] - learning_rate_used * grad[2], 
            w[3] - learning_rate_used * grad[3]
        )
        figs = vis(student=w, teacher=w_star, x = x.item())
        tqdm.tqdm.write(figs)
        tqdm.tqdm.write(
            f"loss: {loss:.3f}"
        )


# # # 
# Perceptron architecture

def loss_fn(w, w_star, x):
    y_star = forward_pass(w_star, x)
    y = forward_pass(w, x)
    return jnp.mean((y_star - y)**2)

def forward_pass(params, x):
    w, b, v, c = params
    x = jnp.expand_dims(x, 0)
    x = w @ x + b
    x = jax.nn.relu(x)
    x = v @ x + c
    x = jnp.squeeze(x)
    return x


def initialize(rng):
    width = 100
    rng, w_rng, b_rng = jax.random.split(rng, 3)
    w = jax.random.normal(w_rng, shape=(width, 1))
    b = jax.random.normal(b_rng, shape=(width, 1))
    rng, v_rng, c_rng = jax.random.split(rng, 3)
    v = jax.random.normal(v_rng, shape=(1, width))
    c = jax.random.normal(c_rng, shape=(1, 1))
    return w, b, v, c

# # # 
# Visualisation


def vis(x=None, overwrite=True, **models):
    # configure plot
    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-4, 4)
    fig.set_y_limits(-3, 3)
    
    # compute data and add to plot
    xs = jnp.linspace(-4, 4)
    for (label, w), color in zip(models.items(), ['cyan', 'magenta']):
        ys = forward_pass(w, xs)
        fig.plot(xs, ys, label=label, lc=color)
    
    # add a marker for the input batch
    if x is not None:
        fig.text([x], [0], ['x'], lc='yellow')
    
    # render to string
    figure_str = str(fig.show(legend=True))
    reset = f"\x1b[{len(figure_str.splitlines())+1}A" if overwrite else ""
    return reset + figure_str


# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)
