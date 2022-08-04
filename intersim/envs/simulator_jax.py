import jax
import jax.numpy as jnp

@jax.jit
def generate_paths(state, ds, xpoly, ypoly, dxpoly, dypoly, smax):
    s = ds + state[:,0:1]
    smax = smax[:, jnp.newaxis]
    nni = (s <= smax)

    x = eval_poly(s, xpoly)
    y = eval_poly(s, ypoly)

    x_max = eval_poly(smax, xpoly)
    y_max = eval_poly(smax, ypoly)

    dx = eval_poly(smax, dxpoly)
    dy = eval_poly(smax, dypoly)
    x_line = x_max + (s - smax) * dx
    y_line = y_max + (s - smax) * dy

    x = jnp.where(~nni, x_line, x)
    y = jnp.where(~nni, y_line, y)
    return x, y

def eval_poly(s, poly):
    return (poly[..., jnp.newaxis, :] * jnp.power(s[..., jnp.newaxis], jnp.arange(poly.shape[-1]))).sum(-1)
