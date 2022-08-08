import numpy as np

def generate_paths(state, ds, xpoly, ypoly, dxpoly, dypoly, smax):
    s = ds + state[:,0:1]
    smax = smax[:, np.newaxis]
    nni = (s <= smax)

    x = eval_poly(s, xpoly)
    y = eval_poly(s, ypoly)

    x_max = eval_poly(smax, xpoly)
    y_max = eval_poly(smax, ypoly)

    dx = eval_poly(smax, dxpoly)
    dy = eval_poly(smax, dypoly)
    x_line = x_max + (s - smax) * dx
    y_line = y_max + (s - smax) * dy

    x = np.where(~nni, x_line, x)
    y = np.where(~nni, y_line, y)
    return x, y

def eval_poly(s, poly):
    coeffs = poly[..., np.newaxis, :]
    bases = s[..., :, np.newaxis]
    lg_abs_coeffs = np.log(np.abs(coeffs))
    lg_abs_powers = np.log(np.abs(bases)) * np.arange(poly.shape[-1])

    coeffs_pos = coeffs >= 0
    even_powers = np.ones_like(lg_abs_powers)
    even_powers[..., np.arange(1, poly.shape[-1], 2)] = 0
    powers_pos = np.logical_or(bases >= 0, even_powers)
    
    coeffs_sgn = -1 + 2 * coeffs_pos
    powers_sgn = -1 + 2 * powers_pos
    sgn = coeffs_sgn * powers_sgn

    return (sgn * np.exp(lg_abs_coeffs + lg_abs_powers)).sum(-1)
