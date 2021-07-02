# box.py

# Adapted from https://github.com/openai/gym/blob/master/gym/spaces/box.py

import numpy as np
import torch
from torch.distributions.exponential import Exponential

class Box:
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)

    """
    def __init__(self, low, high, shape=None, dtype=torch.get_default_dtype()):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = dtype

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match provided shape"
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert np.isscalar(high) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert np.isscalar(low) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        if np.isscalar(low):
            low = torch.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = torch.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low
        self.high = high
        if not isinstance(low, torch.Tensor):
            self.low = torch.tensor(low)
        if not isinstance(high, torch.Tensor):
            self.high = torch.tensor(high)

        self.low = self.low.type(self.dtype)
        self.high = self.high.type(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        

    def is_bounded(self, manner="both"):
        below = torch.all(self.bounded_below)
        above = torch.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """

        high = self.high 
        sample = torch.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded   = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below &  self.bounded_above
        low_bounded =  self.bounded_below & ~self.bounded_above
        bounded     =  self.bounded_below &  self.bounded_above


        # Vectorized sampling by interval type
        sample[unbounded] = torch.randn(self.shape)[unbounded]

        sample[low_bounded] = Exponential(1).sample(
            self.shape)[low_bounded] + self.low[low_bounded]

        sample[upp_bounded] = -Exponential(1).sample(
            self.shape)[upp_bounded] + self.high[upp_bounded]

        sample[bounded] = torch.rand(self.shape)[bounded]*(
                        self.high[bounded]-self.low[bounded]) + self.low[bounded]


        return sample.type(self.dtype)

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        return x.shape == self.shape and torch.all(x >= self.low) and torch.all(x <= self.high)

    def __repr__(self):
        return "Box({}, {}, {}, {})".format(self.low.min(), self.high.max(), self.shape, self.dtype)

    def __eq__(self, other):
        return isinstance(other, Box) and (self.shape == other.shape) and torch.allclose(self.low, other.low) and torch.allclose(self.high, other.high)