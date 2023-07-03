"""Module implementing various samplers."""
import itertools
from typing import Generator
from collections.abc import Iterator

import numpy as np
from scipy import stats
from numba import vectorize, float64, int64
import math
import itertools
import random


# Bell Curve
def gaussian_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding random normal (gaussian) samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """

    while True:
        yield np.random.normal(0.5, 0.5 / 3, size=(d, 1))


def sobol_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a Sobol sequence.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """

    sobol = Sobol(d, np.random.randint(2, max(3, d ** 2)))
    while True:
        yield next(sobol).reshape(-1, 1)


def halton_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a Halton sequence.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """

    halton = Halton(d)
    while True:
        yield next(halton).reshape(-1, 1)


def hammersley_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from Hammersley sequence.

           Parameters
           ----------
           d: int
                An integer denoting the dimenionality of the samples
           n: int
                An integer denoting the population size

           Yields
           ------
           numpy.ndarray

           """
    hammersley = Hammersley(d, n)
    while True:
        yield next(hammersley).reshape(-1, 1)


def uniform_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding random uniform samples.

    Parameters
    ----------
    d: int
        An integer denoting the dimenionality of the samples

    Yields
    ------
    numpy.ndarray

    """

    while True:
        yield np.random.uniform(size=(d, 1))


def latin_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from the Latin Hypercube Sampling method.

       Parameters
       ----------
       d: int
            An integer denoting the dimenionality of the samples
       n: int
            An integer denoting the population size

       Yields
       ------
       numpy.ndarray

       """
    latin = Latin(d, n)
    while True:
        yield next(latin).reshape(-1, 1)


def grid_sampling(d: int, n: int) -> Generator[np.ndarray, None, None]:
    """Generator yielding samples from a grid.

          Parameters
          ----------
          d: int
               An integer denoting the dimenionality of the samples
          n: int
               An integer denoting the population size

          Yields
          ------
          numpy.ndarray

          """
    grid = Grid(d, n)
    while True:
        yield next(grid).reshape(-1, 1)


def mirrored_sampling(sampler: Generator) -> Generator[np.ndarray, None, None]:
    """Generator yielding mirrored samples.

    For every sample from the input sampler (generator), both its
    original and complemented form are yielded.

    Parameters
    ----------
    sampler: generator
        A sample generator yielding numpy.ndarray

    Yields
    ------
    numpy.ndarray

    """
    for sample in sampler:
        yield sample
        yield 1 - sample


class Halton(Iterator):
    """Iterator implementing Halton Quasi random sequences.

    Attributes
    ----------
    d: int
        dimension
    bases: np.ndarray
        array of primes
    index: itertools.count
        current index

    """

    def __init__(self, d, start=1):
        """Compute the bases, and set index to start."""
        self.d = d
        self.bases = self.get_primes(self.d)
        self.index = itertools.count(start)

    @staticmethod
    def get_primes(n: int) -> np.ndarray:
        """Return n primes, starting from 0."""

        def inner(n_):
            sieve = np.ones(n_ // 3 + (n_ % 6 == 2), dtype=bool)
            for i in range(1, int(n_ ** 0.5) // 3 + 1):
                if sieve[i]:
                    k = 3 * i + 1 | 1
                    sieve[k * k // 3:: 2 * k] = False
                    sieve[k * (k - 2 * (i & 1) + 4) // 3:: 2 * k] = False
            return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

        primes = inner(max(6, n))
        while len(primes) < n:
            primes = inner(len(primes) ** 2)
        return primes[:n]

    def __next__(self) -> np.ndarray:
        """Return next Halton sequence."""
        return self.vectorized_next(next(self.index), self.bases)

    @staticmethod
    @vectorize([float64(int64, int64)])
    def vectorized_next(index: int, base: int) -> float:
        """Vectorized method for computing halton sequence."""
        d, x = 1, 0
        while index > 0:
            index, remainder = divmod(index, base)
            d *= base
            x += remainder / d
        return x


class Sobol(Iterator):
    """Iterator implementing Sobol Quasi random sequences.

    This is an iterator version of the version implemented in the python
    package: sobol-seq==0.2.0. This version is 4x faster due to better usage of
    numpy vectorization.

    Attributes
    ----------
    d: int
        dimension
    seed: int
        sample seed
    v: np.ndarray
        array of sample directions
    recipd: int
        1/(common denominator of the elements in v)
    lastq: np.ndarray
        vector containing last sample directions

    """

    def __init__(self, d: int, seed: int = 0):
        """Intialize the v matrix, used for generating Sobol sequences.

        The values for v and poly were taken from the python package sobol-seq.
        """
        self.d = d
        self.seed = np.floor(max(0, seed)).astype(int)
        self.v = np.zeros((40, 30), dtype=int)

        self.v[0:40, 0] = np.ones(40)
        self.v[2:40, 1] = np.r_[
            np.tile([1, 3], 3),
            np.tile(np.r_[np.tile([3, 1], 4), np.tile([1, 3], 4)], 2),
        ]
        self.v[3:40, 2] = [
            7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7,
            5, 1, 3, 3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3
        ]
        self.v[5:40, 3] = [
            1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9,
            13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9
        ]
        self.v[7:40, 4] = [
            9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31,
            11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9
        ]
        self.v[13:40, 5] = [
            37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9,
            49, 33, 19, 29, 11, 19, 27, 15, 25
        ]
        self.v[19:40, 6] = [
            13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3,
            113, 61, 89, 45, 107
        ]
        self.v[37:40, 7] = [7, 23, 39]
        poly = [
            1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109,
            103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299
        ]

        #  Find the number of bits in ATMOST.
        maxcol = Sobol.h1(2 ** 30 - 1)

        #  Initialize row 1 of V.
        self.v[0, :maxcol] = 1

        for i in range(2, self.d + 1):
            j = poly[i - 1]
            m = int(np.log2(j))
            includ = np.fromiter(format(j, "b")[1:], dtype=int)
            powers = 2 ** np.arange(1, m + 1)

            for j in range(m + 1, maxcol + 1):
                mask = np.arange(j - 1)[::-1][:m]
                self.v[i - 1, j - 1] = np.bitwise_xor.reduce(
                    np.r_[
                        self.v[i - 1, j - m - 1], powers * self.v[i - 1, mask] * includ
                    ]
                )

        i = np.arange(maxcol - 1)[::-1]
        powers = 2 ** np.arange(1, len(i) + 1)
        self.v[: self.d, i] = self.v[: self.d, i] * powers

        self.recipd = 1.0 / (2 * powers[-1])
        self.lastq = np.zeros(self.d, dtype=int)

        for loc in map(self.l0, range(self.seed)):
            self.lastq = np.bitwise_xor(self.lastq, self.v[: self.d, loc - 1])

    def __next__(self) -> np.ndarray:
        """Return next Sobol sequence."""
        loc = self.l0(self.seed)
        quasi = self.lastq * self.recipd
        self.lastq = np.bitwise_xor(self.lastq, self.v[: self.d, loc - 1])
        self.seed += 1
        return quasi

    @staticmethod
    def h1(n: int) -> int:
        """Return high 1 bit index for a given integer."""
        return len(format(n, "b")) - abs(format(n, "b").find("1"))

    @staticmethod
    def l0(n: int) -> int:
        """Return low 0 bit index for a given integer."""
        x = format(n, "b")[::-1].find("0")
        if x != -1:
            return x + 1
        return len(format(n, "b")) + 1


class Hammersley(Iterator):
    """Iterator implementing Hammersley Quasi random sequences.

    Closely resembles the Halton sequence, except for the first dimension in which sampling
    points are equidistant.

    Attributes
    ----------
    d: int
        dimension
    bases: np.ndarray
        array of primes
    index: itertools.count
        current index
    lambda:
        the population size


    """

    def __init__(self, d, lambda_, start=1):
        """Compute the bases, and set index to start."""
        self.d = d
        self.lambda_ = lambda_
        self.bases = self.get_primes(self.d)
        self.index = itertools.count(start)

    @staticmethod
    def get_primes(n: int) -> np.ndarray:
        """Return n primes, starting from 0."""

        def inner(n_):
            sieve = np.ones(n_ // 3 + (n_ % 6 == 2), dtype=bool)
            for i in range(1, int(n_ ** 0.5) // 3 + 1):
                if sieve[i]:
                    k = 3 * i + 1 | 1
                    sieve[k * k // 3:: 2 * k] = False
                    sieve[k * (k - 2 * (i & 1) + 4) // 3:: 2 * k] = False
            return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

        primes = inner(max(6, n))
        while len(primes) < n:
            primes = inner(len(primes) ** 2)
        return np.append([1], primes[:n - 1])

    def __next__(self) -> np.ndarray:
        """Return next Hammersley sequence."""
        return self.vectorized_next(next(self.index), self.bases, self.lambda_)

    @staticmethod
    @vectorize([float64(int64, int64, int64)])
    def vectorized_next(index: int, base: int, n: int) -> float:
        """Vectorized method for computing Hammersley sequence."""
        d, x = 1, 0

        # Points are equidistant in the first dimension
        if base == 1:
            return float(index - 1) / float(n - 1)

        # Higher dimensions follow Halton Sequence
        while index > 0:
            index, remainder = divmod(index, base)
            d *= base
            x += remainder / d
        return x


class Latin(Iterator):
    """Iterator implementing Latin HQ sequence

    Every dimension is divided into a number of stratas equal to population size. For
    every strata one sampling point is chosen.

    Attributes
    ----------
    d: int
        dimensionality
    count: itertools.count
        current index
    array: np.ndarray
        contains sampling points, size = lambda
    lambda: int
        population size

    """

    def __init__(self, d, lambda_, start=0):
        self.d = d
        self.lambda_ = lambda_
        self.array = self.getarray(self.d, self.lambda_)
        self.count = itertools.count(start)

    @staticmethod
    def getarray(d: int, n: int) -> np.ndarray:
        lower_limits = np.arange(0, n) / n
        upper_limits = np.arange(1, n + 1) / n

        # Add random numbers between the lower and upper limits to the array
        arr = np.random.uniform(low=lower_limits, high=upper_limits, size=(d, n)).T

        # Create different combinations
        for i in range(arr.shape[1]):
            np.random.shuffle(arr[:, i])

        return arr

    def __next__(self) -> np.ndarray:
        """Return next Latin HQ sequence."""
        return self.array[next(self.count)]


class Grid(Iterator):
    """Iterator implementing Grid sequence

    Forms a grid x^d, where the grid size is chosen such that it is the closest to population size, without
    the population size exceeding it.

    Attributes
    ----------
    d: int
        dimension
    array: np.ndarray
        array of sampling points
    count: itertools.count
        current index
    order: np.ndarray
        array size lambda containing indices corresponding to the array of sampling points
    lambda: int
        population size
    """

    def __init__(self, d, lambda_, start=0):
        self.d = d
        self.lambda_ = lambda_
        self.array = self.getarray(self.d, self.lambda_)
        self.order = self.create_sequence(self.d, self.lambda_)
        self.count = itertools.count(start)

    @staticmethod
    def getarray(d: int, n: int) -> np.ndarray:
        # create a mesh grid that contains the grid
        x = np.linspace(0., 1., math.ceil(n ** (1 / d)))
        mesh = np.meshgrid(*[x] * d)
        positions = np.vstack(map(np.ravel, mesh))

        return positions

    @staticmethod
    def create_sequence(d: int, n: int) -> np.ndarray:
        # compute the highest possible index
        n_points = math.ceil(n ** (1 / d)) ** d

        # draw n numbers from all possible indices
        return random.sample(range(n_points), n)

    def __next__(self) -> np.ndarray:
        # get index for the next sampling point
        new_point = self.order[next(self.count)]
        x = []

        # compute sampling point
        for i in self.array:
            x.append(i[new_point])

        return np.asarray(x)
