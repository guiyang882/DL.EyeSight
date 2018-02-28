# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import numpy as np
from scipy import misc
import copy as copy_module
from collections import defaultdict
from abc import ABCMeta, abstractmethod

import eagle.utils as eu


NP_FLOAT_TYPES = set(np.sctypes["float"])

def handle_continuous_param(
        param, name, value_range=None, tuple_to_uniform=True, list_to_choice=True):
    def check_value_range(v):
        if value_range is None:
            return True
        elif isinstance(value_range, tuple):
            eu.do_assert(len(value_range) == 2)
            if value_range[0] is None and value_range[1] is None:
                return True
            elif value_range[0] is None:
                eu.do_assert(v <= value_range[1], "Parameter '%s' is outside "
                                                 "of the expected value range (x <= %.4f)" % (name, value_range[1]))
                return True
            elif value_range[1] is None:
                eu.do_assert(value_range[0] <= v, "Parameter '%s' is outside "
                                                 "of the expected value range (%.4f <= x)" % (name, value_range[0]))
                return True
            else:
                eu.do_assert(value_range[0] <= v <= value_range[1],
                             "Parameter '%s' is outside of the expected value range (%.4f <= x <= %.4f)" % (name, value_range[0], value_range[1]))
                return True
        elif eu.is_callable(value_range):
            value_range(v)
            return True
        else:
            raise Exception("Unexpected input for value_range, got %s." % (str(value_range),))

    if eu.is_single_number(param):
        check_value_range(param)
        return Deterministic(param)
    elif tuple_to_uniform and isinstance(param, tuple):
        eu.do_assert(len(param) == 2)
        check_value_range(param[0])
        check_value_range(param[1])
        return Uniform(param[0], param[1])
    elif list_to_choice and eu.is_iterable(param):
        for param_i in param:
            check_value_range(param_i)
        return Choice(param)
    elif isinstance(param, StochasticParameter):
        return param
    else:
        raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))

def handle_discrete_param(param, name, value_range=None, tuple_to_uniform=True, list_to_choice=True, allow_floats=True):
    def check_value_range(v):
        if value_range is None:
            return True
        elif isinstance(value_range, tuple):
            eu.do_assert(len(value_range) == 2)
            if value_range[0] is None and value_range[1] is None:
                return True
            elif value_range[0] is None:
                eu.do_assert(v <= value_range[1],
                             "Parameter '%s' is outside of the expected value range (x <= %.4f)" % (name, value_range[1]))
                return True
            elif value_range[1] is None:
                eu.do_assert(value_range[0] <= v,
                             "Parameter '%s' is outside of the expected value range (%.4f <= x)" % (name, value_range[0]))
                return True
            else:
                eu.do_assert(value_range[0] <= v <= value_range[1],
                             "Parameter '%s' is outside of the expected value range (%.4f <= x <= %.4f)" % (name, value_range[0], value_range[1]))
                return True
        elif eu.is_callable(value_range):
            value_range(v)
            return True
        else:
            raise Exception("Unexpected input for value_range, got %s." % (str(value_range),))

    if eu.is_single_integer(param) or (allow_floats and eu.is_single_float(
            param)):
        check_value_range(param)
        return Deterministic(int(param))
    elif tuple_to_uniform and isinstance(param, tuple):
        eu.do_assert(len(param) == 2)
        if allow_floats:
            eu.do_assert(eu.is_single_number(param[0]),
                         "Expected number, got %s." % (type(param[0]),))
            eu.do_assert(eu.is_single_number(param[1]),
                         "Expected number, got %s." % (type(param[1]),))
        else:
            eu.do_assert(eu.is_single_integer(param[0]),
                         "Expected integer, got %s." % (type(param[0]),))
            eu.do_assert(eu.is_single_integer(param[1]),
                         "Expected integer, got %s." % (type(param[1]),))
        check_value_range(param[0])
        check_value_range(param[1])
        return DiscreteUniform(int(param[0]), int(param[1]))
    elif list_to_choice and eu.is_iterable(param):
        for param_i in param:
            check_value_range(param_i)
        return Choice([int(param_i) for param_i in param])
    elif isinstance(param, StochasticParameter):
        return param
    else:
        if allow_floats:
            raise Exception("Expected number, tuple of two number, list of number or StochasticParameter for %s, got %s." % (name, type(param),))
        else:
            raise Exception("Expected int, tuple of two int, list of int or StochasticParameter for %s, got %s." % (name, type(param),))


def force_np_float_dtype(val):
    if val.dtype in NP_FLOAT_TYPES:
        return val
    else:
        return val.astype(np.float32)


def both_np_float_if_one_is_float(a, b):
    a_f = a.dtype in NP_FLOAT_TYPES
    b_f = b.dtype in NP_FLOAT_TYPES
    if a_f and b_f:
        return a, b
    elif a_f:
        return a, b.astype(np.float32)
    elif b_f:
        return a.astype(np.float32), b
    else:
        return a.astype(np.float32), b.astype(np.float32)


def draw_distributions_grid(params, rows=None, cols=None, graph_sizes=(350, 350), sample_sizes=None, titles=None):
    if titles is None:
        titles = [None] * len(params)
    elif titles == False:
        titles = [False] * len(params)

    if sample_sizes is not None:
        images = [param_i.draw_distribution_graph(size=size_i, title=title_i) for param_i, size_i, title_i in zip(params, sample_sizes, titles)]
    else:
        images = [param_i.draw_distribution_graph(title=title_i) for param_i, title_i in zip(params, titles)]

    images_rs = eu.imresize_many_images(np.array(images), sizes=graph_sizes)
    grid = eu.draw_grid(images_rs, rows=rows, cols=cols)
    return grid


def show_distributions_grid(params, rows=None, cols=None, graph_sizes=(350, 350), sample_sizes=None, titles=None):
    misc.imshow(
        draw_distributions_grid(
            params,
            graph_sizes=graph_sizes,
            sample_sizes=sample_sizes,
            rows=rows,
            cols=cols,
            titles=titles
        )
    )


class StochasticParameter(object): # pylint: disable=locally-disabled, unused-variable, line-too-long
    """
    Abstract parent class for all stochastic parameters.

    Stochastic parameters are here all parameters from which values are
    supposed to be sampled. Usually the sampled values are to a degree random.
    E.g. a stochastic parameter may be the range [-10, 10], with sampled
    values being 5.2, -3.7, -9.7 and 6.4.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(StochasticParameter, self).__init__()

    def draw_sample(self, random_state=None):
        """
        Draws a single sample value from this parameter.

        Parameters
        ----------
        random_state : None or np.random.RandomState, optional(default=None)
            A random state to use during the sampling process.
            If None, the libraries global random state will be used.

        Returns
        -------
        out : anything
            A single sample value.

        """
        return self.draw_samples(1, random_state=random_state)[0]

    def draw_samples(self, size, random_state=None):
        """
        Draws one or more sample values from the parameter.

        Parameters
        ----------
        size : tuple of int
            Number of sample values by
            dimension.

        random_state : None or np.random.RandomState, optional(default=None)
            A random state to use during the sampling process.
            If None, the libraries global random state will be used.

        Returns
        -------
        out : (size) iterable
            Sampled values. Usually a numpy ndarray of basically any dtype,
            though not strictly limited to numpy arrays.

        """
        random_state = random_state if random_state is not None else \
            eu.current_random_state()
        samples = self._draw_samples(size, random_state)
        eu.forward_random_state(random_state)

        return samples

    @abstractmethod
    def _draw_samples(self, size, random_state):
        raise NotImplementedError()

    def __add__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Add(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter + %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __sub__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Subtract(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter - %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __mul__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Multiply(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter * %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __pow__(self, other, z=None):
        if z is not None:
            raise NotImplementedError("Modulo power is currently not supported by StochasticParameter.")
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Power(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter ** %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __div__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter / %s. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __truediv__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(self, other)
        else:
            raise Exception("Invalid datatypes in: StochasticParameter / %s (truediv). Expected second argument to be number or StochasticParameter." % (type(other),))

    def __radd__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Add(other, self)
        else:
            raise Exception("Invalid datatypes in: %s + StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rsub__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Subtract(other, self)
        else:
            raise Exception("Invalid datatypes in: %s - StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rmul__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Multiply(other, self)
        else:
            raise Exception("Invalid datatypes in: %s * StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rpow__(self, other, z=None):
        if z is not None:
            raise NotImplementedError("Modulo power is currently not supported by StochasticParameter.")
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Power(other, self)
        else:
            raise Exception("Invalid datatypes in: %s ** StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rdiv__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(other, self)
        else:
            raise Exception("Invalid datatypes in: %s / StochasticParameter. Expected second argument to be number or StochasticParameter." % (type(other),))

    def __rtruediv__(self, other):
        if eu.is_single_number(other) or isinstance(other, StochasticParameter):
            return Divide(other, self)
        else:
            raise Exception("Invalid datatypes in: %s / StochasticParameter (truediv). Expected second argument to be number or StochasticParameter." % (type(other),))

    def copy(self):
        """
        Create a shallow copy of this parameter.

        Returns
        -------
        out : StochasticParameter
            Shallow copy.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """
        Create a deep copy of this parameter.

        Returns
        -------
        out : StochasticParameter
            Deep copy.

        """
        return copy_module.deepcopy(self)

    def draw_distribution_graph(self, title=None, size=(1000, 1000), bins=100):
        """
        Generate a plot (image) that shows the parameter's distribution of
        values.

        Parameters
        ----------
        title : None or False or string, optional(default=None)
            Title of the plot. None is automatically replaced by a title
            derived from `str(param)`. If set to False, no title will be
            shown.

        size : tuple of int
            Number of points to sample. This is always expected to have at
            least two values. The first defines the number of sampling runs,
            the second (and further) dimensions define the size assigned
            to each `draw_samples()` call. E.g. `(10, 20, 15)` will lead
            to `10` calls of `draw_samples(size=(20, 15))`. The results
            will be merged to a single 1d array.

        bins : int
            Number of bins in the plot histograms.

        Returns
        -------
        data : (H,W,3) ndarray
            Image of the plot.

        """
        import matplotlib.pyplot as plt

        points = []
        for _ in range(size[0]):
            points.append(self.draw_samples(size[1:]).flatten())
        points = np.concatenate(points)

        fig = plt.figure()
        fig.add_subplot(111)
        ax = fig.gca()
        heights, bins = np.histogram(points, bins=bins)
        heights = heights / sum(heights)
        ax.bar(
            bins[:-1],
            heights,
            width=(max(bins) - min(bins))/len(bins),
            color="blue",
            alpha=0.75
        )
        #print("[draw_distribution_graph] points", points[0:100])
        #print("[draw_distribution_graph] min/max/avg", np.min(points), np.max(points), np.average(points))
        #print("[draw_distribution_graph] bins", len(bins), bins[0:10], heights[0:10])

        if title is None:
            title = str(self)
        if title != False:
            # split long titles - otherwise matplotlib generates errors
            title_fragments = [title[i:i+50] for i in range(0, len(title), 50)]
            ax.set_title("\n".join(title_fragments))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data

class Binomial(StochasticParameter):
    """
    Binomial distribution.

    Parameters
    ----------
    p : number or tuple of two number or list of number or StochasticParameter
        Probability of the binomial distribution. Expected to be in the
        range [0, 1]. If this is a StochasticParameter, the value will be
        sampled once per call to _draw_samples().

    Examples
    --------
    >>> param = Binomial(Uniform(0.01, 0.2))

    Uses a varying probability `p` between 0.01 and 0.2 per sampling.

    """

    def __init__(self, p):
        super(Binomial, self).__init__()

        """
        if isinstance(p, StochasticParameter):
            self.p = p
        elif ia.is_single_number(p):
            ia.do_assert(0 <= p <= 1.0, "Expected probability p to be in range [0.0, 1.0], got %s." % (p,))
            self.p = Deterministic(float(p))
        else:
            raise Exception("Expected StochasticParameter or float/int value, got %s." % (type(p),))
        """

        self.p = handle_continuous_param(p, "p")

    def _draw_samples(self, size, random_state):
        p = self.p.draw_sample(random_state=random_state)
        eu.do_assert(0 <= p <= 1.0,
                     "Expected probability p to be in range [0.0, 1.0], got %s." % (p,))
        return random_state.binomial(1, p, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if isinstance(self.p, float):
            return "Binomial(%.4f)" % (self.p,)
        else:
            return "Binomial(%s)" % (self.p,)


class Choice(StochasticParameter):
    """
    Parameter that samples value from a list of allowed values.

    Parameters
    ----------
    a : iterable
        List of allowed values.
        Usually expected to be integers, floats or strings.

    replace : bool, optional(default=True)
        Whether to perform sampling with or without
        replacing.

    p : None or iterable, optional(default=None)
        Optional probabilities of each element in `a`.
        Must have the same length as `a` (if provided).

    Examples
    --------
    >>> param = Choice([0.25, 0.5, 0.75], p=[0.25, 0.5, 0.25])

    Parameter of which 50 pecent of all sampled values will be 0.5.
    The other 50 percent will be either 0.25 or 0.75.

    """
    def __init__(self, a, replace=True, p=None):
        super(Choice, self).__init__()

        self.a = a
        self.replace = replace
        self.p = p

    def _draw_samples(self, size, random_state):
        if any([isinstance(a_i, StochasticParameter) for a_i in self.a]):
            seed = random_state.randint(0, 10**6, 1)[0]
            samples = eu.new_random_state(seed).choice(self.a, np.prod(size),
                                                      replace=self.replace, p=self.p)

            # collect the sampled parameters and how many samples must be taken
            # from each of them
            params_counter = defaultdict(lambda: 0)
            #params_keys = set()
            for sample in samples:
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    params_counter[key] += 1
                    #params_keys.add(key)

            # collect per parameter once the required number of samples
            # iterate here over self.a to always use the same seed for
            # the same parameter
            # TODO this might fail if the same parameter is added
            # multiple times to self.a?
            # TODO this will fail if a parameter cant handle size=(N,)
            param_to_samples = dict()
            for i, param in enumerate(self.a):
                key = str(param)
                if key in params_counter:
                    #print("[Choice] sampling %d from %s" % (params_counter[key], key))
                    param_to_samples[key] = param.draw_samples(
                        size=(params_counter[key],),
                        random_state=eu.new_random_state(seed+1+i)
                    )

            # assign the values sampled from the parameters to the `samples`
            # array by replacing the respective parameter
            param_to_readcount = defaultdict(lambda: 0)
            for i, sample in enumerate(samples):
                #if i%10 == 0:
                #    print("[Choice] assigning sample %d" % (i,))
                if isinstance(sample, StochasticParameter):
                    key = str(sample)
                    readcount = param_to_readcount[key]
                    #if readcount%10==0:
                    #    print("[Choice] readcount %d for %s" % (readcount, key))
                    samples[i] = param_to_samples[key][readcount]
                    param_to_readcount[key] += 1

            samples = samples.reshape(size)
        else:
            samples = random_state.choice(self.a, size, replace=self.replace, p=self.p)
        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Choice(a=%s, replace=%s, p=%s)" % (str(self.a), str(self.replace), str(self.p),)


class DiscreteUniform(StochasticParameter):
    """
    Parameter that resembles a discrete range of values [a .. b].

    Parameters
    ----------
    {a, b} : int or StochasticParameter
        Lower and upper bound of the sampling range. Values will be sampled
        from a <= x <= b. All sampled values will be discrete. If a or b is
        a StochasticParameter, it will be queried once per sampling to
        estimate the value of a/b. If a>b, the values will automatically be
        flipped. If a==b, all generated values will be identical to a.

    Examples
    --------
    >>> param = DiscreteUniform(10, Choice([20, 30, 40]))

    Sampled values will be discrete and come from the either [10..20] or
    [10..30] or [10..40].

    """

    def __init__(self, a, b):
        super(DiscreteUniform, self).__init__()

        """
        # for two ints the samples will be from range a <= x <= b
        ia.do_assert(isinstance(a, (int, StochasticParameter)), "Expected a to be int or StochasticParameter, got %s" % (type(a),))
        ia.do_assert(isinstance(b, (int, StochasticParameter)), "Expected b to be int or StochasticParameter, got %s" % (type(b),))

        if ia.is_single_integer(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_integer(b):
            self.b = Deterministic(b)
        else:
            self.b = b
        """
        self.a = handle_discrete_param(a, "a")
        self.b = handle_discrete_param(b, "b")

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.tile(np.array([a]), size)
        return random_state.randint(a, b + 1, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DiscreteUniform(%s, %s)" % (self.a, self.b)


class Normal(StochasticParameter):
    """
    Parameter that resembles a (continuous) normal distribution.

    This is a wrapper around numpy's random.normal().

    Parameters
    ----------
    loc : number or StochasticParameter
        The mean of the normal distribution.
        If StochasticParameter, the mean will be sampled once per call
        to `_draw_samples()`.

    scale : number or StochasticParameter
        The standard deviation of the normal distribution.
        If StochasticParameter, the scale will be sampled once per call
        to `_draw_samples()`.

    Examples
    --------
    >>> param = Normal(Choice([-1.0, 1.0]), 1.0)

    A standard normal distribution, which's mean is shifted either 1.0 to
    the left or 1.0 to the right.

    """
    def __init__(self, loc, scale):
        super(Normal, self).__init__()

        if isinstance(loc, StochasticParameter):
            self.loc = loc
        elif eu.is_single_number(loc):
            self.loc = Deterministic(loc)
        else:
            raise Exception("Expected float, int or StochasticParameter as loc, got %s." % (type(loc),))

        if isinstance(scale, StochasticParameter):
            self.scale = scale
        elif eu.is_single_number(scale):
            eu.do_assert(scale >= 0,
                         "Expected scale to be in range [0, inf) got %s (type %s)." % (scale, type(scale)))
            self.scale = Deterministic(scale)
        else:
            raise Exception("Expected float, int or StochasticParameter as scale, got %s." % (type(scale),))

    def _draw_samples(self, size, random_state):
        loc = self.loc.draw_sample(random_state=random_state)
        scale = self.scale.draw_sample(random_state=random_state)
        eu.do_assert(scale >= 0,
                     "Expected scale to be in range [0, inf), got %s." % (scale,))
        if scale == 0:
            return np.tile(loc, size)
        else:
            return random_state.normal(loc, scale, size=size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Normal(loc=%s, scale=%s)" % (self.loc, self.scale)


class Uniform(StochasticParameter):
    """
    Parameter that resembles a (continuous) uniform range [a, b).

    Parameters
    ----------
    {a, b} : number or tuple of two number or list of number or StochasticParameter
        Lower and upper bound of the sampling range. Values will be sampled
        from a <= x < b. All sampled values will be continuous. If a or b is
        a StochasticParameter, it will be queried once per sampling to
        estimate the value of a/b. If a>b, the values will automatically be
        flipped. If a==b, all generated values will be identical to a.

    Examples
    --------
    >>> param = Uniform(0, 10.0)

    Samples random values from the range [0, 10.0).

    """
    def __init__(self, a, b):
        super(Uniform, self).__init__()

        """
        ia.do_assert(isinstance(a, (int, float, StochasticParameter)), "Expected a to be int, float or StochasticParameter, got %s" % (type(a),))
        ia.do_assert(isinstance(b, (int, float, StochasticParameter)), "Expected b to be int, float or StochasticParameter, got %s" % (type(b),))

        if ia.is_single_number(a):
            self.a = Deterministic(a)
        else:
            self.a = a

        if ia.is_single_number(b):
            self.b = Deterministic(b)
        else:
            self.b = b
        """

        self.a = handle_continuous_param(a, "a")
        self.b = handle_continuous_param(b, "b")

    def _draw_samples(self, size, random_state):
        a = self.a.draw_sample(random_state=random_state)
        b = self.b.draw_sample(random_state=random_state)
        if a > b:
            a, b = b, a
        elif a == b:
            return np.tile(np.array([a]), size)
        return random_state.uniform(a, b, size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Uniform(%s, %s)" % (self.a, self.b)


class Deterministic(StochasticParameter):
    """
    Parameter that resembles a constant value.

    If N values are sampled from this parameter, it will return N times V,
    where V is the constant value.

    Parameters
    ----------
    value : number or string or StochasticParameter
        A constant value to use.
        A string may be provided to generate arrays of strings.
        If this is a StochasticParameter, a single value will be sampled
        from it exactly once and then used as the constant value.

    Examples
    --------
    >>> param = Deterministic(10)

    Will always sample the value 10.

    """
    def __init__(self, value):
        super(Deterministic, self).__init__()

        if isinstance(value, StochasticParameter):
            self.value = value.draw_sample()
        elif eu.is_single_number(value) or eu.is_string(value):
            self.value = value
        else:
            raise Exception("Expected StochasticParameter object or number or string, got %s." % (type(value),))

    def _draw_samples(self, size, random_state):
        return np.tile(np.array([self.value]), size)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if eu.is_single_integer(self.value):
            return "Deterministic(int %d)" % (self.value,)
        elif eu.is_single_float(self.value):
            return "Deterministic(float %.8f)" % (self.value,)
        else:
            return "Deterministic(%s)" % (str(self.value),)


class Clip(StochasticParameter):
    """
    Clips another parameter to a defined value range.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter, which's values are to be
        clipped.

    minval : None or number, optional(default=None)
        The minimum value to use.
        If None, no minimum will be used.

    maxval : None or number, optional(default=None)
        The maximum value to use.
        If None, no maximum will be used.

    Examples
    --------
    >>> param = Clip(Normal(0, 1.0), minval=-2.0, maxval=2.0)

    Defines a standard normal distribution, which's values never go below -2.0
    or above 2.0. Note that this will lead to small "bumps" of higher
    probability at -2.0 and 2.0, as values below/above these will be clipped
    to them.

    """
    def __init__(self, other_param, minval=None, maxval=None):
        super(Clip, self).__init__()

        eu.do_assert(isinstance(other_param, StochasticParameter))
        eu.do_assert(minval is None or eu.is_single_number(minval))
        eu.do_assert(maxval is None or eu.is_single_number(maxval))

        self.other_param = other_param
        self.minval = minval
        self.maxval = maxval

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        if self.minval is not None and self.maxval is not None:
            np.clip(samples, self.minval, self.maxval, out=samples)
        elif self.minval is not None:
            np.clip(samples, self.minval, np.max(samples), out=samples)
        elif self.maxval is not None:
            np.clip(samples, np.min(samples), self.maxval, out=samples)
        else:
            pass
        return samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        if self.minval is not None and self.maxval is not None:
            return "Clip(%s, %.6f, %.6f)" % (opstr, float(self.minval), float(self.maxval))
        elif self.minval is not None:
            return "Clip(%s, %.6f, None)" % (opstr, float(self.minval))
        elif self.maxval is not None:
            return "Clip(%s, None, %.6f)" % (opstr, float(self.maxval))
        else:
            return "Clip(%s, None, None)" % (opstr,)


class Discretize(StochasticParameter):
    """
    Convert values sampled from a continuous distribution into discrete values.

    This will round the values and then cast them to integers.
    Values sampled from discrete distributions are not changed.

    Parameters
    ----------
    other_param : StochasticParameter
        The other parameter, which's values are to be
        discretized.

    Examples
    --------
    >>> param = Discretize(Normal(0, 1.0))

    Generates a discrete standard normal distribution.

    """
    def __init__(self, other_param):
        super(Discretize, self).__init__()
        eu.do_assert(isinstance(other_param, StochasticParameter))
        self.other_param = other_param

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(
            size, random_state=random_state
        )
        if isinstance(samples.dtype, numbers.Integral):
            # integer array, already discrete
            return samples
        else:
            return np.round(samples).astype(np.int32)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Discretize(%s)" % (opstr,)


class Multiply(StochasticParameter):
    """
    Parameter to multiply other parameter's results with.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        multiplied.

    val : number or tuple of two number or list of number or StochasticParameter
        Multiplier to use. If this is a StochasticParameter, either
        a single or multiple values will be sampled and used as the
        multiplier(s).

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and used as
        the constant multiplier.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and multiplied elementwise with the results
        of `other_param`.

    Examples
    --------
    >>> param = Multiply(Uniform(0.0, 1.0), -1)

    Converts a uniform range [0.0, 1.0) to (-1.0, 0.0].

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Multiply, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(size,
                                                random_state=eu.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(size,
                                                random_state=eu.new_random_state(seed+1))
        else:
            val_samples = self.val.draw_sample(
                random_state=eu.new_random_state(seed+1))

        if elementwise:
            return np.multiply(samples, val_samples)
        else:
            return samples * val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Multiply(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)


class Divide(StochasticParameter):
    """
    Parameter to divide other parameter's results with.

    This parameter will automatically prevent division by zero (uses 1.0)
    as the denominator in these cases.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        divided.

    val : number or tuple of two number or list of number or StochasticParameter
        Denominator to use. If this is a StochasticParameter, either
        a single or multiple values will be sampled and used as the
        denominator(s).

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and used as
        the constant denominator.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and used as the elementwise denominators for the
        results of `other_param`.

    Examples
    --------
    >>> param = Divide(Uniform(0.0, 1.0), 2)

    Converts a uniform range [0.0, 1.0) to [0, 0.5).

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Divide, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(
            size,
            random_state=eu.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(
                size,
                random_state=eu.new_random_state(seed+1)
            )

            # prevent division by zero
            val_samples[val_samples == 0] = 1

            return np.multiply(
                force_np_float_dtype(samples),
                force_np_float_dtype(val_samples)
            )
        else:
            val_sample = self.val.draw_sample(
                random_state=eu.new_random_state(seed+1)
            )

            # prevent division by zero
            if val_sample == 0:
                val_sample = 1

            return force_np_float_dtype(samples) / float(val_sample)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Divide(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)


class Add(StochasticParameter):
    """
    Parameter to add to other parameter's results.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    val : number or tuple of two number or list of number or StochasticParameter
        Value to add to the other parameter's results. If this is a
        StochasticParameter, either a single or multiple values will be
        sampled and added.

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and added
        to all values generated by `other_param`.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and added to the results of `other_param`.

    Examples
    --------
    >>> param = Add(Uniform(0.0, 1.0), 1.0)

    Converts a uniform range [0.0, 1.0) to [1.0, 2.0).

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Add, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(
            size,
            random_state=eu.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(
                size,
                random_state=eu.new_random_state(seed+1))
        else:
            val_samples = self.val.draw_sample(
                random_state=eu.new_random_state(seed+1))

        if elementwise:
            return np.add(samples, val_samples)
        else:
            return samples + val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Add(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)


class Subtract(StochasticParameter):
    """
    Parameter to subtract from another parameter's results.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    val : number or tuple of two number or list of number or StochasticParameter
        Value to add to the other parameter's results. If this is a
        StochasticParameter, either a single or multiple values will be
        sampled and subtracted.

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and subtracted
        from all values generated by `other_param`.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and subtracted from the results of `other_param`.

    Examples
    --------
    >>> param = Add(Uniform(0.0, 1.0), 1.0)

    Converts a uniform range [0.0, 1.0) to [1.0, 2.0).

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Subtract, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(
            size,
            random_state=eu.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            val_samples = self.val.draw_samples(
                size,
                random_state=eu.new_random_state(seed+1))
        else:
            val_samples = self.val.draw_sample(
                random_state=eu.new_random_state(seed+1))

        if elementwise:
            return np.subtract(samples, val_samples)
        else:
            return samples - val_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Subtract(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)


class Power(StochasticParameter):
    """
    Parameter to exponentiate another parameter's results with.

    Parameters
    ----------
    other_param : number or tuple of two number or list of number or StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    val : number or tuple of two number or list of number or StochasticParameter
        Value to use exponentiate the other parameter's results with. If this
        is a StochasticParameter, either a single or multiple values will be
        sampled and used as the exponents.

    elementwise : bool, optional(default=False)
        Controls the sampling behaviour when `val` is a StochasticParameter.
        If set to False, a single value will be sampled from val and used as
        the exponent for all values generated by `other_param`.
        If set to True and `_draw_samples(size=S)` is called, `S` values will
        be sampled from `val` and used as the exponents for the results of
        `other_param`.

    Examples
    --------
    >>> param = Power(Uniform(0.0, 1.0), 2)

    Converts a uniform range [0.0, 1.0) to a distribution that is peaked
    towards 1.0.

    """
    def __init__(self, other_param, val, elementwise=False):
        super(Power, self).__init__()

        self.other_param = handle_continuous_param(other_param, "other_param")
        self.val = handle_continuous_param(val, "val")
        self.elementwise = elementwise

    def _draw_samples(self, size, random_state):
        seed = random_state.randint(0, 10**6, 1)[0]
        samples = self.other_param.draw_samples(
            size,
            random_state=eu.new_random_state(seed))

        elementwise = self.elementwise and not isinstance(self.val, Deterministic)

        if elementwise:
            exponents = self.val.draw_samples(
                size,
                random_state=eu.new_random_state(seed+1))
        else:
            exponents = self.val.draw_sample(
                random_state=eu.new_random_state(seed+1))

        # without this we get int results in the case of
        # Power(<int>, <stochastic float param>)
        samples, exponents = both_np_float_if_one_is_float(samples, exponents)
        samples_dtype = samples.dtype

        # float_power requires numpy>=1.12
        #result = np.float_power(samples, exponents)
        # TODO why was float32 type here replaced with complex number
        # formulation?
        result = np.power(samples.astype(np.complex), exponents).real
        if result.dtype != samples_dtype:
            result = result.astype(samples_dtype)

        return result

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Power(%s, %s, %s)" % (str(self.other_param), str(self.val), self.elementwise)


class Absolute(StochasticParameter):
    """
    Converts another parameter's results to absolute values.

    Parameters
    ----------
    other_param : StochasticParameter
        Other parameter which's sampled values are to be
        modified.

    Examples
    --------
    >>> param = Absolute(Uniform(-1.0, 1.0))

    Converts a uniform range [-1.0, 1.0) to [0.0, 1.0].

    """
    def __init__(self, other_param):
        super(Absolute, self).__init__()

        eu.do_assert(isinstance(other_param, StochasticParameter))

        self.other_param = other_param

    def _draw_samples(self, size, random_state):
        samples = self.other_param.draw_samples(size, random_state=random_state)
        return np.absolute(samples)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        opstr = str(self.other_param)
        return "Absolute(%s)" % (opstr,)
