"""
Original code: https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
"""
import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.distributions.multivariate_normal import MultivariateNormal

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class TruncatedStandardNormal(Distribution):
    
    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size([1])
        elif self.a.size() == torch.Size():
            batch_shape = torch.Size([1])
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        device = value.device
        return self._inv_big_phi(self._big_phi_a.to(device) + value * self._Z.to(device))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        device = value.device
        return CONST_LOG_INV_SQRT_2PI - self._log_Z.to(device) - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)
    
    
class TruncatedNormal(TruncatedStandardNormal):

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale
        
    def _to_std_rv(self, value):
        device = value.device
        return (value - self.loc.to(device)) / self.scale.to(device)

    def _from_std_rv(self, value):
        device = value.device
        return value * self.scale.to(device) + self.loc.to(device)

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))
    
    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale
    
    def bound(self):
        return self.loc + self.a * self.scale, self.loc + self.b * self.scale

"""
    Truncated Multivariate Normal distribution
"""
class TruncMultivariateNormal(MultivariateNormal):

    has_rsamples = True

    def __init__(self, loc, scale_tril, a, b):
        """
        loc: mean vector
        scale_tril: cholesky decomposition of covariance matrix (Σ = L @ L^T)
        a, b: truncated region [a1, b1] x [a2, b2] x ...
        """

        assert loc.size(0) == scale_tril.size(0), "The dimension of a covariance matrix does not match with `loc`."
        self.loc, self.a, self.b = broadcast_all(loc, a, b)

        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')

        super(TruncMultivariateNormal, self).__init__(loc=loc, scale_tril=scale_tril)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        out_sample = torch.zeros(shape, device=self.a.device)
        num_samples = 0
        while num_samples < shape[0]:
            temp_sample = super(TruncMultivariateNormal, self).rsample(sample_shape)
            # temp_sample = [x1, x2, ...]
            # check if a1 <= x1 <= b1, a2 <= x2 <= b2, ...
            validate_indicator = ((self.a <= temp_sample) & (temp_sample <= self.b)).all(dim=1)
            curr_num_samples = int(validate_indicator.sum().item())
            last_out_idx = min(num_samples + curr_num_samples, shape[0])
            out_sample[num_samples:last_out_idx, ] = temp_sample[validate_indicator, ][:(last_out_idx - num_samples), ]
            num_samples = last_out_idx
        return out_sample
    
class PushforwardTruncMultivariateNormal(TruncMultivariateNormal):

    has_rsample = True

    def __init__(self, loc, scale_tril, a, b, ftn):
        super(PushforwardTruncMultivariateNormal, self).__init__(loc, scale_tril, a, b)
        self.ftn = ftn

    def rsample(self, sample_shape=torch.Size()):
        # Assume sample_shape == (num_samples, )
        sample = super(PushforwardTruncMultivariateNormal, self).rsample(sample_shape)
        return self.ftn(sample)