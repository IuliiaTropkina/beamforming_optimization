import numpy as np


class PowerSocket:
    def __init__(self, q):
        self.q = q  # the true reward value
        self.Q = 0  # the estimate of this socket's reward value
        self.n = 0  # the number of times this socket has been tried

    # the reward is a random distribution around the initial mean value set for this socket
    # - never allow a charge less than 0 to be returned
    def charge(self):
        # a guassian distribution with unit variance around the true value 'q'
        value = np.random.randn() + self.q
        return 0 if value < 0 else value

    # increase the number of times this socket has been used and improve the estimate of the
    # value (the mean) by combining the new reward 'r' with the current mean
    def update(self, R):
        self.n += 1

        # the new estimate is calculated from the old estimate
        self.Q = (1 - 1.0 / self.n) * self.Q + (1.0 / self.n) * R

class GaussianThompsonSocket(PowerSocket):
    def __init__(self, q):
        self.τ_0 = 0.0001  # the posterior precision
        self.μ_0 = 1  # the posterior mean

        # pass the true reward value to the base PowerSocket
        super().__init__(q)

    def sample(self, t):
        """ return a value from the the posterior normal distribution """
        return (np.random.randn() / np.sqrt(self.τ_0)) + self.μ_0

    def update(self, R):
        """ update this socket after it has returned reward value 'R' """

        # do a standard update of the estimated mean
        super().update(R)

        # update the mean and precision of the posterior
        self.μ_0 = ((self.τ_0 * self.μ_0) + (self.n * self.Q)) / (self.τ_0 + self.n)
        self.τ_0 += 1

def charge(self):
    """ return a random amount of charge """

    # the reward is a guassian distribution with unit variance
    # around the true value 'q'
    value = np.random.randn() + self.q


