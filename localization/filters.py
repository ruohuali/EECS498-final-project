import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

class KalmanFilter:
    def __init__(self, mu, sigma):
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.C = np.eye(2)
        self.R = np.array([[9.24691775e-04, 1.44216463e-06],
                           [1.44216463e-06, 9.49292672e-04]])
        self.Q = np.array([[0.01783823, 0.00071094],
                           [0.00071094, 0.36105314]])
        self.mu = mu
        self.sigma = sigma

    def __call__(self, z, u):
        """
        assume
        x_t+1 = Ax_t + Bu_t+1 + r
        [x, y].T = I2[x, y].T + I2[u1, u2].T + sample(normal(0, R))
        z_t+1 = Cx_t+1 + q
        [z1, z2].T = I2[x, y].T + sample(normal(0, Q))

        @param z ~ (2, 1)
        @param u ~ (2, 1)
        """
        # prediction step
        mu_bar = self.A @ self.mu + self.B @ u
        sigma_bar = self.A @ self.sigma @ self.A.T + self.R

        # update step
        kal_gain = np.linalg.inv(self.C @ sigma_bar @ self.C.T + self.Q)
        K = sigma_bar @ self.C.T @ kal_gain
        self.mu = mu_bar + K @ (z - self.C @ mu_bar)
        self.sigma = (np.eye(2) - K @ self.C) @ sigma_bar

        return self.mu, self.sigma


class ParticleFilter:
    def sampleInitParticles(self, init_pos):
        center = init_pos.reshape(2)
        print("init sample at", center)
        particles = np.random.multivariate_normal(center, [[0.1, 0], [0, 0.1]], self.particle_num)  # (N, 2)
        return particles

    def move(self, u):
        """@param u ~ (2, 1)"""
        noise = np.random.multivariate_normal([0, 0], [[0.2, 0], [0, 0.2]], self.particle_num)  # (N, 2)
        self.particles += u.T
        self.particles += noise

    def resample(self):
        samples = np.random.choice(np.arange(self.particle_num), self.particle_num, p=self.weights)
        self.particles = self.particles[samples]
        self.weights = np.ones(self.particle_num) / self.particle_num  # N,
        # self.weights = self.weights[samples]

    def __init__(self, init_pos, particle_num=100):
        self.particle_num = particle_num
        self.particles = self.sampleInitParticles(init_pos)
        self.weights = np.ones(particle_num) / particle_num     # N,

    def __call__(self, z, u):
        """
        assume
        P(z | x) = normalize( ||x - z||^2 )
        @param u ~ (2, 1)
        @param z ~ (2, 1)
        """
        # prediction step
        self.move(u)

        # update step
        diff = self.particles.T - z  # 2, N
        lh = np.linalg.norm(diff, axis=0)  # N,
        self.weights *= lh  # N,
        self.weights /= np.sum(self.weights)

        # resample
        N_eff = 1 / np.sum(np.square(self.weights))
        if N_eff < self.particle_num / 2:
            print("resample")
            self.resample()

        self.weights /= np.sum(self.weights)
        self.weights = softmax(self.weights)

        # estimate
        loc = np.average(self.particles, weights=self.weights, axis=0)
        loc = loc.reshape(-1, 1)

        return loc


if __name__ == "__main__":
    pos = np.array([[0, 0]]).T
    pf = ParticleFilter(pos)
    for i in range(10):
        u = np.array([[1, 0]]).T
        z = u - np.array([[0.1, 0]]).T
        pos += u
        pf(u, z, pos)
