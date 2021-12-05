import numpy as np


def kalmanFilter(mu, sigma, z, u):
    """
    assume
    x_t+1 = Ax_t + Bu_t+1 + r
    [x, y].T = I2[x, y].T + I2[u1, u2].T + sample(normal(0, R))
    z_t+1 = Cx_t+1 + q
    [z1, z2].T = I2[x, y].T + sample(normal(0, Q))

    @param mu ~ (2, 1)
    @param sigma ~ (2, 2)
    @param z ~ (2, 1)
    @param u ~ (2, 1)
    @param z ~ (2, 1)
    """
    A = np.eye(2)
    B = np.eye(2)
    C = np.eye(2)
    R = np.array([[9.24691775e-04, 1.44216463e-06],
                  [1.44216463e-06, 9.49292672e-04]])
    Q = np.array([[0.01783823, 0.00071094],
                  [0.00071094, 0.36105314]])

    # prediction step
    mu_bar = A @ mu + B @ u
    sigma_bar = A @ sigma @ A.T + R

    # correction step
    kal_gain = np.linalg.inv(C @ sigma_bar @ C.T + Q)
    K = sigma_bar @ C.T @ kal_gain
    mu = mu_bar + K @ (z - C @ mu_bar)
    sigma = (np.eye(2) - K @ C) @ sigma_bar

    return mu, sigma
