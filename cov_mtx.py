import numpy as np
from funcapprox import findz
from scipy.linalg import null_space

def compute_cov_mtx(t, method=None, epsilon=None, delta=None, m_bound=None):
    """
    Compute covariance matrix given d dimensional data points
    :param t: d-by-n data matrix, each column is a data point
    :param method: which private algorithm to use for computing
        covariance matrix; if None, do not add privacy
    :return: d-by-d covariance matrix
    """
    # Standard, non-private way of computing cov
    cov_mtx = np.cov(t)
    d = t.shape[0]

    if method is None:
        assert epsilon is None
        return cov_mtx
    elif method == 'rejection_sampling':
        assert epsilon is not None
        assert delta is None

        # Initialize variables
        C = cov_mtx
        P = np.identity(d)
        eigs = np.linalg.eig(C)
        lambdas = eigs[0].real.reshape(d, 1)
        eps0 = epsilon/2
        lambdas_hat = lambdas + np.random.laplace(scale=2/eps0, size=(d, 1))
        thetas = np.array([])

        for i in range(1, d+1):
            print(f'Eigenvector sampling, iteration {i} ...')
            # Step a) sample u_i
            eps_i = epsilon / (2*d)
            u = rejection_sampling(C, eps_i, m_bound)
            theta = P.T @ u
            if i == 1:
                thetas = theta
            else:
                thetas = np.concatenate((thetas, theta), axis=1)

            # Step b) Find orthonormal basis orthogonal to existing thetas
            P_next = null_space(thetas.T).T

            # Step c) Update C
            C_next = P_next @ cov_mtx @ P_next.T

            P = P_next
            C = C_next

        # Compute private cov mtx
        sum = 0
        for i in range(1, d):
            theta_i = thetas[:, i-1]
            theta_i = theta_i.reshape(d, 1)
            sum = sum + lambdas_hat[i-1] * theta_i @ theta_i.T
        return sum
    elif method == 'analyze_gauss':
        # Compute sensitivity
        assert epsilon is not None and delta is not None
        sensitivity = np.sqrt((2*np.log(1.25/delta)) / epsilon)

        # Sample error matrix (upper triangular part)
        E = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                E[i, j] = np.random.normal(0, sensitivity**2)

        # fill the lower triangular part of the matrix with the upper triangular part
        E += np.triu(E, k=1).T
        assert check_symmetric(E), 'Error matrix must be symmetric!'
        return cov_mtx + E
    elif method == 'laplace':
        # Sample noise matrix
        assert epsilon is not None and delta is None
        n = t.shape[1]
        E = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                E[i, j] = np.random.laplace(scale=(2*d)/(n*epsilon))
        E += np.triu(E, k=1).T
        assert check_symmetric(E), 'Error matrix must be symmetric!'
        return cov_mtx + E
    else:
        raise ValueError('Unsupported covariance computation method!')


def rejection_sampling(C, eps, m_bound):
    d = C.shape[0]
    if d == 1:
        return np.array([1.0]).reshape(1, 1)
    lambdas = np.linalg.eig(C)[0].real
    lambda_d = np.flip(np.sort(lambdas))[-1]
    A = (-eps/4)*C + (eps/4)*lambda_d*np.identity(d)

    b = findz(-2*lambdas, 0.001)
    omega = np.identity(d) + (2*A)/b
    m = np.exp(-(d-b)/2) * ((d/b)**(d/2))
    if not (m > -m_bound and m < m_bound):
        m = m_bound
    # logM = - (d-b)/2 + (d/2)*np.log(np.abs(d/b))

    counter = 0
    while True:
        z = np.random.multivariate_normal(np.zeros(omega.shape[0]), np.linalg.inv(omega))
        u = z / np.linalg.norm(z)
        u = u.reshape(u.shape[0], 1)
        # if m == 0.0:
        #     prob = 1
        # elif np.isnan(0):
        #     prob = 0
        # else:
        #     prob = np.exp(-u.T @ A @ u) / m*(u.T @ Pi @ u)**(d/2)
        # prob = np.max([np.min([1, prob]), 0])

        # log_prob = -u.T @ A @ u - logM - (d/2)*np.log(u.T @ omega @ u)
        # prob = np.exp(log_prob)

        prob = np.exp(-u.T @ A @ u) / (m * (u.T @ omega @ u) ** (d / 2))

        if np.random.binomial(1, prob):
            return u
        else:
            print(f'Prob = {prob}, rejected')

        counter += 1
        if counter >= 100:
            import pdb; pdb.set_trace()

def newton(x0, num_steps, f, grad_f, disconts=None):
    x = x0
    for i in range(num_steps):
        # Take Newton step
        sl = 1
        grad = grad_f(x)
        x_next = x - sl*f(x)/grad

        def check_disconts(disconts, x_next):
            for disc in disconts:
                if abs(disc - x_next) < 1e-8:
                    return True
            return False

        if disconts is not None:
            # Check if we hit a discontinuity
            while True:
                if check_disconts(disconts, x_next):
                    sl = 0.9 * sl
                    x_next = x - sl*f(x)/grad
                else:
                    break
        x = x_next
        print(f"Iteration: {i}, x: {x}, size of grad: {np.linalg.norm(grad)}, f(x): {f(x)}")
    return x

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# if __name__ == '__main__':
#     # Sanity check Newton's method
#     def f(b):
#         return 1/(b+1)
#
#     def grad_f(b):
#         return -1/(b+1)**2
#
#     newton(2, 100, f, grad_f, [-1])