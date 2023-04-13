import numpy as np



def compute_cov_mtx(t, method=None, epsilon=None):
    """
    Compute covariance matrix given d dimensional data points
    :param t: d-by-n data matrix, each column is a data point
    :param method: which private algorithm to use for computing
        covariance matrix; if None, do not add privacy
    :return: d-by-d covariance matrix
    """
    # Standard, non-private way of computing cov
    cov_mtx = np.cov(t)

    if method is None:
        assert epsilon is None
        return cov_mtx
    elif method == 'rejection_sampling':
        # Initialize variables
        C = cov_mtx
        d = t.shape[0]
        P = np.identity(d)
        eigs = np.linalg.eig(C)
        lambdas = eigs[0]
        eps0 = epsilon/2
        lambdas_hat = lambdas + np.random.laplace(2/eps0, size=(d, 1))
        thetas = []

        for i in range(1, d+1):
            # Step a) sample u_i
            eps_i = epsilon / (2*d)
            u = rejection_sampling(C, d, eps_i)
            theta = P.T @ u

            # Step b) Find orthonormal basis orthogonal to existing thetas
    elif method == 'analyze_gauss':
        pass
    else:
        raise ValueError('Unsupported covariance computation method!')

def rejection_sampling(C, d, eps):
    lambdas = np.linalg.eig(C)[0].real
    lambda_d = np.flip(np.sort(lambdas))[-1]
    A = (-eps/4)*C + (eps/4)*lambda_d*np.identity(d)

    # Use Newton's method to solve for b
    def f(b):
        return sum(1/(b + 2*lambdas))

    def grad_f(b):
        return sum(-1/(b + 2*lambdas)**2)

    x0 = np.mean(lambdas)
    b = newton(x0, 100, f, grad_f, disconts=lambdas)
    import pdb; pdb.set_trace()
    Pi = np.identity(d) + (2*A)/b

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


if __name__ == '__main__':
    # Sanity check Newton's method
    def f(b):
        return 1/(b+1)

    def grad_f(b):
        return -1/(b+1)**2

    newton(2, 100, f, grad_f, [-1])