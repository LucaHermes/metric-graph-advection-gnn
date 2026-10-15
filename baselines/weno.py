import numpy as np

def weno5(f, eps=1e-6):
    """
    1-D WENO5 scheme, s. https://www3.nd.edu/~yzhang10/WENO_ENO.pdf
    """
    f = np.pad(f, (2, 2), mode='edge')

    # Create shifted versions for the 5 stencils
    #  Example function     [ X, X, 0, 1, 2, 3, 4, X, X ]
    #  Stencil i-2          [ X, X, 0, 1                ] -> i - 2
    #  Stencil i-1          [    X, 0, 1, 2             ] -> i - 1
    #  Stencil i            [       0, 1, 2, 3          ] -> i
    #  Stencil i+1          [          1, 2, 3, 4       ] -> i + 1
    #  Stencil i+2          [             2, 3, 4, X    ] -> i + 2
    
    f0 = f[:-5]
    f1 = f[1:-4]
    f2 = f[2:-3]
    f3 = f[3:-2]
    f4 = f[4:-1]
    
    # Stencils for reconstruction of f_{i+1/2}
    beta_0 = (13/12) * (f0 - 2 * f1 + f2)**2 + (1/4) * (f0 - 4 * f1 + 3*f2)**2
    beta_1 = (13/12) * (f1 - 2 * f2 + f3)**2 + (1/4) * (f1 - f3)**2
    beta_2 = (13/12) * (f2 - 2 * f3 + f4)**2 + (1/4) * (3 * f2 - 4 * f3 + f4)**2

    gamma0, gamma1, gamma2 = (1/10), (3/5), (3/10)
    
    alpha0 = gamma0 / (eps + beta_0)**2
    alpha1 = gamma1 / (eps + beta_1)**2
    alpha2 = gamma2 / (eps + beta_2)**2
    alphas = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / alphas
    w1 = alpha1 / alphas
    w2 = alpha2 / alphas

    f_hat0 = (2*f0 - 7*f1 + 11*f2) / 6
    f_hat1 = (-f1 + 5*f2 + 2*f3) / 6
    f_hat2 = (2*f2 + 5*f3 - f4) / 6

    result = w0 * f_hat0 + w1 * f_hat1 + w2 * f_hat2
    
    return result

def weno5_biased(f, eps=1e-6):
    """
    1-D WENO5 scheme, s. https://www3.nd.edu/~yzhang10/WENO_ENO.pdf
    """
    f = np.pad(f, (3, 3), mode='edge')

    # Create shifted versions for the 5 stencils
    #  Example function     [ X, X, 0, 1, 2, 3, 4, X, X ]
    #  Stencil i-2          [ X, X, 0, 1                ] -> i - 2
    #  Stencil i-1          [    X, 0, 1, 2             ] -> i - 1
    #  Stencil i            [       0, 1, 2, 3          ] -> i
    #  Stencil i+1          [          1, 2, 3, 4       ] -> i + 1
    #  Stencil i+2          [             2, 3, 4, X    ] -> i + 2
    
    f0 = f[:-5]
    f1 = f[1:-4]
    f2 = f[2:-3]
    f3 = f[3:-2]
    f4 = f[4:-1]
    
    # Stencils for reconstruction of f_{i+1/2}
    beta_0 = (13/12) * (f0 - 2 * f1 + f2)**2 + (1/4) * (f0 - 4 * f1 + 3*f2)**2
    beta_1 = (13/12) * (f1 - 2 * f2 + f3)**2 + (1/4) * (f1 - f3)**2
    beta_2 = (13/12) * (f2 - 2 * f3 + f4)**2 + (1/4) * (3 * f2 - 4 * f3 + f4)**2

    gamma0, gamma1, gamma2 = (1/10), (3/5), (3/10)
    
    alpha0 = gamma0 / (eps + beta_0)**2
    alpha1 = gamma1 / (eps + beta_1)**2
    alpha2 = gamma2 / (eps + beta_2)**2
    alphas = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / alphas
    w1 = alpha1 / alphas
    w2 = alpha2 / alphas

    f_hat0 = (2*f0 - 7*f1 + 11*f2) / 6
    f_hat1 = (-f1 + 5*f2 + 2*f3) / 6
    f_hat2 = (2*f2 + 5*f3 - f4) / 6

    result = w0 * f_hat0 + w1 * f_hat1 + w2 * f_hat2
    
    return result

def weno5_interpolate(f, x_grid, x_target, dx, eps=1e-6):
    """
    1-D WENO5 scheme, s. https://www3.nd.edu/~yzhang10/WENO_ENO.pdf
    """
    f = np.pad(f, (3, 3), mode='edge')
    x_target_hat = (x_target - x_grid[0]) / dx
    j = np.floor(x_target_hat).astype(int)

    xi = x_target_hat - j + 1
    j += 3
    
    f0 = f[j-2]
    f1 = f[j-1]
    f2 = f[j]
    f3 = f[j+1]
    f4 = f[j+2]
    
    # Stencils for reconstruction of f_{i+1/2}
    beta_0 = (13/12) * (f0 - 2 * f1 + f2)**2 + (1/4) * (f0 - 4 * f1 + 3*f2)**2
    beta_1 = (13/12) * (f1 - 2 * f2 + f3)**2 + (1/4) * (f1 - f3)**2
    beta_2 = (13/12) * (f2 - 2 * f3 + f4)**2 + (1/4) * (3 * f2 - 4 * f3 + f4)**2

    gamma0, gamma1, gamma2 = (1/10), (3/5), (3/10)
    
    alpha0 = gamma0 / (eps + beta_0)**2
    alpha1 = gamma1 / (eps + beta_1)**2
    alpha2 = gamma2 / (eps + beta_2)**2
    alphas = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / alphas
    w1 = alpha1 / alphas
    w2 = alpha2 / alphas
    
    f_hat0 = f0 * ((xi - 1)*(xi - 2) / 2) - f1 * ((xi)*(xi - 2)) + f2 * ((xi)*(xi - 1) / 2)
    f_hat1 = f1 * ((xi - 1)*(xi - 2) / 2) - f2 * ((xi)*(xi - 2)) + f3 * ((xi)*(xi - 1) / 2)
    f_hat2 = f2 * ((xi - 1)*(xi - 2) / 2) - f3 * ((xi)*(xi - 2)) + f4 * ((xi)*(xi - 1) / 2)

    result = w0 * f_hat0 + w1 * f_hat1 + w2 * f_hat2
    
    return result