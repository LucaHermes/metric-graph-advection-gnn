import numpy as np

def create_flow_field(x, t, kind='constant', value=0.0021):
    if 'random' in kind:
        return random_1d_flow_field(x, t, kind=kind, value=value)
    else:
        return smooth_1d_flow_field(x, t, kind=kind, value=value)

def smooth_1d_flow_field(x, t, kind='constant', value=0.0021):
    if np.ndim(x) < 1:
        x = np.array([x])
    if np.ndim(t) < 1:
        t = np.array([t])
    if kind == 'constant':
        return np.broadcast_to(value, (*np.shape(x), *np.shape(t)))
    elif kind == 'sine_pos':
        wave = value + value * (np.sin(4*np.pi * t))
        # constant over space: x = cost
        return wave[None] * np.ones_like(x)[:,None]
    elif kind == 'sine':
        wave = value * np.sin(3 * np.pi * t)
        return wave[None] * np.ones_like(x)[:,None]
    
def integral_1d_flow_field(x, t, kind='constant', value=0.0021):
    if np.ndim(x) < 1:
        x = np.array([x])
    if np.ndim(t) < 1:
        t = np.array([t])
    if kind == 'constant':
        return np.broadcast_to(value * t, (*np.shape(x), *np.shape(t)))
    elif kind == 'sine_pos':
        wave = value * (t - (np.cos(4 * np.pi * t)) / (4 * np.pi) + 1 / (4*np.pi))
        # constant over space: x = cost
        return wave[None] * np.ones_like(x)[:,None]
    elif kind == 'sine':
        wave = value * (-np.cos(3 * np.pi * t) / (3 * np.pi) + 1 / (3 * np.pi))
        return wave[None] * np.ones_like(x)[:,None]
    
def random_1d_flow_field(x, t, *args, **kwargs):
    L = np.max(x)   # domain length
    T = np.max(t)   # time duration
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    nt = int(T//dt)
    nx = int(L//dx)
    fx = 2*np.pi*np.fft.fftfreq(nt, d=dt)
    fy = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    K = FX**2 + FY**2

    # Fourier covariance operator
    spectral_density = 625.0 / (K + 0.01)**2

    # sample the GRF
    noise = np.random.normal(size=(nx, nt)) + 1j*np.random.normal(size=(nx, nt))
    field_hat = np.sqrt(spectral_density) * noise

    flow_field = np.fft.ifft2(field_hat).real
    return flow_field