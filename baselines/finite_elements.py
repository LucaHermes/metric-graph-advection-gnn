from .weno import weno5
import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

def flux_split(u, c):
    """Lax-Friedrichs flux splitting"""
    alpha = np.max(np.abs(c))
    flux = c * u
    f_plus = 0.5 * (flux + alpha * u)
    f_minus = 0.5 * (flux - alpha * u)
    return f_plus, f_minus, alpha


def compute_rhs(t, u, flow_field, times, dx, ctrl_in=None, ctrl_idx=None):
    time_index = np.argmin(np.abs(times - t))
    c = flow_field[:, time_index]
    _u = u.copy()
    
    # Create extended domain with padding cells
    padding = 4  # For WENO5
    u_extended = np.zeros(len(_u) + 2*padding)
    
    # Fill interior points
    u_extended[padding:-padding] = _u
    
    # Left boundary (Dirichlet) - use control input value for BC
    if ctrl_in is not None and ctrl_idx is not None and 0 in ctrl_idx:
        left_value = ctrl_in[np.where(np.equal(ctrl_idx, 0))[0][0], time_index]
    else:
        left_value = _u[0]  # Fall back to current value
        
    # Fill left padding cells using anti-symmetric extension around Dirichlet value
    for i in range(padding):
        u_extended[padding-i-1] = 2*left_value - u_extended[padding+i]
    
    # Right boundary (outflow) - use zero-gradient
    for i in range(padding):
        #u_extended[-(i+1)] = u_extended[-(padding+1)]
        # Copy the last interior value
        u_extended[-(i+1)] = _u[-1]
    
    c_extended = np.zeros_like(u_extended)
    c_extended[padding:-padding] = c
    c_extended[:padding] = c[0]
    c_extended[-padding:] = c[-1]
    
    # Compute fluxes with WENO5
    f_plus, f_minus, alpha = flux_split(u_extended, c_extended)
    f_plus_flux = weno5(f_plus)
    f_minus_flux = weno5(f_minus[::-1])[::-1]
    # f_minus_flux = np.roll(f_minus_flux, -1)
    f_hat = f_plus_flux + f_minus_flux
    
    # Calculate derivatives on original domain
    df = np.zeros_like(u)
    df[1:-1] = -(f_hat[padding+1:-padding] - f_hat[padding:-padding-1]) / dx
    
    # Boundary condition (Dirichlet)
    df[0] = 0  # Controlled directly by input, so derivative = 0
    
    # Boundary condition (outflow; no boundary condition)
    # Single-sided difference at the boundary
    df[-1] = -(f_hat[-padding] - f_hat[-padding-1]) / dx
    
    # Apply control inputs
    if ctrl_idx is not None and ctrl_in is not None:
        u[u<0] *= 0  # avoid mutating global state
        u[ctrl_idx.clip(0)] = ctrl_in[:, time_index]
    
    # Freeze control locations
    if ctrl_idx is not None:
        df[ctrl_idx] = 0
        
    return df

import numpy as np
from types import SimpleNamespace

def solve_ivp_fixed_rk4(fun, t_span, y0, t_eval=None, args=None, **kwargs):
    t0, tf = t_span
    y0 = np.array(y0, dtype=float)
    if args is not None:
        f = lambda t, y: fun(t, y, *args)
    else:
        f = fun

    # If t_eval not provided, default to 100 uniform steps
    if t_eval is None:
        num_steps = 100
        t_eval = np.linspace(t0, tf, num_steps)
    else:
        t_eval = np.asarray(t_eval)
    
    dt_all = np.diff(t_eval)
    if not np.allclose(dt_all, dt_all[0]):
        raise ValueError("Non-uniform t_eval not supported in fixed-step RK4")

    dt = dt_all[0]
    t_values = [t_eval[0]]
    y_values = [y0]
    y = y0.copy()
    
    for i in range(1, len(t_eval)):
        t = t_eval[i-1]
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt / 2 * k1)
        k3 = f(t + dt / 2, y + dt / 2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        t_values.append(t_eval[i])
        y_values.append(y.copy())

    # Mimic solve_ivp output using SimpleNamespace
    return SimpleNamespace(
        t=np.array(t_values),
        y=np.stack(y_values, axis=1),  # shape (n, m) like solve_ivp
        success=True,
        message="Fixed-step RK4 completed successfully",
        status=0,
    )

def solve_ivp_prog(rhs, t_span, y0, t_eval=None, progress=True, **kwargs):
    if not progress:
        return solve_ivp_fixed_rk4(rhs, t_span, y0, t_eval=t_eval, **kwargs)

    progress_bar = tqdm(total=t_span[1] - t_span[0], desc="Solving", unit="s")

    def wrapped_rhs(t, y):
        # tqdm expects monotonic increases; clip at the end
        if not hasattr(wrapped_rhs, "last_t"):
            wrapped_rhs.last_t = t_span[0]
        dt = max(0.0, min(t - wrapped_rhs.last_t, t_span[1] - t_span[0]))
        progress_bar.update(round(dt, 2))
        wrapped_rhs.last_t = t
        return rhs(t, y)

    sol = solve_ivp_fixed_rk4(wrapped_rhs, t_span, y0, t_eval=t_eval, **kwargs)
    progress_bar.n = progress_bar.total
    progress_bar.close()
    return sol

def solve_advection_fdm(u0, flow_field, dx, L, times, dt, control_inputs=None, control_indices=None, progress=True):
    flow_field = np.asarray(flow_field)

    def rhs(t, u):
        return compute_rhs(t, u, flow_field, times, dx, ctrl_in=control_inputs, ctrl_idx=control_indices)

    t_span = (times[0], times[-1])
    max_step = min(1.0, 0.4 * dx / np.max(np.abs(flow_field)))
    sol = solve_ivp_prog(rhs, t_span, u0, method='RK45', t_eval=times, max_step=max_step, progress=progress)

    return sol.y

def solve_advection_fdm_fixed_dt(u0, flow_field, dx, L, times, dt, control_inputs=None, control_indices=None, progress=True):
    flow_field = np.asarray(flow_field)
    #print(flow_field.shape, u0.shape, times.shape)
    def rhs(t, u):
        return compute_rhs(t, u, flow_field, times, dx, ctrl_in=control_inputs, ctrl_idx=control_indices)

    t_span = (times[0], times[-1])
    sol = solve_ivp_prog(rhs, t_span, u0, method='RK45', t_eval=times, max_step=dt, min_step=dt, progress=progress)

    return sol.y#[:,1:]