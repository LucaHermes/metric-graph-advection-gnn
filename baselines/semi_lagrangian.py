import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
from .weno import weno5_interpolate

def backward_trace(i, x_grid, flows, dt, dx, L):
    '''
    Computes the integral from t-dt to t over the flows. I.e. calculates the 
    distance traveled over the last time step (dx_t). This is used for the 
    semi-Lagranian parcel tracing. 
    '''
    N = len(flows)
    
    if np.isclose(flows.std(), 0):
        return x_grid[i] - dt * flows[i]
    else:
        # Define characteristic ODE with velocity interpolation
        def char_ode(t, x):
            x_pos = x % L
            idx = min(max(0, int(x_pos/dx)), N-2)
            alpha = (x_pos - x_grid[idx]) / dx
            return -(1-alpha) * flows[idx] - alpha * flows[idx+1]
        
        # Trace characteristic backward
        sol = solve_ivp(
            char_ode, 
            [0, dt], 
            [x_grid[i]], 
            method='RK45',
            rtol=1e-9, 
            atol=1e-10
        )
        
        # Departure point
        return sol.y[0, -1]

def semi_lagrangian_advection(
    initial_state, flow_field, dx, L, output_times, dt, control_inputs=None, 
    control_indices=None, interpolation='linear', progress=True
):
    flow_field = np.asarray(flow_field)
    flow_field = np.pad(flow_field[:,:-1], ((0,0),(1,0)))
    N = len(initial_state)  # Number of grid points
    T = flow_field.shape[1]  # Number of time steps
    
    # Create grid
    x_grid = np.linspace(0, L, N)
    
    # Create time grid (assuming uniform spacing)
    _dt = output_times[-1] / (T - 1)
    #assert _dt == dt
    t_grid = np.linspace(0, output_times[-1], T)
    
    solution = np.zeros((N, len(output_times)))
    current_state = initial_state.copy()
    
    # Store initial state if t=0 is in output_times
    if output_times[0] == 0:
        solution[:, 0] = current_state
        output_index = 1
    else:
        output_index = 0

    const_flows = np.isclose(0, flow_field.std(1)).all()
    
    if const_flows:
        step_shift = x_grid[:,None] - dt * flow_field
        step_shift = step_shift.clip(0, L)

    if progress:
        iterations = tqdm(range(1, T))
    else:
        iterations = range(1, T)

    # For each time step
    for t_idx in iterations:
        velocity = flow_field[:, t_idx]
                
        # Calculate maximum step size (CFL condition)
        max_step = 0.5 * dx / max(np.max(np.abs(velocity)), 1e-6)
        max_step = min(max_step, dt/5)
        new_state = np.zeros_like(current_state)

        # Apply control inputs
        if control_inputs is not None and control_indices is not None:
            for j, idx in enumerate(control_indices):
                new_state[idx] = control_inputs[j, t_idx] #* dt

        x_targets = []

        if const_flows:
            x_targets = step_shift[:,t_idx]
        else:
            for i in range(N):
                x_departure = backward_trace(i, x_grid, velocity, dt, dx, L)
                # We set the boundary non-periodic, for periodic, use x_departure % L
                x_departure = np.clip(x_departure, 0, L)
                x_targets.append(x_departure)
                
        interp_solution = interpolate.interp1d(
            x_grid,
            current_state,
            kind=interpolation,
            bounds_error=False,
            fill_value=(0, 0)
        )(x_targets)
        
        # interp_solution = weno5_interpolate(
        #     current_state, x_grid, x_targets, dx
        # )

        new_state = np.nan_to_num(interp_solution)
        
        # Apply control inputs
        if control_inputs is not None and control_indices is not None:
            for j, idx in enumerate(control_indices):
                new_state[idx] = control_inputs[j, t_idx] #* dt
        
        # Update current state
        current_state = new_state.copy()

        if output_index < len(output_times) and abs(t_grid[t_idx] - output_times[output_index]) < dt/2:
            solution[:, output_index] = current_state
            output_index += 1
    
    return solution
