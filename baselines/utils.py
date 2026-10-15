import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_pulse(loc, scale, N, dN=1.):
    return np.exp(-0.5 * (np.linspace(0, N*dN, N) - loc)**2/scale)

def create_1d_flow_field(domain_length, dx, nsteps, kind='constant', value=0.0021):
    n_gridpoints = int(domain_length / dx)

    if kind == 'constant':
        return np.ones([n_gridpoints, nsteps]) * value
    elif kind == 'sine_pos':
        wave = value + value * (np.sin(2*np.linspace(0, 2*np.pi, nsteps)))
        return wave[None].repeat(n_gridpoints, axis=0)
    
def create_2d_flow_field(domain, nsteps, kind='constant', value=0.0021):
    '''
    kind = 'sine_pos' generates a flow field that varies across time AND space.
    '''
    if kind == 'constant':
        return np.ones([*np.shape(domain)[1:], nsteps]) * value
    elif kind == 'sine_pos':
        wave = value + value * (np.sin((domain[0] * domain[1] * 0.001)[:,:,None] * 2*np.pi + np.linspace(0, 2 * np.pi, nsteps).reshape(1, 1, -1)))
        return wave
    
def animate_solution(solution, xs=None, **kwargs):
    '''
    Animates a 1-D solution, shape: (n_gridpoints, n_timesteps)
    The kwargs are arguments passed to matplotlibs FuncAnimation class.
    '''
    with plt.style.context('seaborn-v0_8'):
        fig = plt.figure(figsize=(12, 3), tight_layout=True)

        if xs is not None:
            plt.xlabel('Domain [m]')
        else:
            xs = np.arange(len(solution))
            plt.xlabel('Domain')
        
        state, = plt.plot(xs, solution[:, 0])
        
        plt.ylim(np.min(solution)-0.05, np.max(solution) + 0.05) 
        plt.ylabel('Mass')

        def update(frame):
            state.set_data((xs, frame))
            return state,

        ani = FuncAnimation(fig, update, frames=solution.T, blit=True, **kwargs)
        plt.close()
    return ani

def animate_solution2d(solution, xs=None, **kwargs):
    '''
    Animates a 2-D solution, shape: (n_gridpoints_y, n_gridpoints_x, n_timesteps)
    The kwargs are arguments passed to matplotlibs FuncAnimation class.
    '''
    fig = plt.figure(figsize=(6, 6), tight_layout=True)

    if xs is not None:
        plt.xlabel('Domain [m]')
    else:
        xs = np.meshgrid(np.arange(len(solution[0])), np.arange(len(solution)))
        plt.xlabel('Domain')
    
    ax = plt.gca()
    extent = (np.min(xs[0]), np.max(xs[0]), np.min(xs[1]), np.max(xs[1]))
    state = ax.imshow(solution[..., 0], extent=extent, vmin=solution.min(), vmax=solution.max(), origin='lower')
    
    #plt.ylim(np.min(solution)-0.05, np.max(solution) + 0.05) 
    #plt.ylabel('Mass')

    def update(frame):
        ax.clear()
        ax.imshow(frame, extent=extent, vmin=solution.min(), vmax=solution.max())

    ani = FuncAnimation(fig, update, frames=solution.transpose(2,1,0), blit=False, **kwargs)
    plt.close()
    return ani