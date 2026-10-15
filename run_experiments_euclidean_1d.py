import warnings
import baselines.semi_lagrangian
warnings.filterwarnings("always", category=UserWarning)
import baselines.epanet
import baselines.finite_elements
import experiments.flow_fields as flow_fields
import model.semi_lagrangian_model_wrapper as mpnn
import numpy as np
import baselines
from modules.semi_lagrangian_backtracing import compute_backward_transit_times_fast
import torch
from tqdm.auto import tqdm
from functools import partial 
from joblib import Parallel, delayed
from copy import deepcopy
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import tree
import pandas as pd
from itertools import starmap
import time

from utils import MODEL_COLOR

from experiment_config_loader import get_euclidean_1d_config

EUCLIDEAN_CONFIG = get_euclidean_1d_config()
EXP_CONFIGS = EUCLIDEAN_CONFIG.get("experiments", [])
RESULTS_DIR = EUCLIDEAN_CONFIG.get("results_dir", "Results")

def gaussian_pulse(t, loc=120, scale=11):
    return np.exp(-0.5 * (t - loc)**2/scale**2)

def sine_wave(t, loc=None, scale=0.2):
    return 0.5 + 0.5 * np.sin(2 * np.pi * t * scale)

def tri_wave(t, loc=None, scale=None):
    return t % loc / (loc*scale)

def windowed_time_series(x, domain_length, style='gaussian', **ts_kwargs):
    # A window function that masks points outside of the domain:
    window = np.isclose(x, domain_length/2, atol=domain_length/2)
    # The Hann window scaled and clipped (flattop)
    window = (10 * window).clip(0, 1)
    if style == 'gaussian':
        ts = gaussian_pulse(x, **ts_kwargs) * window
    if style == 'sine_wave':
        ts = sine_wave(x, **ts_kwargs) * window
    if style == 'tri_wave':
        ts = tri_wave(x, **ts_kwargs) * window
    return ts

def analytical_solver(dt, dx, pde_setup, **kwargs):
    t0 = time.time()
    L = pde_setup['domain']['x']
    T = pde_setup['time']
    iv = pde_setup['initial_value']
    bc = pde_setup['boundary_condition']
    pde_setup = deepcopy(pde_setup)

    xs_fine = np.arange(0, L+dx, dx)
    times = np.arange(0, T+dt, dt)
    nsteps = len(times)

    # nsteps = int(np.ceil(T / dt))
    # times = np.linspace(0, T - T%dt, nsteps)

    flow_field = flow_fields.create_flow_field(
        xs_fine, times / T, **pde_setup['flow_field']
    )
    # solution boundary condition
    res = compute_backward_transit_times_fast(
        1e-12 + xs_fine, flow_field, dt, return_sl=False
    )
    
    bc_solution = 0
    if bc is not None:
        bc.pop('index')
        bc_solution = windowed_time_series(times[None] - res, T, **bc)
        bc_solution = np.nan_to_num(bc_solution)

    # loc_analytical is the distance the the particle at the input boundary traverses
    # solution initial value problem
    # xs_fine = np.arange(0, L+dx, dx)
    iv_solution = 0
    if iv is not None:
        loc_analytical = flow_fields.integral_1d_flow_field(
            xs_fine, times / T, **pde_setup['flow_field']
        ) * T
        iv_solution = windowed_time_series(xs_fine[:,None], L, loc=iv['loc']+loc_analytical, scale=iv['scale'])

    solution = (bc_solution + iv_solution)
    
    runtime = time.time() - t0
    return (dt, dx), (solution, runtime)

def apply_function(dt, dx, fn, pde_setup, **kwargs):
    pde_setup = deepcopy(pde_setup)
    L = pde_setup['domain']['x']
    T = pde_setup['time']
    iv = pde_setup['initial_value']
    bc = pde_setup['boundary_condition']
    #nsteps = int(np.ceil(T / dt))
    #times = np.linspace(0, T - T%dt, nsteps)
    times = np.arange(0, T+dt, dt)
    nsteps = len(times)
    #N = int(np.ceil(L / dx))
    # pulse_in, _ = compute_pulse(dt, nsteps)
    boundary_index = None
    pulse_in = None

    if bc is not None:
        boundary_index = np.array(bc.pop('index'))
        pulse_in = windowed_time_series(times, T, **bc)[None]

    xs = np.arange(0, L+dx, dx)
    N = len(xs) # int(np.ceil(L / dx))

    if iv is not None:
        initial_state = windowed_time_series(xs, L, loc=iv['loc'], scale=iv['scale'])
    else:
        initial_state = np.zeros(N)
    #initial_state = create_pulse(pulse_location-4, 2.5, N) # (dx, N)
    #initial_state = pulse_in# np.zeros(N)
    #boundary_index = np.array([N//2])
    
    flow_field = flow_fields.create_flow_field(
        xs, np.linspace(0, 1, nsteps), **pde_setup['flow_field']
    ) # * np.array([[1,-1],[1,1]]).repeat(int(L/dx/2), 0).repeat(nsteps//2, 1)
    flow_field = torch.tensor(flow_field)#create_1d_flow_field(L, dx, nsteps, value=vel_value, kind=kind))#sine_pos
    
    t0 = time.time()
    solution = fn(
        initial_state, flow_field, dx, L, times, dt, control_indices=boundary_index,
        control_inputs=pulse_in, progress=False, **kwargs #interpolation='nearest'
    )
    runtime = time.time() - t0

    return (dt, dx), (solution, runtime)

def save_result(result, path, file_name):
    # Result contains a list
    # [
    #     ...
    #     (dt, dx), (solution, runtime)
    #     ...
    # ]
    # We save the simulation results separately from the runtime
    dts_dxs, solutions_runtimes = zip(*result)
    solutions, runtimes = zip(*solutions_runtimes)
    sim_result = zip(dts_dxs, solutions)
    with open(os.path.join(path, file_name + '.pkl'), 'wb') as f:
        pickle.dump(sim_result, f)
    timing_results_file = os.path.join(path, file_name + '_runtime.csv')
    timing_results = dict(zip(dts_dxs, runtimes))
    df = pd.DataFrame(runtimes, columns=['seconds'])
    df.index = pd.MultiIndex.from_tuples(dts_dxs, names=['dt', 'dx'])
    df.index = df.index.map(lambda i: (*np.round(i, 2),))
    df.to_csv(timing_results_file)

def run_experiments():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, exp_setup in enumerate(EXP_CONFIGS):
        if i in [0, 1]:
           continue
        dxs = exp_setup['domain']['x'] / np.logspace(np.log(1000), np.log(10), 5, base=np.e).round()
        dts = exp_setup['time'] / np.logspace(np.log(1000), np.log(10), 5, base=np.e).round()
        # dxs = np.logspace(np.log2(0.1), 6 + np.log2(0.1) - 1, 6, base=2)
        # dts = np.logspace(np.log2(0.1)/np.log2(3), 6 + np.log2(0.1)/np.log2(3) - 1, 6, base=3)
        combinations = [ (dt, dx) for dt in dts for dx in dxs ]

        experiment_dir = os.path.join(RESULTS_DIR, f'experiment_{i}')
        os.makedirs(experiment_dir, exist_ok=True)
        #'''
        # run the EPANET solver for advection
        # _apply_function = partial(
        #     apply_function, fn=baselines.epanet.epanet_advection_1d, pde_setup=exp_setup
        # )
        # epanet_result = Parallel(n_jobs=10, backend="loky")(delayed(_apply_function)(dt, dx) for dt, dx in tqdm(combinations))
        # save_result(epanet_result, experiment_dir, 'epanet_result')
        
        # run the Semi-Lagrangian solver for advection
        _apply_function = partial(
            apply_function, fn=baselines.semi_lagrangian.semi_lagrangian_advection, pde_setup=exp_setup
        )
        sl_result = Parallel(n_jobs=10, backend="loky")(delayed(_apply_function)(dt, dx) for dt, dx in tqdm(combinations))
        save_result(sl_result, experiment_dir, 'sl_result')

        # run the runge-kutta-4 solver
        _apply_function = partial(
            apply_function, fn=baselines.finite_elements.solve_advection_fdm_fixed_dt, pde_setup=exp_setup
        )
        fdm_result = Parallel(n_jobs=10, backend="loky")(delayed(_apply_function)(dt, dx) for dt, dx in tqdm(combinations))
        save_result(fdm_result, experiment_dir, 'rk4_result')
        #'''
        # run the analytical solver
        _apply_function = partial(
            analytical_solver, pde_setup=exp_setup
        )
        result = Parallel(n_jobs=10, backend="loky")(
            delayed(_apply_function)(dt, dx) for dt, dx in tqdm(combinations)
        )
        save_result(result, experiment_dir, 'analytical_result')
        
        # run the mpnn solver
        mpnn_bilinear = partial(mpnn.semi_lagrangian_mpnn_1d, interpolation='bilinear', device='cpu')
        _apply_function = partial(
            apply_function, fn=mpnn_bilinear, pde_setup=exp_setup
        )
        result = Parallel(n_jobs=10, backend="loky")(
            delayed(_apply_function)(dt, dx) for dt, dx in tqdm(combinations)
        )
        save_result(result, experiment_dir, 'mpnn_result')

def draw_discretization_axis(fig, axs, dxs, dts):
    x0, y0, _, _ = fig.transFigure.inverted().transform(axs[-1][0].bbox.extents.reshape(-1, 2)).reshape(-1)
    _, _, x1, y1 = fig.transFigure.inverted().transform(axs[0][-1].bbox.extents.reshape(-1, 2)).reshape(-1)
    supax = fig.add_axes((x0, y0, x1-x0, y1-y0))

    dx0, dy0, dx1, dy1 = supax.transAxes.inverted().transform(axs[-1][0].bbox.extents.reshape(-1, 2)).reshape(-1)
    dx0, dy0 = dx0 + 0.5 * (dx1 - dx0), dy0 + 0.5 * (dy1 - dy0)
    _dx0, _dy0, dx1, dy1 = supax.transAxes.inverted().transform(axs[0][-1].bbox.extents.reshape(-1, 2)).reshape(-1)
    dx1, dy1 = _dx0 + 0.5 * (dx1 - _dx0), _dy0 + 0.5 * (dy1 - _dy0)

    supax.set_zorder(-1)
    supax.set_xlabel('dz', color='gray', fontsize=20)
    supax.set_ylabel('dt', rotation=0, color='gray', fontsize=20)
    supax.set_xticks(np.linspace(dx0, dx1, len(axs[0])), np.unique(dxs).round(2), color='gray', fontsize=16)
    supax.set_yticks(np.linspace(dy0, dy1, len(axs[0])), np.unique(dts).round(2)[::-1], color='gray', fontsize=16)
    supax.tick_params(colors='gray')
    supax.set_xlim(0, 1)
    supax.spines['top'].set_visible(False)
    supax.spines['right'].set_visible(False)
    supax.spines['left'].set_linewidth(2)
    supax.spines['bottom'].set_linewidth(2)
    supax.spines['left'].set_color('gray')
    supax.spines['bottom'].set_color('gray')
    supax.spines['bottom'].set_position(position=('outward', 20))
    supax.spines['left'].set_position(position=('outward', 20))
    return supax

def plot_results(exp_results_path, out_file):
    with open(exp_results_path / 'analytical_result.pkl', 'rb') as f:
        analytical_result_dict = dict(pickle.load(f))
    with open(exp_results_path / 'mpnn_result.pkl', 'rb') as f:
        mpnn_result_dict = dict(pickle.load(f))
    with open(exp_results_path / 'rk4_result.pkl', 'rb') as f:
        rk4_result_dict = dict(pickle.load(f))
    with open(exp_results_path / 'sl_result.pkl', 'rb') as f:
        sl_result_dict = dict(pickle.load(f))
    # with open(exp_results_path / 'epanet_result.pkl', 'rb') as f:
    #     epanet_result_dict = dict(pickle.load(f))
        
    dts, dxs = zip(*mpnn_result_dict.keys())
    
    fig, axs = plt.subplots(
        len(np.unique(dxs)[::1]), len(np.unique(dts)[::1]), figsize=(10, 5),
        sharex=True, sharey=True, constrained_layout=False,
    )
    fig.tight_layout(pad=0.5, w_pad=0.0, h_pad=0.)
    
    for i, dx in enumerate(np.unique(dxs)[::1]):
        for j, dt in enumerate(np.unique(dts)[::1]):
            res = analytical_result_dict[(dt, dx)]
            Y = L = res.shape[0] * dx
            X = T = res.shape[1] * dt
            xs = np.arange(0, L, dx)
            ts = np.arange(0, T, dt)
            time_idx = round((0.03125 * 100)/dt)
            res = mpnn_result_dict[(dt, dx)]
            y = res[:,-2]
            
            p1, = axs[j][i].plot(xs, y, color=MODEL_COLOR['MeGA-MP'], linestyle='-', label='MeGA-MP')
            res = analytical_result_dict[(dt, dx)]
            y = res[:,-2]
            p2, = axs[j][i].plot(xs, y, color='k', linestyle=':', label='Analytical')
            res = rk4_result_dict[(dt, dx)]
            y = res[:,-2]
            p3, = axs[j][i].plot(xs, y, color=MODEL_COLOR['RK4'], linestyle='-', label='RK4', zorder=0)
            res = sl_result_dict[(dt, dx)]
            y = res[:,-2]
            p4, = axs[j][i].plot(xs, y, color=MODEL_COLOR['Semi-Lagrangian'], linestyle='-', label='Semi-Lagrangian', zorder=0)
            
            axs[j][0].set_yticks([0, 1], [0, 1])
            axs[j][0].set_ylim(-0.05, 1.05)
            axs[j][0].set_xticks([])
            axs[j][0].set_yticks([])
            axs[j][0].set_ylabel(r'c(z)', rotation=90)
        axs[-1][i].set_xlabel('Space')
    fig.legend(handles=[p1, p3, p4, p2], loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.05))
    draw_discretization_axis(fig, axs, dxs, dts)
    fig.savefig(out_file, bbox_inches='tight')
    plt.close()
    
def plot_spacetime(input_file, out_file, difference_to_file=None):
    with open(input_file, 'rb') as f:
        result_dict = dict(pickle.load(f))
    if difference_to_file is not None:
        with open(difference_to_file, 'rb') as f:
            target_dict = dict(pickle.load(f))
        
    dts, dxs = zip(*result_dict.keys())
    
    fig, axs = plt.subplots(
        len(np.unique(dxs)[::1]), len(np.unique(dts)[::1]), figsize=(8, 8),
        sharex=True, sharey=True, constrained_layout=False,
    )
    fig.tight_layout(pad=0.5, w_pad=0.0, h_pad=0.)
    ims = []
    vmax = 0

    for i, dx in enumerate(np.unique(dxs)[::1]):
        for j, dt in enumerate(np.unique(dts)[::1]):
            res = result_dict[(dt, dx)]
            res = res.reshape(res.shape[:2])
            if difference_to_file is not None:
                res = np.abs(res - target_dict[(dt, dx)])
            X = L = (res.shape[0]-1) * dx
            Y = T = (res.shape[1]-1) * dt
            ims.append(axs[j][i].imshow(
                res.T, interpolation='nearest', extent=(0, X, 0, Y), origin='lower', 
                cmap='jet', aspect='equal', vmin=0, vmax=1
            ))
            vmax = min(1, max(vmax, res.max()))
            axs[j][i].set_ylim(0, Y)
            axs[j][i].set_xlim(0, X)
            axs[j][0].set_xticks([])
            axs[j][0].set_yticks([])
            axs[j][0].set_ylabel('Time')
        axs[-1][i].set_xlabel('Space')
    [ im.set_clim(vmax=vmax) for im in ims ]
    draw_discretization_axis(fig, axs, dxs, dts)

    x1, y1, w1, h1 = axs[-1][0].bbox.bounds
    x2, y2, w2, h2 = axs[0][-1].bbox.bounds
    _, y = fig.transFigure.inverted().transform([x1, y1])
    x, h = fig.transFigure.inverted().transform([x2 + w2, y2 + h2])
    cax = fig.add_axes((x + 0.02, y, 0.02, h - y))
    cb = plt.colorbar(ims[0], cax=cax)
    if difference_to_file is not None:
        cb.set_label('Absolute Error')
    else:
        cb.set_label(r'Solution $c(z,t)$')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close()

def plot_spacetime_dt(input_file, out_file, difference_to_file=None, dx=1.0):
    with open(input_file, 'rb') as f:
        result_dict = dict(pickle.load(f))
    if difference_to_file is not None:
        with open(difference_to_file, 'rb') as f:
            target_dict = dict(pickle.load(f))
        
    dts, dxs = zip(*result_dict.keys())
    
    fig, axs = plt.subplots(
        1, len(np.unique(dts)[::1]), figsize=(8, 3),
        sharey=True, constrained_layout=False,
    )
    fig.tight_layout(pad=0.5, w_pad=0.0, h_pad=0.)
    ims = []
    vmax = 0

    for ax, dt in zip(axs, np.unique(dts)[::1]):
        res = result_dict[(dt, dx)]
        res = res.reshape(res.shape[:2])
        if difference_to_file is not None:
            res = np.abs(res - target_dict[(dt, dx)])
        X = (res.shape[0]-1) * dx
        Y = (res.shape[1]-1) * dt
        ims.append(ax.imshow(
            res.T, interpolation='nearest', extent=(0, X, 0, Y), origin='lower', 
            cmap='jet', aspect='equal', vmin=0, vmax=1
        ))
        vmax = min(1, max(vmax, res.max()))
        ax.set_ylim(0, Y); ax.set_xlim(0, X)
        ax.set_xticks([0, 0.5 * X, X]); ax.set_yticks([0, 0.5 * Y, Y])
        ax.set_title(fr'$dt = {dt:.3f}$')
    axs[2].set_xlabel(fr'Space $z$ ($dz = {dx}$)')
    axs[0].set_ylabel(r'Time $t$')
    [ im.set_clim(vmax=vmax) for im in ims ]

    pos = axs[-1].get_position()
    cax = fig.add_axes((pos.x1 + 0.02, pos.y0, 0.02, pos.height))
    cb = plt.colorbar(ims[0], cax=cax)
    
    cb.set_label('Absolute Error' if difference_to_file is not None else r'Solution $c(z,t)$')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close()

def save_metrics(exp_results_path, out_file):
    keys = ['Semi-Lagrangian', 'RK4', 'MeGA-MP']
    names = ['sl_result.pkl', 'rk4_result.pkl', 'mpnn_result.pkl']

    def mae(pred, y_true):
        ae = np.abs(pred.squeeze() - y_true)
        return ae.mean(), ae.std()

    columns = pd.MultiIndex.from_product([keys, ['mean', 'std']])
    metrics_df = pd.DataFrame(columns=columns)

    with open(exp_results_path / 'analytical_result.pkl', 'rb') as f:
        ground_truth = dict(pickle.load(f))

    for key, filename in zip(keys, names):
        with open(exp_results_path / filename, 'rb') as f:
            result = dict(pickle.load(f))

        l1 = tree.map_structure(mae, result, ground_truth)
        df = pd.DataFrame.from_dict(l1, orient='index', columns=['mean', 'std'])
        df.index = pd.MultiIndex.from_tuples(df.index.map(lambda i: (*np.round(i, 2),)))
        metrics_df[key] = df

    metrics_df.round(4)
    metrics_df.to_csv(out_file)
    print('=' * 100)
    print(exp_results_path)
    print('-' * 100)
    print_latex(metrics_df, out_file.replace('.csv', '.txt'))
    print('=' * 100)

    # Save Runtimes
    keys = ['Semi-Lagrangian', 'RK4', 'MeGA-MP', 'Analytical']
    names = ['sl_result_runtime.csv', 'rk4_result_runtime.csv', 'mpnn_result_runtime.csv', 'analytical_result_runtime.csv']

    
    df_method = pd.read_csv(exp_results_path / names[0], index_col=[0,1])
    df = pd.DataFrame()
    df.index = df_method.index

    for key, filename in zip(keys, names):
        df_method = pd.read_csv(exp_results_path / filename, index_col=[0,1])
        df[key] = df_method
    
    p = Path(out_file)
    df.to_csv(str(p.parent / p.stem) + '_runtime.csv')
    print('-' * 100)
    print_runtime_latex(df, exp_results_path / 'all_runtimes_table.txt')
    print('=' * 100)


def print_latex(metrics_df, out_file):
    TABLE_ROW_FMT = '{}{{\scriptsize$\pm {}$}}'
    
    last = None
    latex = ''

    for row in range(len(metrics_df)):
        row = metrics_df.iloc[row]
        dt_dx = f'  & \hfill{row.name[1]:5.2f}'
        if row.name[0] != last:
            dt_dx = '\hline \multirow[t]{{5}}{{*}}{{${}$}} & \hfill${:5.2f}$'.format(*row.name)
            last = row.name[0]

        values = np.reshape(list(map('{:.4f}'.format, row.values)), (-1, 2)).astype('<U32')
        lowest = row.values[::2].min()
        
        for i, val in enumerate(row.values[::2]):
            if not val.round(4) == lowest.round(4): continue
            values[i][0] = f'\\textbf{{{values[i][0]}}}'
        
        row_cells = np.array(list(starmap(TABLE_ROW_FMT.format, values)))
        row_cells[row.values[::2] > 100] = ['--']
        row_cells[np.isnan(row.values[::2])] = ['--']
        table_row = ' & '.join(row_cells)
        latex += dt_dx + ' & ' + table_row + ' \\\\ \n'
    with open(out_file, 'w') as f:
        f.write(latex)
    print(latex)

def print_runtime_latex(runtime_df, out_file):
    TABLE_ROW_FMT = '{}'
    
    last = None
    not_analytical_mask = runtime_df.columns != 'Analytical'
    latex = ''

    for row in range(len(runtime_df)):
        row = runtime_df.iloc[row]
        dt_dx = f'  & \hfill{row.name[1]:5.2f}'
        if row.name[0] != last:
            dt_dx = '\hline \multirow[t]{{5}}{{*}}{{${}$}} & \hfill${:5.2f}$'.format(*row.name)
            last = row.name[0]

        values = np.array(list(map('{:.4f}'.format, row.values))).astype('<U32')
        lowest = row[not_analytical_mask].values.min()
        
        for i, val in enumerate(row.values):
            if not val.round(4) == lowest.round(4): continue
            values[i] = f'\\textbf{{{values[i]}}}'
        
        row_cells = np.array(list(map(TABLE_ROW_FMT.format, values)))
        
        table_row = ' & '.join(row_cells)
        latex += dt_dx + ' & ' + table_row + ' \\\\ \n'
    with open(out_file, 'w') as f:
        f.write(latex)
    print(latex)

def vizualize_results():
    exp_results = sorted(list(Path(RESULTS_DIR).glob('exp*')))
    experiment_dir = os.path.join(RESULTS_DIR, 'experiment_{}')
    figure_file_comp = os.path.join(experiment_dir, 'solver_comparison_varying_discretization.pdf')
    figure_file_st = os.path.join(experiment_dir, '{}_spacetime_varying_discretization.pdf')
    figure_file_st_sol = os.path.join(experiment_dir, '{}_spacetime_varying_discretization_solution.pdf')
    #figure_analytical_file_st = os.path.join(experiment_dir, 'analytical_solver_spacetime_varying_discretization.pdf')
    metrics_file_st = os.path.join(experiment_dir, 'solver_metrics_varying_discretization.csv')

    method_ids = ['analytical_result', 'mpnn_result']#, 'rk4_result', 'sl_result']
    
    for i, path in enumerate(exp_results):   # <<<<<< For debugging use start=2 to skip the first two experiments (REMOVE LATER!!!)
        plot_results(path, out_file=figure_file_comp.format(i))
        
        for method_id in method_ids:
            plot_spacetime(path / f'{method_id}.pkl', out_file=figure_file_st_sol.format(i, method_id))
            plot_spacetime_dt(
                path / f'{method_id}.pkl', 
                out_file=figure_file_st.format(i, f"{method_id}_dt"),
            )
            plot_spacetime(
                path / f'{method_id}.pkl', 
                out_file=figure_file_st.format(i, method_id),
                difference_to_file=path / 'analytical_result.pkl' # compute error w.r.t. ground truth
            )
        save_metrics(path, out_file=metrics_file_st.format(i))

if __name__ == '__main__':
    run_experiments()
    vizualize_results()
