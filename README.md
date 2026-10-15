# MeGA-MP: Metric Graph Advection Message Passing

## Setup Instructions (Linux/macOS)

After cloning the repository, a conda environment can be setup 
using the `setup_env.sh` script. Simply call

```
bash setup_env.sh
```

This will create a new conda environment called "quality". To change the name of the environment, change the name in `environment.yml`, and the
`ENV_NAME` variable in the `setup_env.sh` script.
It will automatically detect your OS (Linux and macOS) and installs torch CPU if you are on macOS. On Linux, it installs GPU-accelerated versions (with CUDA 11.8). All project dependencies will be installed using `pip` (as specified in `environment.yml`).

After this, activate the environment (with the name that you specified):

```
conda activate quality
```


## Running the Experiments

We have two experiments.

 * **Experiment 1** benchmarks our model on two metric graphs *Hanoi* and *L-Town*. These graphs represent water distribution systems - a perfect example of a real-world dynamical system governed by advection (and reactions) on metric graphs.
 * **Experiment 2** compares the performance of our model to that of an RK4, and a semi-Lagrangian solver. Since these cannot be applied to metric graphs, the second experiment aims to solve initial-value-, and boundary-value-problems on a 1-dimensional Euclidean domain.

**Runtime analysis** is performed for both experiments. 

**Baseline models** we compare to are NNConv, PDE-GNN, GPSConv, and A-DGN (in two variations). For a fair comparison, we run a hyperparameter tuning for each of the baselines.

Run Experiment 1 via

```bash
python run_experiments_pde_gnn.py
```

Run Experiment 2 via

```bash
python run_experiments_euclidean_1d.py
```

Run the runtime analysis for experiment 1 (for experiment 2, runtime is computed when running the experiment) via

```bash
python run_experiments_time_complexity.py
```

Run the hyperparameter tuning via

```bash
python -u tune_hyperparameters.py <model>
```
Replace `<model>` with the respective model to tune. It can be one of `GPSConv`, `NNConv`, `PDE-GNN`, `AntiSymmetricConv`.

***
***


<details>
  <summary>Template</summary>

```python

```
</details>

The message-passing advection solver can be used to solve linear advection problems on
Euclidean and Graph domains. In both cases, first the domain parameters must be 

## Semi-Laplacian Backtracing

**The code that computes the semi-Laplacian Backtracing ($\Delta t_{uv}(t)$) can be found in [`modules.semi_lagrangian_backtracing`](/modules/semi_lagrangian_backtracing.py).**
<details>
  <summary><b>Minimal code example to use this functionality</b></summary>

```python
from modules import semi_lagrangian_backtracing as slb
import matplotlib.pyplot as plt
import numpy as np

# Define temporal discretization (time step)
dt = 1.0
# Define edge lengths
l = [2, 3]
# Define velocities for each edge
v = [
  [1, 0.5, 2, 0.1, 0.8, -0.4, -0.8, -0.9],
  [1, 0.5, 2, 0.1, 0.8, -0.4, -0.8, -0.9],
]

# Compute the transit time steps and a mask, where True means backflow (selfloop)
time_steps, selfloops = slb.compute_backward_transit_times_fast(l, v, dt)

# Plot the results
fig, axs = plt.subplots(1, len(v), figsize=(11, 3), sharey=True, tight_layout=True)

for i, (ax, vi, ti, si) in enumerate(zip(axs, v, time_steps, selfloops)):
    ax.axhline(0, color='k', linestyle=':')
    ax.plot(vi, 'o-', label='Velocity')
    ax.plot(ti, 'o-', label=r'$\Delta t$')
    ax.plot(np.where(si, ti, np.nan), 'o-', color='C5', label='Backflow (selfloop)')
    ax.set_title(f'Edge {i}'); ax.legend(); ax.grid()
```
</details>
<br>

To apply the message-passing scheme, we need to find the time $\Delta t$ that it takes a particle to travel from one node to the next, i.e. to fully traverse a pipe.
Under constant flow conditions, this is trivial and can be computed by

$$
\nu_{uv} \Delta t = l_{uv} \\
\Leftrightarrow \Delta t = \frac{l_{uv}}{\nu_{uv}}
$$

This formula integrates the flow velocity $\nu_{uv}$ at edge $(u, v)$ over time such that the result is the length of that pipe $l_{uv}$. One can equivalently compute this by

$$
\int_{t - \Delta t}^t \nu_{uv} \; ds = \nu_{uv}t - \nu_{uv}(t - \Delta t) = \nu_{uv} \Delta t = l_{uv} \\
\Leftrightarrow \Delta t = \frac{l_{uv}}{\nu_{uv}}
$$

In conditions where flow changes over time, we need to computationally find $\Delta t$, such that the following holds:

$$
\int_{t - \Delta t}^t \nu_{uv}(s) \; ds = l_{uv}
$$

Here, $\nu_{uv}(s)$ is the non-constant function of the flows. The $\Delta t$ depends on $t$ and can be different for each edge, we thus have to compute $\Delta t_{uv}(t)$ for each edge and for each time step $t$.
Equally important is to find the $\Delta t$ where the integral is $-l_{uv}$ or $0$, i.e. when the flow is in opposite (negative) direction, or  the flow inverts and the particle flows into a pipe and out again at the same node:

$$
\min_{\Delta t \gt 0} \Delta t \quad \text{subject to} \quad \int_{t - \Delta t}^t \nu_{uv}(s) \; ds \in \{ -l_{uv}, 0, l_{uv} \} \quad \forall \; t \in [0, T] \; \forall \; (uv) \in E
$$

For small $t$, there will be no solution to this optimization problem, in practice this means that no particle has traversed the pipe yet and the solution is technically infinity. In practice, setting the result to $0$ or to the maximum time $T$ is equivalent to that.

## Shift and Time-Warping Operators

**The code for shift and time-warping can be found in [`modules.torch_advection`](/modules/torch_advection.py).**

The message function is either a shift operator if the flow is constant over time, or a time-warping operation if flow 
is time dependent.
Using the semi-Lagrangian backtracing described above, we can calculate the time a particle needs to travel through an edge.
The resulting $\delta t(t)$, are used to determine the shift or time warping pattern, respectively.

<details>
  <summary>Shift Operation</summary>

```python
from modules.torch_advection import AdvectionModuleGridSample
import matplotlib.pyplot as plt
import numpy as np
import torch

# Setup
T = 500             # number of time steps
shift_steps = 10.0  # number of steps to shift to the left
# The shift may be not an integer timestep, in those cases we interpolate.
# Three methods are available: 'nearest', 'bilinear', and 'bicubic'
interpolation = 'bilinear'

# Some input time series
x = np.sin(np.linspace(0, 10*np.pi, T))

# For constant flows, we have a shift operator:
shift_op = AdvectionModuleGridSample(interpolation_mode=interpolation)

# Convert to torch, add batch dimension
x = torch.as_tensor(x).unsqueeze(0).float()
delta_t_const = torch.as_tensor([shift_steps])

# Apply the shift operator
shifted_x = shift_op(x, -delta_t_const).numpy()

# Plot result
plt.figure(figsize=(15, 3))
plt.plot(x.squeeze(), label='Input sequence')
plt.plot(shifted_x.squeeze(), label=f'Shifted by {shift_steps} time steps')
plt.legend()
```
</details>

<details>
  <summary>Time-Warping Operation</summary>

```python
from modules.torch_advection import AdvectionModuleGridSampleDynamic
import matplotlib.pyplot as plt
import numpy as np
import torch

# Setup
T = 500
# The warping map can contain non-integer timesteps, 
# in those cases we interpolate.
# Three methods are available: 'nearest', 'bilinear', and 'bicubic'
interpolation = 'bilinear'

# Use a time series of delays to perform time warping
warping_map = np.sin(np.linspace(0, np.pi, T)) * 20

# Some input time series
x = np.sin(np.linspace(0, 11*np.pi, T))

# For constant flows, we have a shift operator:
warp_op = AdvectionModuleGridSampleDynamic(interpolation_mode='bilinear')

# Convert to torch, add batch dimension
x = torch.as_tensor(x).unsqueeze(0)
delta_t_dynamic = torch.as_tensor([warping_map])

# Apply the shift operator
warped_x = warp_op(x, -delta_t_dynamic)

# Plot result
plt.figure(figsize=(15, 3))
plt.plot(x.squeeze(), label='Input sequence')
plt.plot(warped_x.squeeze(), label=f'Warped sequence')
plt.legend()
```
</details>

## Code Walkthrough

This code provides functions to run advection dynamics using message-passing in different flavors.
1. General simulation of advection of graphs/networks
2. A concrete practical example of Chlorine traveling through a water distribution system.

> A good starting point to understand the code are the jupyter notebooks. For advection on a simple 1-dimensional domain (i.e. a path graph) s. `SemiLagrangianClassical_vs_MPNN.ipynb`, where we show a comparison of our method to an implementation of a classical Semi-Lagrangian solver.

### Code Structure

 * `baselines`: Code to run baselines and to create baseline scenarios quickly
   * `baselines.semi_lagrangian`: Implementation of a classical semi-Lagrangian solver (only for Euclidean domains, this one only for a 1-dimensional domain)
    * `baselines.utils`: Some handy functions to setup scenarios and evaluate baselines
    * `baselines.epanet`: The EPANET simulator is used for water distribution systems (i.e. the pipe network that transports water to consumers), it implements an advection solver for water quality. EPANET can only be run on `.inp`-files which define a water distribution system layout. This code is a wrapper to apply EPANET to 1-dimensional Euclidean advection problems.
 * `model`: This package contains the code for our message-passing advection solver
   * `model.advection_model_mp`: The class `AdvectionModelMP` implements functionality to solve an advection problem on a graph.
   * `model.semi_lagrangian_model_wrapper`: A wrapper to easily apply the `AdvectionModelMP` to 1-dimensional Euclidean domains.
 * `modules`: This contains functions that are used together with `AdvectionModelMP` or inside of `AdvectionModelMP`.
   * `modules.semi_lagrangian_backtracing`: Implements backward-tracing used in semi-Lagrangian solvers on a graph. For each edge and each timestep, finds the time it took a particle to either fully traverse the edge, or flow into the edge and return (in cases where flow changes direction under changing flow conditions).
   * `modules.torch_advection`: Code used to construct the messages for the message-passing scheme. Implements a differentiable time warping operator.
   * `modules.masking`: Code used for the message-passing scheme. Implements temporal masking which is applied to node histories after every message-passing step.
 * `networks`: Contains several water distribution network `.inp`-files to compare our approach to the EPANET simulator.
   
   
### Minimal Examples

<details>
  <summary>Minimal example Water Domain (Water Distribution System via EPANET inp file)</summary>

```python
from model.semi_lagrangian_model_wrapper import semi_lagrangian_mpnn
from epyt_flow import utils
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import utils as functions
import torch
import simulate

# ------------- 1. Define the Domain -------------

# Space
inp_file = 'networks/SimpleNet.inp'
topology = functions.read_inp(inp_file)
N = topology.num_nodes

# Time
N_SECONDS = utils.to_seconds(hours=12)
HYDRAULIC_STEP = 1 * 60
QUALITY_TIMESTEP = 1
PATTERN_STEP = HYDRAULIC_STEP * 2
nsteps = int(N_SECONDS / HYDRAULIC_STEP)
times = np.linspace(0, N_SECONDS, nsteps)

# ----- 2. Define the Initial Value Problem ------

# Define the Boundary Condition at the inflow node as a Gaussian pulse
n_waves = 3
seed = 42
pattern = functions.create_wavy_pattern(nsteps, n_waves, HYDRAULIC_STEP, seed)
sources_at = ['Injection']

# Define Initial State
initial_state = torch.zeros(N)

# ----- 3. Generate flow field using EPANET's hydraulic model ------
sim_setup_fns = [
    # Here we overwrite the demand values to manipulate the flows
    partial(functions.set_sim_demands, base=1600., pattern=functions.create_wavy_pattern(nsteps, 5, HYDRAULIC_STEP, seed)),
    # set_sim_demands overwrites supply demands required by MSX, so add them here
    partial(functions.set_sim_demands, base=-1600., pattern=functions.create_wavy_pattern(nsteps, 5, HYDRAULIC_STEP, seed), nodelist=sources_at),
]
graph_data = simulate.inp_to_graph_data(inp_file, pattern, sources_at, N_SECONDS, HYDRAULIC_STEP, QUALITY_TIMESTEP, f_msx_in='networks/ltown.msx', sim_setup_fns=sim_setup_fns)
topo =  graph_data['topology']

# -------------- 3. Run the Solver ----------------
pred, edge_passes, _, agg_time, aggs_all = semi_lagrangian_mpnn(
    initial_state, graph_data['flow_field'], graph_data['edge_index'], graph_data['edge_lengths'], 
    output_times=[nsteps], dt=HYDRAULIC_STEP, nsteps=nsteps, edge_capacities=graph_data['edge_diameter'],
    control_indices=graph_data['boundary_index'], control_inputs=graph_data['boundary_values'],
    interpolation='nearest', max_msg_passing_rounds=1000
)

# ------------ 4. Visualize the Results -----------
plt.figure(figsize=(16, 4))

for i, (mpr, er) in enumerate(zip(pred, graph_data['epanet_result'])):
    node = list(topology.nodes)[i]
    color = f'C{i}'
    plt.plot(mpr, color=color, label=f'MP Solution node "{node}"')
    plt.plot(er, color=color, linestyle='--', label=f'EPANET Solution "{node}"')
_ = plt.legend()
plt.show()
```
</details>

<details>
  <summary>Minimal example 1D Domain</summary>

```python
import numpy as np
from baselines.utils import create_pulse, create_1d_flow_field
import model.semi_lagrangian_model_wrapper
import matplotlib.pyplot as plt

# ------------- 1. Define the Domain -------------

# Spatio-Temporal Discretization
dx = dt = 0.1
nsteps = 2000

# Grid
L = 40.
N = int(L / dx)
x = np.linspace(0, L, N)

T = nsteps * dt
times = np.linspace(0, T, nsteps)

# Generate a Flow Field (constant, change via kind argument or create your own)
flow_field = create_1d_flow_field(L, dx, nsteps, kind='constant', value=0.5)

# ----- 2. Define the Initial Value Problem ------

# Define the Boundary Condition at the inflow node as a Gaussian pulse
pulse_location = 100
pulse_scale = 200
pulse_in = create_pulse(pulse_location, pulse_scale, nsteps)
boundary_index = np.array([0])

# Define Initial State
initial_state = np.zeros(N)

# -------------- 3. Run the Solver ----------------
result = model.semi_lagrangian_model_wrapper.semi_lagrangian_mpnn_1d(
    initial_state, flow_field, dx, L, times, dt, control_indices=boundary_index,
    control_inputs=pulse_in
)

# ------------ 4. Visualize the Results -----------
# s. options below
```
<ul>
<li><details>
  <summary><b>Visualize as Plot</b></summary>

```python
# Time step to visualize
t_idx = 800

# Plot results
plt.figure(figsize=(12, 3))
plt.plot(x, result[:,t_idx], label=fr'MPNN $t = {t_idx*dt}$')

# Compute the position of the pulse peak analytically
pulse_start_idx = pulse_location
analytical_peak_pos = flow_field[0, pulse_start_idx:t_idx+1].sum(0) * dt

plt.axvline(analytical_peak_pos, color='k', label='Analytical Peak Position', zorder=0, linestyle=':')
plt.xlabel('Domain [m]'); plt.ylabel('Mass')
plt.legend()
plt.grid(True)
plt.title('Comparison MPNN vs. Classical SL')
plt.xlim(analytical_peak_pos-30*dx, analytical_peak_pos+30*dx)
_ = plt.ylim(-0.05, 1.05)
plt.show()
```
</details></li>
<li><details>
  <summary><b>Visualize as Animation</b></summary>

```python
from baselines.utils import animate_solution
from ipywidgets import HTML

ani = animate_solution(result[:, pulse_location-50:950:2], xs=x, interval=1000 * dt / 3)
HTML(ani.to_html5_video())
```
</details></li>
</ul>
</span>
</details>
