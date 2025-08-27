"""
To generate plot and csv file for quantum memory dip experiment, relaxing at H_relax
"""

from spirit import simulation, state, quantities, hamiltonian, parameters, geometry, configuration, system, io
import numpy as np
import os
import multiprocessing as mp
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import re
from functools import partial
import plotly.graph_objects as go


# Global variable to hold DDI matrices
DDI_interaction_x = None
DDI_interaction_y = None
DDI_interaction_z = None


def get_unique_filename(filepath):
    """Just to make sure file names don't clash and overwrite existing data."""
    if not os.path.exists(filepath):
        return filepath

    dirpath, fname = os.path.split(filepath)
    name, ext = os.path.splitext(fname)
    m = re.search(r'(\d+)$', name)
    if not m:
        raise ValueError("Filename must end with a digit to be incremented")

    prefix = name[:m.start(1)]
    num = int(m.group(1))

    # Increment until we find a filename that doesn't exist
    while True:
        num += 1
        new_fname = f"{prefix}{num}{ext}"
        new_path = os.path.join(dirpath, new_fname)
        if not os.path.exists(new_path):
            return new_path


def get_Hts(H_high, H_low, H_steps_1, H_steps_3):
    """
    Produce the array of Ht for MC iterations.
    H_relax_steps copies of Ht = H_relax and 2 copies of other Hts.
    """
    H1 = [H_high] * H_steps_1
    H2 = [H_low] * H_steps_1
    H3 = [H_high] * H_steps_3
    Hts = H1 + H2 + H3

    return Hts


def get_last_indices(Hts):
    """Get the index of the last occurrence of each unique Ht, where susceptibility should be measured."""
    unique_vals, first_indices = np.unique(Hts, return_index=True)
    last_indices = np.append(first_indices[:-1] - 1, len(Hts) - 1)
    return last_indices


def init_worker(dim):
    """Initialize DDI matrix once per worker to avoid reloading in each loop."""
    global DDI_interaction_x, DDI_interaction_y, DDI_interaction_z
    fn = "dipolar_arr"
    base_path = f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/"
    DDI_interaction_x = np.load(os.path.join(base_path, fn + "_x.npy"))
    DDI_interaction_y = np.load(os.path.join(base_path, fn + "_y.npy"))
    DDI_interaction_z = np.load(os.path.join(base_path, fn + "_z.npy"))


def sparse_matmul(DDI_matrix, spin_component, scale=7 / 1e4):
    """
    Optimized multiplication:
    Most spins are 0 due to vacancies, so we remove those indices before multiplying.
    """
    nonzero_indices = np.nonzero(spin_component)[0]
    spin_nonzero = spin_component[nonzero_indices]
    DDI_submatrix = DDI_matrix[:, nonzero_indices]
    result = DDI_submatrix @ spin_nonzero * scale
    return result


def compute_DDI(spins, DDI_interaction_x, DDI_interaction_y, DDI_interaction_z):
    """Compute dipolar interaction (DDI) field contributions for a given spin configuration."""
    DDI_field_x_from_z = sparse_matmul(DDI_interaction_x, spins[:, 2])
    DDI_field_y_from_z = sparse_matmul(DDI_interaction_y, spins[:, 2])
    DDI_field_z_from_z = sparse_matmul(DDI_interaction_z, spins[:, 2])
    DDI_field_z_from_y = sparse_matmul(DDI_interaction_y.T, spins[:, 1])
    DDI_field_z_from_x = sparse_matmul(DDI_interaction_x.T, spins[:, 0])

    # Sum total z-field contributions
    DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x
    return DDI_field_x_from_z, DDI_field_y_from_z, DDI_field_z_total


def get_paramagnetic_pstate(p_state, Hmax, relax_steps_0, vacancies_idx, nos):
    """
    Spirit's configuration.random gives random spins everywhere.
    We evolve for relax_steps_0 number of steps so anisotropic field makes the initial state approximately Ising.
    """
    configuration.random(p_state)
    Hts_randomise = [Hmax] * relax_steps_0

    for Ht in Hts_randomise:
        hamiltonian.set_field(p_state, Ht, (Ht, 0, 0))

        spins = system.get_spin_directions(p_state)
        spins[vacancies_idx] = 0

        Dx, Dy, Dz = compute_DDI(spins, DDI_interaction_x, DDI_interaction_y, DDI_interaction_z)
        DDI_field_interleave = np.ravel(np.column_stack((Dx, Dy, Dz)))
        system.set_DDI_field(p_state, n_atoms=nos, ddi_fields=DDI_field_interleave)

        simulation.start(p_state, simulation.METHOD_MC, single_shot=False)
        simulation.stop(p_state)


def save_and_plot_results(results, dim, n_cycles, anisotropy, gamma):
    """Save results to CSV and plot averaged susceptibility vs Ht."""
    # Flatten results into list of (H_relax, Ht, chi, gamma)
    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(get_unique_filename(
        f"negative_field_cycle_v3_dim{dim}_anisotropy{anisotropy}_ncycles{n_cycles}_gamma{gamma}_H_high{H_high}_H_low{H_low}.csv"),
        index=False)

    grouped = df_all.groupby(['i', 'Ht']).agg(
        chi_mean=('chi', 'mean'),
        chi_std=('chi', 'std')
    ).reset_index()

    grouped['chi_sem'] = grouped['chi_std'] / (n_cycles ** 0.5)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grouped["i"],
        y=grouped["chi_mean"],
        error_y=dict(
            type="data",
            array=grouped["chi_sem"],
            visible=True
        ),
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=5),
        name="Mean χ",
        hovertemplate="i=%{x}<br>χ=%{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Chi vs MCS Step, gamma = {gamma}",
        xaxis_title="MCS Step (i)",
        yaxis_title="Mean χ",
        template="plotly_white"
    )

    fig.write_html(get_unique_filename(
        f"negative_field_cycle_v3_dim{dim}_anisotropy{anisotropy}_ncycles{n_cycles}_gamma{gamma}_H_high{H_high}_H_low{H_low}.html"))


def compute_chi(Ht, p_state, vacancies_idx, mu, nos):
    """Compute susceptibility χ via linear fit of magnetization vs applied B-field."""

    # Apply longitudinal field to sample to measure susceptibility
    # Randomly select +-z direction to measure susceptibility, to avoid constantly polarizing sample in one direction
    Bfields_positive = np.arange(0, 0.2, 0.05)
    Bfields_negative = np.arange(0, -0.2, -0.05)
    Bfields = Bfields_positive if np.random.rand() < 0.5 else Bfields_negative
    for i, HzB in enumerate(Bfields):
        Hmag = np.sqrt(HzB * HzB + Ht * Ht)
        hamiltonian.set_field(p_state, Hmag, (Ht, 0, HzB))

        spins = system.get_spin_directions(p_state)
        spins[vacancies_idx] = 0

        Dx, Dy, Dz = compute_DDI(spins, DDI_interaction_x, DDI_interaction_y, DDI_interaction_z)
        DDI_field_interleave = np.ravel(np.column_stack((Dx, Dy, Dz)))
        system.set_DDI_field(p_state, n_atoms=nos, ddi_fields=DDI_field_interleave)

        converge_threshold = 1e-9
        converge_max = 1

        if i != 0:
            for j in range(converge_max):
                simulation.start(p_state, simulation.METHOD_MC, single_shot=False)
                simulation.stop(p_state)
                if j == 0:
                    m_temp = quantities.get_magnetization(p_state)[2]
                else:
                    m_prev = m_temp
                    m_temp = quantities.get_magnetization(p_state)[2]
                    ratio = abs((m_temp - m_prev) / mu)
                    if ratio < converge_threshold:
                        break

        if i == 0:
            mz = quantities.get_magnetization(p_state)[2]
        else:
            mz = np.vstack((mz, m_temp))

    chi, intercept, r_value, p_value, std_err = linregress(Bfields, mz[:, 0])
    return chi


def compute_chi_vs_MCS(gamma, H_high, H_low, H_steps_1, H_steps_3, dim, concentration):
    """
    Run MC simulation for susceptibility vs Ht.

    """
    iterations_per_step = 1
    mu = 7
    relax_steps_0 = 10
    Hmax = 8

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg", quiet=True) as p_state:
        types = geometry.get_atom_types(p_state)
        nos = types.size

        locs = geometry.get_positions(p_state)
        vacancies_idx = np.where(types == -1)
        locs[:, 0][vacancies_idx] = 0
        locs[:, 1][vacancies_idx] = 0
        locs[:, 2][vacancies_idx] = 0

        parameters.mc.set_metropolis_cone(p_state, use_cone=True, cone_angle=30, use_adaptive_cone=True)
        parameters.mc.set_metropolis_spinflip(p_state, False)
        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=gamma)
        parameters.mc.set_iterations(p_state, iterations_per_step * types.size, iterations_per_step * types.size)

        get_paramagnetic_pstate(p_state, Hmax, relax_steps_0, vacancies_idx, nos)

        Hts = get_Hts(H_high, H_low, H_steps_1, H_steps_3)

        chis = []
        count = 0

        for k, Ht in enumerate(Hts):

            #Terminal output to see progress of run
            tqdm.write(f'Ht: {Ht:.3f}, k: {k}')
            hamiltonian.set_field(p_state, Ht, (Ht, 0, 0))

            spins = system.get_spin_directions(p_state)
            spins[vacancies_idx] = 0

            #Evolve the system at Ht
            Dx, Dy, Dz = compute_DDI(spins, DDI_interaction_x, DDI_interaction_y, DDI_interaction_z)
            DDI_field_interleave = np.ravel(np.column_stack((Dx, Dy, Dz)))
            system.set_DDI_field(p_state, n_atoms=nos, ddi_fields=DDI_field_interleave)
            simulation.start(p_state, simulation.METHOD_MC, single_shot=False)
            simulation.stop(p_state)

            chi = compute_chi(Ht, p_state, vacancies_idx, mu, nos)
            chis.append((k, Ht, chi))

    return pd.DataFrame(chis, columns=["i", "Ht", "chi"])


if __name__ == '__main__':
    n_cycles = 2400  # Number of cycles to repeat the plot over
    dim = 4
    concentration = 20
    gamma = 1e-7
    anisotropy = 1.5

    H_high = 6.0
    H_low = 2.0
    H_steps_1 = 10
    H_steps_3 = 10

    gammas = [gamma] * n_cycles  # gamma list for each cycle

    worker_fn = partial(
        compute_chi_vs_MCS,
        dim=dim,
        concentration=concentration,
        H_high=H_high,
        H_low=H_low,
        H_steps_1=H_steps_1,
        H_steps_3=H_steps_3,
    )

    mp.set_start_method("spawn", force=True)

    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(dim,)) as pool:
        results = list(tqdm(pool.imap_unordered(worker_fn, gammas), total=n_cycles))

    save_and_plot_results(results, dim, n_cycles, anisotropy, gamma)
