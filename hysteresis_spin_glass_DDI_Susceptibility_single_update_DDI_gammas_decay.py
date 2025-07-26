from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system,io
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm
import time
from datetime import timedelta
from collections import defaultdict

def plot_loop(gamma):
    iterations_per_step = 1  #Spin glass, Take this many Metropolis iterationss per lattice site between each check for convergence
    output_interval = 60  # Interval at which spin configuration files are saved
    fn = "dipolar_arr"
    prefix = "DDI_exp_14_G0p00005_Ht10p0"

    H_relax = 2.3
    mu = 7
    dim = 10
    concentration = 20
    H_relax_steps = 100
    path_arr_x = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/",
                              fn + "_x.npy")  # fn = dipolar_arr
    path_arr_y = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn + "_y.npy")
    path_arr_z = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn + "_z.npy")

    if os.path.exists(path_arr_x) and os.path.exists(path_arr_y) and os.path.exists(path_arr_z):
        tqdm.write("loading DDI interaction data.")
        DDI_interaction_x = np.load(path_arr_x)
        DDI_interaction_y = np.load(path_arr_y)
        DDI_interaction_z = np.load(path_arr_z)
    else:
        tqdm.write("DDI files not found")
    #        break

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg", quiet = True) as p_state:
        types = geometry.get_atom_types(p_state)
        nos = types.size
        # tqdm.write(f"Sites: {nos}  Spins: {nos+np.sum(types)}")
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
    #    write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state,use_cone=True,cone_angle=30,use_adaptive_cone=True)
        parameters.mc.set_metropolis_spinflip(p_state,False)

        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=gamma)

        #We'll evaluate convergence after enough Metropolis steps to hit each site twitce on average
        parameters.mc.set_iterations(p_state,iterations_per_step*types.size,iterations_per_step*types.size)

        #Initialize from unordered state
        configuration.random(p_state)


    #Filter out any vacant sites
        vacancies_idx = np.where(types == -1)
        locs[:,0][vacancies_idx] = 0
        locs[:,1][vacancies_idx] = 0
        locs[:,2][vacancies_idx] = 0


        Hts = [H_relax]*H_relax_steps

        chis = []

        count = 0
        for k,Ht in enumerate(Hts):
            count+=1
            tqdm.write(f'Ht: {Ht:.3f}, count: {count} gamma: {gamma} concentration: {concentration}')


            hamiltonian.set_field(p_state,H_relax,(H_relax,0,0)) #Inside set_field, the vector is normalized, so we don't have to do that here

            spins = system.get_spin_directions(p_state)  #Get the current spin state to update the DDI fields from the Ewald sum
            spins[:,2][vacancies_idx] = 0   #For LHF, we only care about Sz, but zero out the moments for vacancy site
            spins[:, 1][vacancies_idx] = 0
            spins[:, 0][vacancies_idx] = 0
            #Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
            #uses T, so need to scale accordingly.
            #Get the field at each spin.
            #V_xz = V_zx.T etc

            DDI_field_x_from_z = np.matmul(DDI_interaction_x,spins[:,2]) *7/1e4 #V_xz
            DDI_field_y_from_z = np.matmul(DDI_interaction_y,spins[:,2]) *7/1e4 #V_yz
            DDI_field_z_from_z = np.matmul(DDI_interaction_z,spins[:,2]) *7/1e4 #V_zz

            DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4 #V_zy

            DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4 #V_zx

            DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

            #Pass into SPIRIT
            DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x_from_z,DDI_field_y_from_z,DDI_field_z_total)))
            system.set_DDI_field(p_state,n_atoms=nos,ddi_fields=DDI_field_interleave)


            # for j in range(converge_max):
            simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
            simulation.stop(p_state)

            # Define both options
            Bfields_positive = np.arange(0, 0.3, 0.1)
            Bfields_negative = np.arange(0, -0.3, -0.1)

            # Randomly choose one, to avoid always polarising spins in one direction
            Bfields = Bfields_positive if np.random.rand() < 0.5 else Bfields_negative

            for i, HzB in enumerate(Bfields):
                # tqdm.write(f'Hz: {Hz:.3f}', concentration)

                Hmag = np.sqrt(HzB * HzB + Ht * Ht)
                hamiltonian.set_field(p_state, Hmag, (Ht, 0,
                                                        HzB))  # Inside set_field, the vector is normalized, so we don't have to do that here

                spins = system.get_spin_directions(
                    p_state)  # Get the current spin p_state to update the DDI fields from the Ewald sum
                spins[:, 2][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
                spins[:, 1][vacancies_idx] = 0
                spins[:, 0][vacancies_idx] = 0

                # Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
                # uses T, so need to scale accordingly.
                # Get the field at each spin.
                DDI_field_x_from_z = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4  # V_xz
                DDI_field_y_from_z = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4  # V_yz
                DDI_field_z_from_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4  # V_zz

                DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy

                DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

                DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

                # Pass into SPIRIT
                DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x_from_z, DDI_field_y_from_z, DDI_field_z_total)))
                system.set_DDI_field(p_state, n_atoms=nos, ddi_fields=DDI_field_interleave)

                # converge_threshold = 0.01  # Fractional change in magnetization between steps to accept convergence
                converge_threshold = 0.000000001
                converge_max = 1  # Maximum number of steps to take before moving on
                if i != 0:  # i = 0 state already went through one round of convergence before entering susceptibility loop
                    # Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files
                    for j in range(converge_max):
                        simulation.start(p_state, simulation.METHOD_MC,
                                         single_shot=False)  # solver_type=simulation.MC_ALGORITHM_METROPOLIS
                        simulation.stop(p_state)
                        if j == 0:
                            m_temp = quantities.get_magnetization(p_state)[2]
                        else:
                            m_prev = m_temp
                            m_temp = quantities.get_magnetization(p_state)[2]
                            #               ratio = abs((m_temp-m_prev)/m_prev)
                            ratio = abs((m_temp - m_prev) / mu)
                            tqdm.write(f"########Susceptibility measurement: Iteration: {j:d}, Convergence: {ratio:.4f}, M_z: {m_temp:.4f}")
                            if ratio < converge_threshold:
                                break

                if i == 0:
                    mz = quantities.get_magnetization(p_state)[2]
                else:
                    mz = np.vstack((mz, m_temp))

            # Fit a straight line
            chi, intercept, r_value, p_value, std_err = linregress(Bfields, mz[:, 0])


            chis.append((k, chi))

    return pd.DataFrame(chis, columns=["i", "chi"]).assign(gamma=gamma)


if __name__ == '__main__':
    start_time = time.time()  # Start timer
    mp.set_start_method("spawn", force=True)

    n_cycles = 88
    # H_relax = 1.2
    H_relax = 2.3
    # H_relax_steps = 200
    H_relax_steps = 100
    dim = 10
    concentration = 20
    # gamma = 0.000001
    gammas = [0.0001, 0.0002]*n_cycles

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(plot_loop, gammas), total=len(gammas), desc="Running simulations"))

    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(f"MCS_decay_gammas_dim_{dim}_with_SEM_errorbars.csv", index = False)

    grouped = df_all.groupby(['gamma', 'i']).agg(
        chi_mean=('chi', 'mean'),
        chi_std=('chi', 'std')
    ).reset_index()

    grouped['chi_sem'] = grouped['chi_std'] / (n_cycles ** 0.5)

    fig = go.Figure()

    unique_gammas = grouped['gamma'].unique()
    for gamma in unique_gammas:
        df_gamma = grouped[grouped['gamma'] == gamma]

        fig.add_trace(go.Scatter(
            x=df_gamma['i'],
            y=df_gamma['chi_mean'],
            mode='lines+markers',
            name=f"γ = {gamma:.1e}",
            error_y=dict(
                type='data',
                array=df_gamma['chi_sem'],
                visible=True
            )
        ))

    fig.update_layout(
        title="Average Susceptibility χ vs i for different γ, relaxed at Ht = 2.3T",
        xaxis_title="MCS (step index i)",
        yaxis_title="χ (chi)",
        legend_title="Tunneling γ",
        template="plotly_white",
        width=900,
        height=600
    )

    fig.write_html(f"MCS_decay_gammas_dim_{dim}_with_SEM_errorbars_Ht2.3.html")

    fig_ln = go.Figure()

    for gamma in unique_gammas:
        df_gamma = grouped[grouped['gamma'] == gamma]

        # Avoid log of zero or negative numbers
        df_gamma = df_gamma[df_gamma['chi_mean'] > 0]

        fig_ln.add_trace(go.Scatter(
            x=df_gamma['i'],
            y=np.log(df_gamma['chi_mean']),
            mode='lines+markers',
            name=f"γ = {gamma:.1e}",
            error_y=dict(
                type='data',
                array=df_gamma['chi_sem'] / df_gamma['chi_mean'],  # Propagation of error in log(chi)
                visible=True
            )
        ))

    fig_ln.update_layout(
        title="ln(χ) vs i for different γ, relaxed at Ht = 2.3T",
        xaxis_title="MCS (step index i)",
        yaxis_title="ln(χ)",
        legend_title="Tunneling γ",
        template="plotly_white",
        width=900,
        height=600
    )

    fig_ln.write_html(f"ln_MCS_decay_gammas_dim_{dim}_with_SEM_errorbars_Ht2.3.html")

    fig_lnln = go.Figure()

    for gamma in unique_gammas:
        df_gamma = grouped[grouped['gamma'] == gamma].copy()

        # Filter out invalid values
        df_gamma = df_gamma[(df_gamma['chi_mean'] > 0) & (np.log(df_gamma['chi_mean']) < 0)]

        # Compute x = ln(i), y = ln(-ln(chi))
        df_gamma['ln_i'] = np.log(df_gamma['i'])
        df_gamma['ln_ln_chi'] = np.log(-np.log(df_gamma['chi_mean']))

        # Error propagation
        df_gamma['ln_ln_chi_sem'] = (
                np.abs(1 / (np.log(df_gamma['chi_mean']) * df_gamma['chi_mean'])) * df_gamma['chi_sem']
        )

        fig_lnln.add_trace(go.Scatter(
            x=df_gamma['ln_i'],
            y=df_gamma['ln_ln_chi'],
            mode='lines+markers',
            name=f"γ = {gamma:.1e}",
            error_y=dict(
                type='data',
                array=df_gamma['ln_ln_chi_sem'],
                visible=True
            )
        ))

    fig_lnln.update_layout(
        title="ln(-ln(χ)) vs ln(i) for different γ, relaxed at Ht = 2.3T",
        xaxis_title="ln(MCS step index i)",
        yaxis_title="ln(-ln(χ))",
        legend_title="Tunneling γ",
        template="plotly_white",
        width=900,
        height=600
    )

    fig_lnln.write_html(f"lnln_MCS_decay_gammas_dim_{dim}_with_SEM_errorbars_Ht2.3.html")

