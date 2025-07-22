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
    output_interval = 30  # Interval at which spin configuration files are saved
    fn = "dipolar_arr"
    prefix = "DDI_exp_14_G0p00005_Ht10p0"

    H_relax = 2.3 #around 0.5*H_max
    mu = 7
    dim = 10
    concentration = 20
    H_relax_steps = 400
    relax_steps_0 = 50

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

        Hts_randomise = [0] * relax_steps_0
        # configuration.random give spins pointing anywhere on a sphere, want to align it more along anisotropic field by evolving with Ht = 0
        for i, Ht in enumerate(Hts_randomise):
            Hmag = Ht
            hamiltonian.set_field(p_state, Hmag, (Ht, 0,
                                                  0))  # Inside set_field, the vector is normalized, so we don't have to do that here

            spins = system.get_spin_directions(
                p_state)  # Get the current spin state to update the DDI fields from the Ewald sum
            spins[:, 2][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
            spins[:, 1][vacancies_idx] = 0
            spins[:, 0][vacancies_idx] = 0
            # Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
            # uses T, so need to scale accordingly.
            # Get the field at each spin.
            # V_xz = V_zx.T etc

            DDI_field_x_from_z = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4  # V_xz
            DDI_field_y_from_z = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4  # V_yz
            DDI_field_z_from_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4  # V_zz

            DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy

            DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

            DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

            # Pass into SPIRIT
            DDI_field_interleave = np.ravel(
                np.column_stack((DDI_field_x_from_z, DDI_field_y_from_z, DDI_field_z_total)))
            system.set_DDI_field(p_state, n_atoms=nos, ddi_fields=DDI_field_interleave)

            # for j in range(converge_max):
            simulation.start(p_state, simulation.METHOD_MC,
                             single_shot=False)  # solver_type=simulation.MC_ALGORITHM_METROPOLIS
            simulation.stop(p_state)

        Hmax = 4.5
        H_step = 0.1
        # Sweep down to just above H_relax (exclusive)
        Hts_above = np.arange(Hmax, H_relax - 1e-8, -H_step)
        Hts_above = np.repeat(Hts_above, 2)  # repeat each value twice

        # Insert H_relax multiple times
        H_relax_insert = np.full(H_relax_steps, H_relax)

        # Sweep just below H_relax down to just above 0 (inclusive)
        Hts_below = np.arange(H_relax - H_step, -H_step, -H_step)
        Hts_below = np.repeat(Hts_below, 2)  # repeat each value twice

        # Combine all
        Hts = np.concatenate((Hts_above, H_relax_insert, Hts_below))

        # Sort in descending order
        Hts = np.sort(Hts)[::-1]

        Hts = np.round(Hts, decimals=3)
        # print('Hts', Hts, len(Hts))

        # Get all unique values and the *first* indices where they occur
        unique_vals, first_indices = np.unique(Hts, return_index=True)
        # print('first_indices', first_indices, len(first_indices))
        # print('unique_vals', unique_vals, len(unique_vals))

        # Compute the last index of each group: next first - 1
        # For the last group, just use the last index in Hts
        last_indices = np.append(first_indices[:-1] - 1, len(Hts) - 1)
        # print('last indices:', last_indices)

        chis = {}
        count = 0

        for i,Ht in enumerate(Hts):

            if abs(Ht - H_relax) < 1e-6:
                count += 1
                tqdm.write(f'Ht: {Ht:.3f}, count: {count} concentration: {concentration}')
            else:
                tqdm.write(f'Ht: {Ht:.3f}, concentration: {concentration}')

            Hmag = Ht
            hamiltonian.set_field(p_state,Hmag,(Ht,0,0)) #Inside set_field, the vector is normalized, so we don't have to do that here

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

            # # For relaxing at H_relax
            # if abs(Ht - H_relax) < 1e-6:
            #     tqdm.write(f"~~~~~~~~~~~~~~~~~~~~~~~Changed converge_max~~~~~~~~~~~~~~~~~~~~~~~")
            #     converge_max = 200
            #     # iterations_per_step = 1
            #     # parameters.mc.set_iterations(p_state, iterations_per_step * types.size,
            #     #                              iterations_per_step * types.size)
            #
            # else:
            #     converge_max = 2

            #Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files

            # for j in range(converge_max):
            simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
            simulation.stop(p_state)


            # if output_interval>0: #Output the spin configuration.
            #     #TODO: Use system.get_spin_directions to pull the configuration array into Python, and then save out as an npy or similar
            #     if (i % output_interval == 0):
            #         tag = prefix+f'N{i:d}_H{Ht:.3f}'
            #         name = "output/" + tag + "_Image-00_Spins_0.ovf" #To match the internally-generated naming format
            #         io.image_write(p_state,filename=name)

            # Check susceptibility every 4th Ht datapoint
            # if i%4 ==0 or Ht <= H_relax + H_step:
            # if i % 1 == 0:
            if i in last_indices: #Only calculate chi if it's last iteration of each unique Ht

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


                chis[(H_relax, Ht)] = chi  # replaces existing entry with same key

    return chis


if __name__ == '__main__':
    start_time = time.time()  # Start timer
    mp.set_start_method("spawn", force=True)

    n_cycles = 240
    # H_relax = 1.2
    H_relax = 0.8
    # H_relax_steps = 200
    H_relax_steps = 400
    dim = 10
    concentration = 20
    gamma = 0.0001
    gammas = [gamma]

    all_results = []

    # Loop over different gamma values
    for gamma in gammas:
        gammas = [gamma] * n_cycles  # gamma list for each cycle

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(plot_loop, gammas), total=n_cycles))

        # Flatten and tag with gamma
        flat = [(round(k[0], 2), round(k[1], 2), v, gamma) for result in results for k, v in result.items()]

        all_results.extend(flat)

    # Make DataFrame
    df_all = pd.DataFrame(all_results, columns=["H_relax", "Ht", "chi", "gamma"])

    # Save raw data
    df_all.to_csv(
        f'Susceptibility_multi_gammas_{dim}_{n_cycles}_per_gamma_{concentration}_anisotropy_0.7_relax_step_{H_relax_steps}_gammas_{-5}_relax_0.8_gamma_{gamma}.csv',
        index=False)

    # Average over cycles, std divided by sqrt(n_cycles)
    df_avg = (
        df_all.groupby(["gamma", "Ht"], as_index=False)
        .agg(
            chi_mean=("chi", "mean"),
            chi_std=("chi", lambda x: x.std(ddof=1) / np.sqrt(n_cycles))
        )
    )

    # Plot
    fig = px.line(
        df_avg,
        x="Ht",
        y="chi_mean",
        error_y="chi_std",
        color="gamma",  # Different color per gamma
        markers=True,
        labels={
            "Ht": "Ht (T)",
            "chi_mean": "Susceptibility χ",
            "gamma": "Gamma",
        },
        title="Susceptibility χ vs Ht for different Γ (gamma)"
    )

    fig.write_html(
        f'Susceptibility_multi_gamma_{dim}_{n_cycles}_{concentration}_anisotropy_0.7_relax_step_{H_relax_steps}_gammas_DDI_{-5}_relax_0.8_gamma_{gamma}.html')

    # Timer
    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    tqdm.write(f"\n✅ Finished in {str(elapsed_time)} (hh:mm:ss)")

    #240/240 [11:14:04<00:00, 168.52s/it]
