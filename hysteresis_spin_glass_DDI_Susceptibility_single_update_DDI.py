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

def plot_loop(H_relax):
    iterations_per_step = 1  #Spin glass, Take this many Metropolis iterationss per lattice site between each check for convergence
    output_interval = 20  # Interval at which spin configuration files are saved
    fn = "dipolar_arr"
    prefix = "DDI_exp_14_G0p00005_Ht10p0"

    mu = 7
    dim = 4
    concentration = 20
    H_relax_step = 500
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

        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=0.00001)

        #We'll evaluate convergence after enough Metropolis steps to hit each site twitce on average
        parameters.mc.set_iterations(p_state,iterations_per_step*types.size,iterations_per_step*types.size)

        #Initialize in the saturated state
        # configuration.plus_z(p_state)

        #Initialize from unordered state
        configuration.random(p_state)


    #Filter out any vacant sites
        vacancies_idx = np.where(types == -1)
        locs[:,0][vacancies_idx] = 0
        locs[:,1][vacancies_idx] = 0
        locs[:,2][vacancies_idx] = 0


        Hmax = 1.5
        H_step = 0.1
        # Sweep down to just above H_relax (exclusive)
        Hts_above = np.arange(Hmax, H_relax- 1e-8, -H_step)  #1e-8 to make sure floating point error doesn't include in another H_relax, which will be slightly off H_relax and give multiple points on H_relax in the graph
        # Insert 20 copies of H_relax
        H_relax_insert = np.full(H_relax_step, H_relax)
        # Sweep just below H_relax down to just above 0 (inclusive)
        Hts_below = np.arange(H_relax - H_step, -H_step, -H_step)
        # Combine all
        Hts = np.concatenate((Hts_above, H_relax_insert, Hts_below))

        chis = {}
        count = 0

        for i,Ht in enumerate(Hts):
            # print(f'Ht: {Ht:.3f}', concentration)
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


            converge_max = 1

            #Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files

            for j in range(converge_max):
                simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
                simulation.stop(p_state)


            if output_interval>0: #Output the spin configuration.
                #TODO: Use system.get_spin_directions to pull the configuration array into Python, and then save out as an npy or similar
                if (i % output_interval == 0):
                    tag = prefix+f'N{i:d}_H{Ht:.3f}'
                    name = "output/" + tag + "_Image-00_Spins_0.ovf" #To match the internally-generated naming format
                    io.image_write(p_state,filename=name)

            # Check susceptibility every 4th Ht datapoint
            # if i%4 ==0 or Ht <= H_relax + H_step:
            if i % 1 == 0:

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

    n_cycles = 160
    H_relax = 0.8
    H_relaxes = [H_relax]*n_cycles
    H_relax_step = 500
    dim = 4
    concentration = 20


    with mp.Pool(processes=mp.cpu_count()) as pool:
        # results = pool.map(plot_loop, H_relaxes)
        results = list(tqdm(pool.imap_unordered(plot_loop, H_relaxes), total=len(H_relaxes)))

    # Merge all dicts
    merged = defaultdict(list)

    for result in results:
        for (H_relax, Ht), chi in result.items():
            merged[(H_relax, round(Ht, 2))].append(chi) #round to ensure all keys are the same, no floating point errors

    # Now average the values and compute std dev
    data = []
    for (H_relax, Ht), chis in merged.items():
        chi_mean = np.mean(chis)
        chi_std = np.std(chis)
        data.append((H_relax, Ht, chi_mean, chi_std))

    df_avg = pd.DataFrame(data, columns=["H_relax", "Ht", "chi_mean", "chi_std"])

    df_avg.to_csv(
        f'Susceptibility_{dim}_{n_cycles}_{concentration}_anisotropy_0.7_Hrelax_{H_relax}_relax_step_{50}_reference.csv')

    # Plot with error bars
    fig = px.line(
        df_avg,
        x="Ht",
        y="chi_mean",
        error_y="chi_std",
        color="H_relax",
        markers=True,
        labels={
            "Ht": "Ht (T)",
            "chi_mean": "Susceptibility χ",
            "H_relax": "H_relax (T)",
        },
        title="Susceptibility χ vs Ht for different H_relax"
    )

    fig.write_html(
        f'Susceptibility_{dim}_{n_cycles}_{concentration}_anisotropy_0.7_Hrelax_{H_relax}_relax_step_{H_relax_step}.html')

    # Timer
    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    tqdm.write(f"\n✅ Finished in {str(elapsed_time)} (hh:mm:ss)")
    #2:31:36.733138 for 160 cycles 8 cpus, 10x10x10, H_relax_steps = 50
