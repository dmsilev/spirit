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

def plot_loop(H_relax):
    iterations_per_step = 1  #Spin glass, Take this many Metropolis iterationss per lattice site between each check for convergence
    output_interval = 1  # Interval at which spin configuration files are saved
    fn = "dipolar_arr"
    prefix = "DDI_exp_14_G0p00005_Ht10p0"

    mu = 7
    dim = 10
    concentration = 15
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

        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=0.00012)

        #For trying without tunneling and at base temperature ~0K
        # parameters.mc.set_use_tunneling(p_state, False)
        # parameters.mc.set_temperature(p_state, 0.000035)

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

        Hz = 0.0
        Hmax = 4
        # H_step = 0.1 #Coarser step since there susceptibility not changing much, and also to reduce number of times simulation is ran to simulate spin glass where spins are not flipped that often
        H_step_fine = 0.01
        # # Boundary between coarse and fine steps
        # H_switch = H_relax + H_step
        #
        # # Coarse part: from H_max to just above H_switch
        # H_coarse = np.arange(Hmax, H_switch - 1e-10, -H_step)
        #
        # # Fine part: from just below H_switch to around H_relax
        # H_fine = np.arange(H_switch, H_relax - 1e-10, -H_step_fine)
        #
        # # Concatenate both
        # Hts = np.concatenate([H_coarse, H_fine])

        # Hts = np.arange(Hmax, 0, -H_step)
        Hts = [H_relax]
        chis = []

        for i,Ht in enumerate(Hts):
            # print(f'Ht: {Ht:.3f}', concentration)
            # tqdm.write(f'Ht: {Ht:.3f}, concentration: {concentration}')

            Hmag = np.sqrt(Hz*Hz + Ht*Ht)
            hamiltonian.set_field(p_state,Hmag,(Ht,0,Hz)) #Inside set_field, the vector is normalized, so we don't have to do that here

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

            # DDI_field_x_from_y = np.matmul(DDI_interaction_x, spins[:, 1]) * 7 / 1e4 #V_xy
            # DDI_field_y_from_y = np.matmul(DDI_interaction_y, spins[:, 1]) * 7 / 1e4 #V_yy
            DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4 #V_zy

            # DDI_field_x_from_x = np.matmul(DDI_interaction_x, spins[:, 0]) * 7 / 1e4 #V_xx
            # DDI_field_y_from_x = np.matmul(DDI_interaction_y, spins[:, 0]) * 7 / 1e4 #V_yx
            DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4 #V_zx

            DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

            #Pass into SPIRIT
            DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x_from_z,DDI_field_y_from_z,DDI_field_z_total)))
            system.set_DDI_field(p_state,n_atoms=nos,ddi_fields=DDI_field_interleave)

            ####Less strict convergence check conditions to simulate spin glass not relaxing to equilibrium state
            # converge_threshold = 0.01 #Fractional change in magnetization between steps to accept convergence
            # converge_threshold = 1  #Spin glass, Fractional change in magnetization between steps to accept convergence
            converge_threshold = 0.0000000000000000001 #So to fix number of iterations per unit of Ht moved
            # converge_max = 20 #Maximum number of steps to take before moving on
            converge_max = 125  # Maximum number of steps to take before moving on

            #Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files

            for j in range(converge_max):
                simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
                simulation.stop(p_state)
                if j == 0:
                    m_temp = quantities.get_magnetization(p_state)[2]
                else :
                    m_prev = m_temp
                    m_temp = quantities.get_magnetization(p_state)[2]
     #               ratio = abs((m_temp-m_prev)/m_prev)
                    ratio = abs((m_temp-m_prev)/mu)

                    tqdm.write(f"Iteration: {j:d}, Convergence: {ratio:.4f}, M_z: {m_temp:.4f}")
                    if ratio<converge_threshold :
                        # tqdm.write(f'++++++++++++++++++++++++++++++++Number of iterations needed for ratio<convergence_threshold: {j}++++++++++++++++++++++++++++++++')
                        break

            # if output_interval>0: #Output the spin configuration.
            #     #TODO: Use system.get_spin_directions to pull the configuration array into Python, and then save out as an npy or similar
            #     if (i % output_interval == 0):
            #         tag = prefix+f'N{i:d}_H{Ht:.3f}'
            #         name = "output/" + tag + "_Image-00_Spins_0.ovf" #To match the internally-generated naming format
            #         io.image_write(p_state,filename=name)

            # Check susceptibility every 4th Ht datapoint
            # if i%4 ==0 or Ht <= H_relax + H_step:
            # if i % 1 == 0:

                # filename = f'state_configs/p_state_{Ht:.3f}_Hrelax_{H_relax:.3f}.ovf'
                # io.image_write(p_state, filename)
                # chi = get_susceptibility(filename, H_relax, Ht)

                Bfields = np.arange(0, 0.03, 0.01)
                for i, Hz in enumerate(Bfields):
                    # tqdm.write(f'Hz: {Hz:.3f}', concentration)

                    Hmag = np.sqrt(Hz * Hz + Ht * Ht)
                    hamiltonian.set_field(p_state, Hmag, (Ht, 0,
                                                            Hz))  # Inside set_field, the vector is normalized, so we don't have to do that here

                    spins = system.get_spin_directions(
                        p_state)  # Get the current spin p_state to update the DDI fields from the Ewald sum
                    spins[:, 2][
                        vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site

                    # Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
                    # uses T, so need to scale accordingly.
                    # Get the field at each spin.
                    DDI_field_x_from_z = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4  # V_xz
                    DDI_field_y_from_z = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4  # V_yz
                    DDI_field_z_from_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4  # V_zz

                    # DDI_field_x_from_y = np.matmul(DDI_interaction_x, spins[:, 1]) * 7 / 1e4 #V_xy
                    # DDI_field_y_from_y = np.matmul(DDI_interaction_y, spins[:, 1]) * 7 / 1e4 #V_yy
                    DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy

                    # DDI_field_x_from_x = np.matmul(DDI_interaction_x, spins[:, 0]) * 7 / 1e4 #V_xx
                    # DDI_field_y_from_x = np.matmul(DDI_interaction_y, spins[:, 0]) * 7 / 1e4 #V_yx
                    DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

                    DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

                    # Pass into SPIRIT
                    DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x_from_z, DDI_field_y_from_z, DDI_field_z_total)))
                    system.set_DDI_field(p_state, n_atoms=nos, ddi_fields=DDI_field_interleave)
                    # tqdm.write(f"X: mean: {np.mean(DDI_field_x_from_z):.4e} Std. Dev: {np.std(DDI_field_x_from_z):.4e}")
                    # tqdm.write(f"Y: mean: {np.mean(DDI_field_y_from_z):.4e} Std. Dev: {np.std(DDI_field_y_from_z):.4e}")
                    # tqdm.write(f"Z: mean: {np.mean(DDI_field_z_from_z):.4e} Std. Dev: {np.std(DDI_field_z_from_z):.4e}")

                    # converge_threshold = 0.01  # Fractional change in magnetization between steps to accept convergence
                    converge_threshold = 0.000000000000000000000000000001
                    converge_max_sus = 2  # Maximum number of steps to take before moving on
                    # Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files
                    for k in range(converge_max_sus):
                        simulation.start(p_state, simulation.METHOD_MC,
                                         single_shot=False)  # solver_type=simulation.MC_ALGORITHM_METROPOLIS
                        simulation.stop(p_state)
                        if k == 0:
                            m_temp = quantities.get_magnetization(p_state)[2]
                        else:
                            m_prev = m_temp
                            m_temp = quantities.get_magnetization(p_state)[2]
                            #               ratio = abs((m_temp-m_prev)/m_prev)
                            ratio = abs((m_temp - m_prev) / mu)
                            tqdm.write(f"########Susceptibility measurement: Iteration: {k:d}, Convergence: {ratio:.4f}, M_z: {m_temp:.4f}")
                            if ratio < converge_threshold:
                                break

                    if i == 0:
                        mz = m_temp
                    else:
                        mz = np.vstack((mz, m_temp))

                # Fit a straight line
                chi, intercept, r_value, p_value, std_err = linregress(Bfields, mz[:, 0])

                chis.append((chi, j))

    return chis


if __name__ == '__main__':
    start_time = time.time()  # Start timer
    mp.set_start_method("spawn", force=True)
    H_relax = 2
    n_cycles = 240
    H_relaxes = [H_relax]*n_cycles

    dim = 10
    concentration = 15

    #~12 minutes per cycle
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # results = pool.map(plot_loop, H_relaxes)
        results = list(tqdm(pool.imap_unordered(plot_loop, H_relaxes), total=len(H_relaxes)))
    # results is a list of lists of (Ht, chi, j) tuples from all processes
    flat = [item for sublist in results for item in sublist]  # flatten nested list

    # Create DataFrame
    df = pd.DataFrame(flat, columns=["chi", "j"])

    df.to_csv(f'MCS_Susceptibility_{dim}_{n_cycles}_{concentration}_anisotropy_0.7.csv', index=False)

    # Group by j and compute mean and std of chi
    df_avg = df.groupby("j", as_index=False).agg(
        chi_mean=("chi", "mean"),
        chi_std=("chi", "std")
    )

    # Plot with Plotly
    fig = px.line(
        df_avg,
        x="j",
        y="chi_mean",
        error_y="chi_std",
        markers=True,
        labels={"j": "Iteration Count (j)", "chi_mean": "Average Susceptibility χ"},
        title=f"Average Susceptibility χ vs Iteration Count j at Ht = {H_relax}"
    )
    fig.write_html(f'MCS_Susceptibility_{dim}_{n_cycles}_{concentration}_anisotropy_0.7.html')

    # Timer
    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    tqdm.write(f"\n✅ Finished in {str(elapsed_time)} (hh:mm:ss)")
