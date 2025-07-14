from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system,io
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import plotly.graph_objects as go
import pandas as pd
from tqdm import tqdm
import time
from datetime import timedelta

Hmax = 5
Hstep_coarse = 0.125
Hstep_fine = 0.01
Hfine1 = 4
Hfine2 = -4

fields = np.arange(Hmax,Hfine1,-1*Hstep_coarse,dtype=float)
fields = np.append(fields,np.arange(Hfine1,Hfine2,-1*Hstep_fine,dtype=float))
fields = np.append(fields,np.arange(Hfine2,-1*Hmax,-1*Hstep_coarse,dtype=float))
fields_hyst = np.append(fields,-1*fields)
# fields_hyst = np.tile(fields_hyst, n_cycles)

Ht = 1  #Transverse field

output_interval = 2 #Interval at which spin configuration files are saved
fn = "dipolar_arr"
prefix = "DDI_exp_14_G0p00005_Ht10p0"

iterations_per_step = 1 #Take this many Metropolis iterationss per lattice site between each check for convergence
converge_threshold = 0.01 #Fractional change in magnetization between steps to accept convergence
converge_max = 20 #Maximum number of steps to take before moving on
mu = 7
dim = 10
concentration = 45
gamma = 0.0003
#with state.State("input/test_Ising_largelattice.cfg") as p_state:


def plot_loop(concentration):
    with (state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg", quiet = True) as p_state):
        types = geometry.get_atom_types(p_state)
        nos = types.size
        tqdm.write(f"Sites: {nos}  Spins: {nos+np.sum(types)}")
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
    #    write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state,use_cone=True,cone_angle=30,use_adaptive_cone=True)
        parameters.mc.set_metropolis_spinflip(p_state,False)
        # parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=0.00012)
        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=gamma)

        #We'll evaluate convergence after enough Metropolis steps to hit each site twitce on average
        parameters.mc.set_iterations(p_state,iterations_per_step*types.size,iterations_per_step*types.size)

        #Initialize in the saturated state
        configuration.plus_z(p_state)

    ## Import the DDI interaction arrays
    #    path_arr_x = os.path.join("input/", fn +"_DDI_x.npy")
    #    path_arr_y = os.path.join("input/", fn +"_DDI_y.npy")
    #    path_arr_z = os.path.join("input/", fn +"_DDI_z.npy")
        path_arr_x = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn +"_x.npy") #fn = dipolar_arr
        path_arr_y = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn +"_y.npy")
        path_arr_z = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn +"_z.npy")

        path_arr_x_x = os.path.join(f"dipolar_interaction_matrices_reordered_x/{dim}_{dim}_{dim}/", fn + "_x_x.npy")
        path_arr_y_x = os.path.join(f"dipolar_interaction_matrices_reordered_x/{dim}_{dim}_{dim}/", fn + "_y_x.npy")

        if os.path.exists(path_arr_x) and os.path.exists(path_arr_y) and os.path.exists(path_arr_z) and os.path.exists(path_arr_x_x) and os.path.exists(path_arr_y_x):
            tqdm.write("loading DDI interaction data.")
            DDI_interaction_x = np.load(path_arr_x)
            DDI_interaction_y = np.load(path_arr_y)
            DDI_interaction_z = np.load(path_arr_z)

            DDI_interaction_x_x = np.load(path_arr_x_x)
            DDI_interaction_y_x = np.load(path_arr_y_x)

        else:
            tqdm.write("DDI files not found")
    #        break

    #Check that the size of the DDI arrays matches NOS (extracted from the types array above).
        if (nos != DDI_interaction_x.shape[0]) or (nos != DDI_interaction_y.shape[0]) or (nos != DDI_interaction_z.shape[0]) or (nos != DDI_interaction_x_x.shape[0]) or (nos != DDI_interaction_y_x.shape[0]) \
            or (nos != DDI_interaction_x.shape[1]) or (nos != DDI_interaction_y.shape[1]) or (nos != DDI_interaction_z.shape[1]) or (nos != DDI_interaction_x_x.shape[1]) or (nos != DDI_interaction_y_x.shape[1]):
            tqdm.write("Size mismatch between DDI and spin array")
            # break
    #Filter out any vacant sites
        vacancies_idx = np.where(types == -1)
        locs[:,0][vacancies_idx] = 0
        locs[:,1][vacancies_idx] = 0
        locs[:,2][vacancies_idx] = 0

        for i,Hz in enumerate(fields_hyst):
            tqdm.write(f'Hz: {Hz:.3f}, Concentration: {concentration}')

            Hmag = np.sqrt(Hz*Hz + Ht*Ht)
            hamiltonian.set_field(p_state,Hmag,(Ht,0,Hz)) #Inside set_field, the vector is normalized, so we don't have to do that here

            spins = system.get_spin_directions(p_state)  #Get the current spin state to update the DDI fields from the Ewald sum
            spins[:,2][vacancies_idx] = 0   #For LHF, we only care about Sz, but zero out the moments for vacancy site

            #Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
            #uses T, so need to scale accordingly.
            #Get the field at each spin. Only care about DDI due to z-spin so use spins[:,2]
            DDI_field_x_from_z = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4  # V_xz
            DDI_field_y_from_z = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4  # V_yz
            DDI_field_z_from_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4  # V_zz


            DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy
            DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

            DDI_field_x_from_x = np.matmul(DDI_interaction_x_x, spins[:, 0]) * 7 / 1e4
            DDI_field_y_from_x = np.matmul(DDI_interaction_y_x, spins[:, 0]) * 7 / 1e4

            DDI_field_x_from_y = np.matmul(DDI_interaction_y_x.T, spins[:, 1]) * 7 / 1e4

            DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x
            DDI_field_x_total = DDI_field_x_from_z + DDI_field_x_from_x + DDI_field_x_from_y
            DDI_field_y_total = DDI_field_y_from_z + DDI_field_y_from_x


            # Pass into SPIRIT
            DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x_total, DDI_field_y_total, DDI_field_z_total)))
            system.set_DDI_field(p_state,n_atoms=nos,ddi_fields=DDI_field_interleave)
            # tqdm.write(f"X: mean: {np.mean(DDI_field_x):.4e} Std. Dev: {np.std(DDI_field_x):.4e}")
            # tqdm.write(f"Y: mean: {np.mean(DDI_field_y):.4e} Std. Dev: {np.std(DDI_field_y):.4e}")
            # tqdm.write(f"Z: mean: {np.mean(DDI_field_z):.4e} Std. Dev: {np.std(DDI_field_z):.4e}")

            #Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files
            for j in range(converge_max) :
                simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
                simulation.stop(p_state)
                if j == 0 :
                    m_temp = quantities.get_magnetization(p_state)[2]
                else :
                    m_prev = m_temp
                    m_temp = quantities.get_magnetization(p_state)[2]
     #               ratio = abs((m_temp-m_prev)/m_prev)
                    ratio = abs((m_temp-m_prev)/mu)
                    tqdm.write(f"Iteration: {j:d}, Convergence: {ratio:.4f}, M_z: {m_temp:.4f}")
                    if ratio<converge_threshold :
                        break


            if output_interval>0: #Output the spin configuration.
                #TODO: Use system.get_spin_directions to pull the configuration array into Python, and then save out as an npy or similar
                if (i % output_interval == 0):
                    tag = prefix+f'N{i:d}_H{Hz:.3f}'
                    name = "output/" + tag + "_Image-00_Spins_0.ovf" #To match the internally-generated naming format
                    io.image_write(p_state,filename=name)

            #Using tunneling, so set tunneling flag in cfg files to 1
            if i == 0:
                # mx = quantities.get_magnetization(p_state)[0]
                # my = quantities.get_magnetization(p_state)[1]
                mz = m_temp
                iter_count = j
                spin_flip_count = parameters.mc.get_tunneling_spin_flip(p_state)
            else:
                # mx = np.vstack( (mx,quantities.get_magnetization(p_state)[0]) )
                # my = np.vstack( (my,quantities.get_magnetization(p_state)[1]) )
                mz = np.vstack( (mz,m_temp) )
                iter_count = np.vstack( (iter_count,j) )
                spin_flip_count = np.vstack((spin_flip_count,parameters.mc.get_tunneling_spin_flip(p_state)) )

    #np.savetxt("output/"+prefix+"mh.csv",np.transpose((fields_hyst, mz[:,0],iter_count[:,0],spin_flip_count[:,0])),delimiter=',')
    arr = np.transpose((fields_hyst, mz[:, 0]))
    df = pd.DataFrame(
        arr,
        columns=['B_Field', 'M_z']
    )
    return df



if __name__ == '__main__':
    start = time.time()  # ⏱️ Start timing
    # spin_concentrations = [85, 95, 15, 25, 35, 45, 55, 65, 75]
    n_cycles = 18
    spin_concentrations = [concentration] * n_cycles

    # Assuming fields_hyst and mz are already defined

    #~7hrs for 18 cycles 6 cpus
    with mp.Pool(processes=6) as pool:
        # results = pool.map(plot_loop, spin_concentrations)
        # imap returns an iterator, allowing tqdm to track progress
        results = list(
            tqdm(
                pool.imap(plot_loop, spin_concentrations),
                total=len(spin_concentrations),
            )
        )

    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(f"Bfield_Mz_{concentration}_{dim}_{dim}_{dim}_Ht_{Ht}_ncycles_{n_cycles}_with_Vx_gamma_{gamma}.csv", index=False)

    # # Create an interactive figure
    # fig = go.Figure()
    #
    # for fields, m_per_spin, conc in results:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=fields,
    #             y=m_per_spin,
    #             mode='lines',
    #             name=str(conc)
    #         )
    #     )
    #
    #     # Set layout
    #     fig.update_layout(
    #         title="Hysteresis Curve",
    #         xaxis_title="Magnetic Field",
    #         yaxis_title="Avg Magnetization Per Spin",
    #         legend_title="Concentration",
    #         template="plotly_white",
    #         width=800,
    #         height=500,
    #     )
    #
    #     fig.write_html(f'Bfield_Mz_{conc}_{dim}_{dim}_{dim}_Ht_{Ht}_ncycles_{n_cycles}_with_Vx_gamma_{gamma}.html')
        # plt.savefig(f'hysteresis_loop_tunnel_{dim}_new_gamma.png')
        # ⏱️ End timing
    end = time.time()
    tqdm.write(f"Total execution time: {end - start:.2f} seconds")