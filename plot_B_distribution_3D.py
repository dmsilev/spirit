from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system,io
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from tqdm import tqdm

dim = 10
concentration = 20

def plot_loop(H_relax):
    Hmax = 1.2
    H_step = 0.1
    iterations_per_step = 1  # Take this many Metropolis iterationss per lattice site between each check for convergence
    # Now is Ht transverse fields, max is  = 1T

    # fields = np.arange(Hmax,Hfine1,-1*Hstep_coarse,dtype=float)
    # fields = np.append(fields,np.arange(Hfine1,Hfine2,-1*Hstep_fine,dtype=float))
    # fields = np.append(fields,np.arange(Hfine2,-1*Hmax,-1*Hstep_coarse,dtype=float))
    # fields_hyst = np.append(fields,-1*fields)
    # fields_hyst = np.tile(fields_hyst, n_cycles)

    output_interval = 20  # Interval at which spin configuration files are saved
    fn = "dipolar_arr"
    prefix = "DDI_exp_14_G0p00005_Ht10p0"

    mu = 7

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg") as p_state:
        types = geometry.get_atom_types(p_state)
        nos = types.size
        print(f"Sites: {nos}  Spins: {nos+np.sum(types)}")
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
    #    write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state, use_cone=True, cone_angle=30, use_adaptive_cone=True)
        parameters.mc.set_metropolis_spinflip(p_state, True)

        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=0.00012)

        # For trying without tunneling and at base temperature ~0K
        # parameters.mc.set_use_tunneling(p_state, False)
        # parameters.mc.set_temperature(p_state, 0.0000035)

        #We'll evaluate convergence after enough Metropolis steps to hit each site twitce on average
        parameters.mc.set_iterations(p_state,iterations_per_step*types.size,iterations_per_step*types.size)

        #Initialize in the random state
        configuration.random(p_state)

    ## Import the DDI interaction arrays
    #    path_arr_x = os.path.join("input/", fn +"_DDI_x.npy")
    #    path_arr_y = os.path.join("input/", fn +"_DDI_y.npy")
    #    path_arr_z = os.path.join("input/", fn +"_DDI_z.npy")
        path_arr_x = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn +"_x.npy") #fn = dipolar_arr
        path_arr_y = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn +"_y.npy")
        path_arr_z = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn +"_z.npy")

        if os.path.exists(path_arr_x) and os.path.exists(path_arr_y) and os.path.exists(path_arr_z):
            print("loading DDI interaction data.")
            DDI_interaction_x = np.load(path_arr_x)
            DDI_interaction_y = np.load(path_arr_y)
            DDI_interaction_z = np.load(path_arr_z)
        else:
            print("DDI files not found")
    #        break

    #Check that the size of the DDI arrays matches NOS (extracted from the types array above).
        if (nos != DDI_interaction_x.shape[0]) or (nos != DDI_interaction_y.shape[0]) or (nos != DDI_interaction_z.shape[0]) :
            print("Size mismatch between DDI and spin array")
    #        break
    #Filter out any vacant sites
        vacancies_idx = np.where(types == -1)
        locs[:,0][vacancies_idx] = 0
        locs[:,1][vacancies_idx] = 0
        locs[:,2][vacancies_idx] = 0

        Hz = 0.0
        Hts = np.arange(Hmax, H_relax-1 * H_step, -1 * H_step, dtype=float)
        for i,Ht in enumerate(Hts):
            print(f'Ht: {Ht:.3f}', concentration)

            Hmag = np.sqrt(Hz*Hz + Ht*Ht)
            hamiltonian.set_field(p_state,Hmag,(Ht,0,Hz)) #Inside set_field, the vector is normalized, so we don't have to do that here

            spins = system.get_spin_directions(p_state)  #Get the current spin state to update the DDI fields from the Ewald sum
            spins[:,2][vacancies_idx] = 0   #For LHF, we only care about Sz, but zero out the moments for vacancy site
            spins[:,1][vacancies_idx] = 0   #For LHF, we only care about Sz, but zero out the moments for vacancy site
            spins[:,0][vacancies_idx] = 0   #For LHF, we only care about Sz, but zero out the moments for vacancy site

            #Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
            #uses T, so need to scale accordingly.
            #Get the field at each spin. Only care about DDI due to z-spin so use spins[:,2]
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
            system.set_DDI_field(p_state,n_atoms=nos,ddi_fields=DDI_field_interleave)

            ####Less strict convergence check conditions to simulate spin glass not relaxing to equilibrium state
            # converge_threshold = 0.01 #Fractional change in magnetization between steps to accept convergence
            converge_threshold = 0.1  # Fractional change in magnetization between steps to accept convergence
            # converge_max = 20 #Maximum number of steps to take before moving on
            converge_max = 10  # Maximum number of steps to take before moving on

            # Stricter convergence check for relaxation step
            converge_threshold_relax = 0.01
            converge_max_relax = 20

            #For relaxing at H_relax
            if Ht <= H_relax:
                converge_max = converge_max_relax
                converge_threshold = converge_threshold_relax

            #Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files
            for j in range(converge_max):
                simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
                simulation.stop(p_state)
                if j == 0 :
                    m_temp = quantities.get_magnetization(p_state)[2]
                else :
                    m_prev = m_temp
                    m_temp = quantities.get_magnetization(p_state)[2]
     #               ratio = abs((m_temp-m_prev)/m_prev)
                    ratio = abs((m_temp-m_prev)/mu)
                    print(f"Iteration: {j:d}, Convergence: {ratio:.4f}, M_z: {m_temp:.4f}")
                    if ratio<converge_threshold :
                        break

            if output_interval>0: #Output the spin configuration.
                #TODO: Use system.get_spin_directions to pull the configuration array into Python, and then save out as an npy or similar
                if (i % output_interval == 0):
                    tag = prefix+f'N{i:d}_H{Ht:.3f}'
                    name = "output/" + tag + "_Image-00_Spins_0.ovf" #To match the internally-generated naming format
                    io.image_write(p_state,filename=name)

        spins = system.get_spin_directions(p_state)  # Get the current spin state to update the DDI fields from the Ewald sum
        spins[:, 2][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
        spins[:, 1][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
        spins[:, 0][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site

        DDI_field_x_from_z = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4
        DDI_field_y_from_z = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4
        DDI_field_z_from_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4

        DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy
        DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

        DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x



    return DDI_field_z_total


if __name__ == '__main__':

    # Assuming fields_hyst and mz are already defined
    n_cycles = 8
    H_relaxes = [5]
    H_relaxes = H_relaxes * n_cycles

    with mp.Pool(processes=4) as pool:
        results = pool.map(plot_loop, H_relaxes)
        # results = list(tqdm(pool.imap_unordered(plot_loop, H_relaxes), total=len(H_relaxes)))
    # df_all = pd.concat(results, ignore_index=True)
    # df_all.to_csv(f'B_z_distribution_{dim}.csv', index=False)

    # Stack all 1D arrays to make a 2D array of shape (M, N)
    stacked = np.stack(results)  # shape: (n_cycles, vector_length)

    # Average along axis 0 (over the n_cycles)
    average_vector = np.mean(stacked, axis=0)

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg") as p_state:
        positions = geometry.get_positions(p_state)
        types = geometry.get_atom_types(p_state)
        vacancies_idx = np.where(types == -1)

    # Mask to exclude vacancies
    mask = np.ones(len(positions), dtype=bool)
    mask[vacancies_idx] = False

    filtered_positions = positions[mask]
    filtered_values = average_vector[mask]

    # Custom nonlinear colorscale to emphasize extremes near -1 and 1
    colorscale = [
        [0.0, 'darkblue'],  # min
        [0.1, 'blue'],
        [0.4, 'lightblue'],
        [0.5, 'white'],  # midpoint (0)
        [0.6, 'lightcoral'],
        [0.9, 'red'],
        [1.0, 'darkred']  # max
    ]

    # Normalize color range symmetrically around zero (assuming approx. [-1, 1])
    cmin = -1.0
    cmax = 1.0

    # Plot
    fig = go.Figure(data=[go.Scatter3d(
        x=filtered_positions[:, 0],
        y=filtered_positions[:, 1],
        z=filtered_positions[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=filtered_values,
            colorscale=colorscale,
            colorbar=dict(title='Average Value'),
            cmin=cmin,
            cmax=cmax,
            opacity=0.8
        )
    )])

    fig.update_layout(
        title="3D Spin Heatmap (Extreme Values Highlighted)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html(f'B_z_distribution_3D_{dim}_{n_cycles}_{concentration}.html')
