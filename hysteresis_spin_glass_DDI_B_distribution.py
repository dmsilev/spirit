from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system,io
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from tqdm import tqdm

dim = 4
concentration = 20
gamma = 0.000001
def plot_loop(H_relax):

    iterations_per_step = 1  # Take this many Metropolis iterationss per lattice site between each check for convergence

    output_interval = 20  # Interval at which spin configuration files are saved
    fn = "dipolar_arr"
    prefix = "DDI_exp_14_G0p00005_Ht10p0"

    mu = 7
    H_relax_steps = 2 if abs(0.1 - H_relax) < 1e-6 else 200

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg", quiet=True) as p_state:
        types = geometry.get_atom_types(p_state)
        nos = types.size
        # print(f"Sites: {nos}  Spins: {nos+np.sum(types)}")
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
    #    write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state, use_cone=True, cone_angle=30, use_adaptive_cone=True)
        parameters.mc.set_metropolis_spinflip(p_state, True)

        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=gamma)

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
            raise

    #Check that the size of the DDI arrays matches NOS (extracted from the types array above).
        if (nos != DDI_interaction_x.shape[0]) or (nos != DDI_interaction_y.shape[0]) or (nos != DDI_interaction_z.shape[0]) :
            print("Size mismatch between DDI and spin array")
    #        break
    #Filter out any vacant sites
        vacancies_idx = np.where(types == -1)
        locs[:,0][vacancies_idx] = 0
        locs[:,1][vacancies_idx] = 0
        locs[:,2][vacancies_idx] = 0

        Hts = [H_relax]*H_relax_steps
        for i,Ht in enumerate(Hts):

            Hmag = Ht
            hamiltonian.set_field(p_state,Hmag,(Ht,0,0)) #Inside set_field, the vector is normalized, so we don't have to do that here

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

            DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy

            DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

            DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

            # Pass into SPIRIT
            DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x_from_z, DDI_field_y_from_z, DDI_field_z_total)))
            system.set_DDI_field(p_state,n_atoms=nos,ddi_fields=DDI_field_interleave)

            #Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files
            # for j in range(converge_max):
            tqdm.write(f'Ht: {Ht:.3f}, i: {i}')
            simulation.start(p_state, simulation.METHOD_MC, single_shot=False) #solver_type=simulation.MC_ALGORITHM_METROPOLIS
            simulation.stop(p_state)

            # if output_interval>0: #Output the spin configuration.
            #     #TODO: Use system.get_spin_directions to pull the configuration array into Python, and then save out as an npy or similar
            #     if (i % output_interval == 0):
            #         tag = prefix+f'N{i:d}_H{Ht:.3f}'
            #         name = "output/" + tag + "_Image-00_Spins_0.ovf" #To match the internally-generated naming format
            #         io.image_write(p_state,filename=name)

        spins = system.get_spin_directions(p_state)  # Get the current spin state to update the DDI fields from the Ewald sum
        spins[:, 2][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
        spins[:, 1][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
        spins[:, 0][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site

        # DDI_field_x_from_z = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4
        # DDI_field_y_from_z = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4
        DDI_field_z_from_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4

        DDI_field_z_from_y = np.matmul(DDI_interaction_y.T, spins[:, 1]) * 7 / 1e4  # V_zy
        DDI_field_z_from_x = np.matmul(DDI_interaction_x.T, spins[:, 0]) * 7 / 1e4  # V_zx

        DDI_field_z_total = DDI_field_z_from_z + DDI_field_z_from_y + DDI_field_z_from_x

        # DDI_field_trans = np.sqrt(DDI_field_x_from_z ** 2 + DDI_field_y_from_z ** 2)

        # Remove vacant site data from output arrays
        valid_idx = np.setdiff1d(np.arange(spins.shape[0]), vacancies_idx)

        return pd.DataFrame({
            'Ht': H_relax * np.ones(len(valid_idx)),
            'DDI_field_z': DDI_field_z_total[valid_idx],
            # 'DDI_field_trans': DDI_field_trans[valid_idx]
        })


if __name__ == '__main__':

    # Assuming fields_hyst and mz are already defined
    n_cycles = 720
    H_relaxes = [0.1, 1.0]
    H_relaxes = H_relaxes * n_cycles
    H_relax_steps = 5

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(plot_loop, H_relaxes), total=len(H_relaxes)))

    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(f'B_z_distribution_{dim}_H_relax_steps_{H_relax_steps}_ncycles_{n_cycles}_gamma_{gamma}.csv', index=False)

    hist_data = {}

    for Ht in df_all["Ht"].unique():
        data = df_all[df_all["Ht"] == Ht]["DDI_field_z"]
        counts, bin_edges = np.histogram(data, bins=80, density= True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_data[Ht] = (bin_centers, counts)

    fig = go.Figure()

    for Ht, (x, y) in hist_data.items():
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name=f"Ht={Ht}",
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="DDI_field_z Distribution for different Ht",
        xaxis_title="DDI_field_z (T)",
        yaxis_title="Density",
    )

    fig.write_html(f'B_z_distribution_{dim}_H_relax_steps_{H_relax_steps}_ncycles_{n_cycles}_gamma_{gamma}.html')
