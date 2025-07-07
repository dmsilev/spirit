from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system,io
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

Hmax = 10
H_step = 0.1
iterations_per_step = 1  # Take this many Metropolis iterationss per lattice site between each check for convergence
#Now is Ht transverse fields, max is  = 1T

# fields = np.arange(Hmax,Hfine1,-1*Hstep_coarse,dtype=float)
# fields = np.append(fields,np.arange(Hfine1,Hfine2,-1*Hstep_fine,dtype=float))
# fields = np.append(fields,np.arange(Hfine2,-1*Hmax,-1*Hstep_coarse,dtype=float))
# fields_hyst = np.append(fields,-1*fields)
# fields_hyst = np.tile(fields_hyst, n_cycles)

Hz = 0.0  #Longitudinal field
output_interval = 2 #Interval at which spin configuration files are saved
fn = "dipolar_arr"
prefix = "DDI_exp_14_G0p00005_Ht10p0"

mu = 7
dim = 4
concentration = 25

#with state.State("input/test_Ising_largelattice.cfg") as p_state:


def plot_loop(H_relax):

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg") as p_state:
        types = geometry.get_atom_types(p_state)
        nos = types.size
        print(f"Sites: {nos}  Spins: {nos+np.sum(types)}")
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
    #    write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state,use_cone=True,cone_angle=0.00000001,use_adaptive_cone=False)
        parameters.mc.set_metropolis_spinflip(p_state,True)
        parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=0.00012)

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
        Hts = np.arange(Hmax, H_relax, -1 * H_step, dtype=float)
        for i,Ht in enumerate(Hts):
            print(f'{Ht:.3f}', concentration)

            Hmag = np.sqrt(Hz*Hz + Ht*Ht)
            hamiltonian.set_field(p_state,Hmag,(Ht,0,Hz)) #Inside set_field, the vector is normalized, so we don't have to do that here

            spins = system.get_spin_directions(p_state)  #Get the current spin state to update the DDI fields from the Ewald sum
            spins[:,2][vacancies_idx] = 0   #For LHF, we only care about Sz, but zero out the moments for vacancy site

            #Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
            #uses T, so need to scale accordingly.
            #Get the field at each spin. Only care about DDI due to z-spin so use spins[:,2]
            DDI_field_x = np.matmul(DDI_interaction_x,spins[:,2]) *7/1e4
            DDI_field_y = np.matmul(DDI_interaction_y,spins[:,2]) *7/1e4
            DDI_field_z = np.matmul(DDI_interaction_z,spins[:,2]) *7/1e4

            #Pass into SPIRIT
            DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x,DDI_field_y,DDI_field_z)))
            system.set_DDI_field(p_state,n_atoms=nos,ddi_fields=DDI_field_interleave)
            # print(f"X: mean: {np.mean(DDI_field_x):.4e} Std. Dev: {np.std(DDI_field_x):.4e}")
            # print(f"Y: mean: {np.mean(DDI_field_y):.4e} Std. Dev: {np.std(DDI_field_y):.4e}")
            print(f"Z: mean: {np.mean(DDI_field_z):.4e} Std. Dev: {np.std(DDI_field_z):.4e}")

            ####Less strict convergence check conditions to simulate spin glass not relaxing to equilibrium state
            # converge_threshold = 0.01 #Fractional change in magnetization between steps to accept convergence
            converge_threshold = 0.01  # Fractional change in magnetization between steps to accept convergence
            # converge_max = 20 #Maximum number of steps to take before moving on
            converge_max = 10  # Maximum number of steps to take before moving on

            # Stricter convergence check for relaxation step
            converge_threshold_relax = 0.001
            converge_max_relax = 80

            #For relaxing at H_relax
            if Ht == H_relax:
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

        spins = system.get_spin_directions(p_state)  # Get the current spin state to update the DDI fields from the Ewald sum
        spins[:, 2][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site
        DDI_field_x = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4
        DDI_field_y = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4
        DDI_field_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4
        DDI_field_trans = np.sqrt(DDI_field_x ** 2 + DDI_field_y ** 2)

    return pd.DataFrame({
    'Ht': Ht * np.ones_like(DDI_field_z),
    'DDI_field_z': DDI_field_z,
    'DDI_field_trans': DDI_field_trans
})


if __name__ == '__main__':

    # Assuming fields_hyst and mz are already defined
    n_cycles = 8
    H_relaxes = [0, 1, 5, 7, 9, 0.25, 0.5, 0.1]
    H_relaxes = H_relaxes * n_cycles

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(plot_loop, H_relaxes)

    df_all = pd.concat(results, ignore_index=True)
    df_all.to_csv(f'B_z_distribution_ISING_{dim}_trial.csv', index=False)

    hist_data = {}

    for Ht in df_all["Ht"].unique():
        data = df_all[df_all["Ht"] == Ht]["DDI_field_z"]
        counts, bin_edges = np.histogram(data, bins=50)
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
        title="DDI_field_z Distribution (as points) for different Ht",
        xaxis_title="DDI_field_z (T)",
        yaxis_title="Density",
    )

    fig.write_html(f'B_z_distribution_ISING_{dim}_{n_cycles}_trial.html')
