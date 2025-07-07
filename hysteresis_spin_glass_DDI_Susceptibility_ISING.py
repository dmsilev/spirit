from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system,io
import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import linregress


iterations_per_step = 1  # Take this many Metropolis iterationss per lattice site between each check for convergence
#Now is Ht transverse fields, max is  = 1T


Hz = 0.0  #Longitudinal field
output_interval = 2 #Interval at which spin configuration files are saved
fn = "dipolar_arr"
prefix = "DDI_exp_14_G0p00005_Ht10p0"

mu = 7
dim = 10
concentration = 25

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

# #Check that the size of the DDI arrays matches NOS (extracted from the types array above).
# if (nos != DDI_interaction_x.shape[0]) or (nos != DDI_interaction_y.shape[0]) or (nos != DDI_interaction_z.shape[0]) :
#     print("Size mismatch between DDI and spin array")
# #        break

#with state.State("input/test_Ising_largelattice.cfg") as p_state:

def get_susceptibility(p_state_file, H_relax, Ht):
    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg") as p_state_1:
        #For a given state vary longitudinal H to get dc susceptibility
        io.image_read(p_state_1, p_state_file)
        types = geometry.get_atom_types(p_state_1)
        nos = types.size
        # print(f"Sites: {nos}  Spins: {nos + np.sum(types)}")
        locs = geometry.get_positions(p_state_1)
        np.savetxt("output/" + prefix + "atom_locs.csv", locs, delimiter=",")
        np.savetxt("output/" + prefix + "atom_types.csv", types, delimiter=",")
        #    write_config(p_state_1,prefix)
        parameters.mc.set_metropolis_cone(p_state_1, use_cone=True, cone_angle=0.1, use_adaptive_cone=False)
        parameters.mc.set_metropolis_spinflip(p_state_1, True)
        # parameters.mc.set_tunneling_gamma(p_state_1, tunneling_gamma=0.00012)

        # For trying without tunneling and at base temperature ~0K
        parameters.mc.set_use_tunneling(p_state_1, False)
        parameters.mc.set_temperature(p_state_1, 0.0000000001)

        # We'll evaluate convergence after enough Metropolis steps to hit each site twitce on average
        parameters.mc.set_iterations(p_state_1, iterations_per_step * types.size, iterations_per_step * types.size)

        # Initialize in the saturated p_state_1
        configuration.plus_z(p_state_1)

        ## Import the DDI interaction arrays
        #    path_arr_x = os.path.join("input/", fn +"_DDI_x.npy")
        #    path_arr_y = os.path.join("input/", fn +"_DDI_y.npy")
        #    path_arr_z = os.path.join("input/", fn +"_DDI_z.npy")
        path_arr_x = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/",
                                  fn + "_x.npy")  # fn = dipolar_arr
        path_arr_y = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn + "_y.npy")
        path_arr_z = os.path.join(f"dipolar_interaction_matrices_reordered/{dim}_{dim}_{dim}/", fn + "_z.npy")

        # Check that the size of the DDI arrays matches NOS (extracted from the types array above).
        if (nos != DDI_interaction_x.shape[0]) or (nos != DDI_interaction_y.shape[0]) or (
                nos != DDI_interaction_z.shape[0]):
            print("Size mismatch between DDI and spin array")
        #        break
        # Filter out any vacant sites
        vacancies_idx = np.where(types == -1)
        locs[:, 0][vacancies_idx] = 0
        locs[:, 1][vacancies_idx] = 0
        locs[:, 2][vacancies_idx] = 0

        # Bfields = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
        Bfields = np.arange(0, 0.1, 0.01)
        for i, Hz in enumerate(Bfields):
            # print(f'Hz: {Hz:.3f}', concentration)

            Hmag = np.sqrt(Hz * Hz + Ht * Ht)
            hamiltonian.set_field(p_state_1, Hmag, (Ht, 0,
                                                  Hz))  # Inside set_field, the vector is normalized, so we don't have to do that here

            spins = system.get_spin_directions(
                p_state_1)  # Get the current spin p_state_1 to update the DDI fields from the Ewald sum
            spins[:, 2][vacancies_idx] = 0  # For LHF, we only care about Sz, but zero out the moments for vacancy site

            # Calculate the DDI field components, and then send to the SPIRIT engine. DDI interaction calculations are in Oe, SPIRIT
            # uses T, so need to scale accordingly.
            # Get the field at each spin. Only care about DDI due to z-spin so use spins[:,2]
            DDI_field_x = np.matmul(DDI_interaction_x, spins[:, 2]) * 7 / 1e4
            DDI_field_y = np.matmul(DDI_interaction_y, spins[:, 2]) * 7 / 1e4
            DDI_field_z = np.matmul(DDI_interaction_z, spins[:, 2]) * 7 / 1e4

            # Pass into SPIRIT
            DDI_field_interleave = np.ravel(np.column_stack((DDI_field_x, DDI_field_y, DDI_field_z)))
            system.set_DDI_field(p_state_1, n_atoms=nos, ddi_fields=DDI_field_interleave)
            # print(f"X: mean: {np.mean(DDI_field_x):.4e} Std. Dev: {np.std(DDI_field_x):.4e}")
            # print(f"Y: mean: {np.mean(DDI_field_y):.4e} Std. Dev: {np.std(DDI_field_y):.4e}")
            # print(f"Z: mean: {np.mean(DDI_field_z):.4e} Std. Dev: {np.std(DDI_field_z):.4e}")

            converge_threshold = 0.01  # Fractional change in magnetization between steps to accept convergence
            converge_max = 20  # Maximum number of steps to take before moving on
            # Check convergence, same as old hysteresis_loop. But METHOD_MC now uses tunnelling since we set tunnel flag to 1 in cfg files
            for j in range(converge_max):
                simulation.start(p_state_1, simulation.METHOD_MC,
                                 single_shot=False)  # solver_type=simulation.MC_ALGORITHM_METROPOLIS
                simulation.stop(p_state_1)
                if j == 0:
                    m_temp = quantities.get_magnetization(p_state_1)[2]
                else:
                    m_prev = m_temp
                    m_temp = quantities.get_magnetization(p_state_1)[2]
                    #               ratio = abs((m_temp-m_prev)/m_prev)
                    ratio = abs((m_temp - m_prev) / mu)
                    # print(f"Iteration: {j:d}, Convergence: {ratio:.4f}, M_z: {m_temp:.4f}")
                    if ratio < converge_threshold:
                        break

            if i == 0:
                mz = m_temp
            else:
                mz = np.vstack((mz, m_temp))

        # Fit a straight line
        susceptibility, intercept, r_value, p_value, std_err = linregress(Bfields, mz[:, 0])

    return susceptibility


def plot_loop(H_relax):

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_tunnel_{dim}.cfg") as p_state:
        types = geometry.get_atom_types(p_state)
        nos = types.size
        # print(f"Sites: {nos}  Spins: {nos+np.sum(types)}")
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
    #    write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state,use_cone=True,cone_angle=0.1,use_adaptive_cone=False)
        parameters.mc.set_metropolis_spinflip(p_state,True)

        # parameters.mc.set_tunneling_gamma(p_state, tunneling_gamma=0.00012)

        #For trying without tunneling and at base temperature ~0K
        parameters.mc.set_use_tunneling(p_state, False)
        parameters.mc.set_temperature(p_state, 0.000035)

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
        Hmax = 1
        H_step = 0.01
        Hts = np.arange(Hmax, H_relax, -1 * H_step, dtype=float)
        chis = []
        #Check susceptibility every 5th Ht datapoint
        for i,Ht in enumerate(Hts):
            print(f'Ht: {Ht:.3f}', concentration)

            #To randomise field, set very high temperature
            if i % output_interval == 0 :
                tag = prefix+f'N{i:d}_H{Ht:.3f}'
                parameters.llg.set_output_tag(p_state,tag)
                parameters.llg.set_temperature(p_state, 1000000)
                simulation.start(p_state, simulation.METHOD_LLG,simulation.SOLVER_DEPONDT, single_shot=False)
                simulation.stop(p_state)
                parameters.llg.set_temperature(p_state, 0.000035)

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
            # print(f"Z: mean: {np.mean(DDI_field_z):.4e} Std. Dev: {np.std(DDI_field_z):.4e}")

            ####Less strict convergence check conditions to simulate spin glass not relaxing to equilibrium state
            # converge_threshold = 0.01 #Fractional change in magnetization between steps to accept convergence
            converge_threshold = 0.05  # Fractional change in magnetization between steps to accept convergence
            # converge_max = 20 #Maximum number of steps to take before moving on
            converge_max = 10  # Maximum number of steps to take before moving on

            # Stricter convergence check for relaxation step
            converge_threshold_relax = 0.01
            converge_max_relax = 40

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
                        print(f'++++++++++++++++++++++++++++++++Number of iterations needed for ratio<convergence_threshold: {j}++++++++++++++++++++++++++++++++')
                        break

            if i%5 ==0:
                filename = f'state_configs/p_state_{Ht:.3f}_Hrelax_{H_relax:.3f}.ovf'
                io.image_write(p_state, filename)
                chi = get_susceptibility(filename, H_relax, Ht)

                chis.append((H_relax, Ht, chi))

    return chis


if __name__ == '__main__':

    # Assuming fields_hyst and mz are already defined
    n_cycles = 16
    H_relaxes = [0]*n_cycles


    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(plot_loop, H_relaxes)

    # flatten results
    flat = [item for sublist in results for item in sublist]

    # make dataframe
    df_all = pd.DataFrame(flat, columns=["H_relax", "Ht", "chi"])
    df_avg = (
        df_all.groupby(["H_relax", "Ht"], as_index=False)
        .agg({"chi": "mean"})
    )
    fig = px.line(
        df_avg,
        x="Ht",
        y="chi",
        color="H_relax",
        markers=True,
        labels={"Ht": "Ht (T)", "chi": "Susceptibility χ"},
        title="Susceptibility χ vs Ht for different H_relax"
    )

    fig.write_html(f'Susceptibility_ISING_{dim}_{n_cycles}_{concentration}.html')
