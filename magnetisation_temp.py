from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration
import numpy as np
import os
import matplotlib.pyplot as plt

def write_config(p_state,prefix="") :
    fn = prefix + "_config.txt"
    with open(fn,"w") as cfile :
        print(f"Temperature: {parameters.mc.get_temperature(p_state)}", file=cfile)
        print(f"Basis Cell size: {geometry.get_n_cell_atoms(p_state)}", file=cfile)
        nsize = geometry.get_n_cells(p_state)
        print(f"Lattice size: {nsize[0]}, {nsize[1]}, {nsize[2]}",file=cfile)
        print(f"Lattice constant: {geometry.get_lattice_constant(p_state)}",file=cfile)
        anisotropy_mag,anisotropy_vec = hamiltonian.get_anisotropy(p_state)
        print(f"Standard spin moment: {geometry.get_mu_s(p_state)[0]}",file=cfile)
        print(f"Uniaxial anisotropy magnitude: {anisotropy_mag}",file=cfile)
        print(f"Uniaxial anisotropy vector: {anisotropy_vec[0]}, {anisotropy_vec[1]}, {anisotropy_vec[2]}",file=cfile)
        field_mag,field_vec = hamiltonian.get_field(p_state)
        print(f"Applied field magnitude: {field_mag}",file=cfile)
        print(f"Applied field vector: {field_vec[0]}, {field_vec[1]}, {field_vec[2]}",file=cfile)

def make_output_dir(prefix="") :
    HOME = os.path.dirname(os.getcwd())
    OUTPUT_PATH = os.path.join(HOME,"Output{}".format(prefix))
    if not os.path.isdir(OUTPUT_PATH) : os.makedirs(OUTPUT_PATH)
    return HOME,OUTPUT_PATH

#When we're done, go back to the original home directory
#os.path.setcwd(HOME)

# Hmax = 8
# Hstep_coarse = 0.125
# Hstep_fine = 0.05
# Hfine1 = 2
# Hfine2 = -3
# fields = np.arange(Hmax,Hfine1,-1*Hstep_coarse,dtype=float)
# fields = np.append(fields,np.arange(Hfine1,Hfine2,-1*Hstep_fine,dtype=float))
# fields = np.append(fields,np.arange(Hfine2,-1*Hmax,-1*Hstep_coarse,dtype=float))
# fields_hyst = np.append(fields,-1*fields)

Tmax = 40
Tstep_coarse = 0.5
Tstep_fine = 0.1
Tfine1 = 20
Tfine2 = -0.05
temps = np.arange(Tmax,Tfine1,-1*Tstep_coarse,dtype=float)
temps = np.append(temps,np.arange(Tfine1,Tfine2,-1*Tstep_fine,dtype=float))
# fields = np.append(fields,np.arange(Hfine2,-1*Hmax,-1*Hstep_coarse,dtype=float))
# fields_hyst = np.append(fields,-1*fields)




output_interval = 1 #Interval at which spin configuration files are saved
#prefix = "Ising_Dilute90_lattice15_ddi"
prefix = "LHF_DDI_glass_14"

iterations_per_step = 1 #Take this many Metropolis iterationss per lattice site between each check for convergence
converge_threshold = 0.01 #Fractional change in magnetization between steps to accept convergence
converge_max = 20 #Maximum number of steps to take before moving on
mu = 7


#with state.State("input/test_Ising_largelattice.cfg") as p_state:

spin_concentrations = [25, 50, 75]
plt.figure(figsize=(8, 5))

for concentration in spin_concentrations :

    with state.State(f"input/LHF_DDI_glass_14_{concentration}_temp.cfg") as p_state:
        types = geometry.get_atom_types(p_state)
        locs = geometry.get_positions(p_state)
        np.savetxt("output/"+prefix+"atom_locs.csv",locs,delimiter=",")
        np.savetxt("output/"+prefix+"atom_types.csv",types,delimiter=",")
        write_config(p_state,prefix)
        parameters.mc.set_metropolis_cone(p_state,use_cone=True,cone_angle=30,use_adaptive_cone=True)
        parameters.mc.set_metropolis_spinflip(p_state,False)

        #We'll evaluate convergence after enough Metropolis steps to hit each site twitce on average
        parameters.mc.set_iterations(p_state,iterations_per_step*types.size,iterations_per_step*types.size)

        #Initialize in the saturated state
        configuration.plus_z(p_state)

        for i,T in enumerate(temps):

            print(f'{T:.3f}')

            #Saving spin configuration is broken for MC. Instead, do one iteration of LLG just to write out the spins
            if i % output_interval == 0 :
                tag = prefix+f'N{i:d}_H{T:.3f}'
                parameters.llg.set_output_tag(p_state,tag)
                simulation.start(p_state, simulation.METHOD_LLG,simulation.SOLVER_DEPONDT, single_shot=True)
                simulation.stop(p_state)

            hamiltonian.set_field(p_state, 0.5, (0, 0, 1))
            parameters.mc.set_temperature(p_state, T)


            for j in range(converge_max) :
                simulation.start(p_state, simulation.METHOD_MC, solver_type=simulation.SOLVER_DEPONDT, single_shot=False)
                simulation.stop(p_state)
                if j == 0 :
                    m_temp = quantities.get_magnetization(p_state)[2]
                else :
                    m_prev = m_temp
                    m_temp = quantities.get_magnetization(p_state)[2]
     #               ratio = abs((m_temp-m_prev)/m_prev)
                    ratio = abs((m_temp-m_prev)/mu)
                    print(j,ratio)
                    if ratio<converge_threshold :
                        break

            if i == 0:
                mx = quantities.get_magnetization(p_state)[0]
                my = quantities.get_magnetization(p_state)[1]
                mz = m_temp
                iter_count = j
            else:
                mx = np.vstack((mx,quantities.get_magnetization(p_state)[0]))
                my = np.vstack((my,quantities.get_magnetization(p_state)[1]))
                mz = np.vstack((mz,m_temp))
                iter_count = np.vstack((iter_count,j))

        #np.savetxt("output/"+prefix+"mh.csv",np.transpose((fields_hyst,mx[:,0],my[:,0],mz[:,0],iter_count[:,0])),delimiter=',')


        data = np.transpose((temps,mx[:,0],my[:,0],mz[:,0],iter_count[:,0]))
        plt.plot(temps, mz[:, 0] / (concentration /100), label = str(concentration))

# Assuming fields_hyst and mz are already defined



plt.xlabel('Temperature')
plt.ylabel('Avg Magnetization Per Spin')
plt.title('Hysteresis Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('magnetisation_temp.png')
