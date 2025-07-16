#from spirit import simulation, state,quantities, hamiltonian,parameters,geometry,configuration,system
import numpy as np
import dipolar_parallel_2_x


#import pandas as pd
#import paths

def run_DDI_calculation(fn, atom_locs_file = "", lattice_dims_file=""): # Filtered from vacancies
						# atom locations nx3, crystal dims nx3, atom spin vectors nx3
						# H serves as a title distinguisher, placed in preparation for hysteresis loop calculations with multiple DDI calculations
	

	OVERWRITE_FLAG = False
	atom_locs = np.loadtxt(atom_locs_file, delimiter = ',')
	lattice_dims = np.loadtxt(lattice_dims_file, delimiter = ',')

	if OVERWRITE_FLAG:
		print("MAY BE OVERWRITING DATA")
		#check = input("Continue? (y/n) ")
		
## If not already existing as saved file, DDI interaction calculation will be executed. If exists, existing saved file will load. 
	# path_arr_x = os.path.join(paths.PATH_DATA, fn +"_DDI_x.npy")
	# path_arr_y = os.path.join(paths.PATH_DATA, fn +"_DDI_y.npy")
	# path_arr_z = os.path.join(paths.PATH_DATA, fn +"_DDI_z.npy")

	# if os.path.exists(path_arr_x) and os.path.exists(path_arr_y) and os.path.exists(path_arr_z) and not OVERWRITE_FLAG:
	# 	print("loading DDI interaction data.")
	# 	DDI_interaction_x = np.load(path_arr_x)
	# 	DDI_interaction_y = np.load(path_arr_y)
	# 	DDI_interaction_z = np.load(path_arr_z)
		
		#DDI_arr = np.array([DDI_interaction_x, DDI_interaction_y, DDI_interaction_z])


	else:

		print("Calculating DDI interaction.")

		# print("The size of atom_locs is:" + str(atom_locs.size))
		# print(atom_locs)
		DDI_interaction_x = dipolar_parallel_2_x.calculate_dipolar_arr(atom_locs / 10, 0,
																	   lattice_dims)  # atom positions need to be in nm so /10
		print("Calculation for DDI Interaction in x completed.")
		DDI_interaction_y = dipolar_parallel_2_x.calculate_dipolar_arr(atom_locs / 10, 1, lattice_dims)
		print("Calculation for DDI Interaction in y completed.")

		# DDI_interaction_z = dipolar_parallel_2_x.calculate_dipolar_arr(atom_locs/10, 2, lattice_dims)
		# print("Calculation for DDI Interaction in z completed.")
		# DDI_arr = np.array([DDI_interaction_x, DDI_interaction_y, DDI_interaction_z])

		folder = 'dipolar_interaction_matrices_reordered_x/'

		np.save(folder + fn + "_DDI_x.npy", DDI_interaction_x)
		np.save(folder + fn + "_DDI_y.npy", DDI_interaction_y)
		# np.save(fn + "_DDI_z.npy", DDI_interaction_z)
		# pd.DataFrame({"DDI_x": DDI_interaction_x, "DDI_y": DDI_interaction_y, "DDI_Z": DDI_interaction_z}).to_csv(os.path.join(path_data, path_arr), index=False, sep = ',')

		return DDI_interaction_x, DDI_interaction_y  # , DDI_interaction_z


if __name__ == "__main__":
	dim = 4
	run_DDI_calculation(f"LHF_{dim}", f"LHF_{dim}_HPC_atom_locs.csv", f"LHF_{dim}_HPC_lattice_dims.csv")
