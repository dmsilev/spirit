"""Script for calculating the local field on each ion due to the dipolar interaction between all other ions"""

import sys
import datetime

import numpy as np
import scipy
from scipy import special

#from matplotlib import pyplot as plt

from functools import partial
import itertools
from itertools import repeat
from multiprocessing import Pool, freeze_support, cpu_count, Array
from functools import partial
from tqdm import tqdm

########################################################################
# Constants

# Bohr Magneton in (K / G)
mu_B = 6.717e-5

# vacuum permeability in (G^2 nm^3 / K)
mu_0_4pi = 1.3806e5

# Lande g-factor
g_L = 1.25

# renormalized Ising spin
C_zz = 5.51

DIPOLAR_PREFACTOR_LIHO = mu_0_4pi * g_L * mu_B * C_zz

# tolerance to terminate Ewald summation
TOL_DEFAULT = 1e-5

cpu_count = cpu_count()
# cpu_count = 1
########################################################################
# self term

def self_term(field_axis,
				ewald_factor):
	"""
	calculates the self-interaction term of the ewald summation of the dipolar interaction

	args:
		field_axis: which field component to calculate (0 == x, 1 == y, 2 == z)
		ewald_factor: ewald factor in (nm^-1) that controls how quickly the sums converge

	returns:
		self_term: self-interaction term of the Ewald sum
	"""

	ret = 0.

	if field_axis == 2:
		ret = (4. * (ewald_factor**3)) / (3. * np.sqrt(np.pi))

	return ret
########################################################################
# real-space term in Ewald sum

def b_fun(r, alpha):
	"""B(r) as defined in Wang and Holm, J. Chem. Phys. 115, 6351 (2001)"""

	return (1./r**3) * ( special.erfc(alpha*r) + ((2*alpha*r/np.sqrt(np.pi)) * np.exp(-1.*(alpha**2)*(r**2))) )

def c_fun(r, alpha):
	"""C(r) as defined in Wang and Holm, J. Chem. Phys. 115, 6351 (2001)"""

	return (1./r**5) * ( (3.*special.erfc(alpha*r)) + ((2*alpha*r/np.sqrt(np.pi)) * (3. + 2.*(alpha**2)*(r**2)) * np.exp(-1.*(alpha**2)*(r**2))) )

def real_space_sum(r_ij,
				field_axis,
				lattice_dims,
				ewald_factor,
				n_truncate,
				self_int_flag):
	"""
	calculates the real-space term of the ewald summation of the dipolar interaction

	args:
		r_ij: displacement vector in (nm)
		field_axis: which field component to calculate (0 == x, 1 == y, 2 == z)
		lattice_dims: dimensions of full lattice (not just single unit cell) in (nm)
		ewald_factor: ewald factor in (nm^-1) that controls how quickly the sums converge
		n_truncate: integer number of terms to use in the summation truncation
		self_int_flag: boolean to set whether you're calculating the self-interaction (skip divergent n=0 term)

	returns:
		real_space_term: real-space part of the Ewald sum
	"""

	ret = 0.

	for i in range(-n_truncate, n_truncate+1):
		for j in range(-n_truncate, n_truncate+1):
			for k in range(-n_truncate, n_truncate+1):

				# print(f"{self_int_flag}, {i}, {j}, {k}")


				# omit divergent i=j=k=0 term for r_ij = 0
				if self_int_flag and (i == 0) and (j == 0) and (k == 0):
					continue

				# create the ith,jth,kth image of dest spin
				n_vec = np.array([i, j, k]) * lattice_dims
				r_n = r_ij + n_vec
				r_n_mag = np.linalg.norm(r_n)

				term = r_n[2] * r_n[field_axis] * c_fun(r_n_mag, ewald_factor)

				if field_axis == 2:
					term = term - b_fun(r_n_mag, ewald_factor)

				ret = ret + term

	return ret
########################################################################
# fourier-space term in Ewald sum

def k_space_sum(r_ij,
				field_axis,
				recip_lattice_dims,
				sample_vol,
				ewald_factor,
				n_truncate):
	"""
	calculates the k-space term of the ewald summation of the dipolar interaction

	args:
		r_ij: displacement vector in (nm)
		field_axis: which field component to calculate (0 == x, 1 == y, 2 == z)
		recip_lattice_dims: dimensions of full reciprocal lattice (not just single unit cell) in (nm)
		sample_vol: sample volume in (nm^3)
		ewald_factor: ewald factor in (nm^-1) that controls how quickly the sums converge
		n_truncate: integer number of terms to use in the summation truncation

	returns:
		k_space_term: fourier-space part of the Ewald sum
	"""

	ret = 0.

	for i in range(-n_truncate, n_truncate+1):
		for j in range(-n_truncate, n_truncate+1):
			for k in range(-n_truncate, n_truncate+1):

				if i == 0 and j == 0 and k == 0:
					continue

				k_vec = np.array([i, j, k]) * recip_lattice_dims
				k_mag = np.linalg.norm(k_vec)

				term = -1. * np.exp(-1.*(k_mag**2)/(4.*ewald_factor**2)) * (k_vec[2] * k_vec[field_axis]) * np.cos(np.dot(k_vec, r_ij)) / (k_mag**2)
				
				ret = ret + term

	return (4. * np.pi / sample_vol) * ret
########################################################################
# effective dipolar interaction between two spins
def ewald_sum(i, 
				j,
				r_ij,
				field_axis,
				lattice_dims,
				recip_lattice_dims,
				sample_vol,
				ewald_factor,
				n_truncate_r,
				n_truncate_k,
				self_int_flag,
				dipolar_prefactor,
				n_spins):
	"""
	calculates the effective dipolar interaction between two spins using ewald summation

	args:
		r_ij: displacement vector in (nm)
		field_axis: which field component to calculate (0 == x, 1 == y, 2 == z)
		lattice_dims: dimensions of full lattice (not just single unit cell) in (nm)
		recip_lattice_dims: dimensions of full reciprocal lattice (not just single unit cell) in (nm)
		sample_vol: sample volume in (nm^3)
		ewald_factor: ewald factor in (nm^-1) that controls how quickly the sums converge
		n_truncate_r: integer number of terms to use in the real-space summation truncation
		n_truncate_k: integer number of terms to use in the real-space summation truncation
		self_int_flag: boolean to set whether you're calculating the self-interaction (skip divergent n=0 term)
		dipolar_prefactor: dipolar prefactor = (mu_0 / 4pi) g_L mu_B C_zz in (G nm^3)

	returns:
		dipolar_int: effective dipolar interaction between spins i,j in (G)
	"""

	#global dipolar_arr
	#dipolar_arr = np.zeros((n_spins,n_spins))

	real_space_term = real_space_sum(r_ij=r_ij,
								field_axis=field_axis,
										lattice_dims=lattice_dims,
										ewald_factor=ewald_factor,
										n_truncate=n_truncate_r,
										self_int_flag=self_int_flag)

	k_space_term = k_space_sum(r_ij=r_ij,
								field_axis=field_axis,
								recip_lattice_dims=recip_lattice_dims,
								sample_vol=sample_vol,
								ewald_factor=ewald_factor,
								n_truncate=n_truncate_k)


	dipolar_int = dipolar_prefactor * (real_space_term + k_space_term)

	if self_int_flag:
		dipolar_int = dipolar_int + self_term(field_axis=field_axis,
												ewald_factor=ewald_factor)


	# dipolar_arr[i][j] = dipolar_int
	# if i != j:
	# 	dipolar_arr[j][i] = dipolar_int
	return dipolar_int


########################################################################
# array_dict = {}

# def init_worker(X, X_shape):
# 	global array_dict
# 	array_dict['X'] = X
# 	array_dict['X_shape'] = X_shape

########################################################################

def processing_dipolar(params, const):

	#X_np = np.frombuffer(array_dict.X.get_obj()).reshape(array_dict.X_shape)
	i = params[0]
	j = params[1]


	field_axis = const[0]
	lattice_dims =const[1]
	recip_lattice_dims = const[2]
	sample_vol =const[3]
	ewald_factor = const[4]
	n_truncate_r = const[5]
	n_truncate_k = const[6]
	dipolar_prefactor = const[7]
	pos_arr = const[8]
	n_spins = const[9]

	#cw.update_prog_loop("complete...")

# matrix is symmetric--only calculate upper-triangle
	#if j > i: 
		
# displacement vector
	r_ij = pos_arr[j] - pos_arr[i]

# self-interaction
	self_int_flag = (i == j)
	DDI_value = ewald_sum(i=i, 
						j=j,
						r_ij=r_ij,
						field_axis=field_axis,
						lattice_dims=lattice_dims,
						recip_lattice_dims=recip_lattice_dims,
						sample_vol=sample_vol,
						ewald_factor=ewald_factor,
						n_truncate_r=n_truncate_r,
						n_truncate_k=n_truncate_k,
						self_int_flag=self_int_flag,
						dipolar_prefactor=dipolar_prefactor,
						n_spins = n_spins)


	return DDI_value
########################################################################
def calculate_dipolar_arr(pos_arr, 
							field_axis,
							lattice_dims,
							dipolar_prefactor=DIPOLAR_PREFACTOR_LIHO,
							tol=TOL_DEFAULT):
	"""
	calculates the dipolar interaction between each spin

	args:
		pos_arr: (n x 3) array of spin locations (nm)
		field_axis: which field component to calculate (0 == x, 1 == y, 2 == z)
		lattice_dims: dimensions of full lattice (not just single unit cell) in (nm)
		dipolar_prefactor: dipolar prefactor = (mu_0 / 4pi) g_L mu_B C_zz in (G nm^3)
		tol: tolerance to use for terminating Ewald summation

	returns:
		dipolar_arr: (n x n) array of dipolar interactions between spins in (G)
	"""
	cw = ConsoleWriter()

	n_spins = pos_arr.shape[0]

	# sample volume
	sample_vol = np.prod(lattice_dims)

	# k-space vectors for creating image spins
	recip_lattice_dims = 2. * np.pi / lattice_dims

	# Ewald factor to make each part of the sum converge approximately equally quickly
	ewald_factor = 2. * np.sqrt(np.pi) / np.amax(lattice_dims)

	# number of terms to use in the sum truncation
	n_truncate_r = int(np.ceil( np.sqrt(-1. * np.log(tol)) / (ewald_factor * lattice_dims[0]) ))
	n_truncate_k = int(np.ceil( (ewald_factor * lattice_dims[0]) * np.sqrt(-1. * np.log(tol)) / (np.pi) ))



	dipolar_arr = np.zeros((n_spins, n_spins))

	# i = range(n_spins)
	# j = range(n_spins)

	constants = (field_axis,
		lattice_dims,
		recip_lattice_dims,
		sample_vol,
		ewald_factor,
		n_truncate_r,
		n_truncate_k,
		dipolar_prefactor,
		pos_arr,
		n_spins)

	# paramlist = list(itertools.product(i, j))

	# matrix is symmetric--only calculate upper-triangle
	paramlist = [(i, j) for i in range(n_spins) for j in range(i, n_spins)]

	with Pool() as pool:
		# Create a partial function with the constants
		func = partial(processing_dipolar, const=constants)
		# dipolar_arr_result = pool.map(partial(processing_dipolar, const = constants), paramlist)
		# Use imap instead of map for progress tracking
		dipolar_arr_result = list(
			tqdm(pool.imap(func, paramlist), total=len(paramlist), desc="Processing")
		)

	# Fill upper dipolar_arr, mirror to the lower triangle
	for (i, j), value in zip(paramlist, dipolar_arr_result):
		dipolar_arr[i, j] = value
		if i != j:
			dipolar_arr[j, i] = value

	# dipolar_arr = np.array(dipolar_arr_result).reshape(n_spins,n_spins)


	# dipolar_arr_summed = np.sum(dipolar_arr, axis = 1) #????

	# return dipolar_arr_summed
	return dipolar_arr
########################################################################

########################################################################
class ConsoleWriter:
	"""Object that will print display messages to console while over-writing previous message"""

	def __init__(self, timestamp_flag=True):
		self.n_chars = 0
		self.timestamp_flag = timestamp_flag
		self.n_loop = 0
		self.ind_cur = 0
		self.prog_percent = 0

	def write(self, msg):
		"""Writes a message to the console while overwriting the previous message"""
		if self.timestamp_flag: 
			msg = "{}: {}".format( datetime.datetime.now().strftime("%H:%M:%S"), msg )
		sys.stdout.write("\r{}".format(" "*self.n_chars))
		sys.stdout.flush()
		sys.stdout.write("\r{}".format(msg))
		sys.stdout.flush()
		self.n_chars = len(msg)
	def display(self, msg): self.write(msg)

	def print(self, msg): 
		self.close()
		if self.timestamp_flag:
			msg = "{}: {}".format( datetime.datetime.now().strftime("%H:%M:%S"), msg )
		print("{}".format(msg))

	def close(self):
		"""Clears the current message"""
		sys.stdout.write("\r{}\r".format(" "*self.n_chars))
		sys.stdout.flush()
		self.n_chars = 0
	def clear(self): self.close()


	def init_prog_loop(self, n_loop):
		"""prepares console writer to loop over something and only print every 1%"""

		self.n_loop = n_loop
		self.ind_cur = 0
		self.prog_percent = 0

	def update_prog_loop(self, msg=""):
		"""writes a message every 1%"""

		self.ind_cur = self.ind_cur + 1

		new_prog_percent = int(1e2 * float(self.ind_cur) / float(self.n_loop))
		if new_prog_percent > self.prog_percent:
			self.prog_percent = new_prog_percent
			self.write(f"{self.prog_percent:02d}%: {msg}")


########################################################################

