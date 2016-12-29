#pragma once
#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <random>
#include <vector>

#include "Core_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

namespace Engine
{
	/*
		The Hamiltonian contains the interaction parameters of a System.
		It also defines the functions to calculate the Effective Field and Energy.
	*/
	class Hamiltonian
	{
	public:
		Hamiltonian(std::vector<bool> boundary_conditions);

		/*
			Update the Energy array.
			This needs to be done every time the parameters are changed, in case an energy
			contribution is now non-zero or vice versa.
		*/
		virtual void Update_Energy_Contributions();

		/*
			Calculate the Hessian matrix of a spin configuration.
			This function uses finite differences and may thus be quite inefficient. You should
			override it if you want to get proper performance.
			This function is the fallback for derived classes where it has not been overridden.
		*/
		virtual void Hessian(const vectorfield & spins, MatrixX & hessian);

		/*
			Calculate the effective field of a spin configuration.
			This function uses finite differences and may thus be quite inefficient. You should
			override it if you want to get proper performance.
			This function is the fallback for derived classes where it has not been overridden.
		*/
		virtual void Effective_Field(const vectorfield & spins, vectorfield & field);

		// Calculate the Energy of a spin configuration
		virtual scalar Energy(const vectorfield & spins);

		// Calculate the Energies of the spins of a configuration
		virtual std::vector<std::vector<scalar>> Energy_Array_per_Spin(const vectorfield & spins);

		// Calculate the Effective Field of a spin configuration
		virtual std::vector<std::pair<std::string, scalar>> Energy_Array(const vectorfield & spins);

		// Hamiltonian name as string
		virtual const std::string& Name();

		// Boundary conditions
		std::vector<bool> boundary_conditions; // [3] (a, b, c)
	
	private:
		std::mt19937 prng;
		std::uniform_int_distribution<int> distribution_int;
		scalar delta;
	};
}
#endif