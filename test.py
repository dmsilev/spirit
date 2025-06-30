from spirit import chain, configuration, transition, simulation, system, quantities, state

def evaluate(p_state):
    M = quantities.get_magnetization(p_state)
    E = system.get_energy(p_state)
    return M, E

with state.State("input/input.cfg") as p_state:
    ### Copy the system and set chain length
    chain.image_to_clipboard(p_state)
    noi = 7
    chain.set_length(p_state, noi)

    ### First image is homogeneous with a Skyrmion in the center
    configuration.plus_z(p_state, idx_image=0)
    configuration.skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=0)
    ### Last image is homogeneous
    configuration.plus_z(p_state, idx_image=noi-1)
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=noi-1)

    ### Create transition of images between first and last
    transition.homogeneous(p_state, 0, noi-1)

    ### GNEB calculation
    simulation.start(p_state, simulation.METHOD_MC, simulation.SOLVER_VP)

