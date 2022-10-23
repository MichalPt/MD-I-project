import simplemd_2 as smd
import numpy as np
from contextlib import ExitStack

def simulate(settings):

    print('SimpleMD')
    print('========')
    print()
    
    # extract settings
    s = settings
    N = s['N_particles']
    names = list(s['name']) * N
    settings.update({'names':names})
    masses = s['masses']
    
    # cutoff and potential settings:
    sc = s['cutoff']
    
    if sc['use_cutoff'] is True:
        potential = smd.Potential(sc['potential'])
        cutoff = smd.Cutoff(sc['cutoff_distance'],
                        smooth_length = sc['smoothing_length'],
                        smooth_type = sc['cutoff_smoothing'],
                        shift = sc['shift'],
                        n = sc['n'],
                        c = sc['c'])
        interactions = smd.Interactions(potential, cutoff)
    
    else:
        from .utils import format_interaction_function
        print("")
        interactions = format_interaction_function(s['interactions'])

    
    # print potential:
    if isinstance(interactions, smd.Interactions) and sc['plot'] is True:
        print("Plotting graph ...")
        interactions.plot(ylim=sc['ylim'], xlim=sc['xlim'])
    
    use_sim_time = s['use_sim_time']
    sim_time = s['sim_time']
    dt = s['dt']
    
    if use_sim_time is True:
        n_steps = int(sim_time / dt)
    else:
        n_steps = s['n_steps']
        
    stride_progress = s['stride_progress']
    do_wrap = s['do_wrap']
    k_B = s['k_B']
    
    
    # derived settings
    dth = 0.5 * dt
    L = (N / s['rho'])**(1.0/3)
    box = np.array((L, L, L))
    print("box: ",box)
    
    
    # simple neighbourlist setup:    
    use_nb = s['neighbourlist']['use_neighbourlist'] and sc['use_cutoff']
    
    if use_nb is True:
        buffer_length = s['neighbourlist']['buffer_length']
        nbhood = smd.Neighbourhood(interactions, buffer_length, box ) 
    
    
    # check numerical values
    smd.check_values(box, sc['cutoff_distance'], sc['smoothing_length'], s['neighbourlist']['buffer_length'],
                     sc['use_cutoff'] , sc['cutoff_smoothing'] != None, use_nb
                    )
    
        
    # print some information on the run
    print('L = {:.3f}'.format(L))
    #print('[print other settings here]')
    print()

    #
    # initialize everything
    #

    # simulation time starts at 0
    t = 0.0

    # initial conditions - positions on a regular grid
    N0 = int(round(N**(1.0/3)))
    assert N0**3 == N
    x = smd.regular_grid(N0, L)

    # prepare masses and their broadcasting version
    m = np.array([masses[name] for name in names], dtype=float)
    m_DOF = m[:, np.newaxis]

    # thermostat settings
    use_thermo = s['thermostat']['use_thermostat']
    
    if use_thermo:
        T = s['thermostat']['T']
        gamma = 1.0 / s['thermostat']['tau']
        A = np.exp(-gamma * dt)
        B = np.sqrt(k_B * T * (1-A**2) / m_DOF)

    # initial conditions - thermal velocities
    v = np.random.normal(0, np.sqrt(k_B * s['T_init'] / np.repeat(m_DOF, 3, axis=1)))

    # update interactions for initial positions
    E_pot, dU_dx = interactions(x, box)
    a = - dU_dx / m_DOF
    
    timerformat = '   >>> step {:6d} | {:.2f}'
    
    # prepare output
    output = smd.Output(settings, x, v)
    
    if use_nb is True:
        #disp_monitor = smd.io.DisplacementMonitor(settings, x, nbhood)
        disp_monitor = smd.DisplacementMonitor(settings, x, nbhood)

    # main loop with a context manager
    with ExitStack() as stack:

        # context manage output
        stack.enter_context(output)
        
        # displacement monitor for managing neighbourlist updates
        if use_nb is True:
            stack.enter_context(disp_monitor)

        # loop over all propagation steps
        for i in smd.steps(n_steps, stride_progress):

            # write output, possibly
            output.run(i, t, E_pot)

            # propagator
            v += a * dth
            x += v * dth
            
            if use_thermo:
                v *= A
                v += B * np.random.normal(0.0, 1.0, size=v.shape)
                
            x += v * dth

            E_pot, dU_dx = interactions(x, box)         
            
            a[:] = - dU_dx / m_DOF
            v += a * dth

            # wrap atoms in the box
            if do_wrap:
                x -= box * np.floor(x / box)
                
            # checking if any particle has moved more than half the buffer zone length 
            if use_nb is True:
                disp_monitor.evaluate(i, t)
            
            # update time
            t += dt
            
        i += 1

        # write output for last frame
        output.run(i, t, E_pot)

    # store final positions, possibly
    output.store_final_positions()

    print('SimpleMD finished.')