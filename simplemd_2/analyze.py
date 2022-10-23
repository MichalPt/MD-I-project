from matplotlib import pyplot as plt
import numpy as np

def plot_energy(file, x_axis, *y_axis, labels=None, evolution=False):
    """ Plot energy related or in fact any other numerical variable from a text file.

    Args:
    >>> file      ... str, filepath
    >>> x_axis    ... str, header of 'column' of x-axis data, e.g. time, steps
    >>> y_axis    ... str, one or more headers of 'columns' of y-axis data
                  ... if two headers are given, the plot will adjust and show two independ y-axis, for the sake of intelligibility
    >>> evolution ... bool, if True -> the evolution of data from the initial value wil be shown


    Optional args:
    >>> labels    ... list of str, the list of column headers in corresponding order
                  ... if none is specified, the default list used in this project output file will be used

    """
    from .io import read_energy_file
    
    if labels is None:
        labels = ['step', 'time', 'E_kin', 'E_pot', 'E_tot', 'T_kin']
    
    data = read_energy_file(file, labels)
    
    plt.style.use({'figure.dpi': 200, 'legend.frameon': True, 'figure.figsize': (8,5), })
    
    with plt.style.context('seaborn'):
        if evolution is True:
            def subtract_first(dct, key):
                dct[key] = [x - dct[key][0] for x in dct[key]]
            any(subtract_first(data, key) for key in y_axis)

        else:
            pass

        if len(y_axis) == 2:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            lab1, lab2 = y_axis
            line1 = ax1.plot(data[x_axis], data[lab1], '.-', label=lab1)
            next(ax2._get_lines.prop_cycler)
            line2 = ax2.plot(data[x_axis], data[lab2], '.--', label=lab2)

            ax1.set_ylabel(lab1)
            ax2.set_ylabel(lab2)
            ax2.legend(line1+line2, y_axis, frameon=True)

            fig.tight_layout()

        else:
            for y in y_axis:
                plt.plot(data[x_axis], data[y], '.-', label=y)

            plt.xlabel(x_axis)
            plt.legend(frameon=True)

        plt.show()
        
        

def get_rdf(f_in, dr=0.05, skip_frames=10, box=None, rho=None, 
            step_function=True, plot=True, **kwargs):
    """ Plot the simple radial distribution function:

    Args:
    >>> f_in        ... .xyz filepath
    >>> dr          ... x-axis resolution
    >>> skip_frames ... int, number of xyz frames to be skipped, keep in mind that the id of frame does not correspond to id of step
    >>> box, rho    ... either density rho or cubic box (L,L,L) can be enterd as an argument, the other will be calculated by itself
                    ... if none of them are specified
                        -> both will be calculated dynamicaly from the number of particles and maximal position encountred
    >>> step_function ... bool, if True -> step-like function is plotted, if False -> somewhat smoothened default function is plotted
    >>> kwargs      ... passed to plt.plot()

    """
    from .io import read_xyz_frame
    
    def evaluate_frame(array, dr, axis, data):
        assert len(axis) == len(data)
            
        for i, r in enumerate(axis):
            V_shell = 4.0 / 3.0 * np.pi * ((r+dr)**3 - r**3)
            n = np.where((array <= r+dr) & (array > r), 1, 0).sum()
            
            data[i] += n / V_shell
    
    calc_box = False
    calc_rho = False
    
    if rho is None and box is None:
        calc_box = True
        calc_rho = True
    elif box is not None:
        calc_rho = True
    elif rho is not None:
        calc_box = True
    else:
        pass
        
    axis = np.array([])
    data = np.array([])
    N = 0
    frame_counter = 0

    with open(f_in) as file:
        while True:
            frame = read_xyz_frame(file)
            
            if not frame:
                break
            
            frame_counter += 1
            
            if frame_counter < skip_frames:
                continue
                
            x = frame[2]
            N = len(x)
            dx = x[:,np.newaxis,:] - x[np.newaxis,:,:]

            if calc_box is True:
                if rho is not None:
                    box_L = np.cbrt(N / rho)
                    box = ([box_L]*3)
                    calc_box = False
                    print("Box set up from rho:", box)
                elif box is None:
                    box = np.full(3, np.max(x))
                    print("Box set up maximal positions: ", box)
                elif np.max(x) > np.max(box):
                    box = np.full(3, np.max(x))
                    print("Box updated from maximum positions: ", box)
                    
            if calc_rho is True:
                rho = N / box[0]**3
                print("Rho set up from box: ", rho)
                if calc_box is False:
                    calc_rho = False
                    
            dx -= box * np.round(dx / box)
            d = np.sqrt((dx**2).sum(axis=2))
            
            if len(axis) == 0:
                axis = np.arange(0.0, box[0]/2.0, dr)
                data = np.zeros_like(axis)
            
            evaluate_frame(d, dr, axis, data)
     
    data /= (frame_counter - skip_frames +1) * N * rho
    
    if plot is True:
        plt.style.use({'figure.dpi': 300, 'legend.frameon': False, 'figure.figsize': (8,5)})

        with plt.style.context('seaborn'):
            if step_function is True:
                plt.step(axis, data, **kwargs)            
            else:
                plt.plot(axis, data, '.-', **kwargs)

            plt.xlabel("radius")
            plt.ylabel("RDF")
            plt.xlim(0, min(box)/2 )
            plt.show()
            
    else:
        return axis, data
    
