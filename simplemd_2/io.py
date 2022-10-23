import numpy as np
import numba as nb

__all__ = ['write_xyz_frame', 'read_xyz_frame', 'read_xyz', 'Output', 'DisplacementMonitor', 'read_energy_file']


def write_xyz_frame(f_out, x, names, comment=''):
    """Write one XYZ frame to an open file."""

    N = x.shape[0]

    # number of atoms
    f_out.write('{:d}\n'.format(N))

    # comment line
    f_out.write(comment + '\n')

    # one line per atom
    for i in range(N):
        data = names[i], x[i, 0], x[i, 1], x[i, 2]
        f_out.write('{:s} {:12.6f} {:12.6f} {:12.6f}\n'.format(*data))

        

def read_xyz_frame(f_in):
    """Read one frame from an open XYZ file.

    Returns `None` if there are no more frames.
    """

    line = f_in.readline()
    if line == '':
        return None
    N = int(line)
    comment = f_in.readline()[:-1]
    names = []
    data = []
    for i in range(N):
        items = f_in.readline().split()
        names.append(items[0])
        data.append([float(item) for item in items[1:]])

    return comment, names, np.array(data)



def read_xyz(fn_in):
    """Read all frames from an XYZ file."""

    frames = []

    with open(fn_in) as f_in:
        while True:
            frame = read_xyz_frame(f_in)
            if not frame:
                break
            frames.append(frame)

    return frames



class Output:
    """Prepare and perform output from SimpleMD.

    A rough sketch of how one might handle output in a somewhat robust way."""

    def __init__(self, settings, x, v):
        
        labels = ('energy', 'positions', 'velocities')
        self.fns = {label: settings['output'][label] for label in labels}

        if 'positions_final' in settings['output']:
            self.fn_positions_final = settings['output']['positions_final']
        else:
            self.fn_positions_final = None

        self.stride = settings['output']['stride']
        names = settings['names']
        self.names = names
        self.N = len(names)
        self.k_B = settings['k_B']
        self.x = x
        self.v = v

        masses = settings['masses']
        m = np.array([masses[name] for name in names], dtype=float)
        self.m_DOF = m[:, np.newaxis]

        self.fs_out = None

        # format string for one energy file line
        self.fmt_energy = '{:6d} {:10.3f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n'

    def __enter__(self):
        
        # filenames:
        fns = self.fns

        # open all files
        fs_out = {}
        for label, fn in fns.items():
            if fn is None:
                fs_out[label] = None
            else:
                fs_out[label] = open(fn, 'w')
        self.fs_out = fs_out

    def __exit__(self, exception_type, exception_value, traceback):

        # close all files
        for label, f_out in self.fs_out.items():
            if f_out is not None:
                f_out.close()
        self.fs_out = None

    def run(self, i, t, E_pot):

        if i % self.stride != 0:
            return

        assert self.fs_out is not None, 'Need to enter context before running output.'

        fs_out = self.fs_out

        # calculate kinetic energy and temperature
        E_kin = (self.m_DOF * self.v**2).sum() / 2
        T_kin = 2 * E_kin / (self.k_B * 3 * self.N)

        f_out_energy = fs_out['energy']
        if f_out_energy is not None:
            f_out_energy.write(self.fmt_energy.format(i, t, E_kin, E_pot, E_kin+E_pot, T_kin))

        # comment line for XYZ files
        comment = 'step {:d} time {:.3f}'.format(i, t)
        #comment = 'step {:d}'.format(i)

        # write XYZ files - positions and velocities
        names = self.names
        f_out_positions = fs_out['positions']
        if f_out_positions is not None:
            write_xyz_frame(f_out_positions, self.x, names, comment=comment)
        f_out_velocities = fs_out['velocities']
        if f_out_velocities is not None:
            write_xyz_frame(f_out_velocities, self.v, names, comment=comment)

    def store_final_positions(self):
        if self.fn_positions_final is not None:
            with open(self.fn_positions_final, 'w') as f_out:
                write_xyz_frame(f_out, self.x, names=self.names)
                
                       

class DisplacementMonitor:
    """ Class for monitoring the movements of particles for the sake of creating neighbourlist:

    structure is based on the Output class 

    Init args:
    >>> settings ... dict, settings of the MDS
    >>> x_init   ... numpy.ndarray, initial coordinates of the particles
    >>> nbhood   ... object, instance of the Neighbourhood class (see cutoff.py file)

    """
    def __init__(self, settings, x_init, nbhood):
        from .cutoff import Neighbourhood
        
        assert isinstance(nbhood, Neighbourhood)
        self.nbhood = nbhood
        self.x_saved = x_init.copy()
        self.x = x_init
        self.filename = settings['neighbourlist']['filename']
        self.names = settings['names']
        #self.limit_disp = nbhood.buffer / 2.0
        self.limit_disp = nbhood.buffer / (2.0 * np.sqrt(3))
        assert self.filename is not None
    
        
    def __enter__(self):
        self.file = open(self.filename, 'w')
        comment= 'step {:d}, time {:.1f}  ...  initial frame'.format(1, 0.0)
        write_xyz_frame(self.file, self.x, self.names, comment=comment)
        self.nbhood.update_nblist(self.x)

        
    def __exit__(self, exception_type, exception_value, traceback):
        self.file.close()
        self.file = None
        
        
    def print_out(self, i, t, indexes=[]):
        comment= 'step {:d}, time {:.1f}'.format(i, t) + " "*7 + str(indexes)
        write_xyz_frame(self.file, self.x, self.names, comment=comment)
        
    # method for evaluating the displacements of particles at step i 
    ## and comparing them with the reference array from the last neighbourlist update
    ## if the treshold of half the buffer zone length is exceeded -> the neighbourlist update is triggered
    def evaluate(self, i, t):

        def check(ar, dist):
            out1 = np.array(np.where(np.abs(ar) >= dist))
            out2 = np.any(out1)
            return out1.flatten().tolist(), out2
        
        disp = self.x - self.x_saved
        
        if self.nbhood.box is not None:
            disp -= self.nbhood.box * np.round(disp / self.nbhood.box)
        #disp2 = (disp**2).sum(axis=1)
        #indexes, answ = check(disp2, self.limit_disp**2)
        indexes, answ = check(disp, self.limit_disp)
        
        if answ:
            self.nbhood.update_nblist(self.x)
            self.x_saved = self.x.copy()
            self.print_out(i, t, indexes)
            
            

def read_energy_file(file, labels):
    """ Function for reading the energy.txt file, or in fact any another formatted numerical textfile

    Args:
    >>> file  ... str, filepath
    >>> labels ... list of str, list of 'column headers' in the corresponding order as in the file

    """
    import re
    reg_num = re.compile("-?\d+[\.\d]*")
    data = np.array([])
    
    with open(file, 'r') as f:
        for line in f:
            found = reg_num.findall(line)
            numerics = np.array([np.float(x) for x in found])
            if data.size == 0.0:
                data = numerics
            else:
                data = np.vstack((data, numerics))
    
    assert len(labels) == data.shape[1], "List of labels should be of the same length as the number of imported num. containing columns!"
    out = dict()
    
    for label, column in zip(labels, data.T):
        out[label] = column
        
    return out




def save_settings_to_json(settings, filename):
    """ Function for saving the dictionary of settings into JSON file

    Args:
    >>> settings ... dict, dictionary of settings
    >>> filename ... str

    """
    import json
    assert type(settings) == dict
    assert type(filename) == str
    
    functions = [key for key,val in settings.items() if callable(val)]
    for i in functions:
        settings[i] = settings[i].__name__
        
    try:
        with open(filename, 'w') as file:
            json.dump(settings, file, indent=4)
    except:
        raise Exception("ERROR in saving settings to JSON.")
    
    

def load_settings_from_json(filepath):
    """ Function for importing dictionary saved as a JSON file

    Arg:
    >> filepath ... str, path to the JSON file

    """
    import json
    assert filepath.endswith(".json")
    
    with open(filepath, 'r') as file:
        try:
            dict_data = json.load(file)
        except:
            raise Exception("ERROR in importing JSON file. Please check its format.")
        
    return dict_data

        
        


        