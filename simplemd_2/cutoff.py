import numpy as np
import numba as nb
from functools import reduce
import time

from numba import float64, int64
import numba as nb


class Cutoff:
    """ Class for managing cutoffs being applied on potentials:

    Init args:
    >>> cut_dist      ... float, the distance of cutoff
    >>> smooth_type   ... str, one of supported smoothing functions, default is None
    >>> smooth_length ... float, the length of smoothing the potential before it drops to zero at the cutoof distance
    >>> shift         ... bool, if True -> the cut potential will be shifted to zero at the cutoff distance
    >>> kwargs        ... passed to individual smoothing functions, to set up various parameters

    """  
    def __init__(self, cut_dist, smooth_length=0.0, smooth_type=None, shift=False, **kwargs):
        self.cut_dist = cut_dist
        self.shift = shift
        self.type_name = smooth_type
                
        # sanity check - distances
        assert smooth_length <= cut_dist
        
        if smooth_type == "shifted":
            self.shift = True
        
        if shift is True:
            self.type_name = "shifted"

        if smooth_type is None:
            self.smooth = False    # smoothing in this sense involves any postprocessing to cutted potential, i.e. shifting as well 
        else:
            self.smooth = True
        
        # set default value for smooth_length if only smooth_type was given
        if smooth_type is not None and smooth_length == 0.0:
            smooth_length = cut_dist / 6.0
        
        # setting checked smoothing lenght
        self.smooth_length = smooth_length
        
        # keword arguments for smoothing functions:
        if "n" in kwargs.keys():
            self.n = kwargs["n"]
        else:
            self.n = 1
        
        if "c" in kwargs.keys():
            self.c = np.abs(kwargs["c"])
        else:
            self.c = 1
            
            
        # predefined smoothing functions:
        def linear(x, r1, r2):
            fx = 1.0 - (x-r1) / (r2-r1)
            dfx = 1.0 - 1.0 / (r2-r1)
            return fx, dfx
        
        def goniometric(x, r1, r2):
            fx = (1.0 + np.cos(np.pi * (x-r1) / (r2-r1)) ) / 2.0
            dfx = - np.sin(np.pi * (x-r1) / (r2-r1)) * (np.pi / (r2-r1)) / 2.0
            return fx, dfx
        
        def polynomial(x, r1, r2, n=1):
            xr1 = x-r1
            xr2 = x-r2
            qxr1 = xr1**(2*n)
            qxr2 = xr2**(2*n)
            fx = 1.0 - qxr1 / (qxr1 + qxr2)
            dfx = -2*n*qxr1/(xr1*(qxr1 + qxr2)) - qxr1*(-2*n*qxr2/xr2 - 2*n*qxr1/xr1)/(qxr1 + qxr2)**2
            return fx, dfx

        def exponential(x, r1, r2, c=1, n=1):
            xr2 = x-r2
            xr1 = x-r1
            xr22n = xr2**(-2*n)
            xr12n = xr1**(-2*n)
            exr1xr2 = np.exp(-c * ( xr12n - xr22n ))
            fx = 1.0 / (1.0 + exr1xr2 )
            dfx = -2.0*c*n *(xr12n / xr1 - xr22n / xr2) * exr1xr2 * fx**2
            return fx, dfx 
        
        # list of the predefined smoothing functions
        types = [linear, goniometric, polynomial, exponential]
        
        # function for matching the given string argument with the smoothing function
        def search_types(name, types):
            for i in types:
                if name == i.__name__:
                    return i 
        
        # setting the smoothing function into an attribute
        self.smoothing_f = search_types(smooth_type, types)
        
        
    def __call__(self, x):
        x = abs(x)
        r1 = self.cut_dist - self.smooth_length
        r2 = self.cut_dist
        
        if x <= r1:
            return 1.0, 1.0
        elif x < r2:
            return self.smoothing_f(x, r1, r2)
        else:
            return 0.0, 0.0
        
        
    # method for plotting the cutoff masking-function quickly
    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np
        
        r1 = self.cut_dist - self.smooth_length
        r2 = self.cut_dist
        
        data_x = np.linspace(0.0, 1.2*r2, 100)
        data_y = list(map(self, data_x))
        plt.plot(data_x, data_y, color="red", label="Cutoff function")
        
        settings = dict(linestyle="dashed", color="gray", linewidth=0.6)
        
        plt.plot([r1]*2,[0,1], **settings, label="r1")
        plt.plot([r2]*2,[0,1], **settings, label="r2")
        plt.legend()
        


class Potential:
    """ Class for managing potentials:

    Init args:
    >>> function ... str, the name of desired predefined potential, 
                 ... alternatively ... function(x), custom potential function of single positional variable 
                                       - experimental, not fully tested

    Optional args:
    >>> bunch of preset parameters for the potentials ...

    """   
    def __init__(self, function, eps=1.0, sigma=1.0, n=1, alpha=0.1):
        self.sigma = sigma
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.name = None
        
        def search_potential(name, predef):
            for i in predef:
                if i.__name__ == name:
                    return i
        
        # predefined potentials:
        def Lennard_Jones(x, eps=self.eps, sigma=self.sigma):
            xinv = sigma / x
            xinv6 = xinv**6
            Ux = 4.0 * eps * (xinv6**2 - xinv6)
            dUx = - 24.0 * eps * (2.0 * xinv6**2 - xinv6) * xinv 
            return Ux, dUx
        
        # see Toxvaerd, Heilman & Dyre (2012): https://doi.org/10.1063/1.4726728
        def Toxvaerd_2012(x, r2, eps=self.eps, sigma=self.sigma, n=self.n, alpha=self.alpha):
            xinv = sigma / x
            xinv6 = xinv**6
            xinv12 = xinv6**2

            r2inv = sigma / r2
            r2inv6 = r2inv**6

            Ux_LJ = 4.0 * eps * (xinv6**2 - xinv6)
            Ux_LJr2 = 4.0 * eps * (r2inv6**2 - r2inv6)

            Ux_A = Ux_LJ - Ux_LJr2

            r2x2n = (r2 - x)**(2*n)
            a2n = alpha**(2*n)

            Ux_B = r2x2n / (r2x2n + a2n)
            Ux = Ux_A * Ux_B
            
            # derivative:
            dUx_A = - 24.0 * eps * r2x2n * (2.0 * xinv12 - xinv6) / (a2n + r2x2n) / x
            dUx_B = - 2*n * r2x2n * Ux_A / ((a2n + r2x2n) * (r2-x))
            dUx_C = 2*n * r2x2n**2 * Ux_A / ((a2n + r2x2n)**2 * (r2-x))

            dUx = dUx_A + dUx_B + dUx_C
            return Ux, dUx
            
        
        # list of predefined potentials:
        predefined = [Lennard_Jones, Toxvaerd_2012]
                      
        if callable(function):
            self.potential = function   
        elif isinstance(function, str):
            self.potential = search_potential(function, predefined)
            self.name = function
        else:
            pass
                 
    # calling the Potential object calls the potential function instead  
    def __call__(self, x):
        return self.potential(x)

    

class Interactions:
    """ Class for manging the interaction, involving both the cutoff and potential eventually:

    Init args:
    >>> potential ... object, instance of Potential class
    >>> cutoff    ... object, instance of Cutoff class, if none is given, no cutoff is applied

    """
    def __init__(self, potential, cutoff=None):
        if cutoff is not None and cutoff.smooth is not None:
            self.cut = True
            assert isinstance(cutoff, Cutoff)
        else:
            self.cut = False
            
        assert isinstance(potential, Potential)
        
        # ensuring compatibility of the Toxvaerds potential with other functions ... changing it to a single variable function, fixing r2
        if potential.name == "Toxvaerd_2012":
            assert cutoff is not None, "Cutoff object has to be initialized to enable usage of >>"+potential.name+"<< potential!"
            r2 = cutoff.cut_dist
            
            def helper(func, num):
                f = lambda x: func(x, num)
                return f
            
            potential.potential = helper(potential.potential, float(r2))
            cutoff.smooth = True
            cutoff.smooth_type = potential.name
        
        self.cutoff = cutoff
        self.potential = potential
        self.vectorized = np.vectorize(self.get_interaction)
        
        if self.potential.name == "Toxvaerd_2012": 
            self.r2 = self.cutoff.cut_dist
        if self.cutoff.shift is True:
            self.r2 = self.cutoff.cut_dist
        if self.cutoff.smooth is True:
            self.r1 = self.cutoff.cut_dist - self.cutoff.smooth_length
            self.r2 = self.cutoff.cut_dist
            
            if self.cutoff.type_name == "exponential":
                self.closest_to_boundary = (self.cutoff.c / 300.0)**(1.0 / (2 * self.cutoff.n))
                self.r2_min = self.r2 - self.closest_to_boundary
          
        
    def get_interaction(self, r):
        if isinstance(r, np.ndarray):
            return self.vectorized(r)
        else:
            return self.get_interaction_full(r)
        
    # method for evaluating the potential and its derivative in point r
    ## r can be either a numerical value or a numpy array (method will be vectorized then)
    ## a tuple of U(r) and dU(r) values is returned
    def get_interaction_full(self, r):
            
        # prevents zero division error
        if r == 0.0:
            return 0.0, 0.0
        
        # special case of somewhat special Toxvaerds potential, smoothened till 4. derivative by definition
        # based on shifted Lennard-Jones
        if self.potential.name == "Toxvaerd_2012":            
            r2 = self.r2
            
            if r < r2:        
                U, dU = self.potential(r)
                return U, dU
            else:
                return 0.0, 0.0
        
        # case without any cutoff
        elif self.cut is False:
            U, dU = self.potential(r)
            return U, dU
        
        # case with a cutoff, then shifted to zero
        elif self.cutoff.shift is True:
            r2 = self.r2
            
            if r < r2:
                U, dU = self.potential(r)
                shift_const, _foo = self.potential(r2)
                return U - shift_const, dU
            else:
                return 0.0, 0.0
        
        # case with a cutoff, smoothened
        elif self.cutoff.smooth is True:
            r1 = self.r1
            r2 = self.r2
                    
            # managing the overflow error in np.exp near the cutoff distance r2:
            ## the same error near the beginning of smoothing zone was hopefully managed conveniently
            ## by algebraic manipulation of the exponential smoothing function 
            if self.cutoff.type_name == "exponential":
                closest_to_boundary = self.closest_to_boundary
                r2_min = self.r2_min
                
                if r >= r2_min and r < r2:
                    r = r2
                    
            if r <= r1:
                U, dU = self.potential(r)
                return U, dU
            
            elif r < r2:
                U, dU = self.potential(r)
                smf, dsmf = self.cutoff(r)
                U_r2 = U * smf
                dU_r2 = U * dsmf + dU * smf
                return U_r2, dU_r2
            
            else:
                return 0.0, 0.0
        
        # case with a simple cutoff
        else:
            if r < self.cutoff.cut_dist:
                U, dU = self.potential(r)
                return U, dU
            else:
                return 0.0, 0.0
            
            
    ### ### ### ### ### 
    # method for printing the analytical form of potential derivative evaluated in point r
    ## if analytic is True -> the whole equation is printed, before as well as after the differentiation
    ## if analytic is set to False -> single numerical value of dU(r) is returned
    ### derived from get_interaction() method above
#     def get_derivative(self, r, analytic=True):
#         x = sympy.symbols('x')
        
#         # enabling np.arrays as argument
#         if isinstance(r, np.ndarray):
#             vectorized = np.vectorize(self.get_interaction)
#             return vectorized(r)
            
#         # prevents zero division error
#         elif r == 0.0:
#             return 0.0, 0.0
        
#         def init_both_diffs(U, x):
#             from IPython.display import display
#             dU = sympy.Derivative(U(x), x)
#             dU_dx =sympy.diff(U(x), x)
#             equals = sympy.symbols('=')
#             return display(dU, equals, dU_dx)    

#         # special case of somewhat special Toxvaerds potential, smoothened till 4. derivation by definition
#         # based on shifted Lennard-Jones
#         if self.potential.name == "Toxvaerd_2012":           
#             r2 = self.cutoff.cut_dist
#             #U = lambda x: self.potential.potential(x, r2)
#             U = self.potential.potential
            
#             if r < r2:
#                 if not hasattr(self, "dU_dx"):
#                     self.dU_dx = differentiate(x, U)
#                 return U(r), self.dU_dx(r)
#             else:
#                 return 0.0, 0.0
        
#         # case without any cutoff
#         elif self.cut is False:
#             U = self.potential.potential
#             if not hasattr(self, "dU_dx"):
#                 self.dU_dx = differentiate(x, U)
#             if analytic is True:
#                 return init_both_diffs(U, x)
#             else:
#                 return self.dU_dx(r)
            
#         # case with a cutoff, then shifted to zero
#         elif self.cutoff.shift is True:
#             r1 = self.cutoff.cut_dist
#             U = self.potential.potential
            
#             if r < r1:
#                 shift_const = self.potential.potential(r1)
#                 if not hasattr(self, "dU_dx"):
#                     self.dU_dx = differentiate(x, U)
#                 if analytic is True:
#                     return init_both_diffs(U, x)
#                 else:
#                     return self.dU_dx(r)
#             else:
#                 return 0.0, 0.0
        
#         # case with a cutoff, smoothened
#         elif self.cutoff.smooth is True:
#             r1 = self.cutoff.cut_dist - self.cutoff.smooth_length
#             r2 = self.cutoff.cut_dist
            
#             if r <= r1:
#                 U = self.potential.potential
#                 if not hasattr(self, "dU_dx_r1"):
#                     self.dU_dx_r1 = differentiate(x, U)
                
#                 if analytic is True:
#                     return init_both_diffs(U, x)
#                 else:
#                     return self.dU_dx_r1(r)
#             elif r < r2:
#                 if not hasattr(self, "U_r2"):
#                     self.U_r2 = lambda x: self.potential.potential(x) * self.cutoff.smoothing_f(x, r1, r2)
#                 if not hasattr(self, "dU_dx_r2"):
#                     self.dU_dx_r2 = differentiate(x, self.U_r2)

#                 if analytic is True:
#                     return init_both_diffs(self.U_r2, x)
#                 else:
#                     return self.dU_dx_r2(r)
#             else:
#                 return 0.0, 0.0
        
#         # case with a simple dire cutoff
#         else:
#             if r < self.cutoff.cut_dist:
#                 U = self.potential.potential
#                 if not hasattr(self, "dU_dx"):
#                     self.dU_dx = differentiate(x, U)
#                 if analytic is True:
#                     return init_both_diffs(U, x)
#                 else:
#                     return self.dU_dx(r)
#             else:
#                 return 0.0, 0.0
    ### ### ### ### ###
    
            
    # method for quick plotting of the resulting potential and its derivative 
    def plot(self, ylim=(-2, 0.1), xlim=(0.85, 1.85), figsize=(8,4), dpi=300):
        import matplotlib.pyplot as plt
        import numpy as np
        from .utils import bracketise
        
        data_x = np.linspace(0.0, max(xlim), 500)
        data_x = data_x[1:]
        data_U, data_dU = np.array(list(map(self.get_interaction, data_x))).T
        data_orig_U, data_orig_dU = np.array(list(map(self.potential.potential, data_x))).T
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # case without any cutoff
        if self.cut is False:
            plt.plot(data_x, data_U, color="red", label="Potential")
            plt.plot(data_x, data_dU, color="green", label="Derivative of potential")
            
        else:
            r2 = self.cutoff.cut_dist
            pot_label = "Potential"+bracketise(self.cutoff.type_name)
            data_x_orig = data_x.copy()
            
            if self.cutoff.smooth is False or self.cutoff is None: # and self.cutoff.shift is False and self.potential.name != "Toxvaerd_2012":
                from simplemd_2.utils import add_points    
                data_x, data_U, data_dU = add_points(r2, self.potential, data_x, data_U, data_dU)
                
            settings = dict(color="gray", linewidth=0.6)  
                
            if self.potential.name == "Toxvaerd_2012":
                plt.plot(data_x, data_U, color="red", label=pot_label)
                plt.plot(data_x, data_dU, color="green", label="Derivative of potential")
                data_orig, data_orig_dx = np.array(list(map(Potential("Lennard_Jones").potential, data_x))).T
                plt.plot(data_x_orig, data_orig, color="red", linestyle="dashed", linewidth=0.7, label="Original potential (LJ)")
                plt.plot(data_x_orig, data_orig_dx, color="green", linestyle="dashed", linewidth=0.7, label="Derivative of original potential (LJ)")
                
            else:          
                # case with a cutoff
                plt.plot(data_x, data_U, color="red", label=pot_label)
                plt.plot(data_x, data_dU, color="green", label="Derivative of potential")
                plt.plot(data_x_orig, data_orig_U, color="red", linestyle="dashed", linewidth=0.7, label="Original potential")
                plt.plot(data_x_orig, data_orig_dU, color="green", linestyle="dashed", linewidth=0.7, label="Derivative of original potential")

                if self.cutoff.smooth is True and self.cutoff.shift is False :
                    r1 = self.cutoff.cut_dist - self.cutoff.smooth_length
                    plt.plot([r1]*2,ylim, **settings, linestyle="dotted", label="r1")

            plt.plot([r2]*2,ylim, **settings, linestyle="dashed", label="r2")
        
        plt.plot(xlim, [0]*2, color="gray", linewidth=0.2)
        plt.legend()
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.show()
        
        
    # making the class callable - for the sake of back-compatibility with MD code from lectures
    ## only an array can be given as an argument
    ## a total potential energy value and a numpy array of gradients of potential energy for each particle are returned 
    def __call__(self, x, box=None):
        assert isinstance(x, np.ndarray)
        
        if hasattr(self, "neighbourhood") and self.neighbourhood.nblist is not None:
            nblist = self.neighbourhood.nblist  
            
            # if there are no particle-pairs on neighbourlist, i.e. the ignore-cutoff is shorter than any intermolec. dist. -> return zeros
            if len(nblist) == 0:
                print("WARNING:  Zero-pair neigbourlist encountered (i.e. no particles were close enough to interact). Zero-lists are being returned.\n")
                return 0, np.zeros_like(x)
            
            evaluator = lambda pair: eval_pair(pair, x, box)
            dxd = np.array(list(map(evaluator, nblist)), dtype=object)
            
            U, dU = self.get_interaction(dxd[:,1])
            U_total = U.sum()
            
            stacked = np.hstack((nblist, dxd, dU[:,np.newaxis]))
            
            def get_dU_dx(data, line):
                m, n, dx, d, dU = line

                if d!= 0:
                    val = get_line(dU, d, dx)
                else:
                    val = np.zeros_like(dx)

                return func1(data, (m,n), val)
            
            dU_dx = np.zeros_like(x)
            dU_fin = np.array(list(reduce(get_dU_dx, stacked, dU_dx ))) 
                 
        else:
            dx = x[:,np.newaxis,:] - x[np.newaxis,:,:]
            if box is not None:
                dx -= box * np.round(dx / box)
            d = np.sqrt((dx**2).sum(axis=2))
            #print("  > d.shape: ", dx.shape)
            U, dU = self.get_interaction(d)
            
            U_total = U.sum() * 0.5
            dU_pairs = np.divide(dU, d, out=np.zeros_like(dU), where= d!=0.0)[:,:, np.newaxis] * dx
            dU_fin = dU_pairs.sum(axis=1)
        
        return U_total, dU_fin

    



@nb.jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)

@nb.jit(nopython=True)
def eval_pair(pair, positions, box=None):
    m, n = pair
    #print(pair)
    dx = positions[m,:] - positions[n,:]
    if box is not None:
        dx -= box * rnd1(dx / box, 0, np.empty_like(dx))
    d = np.sqrt((dx**2).sum())
    return dx, d

@nb.jit(nopython=True)
def func1(mat, index, line):
    mat_out = mat
    i, j = index
    mat_out[i] += line
    mat_out[j] -= line
    return mat_out

@nb.jit(nopython=True)
def get_line(dU, d, dx):
    return dU / d * dx


class Neighbourhood:
    """ Class for introduction of basic neighbourlist:

    Init args:
    >>> interactions  ... object, instance of the Interactions class
    >>> buffer_length ... float, length of the buffer zone behind the cutoff distance
    >>> box           ... list of floats, dimensions of the pbc cubic box

    """  
    def __init__(self, interactions, buffer_length, box=None):
        assert isinstance(interactions, Interactions)
        
        if interactions.cut is False:
            print("WARNING: The given interaction object doesn't involve cutoff. Neighbourlist is not supported. :-(")
        else:
            self.buffer = buffer_length
            self.cut = interactions.cutoff.cut_dist
            print("Cutting distance: ",self.cut)
            
            self.nb_dist = self.cut + self.buffer
            print("Neighbourlist outer distance: ", self.nb_dist)
            
            self.box = box
            self.nblist = None
            assert self.nb_dist < min(self.box)
            
            print("NB_dist: ", self.nb_dist, " BOX: ", self.box)
            
            self.interactions = interactions
            interactions.neighbourhood = self
            
            
    # method for updating the stored neighbourlist, it's being triggered by the DisplacementMonitor object (see io.py file)
    def update_nblist(self, x):
        dx = x[:,np.newaxis,:] - x[np.newaxis,:,:]
        if self.box is not None:
            dx -= self.box * np.round(dx / self.box)
        d2 = (dx**2).sum(axis=2)
        d2 = np.triu(d2)
        indexes = np.where((d2 <= self.nb_dist**2) & (d2 != 0.0))        
        pairs = np.stack(indexes, axis=1)
        self.nblist = pairs

        