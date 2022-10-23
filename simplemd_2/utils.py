import time
import numpy as np

__all__ = ['regular_grid', 'steps', 'bracketise', 'check_values']


def regular_grid(n, L):
    """Create atomic positions on a regular cubic grid.

    Args:
        n: number of atoms in each direction
        L: periodicity length
    """
    d = (L / n)
    x = d * np.arange(n, dtype=float) + 0.5 * d
    y = x.copy()
    z = x.copy()

    X, Y, Z = np.meshgrid(x, y, z)
    positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

    return positions


def steps(n_steps, stride=None):
    """Step iterator and progress printer for SimpleMD runs."""
    
    if stride is None:
        stride = n_steps + 1

    # formatting strings
    fmt_progress = '\rstep {:6d} | {:5.1f} % | {:.2f} ms/step'
    fmt_done = fmt_progress + '\n'

    # stash start time
    t00 = time.time()
    t0 = t00

    for i_step in range(n_steps):

        # yield the item itself
        yield i_step

        # print progress and time per frame in milliseconds
        if i_step % stride == 0:
            t1 = time.time()
            pct = 100.0 * i_step / n_steps
            time_step = 1000 * (t1 - t0) / stride
            print(fmt_progress.format(i_step, pct, time_step), end='', flush=True)
            t0 = t1

    i_step += 1

    # summarize progress when done
    t_total = time.time() - t00
    print(fmt_done.format(i_step, 100.0 * i_step / n_steps, 1000*t_total/n_steps))
    

# put string into brackets 
def bracketise(label):
    if label is None:
        label = "none"

    if len(label) > 0:
        return " ("+label+")"
        
# sanity checker:
def check_values(box, val_cut, val_smooth, val_nb, do_cut, do_smooth, do_nb):
    if do_cut:
        assert val_cut < min(box)/2, "Cutoff distance exceeded the half box length!"
    if do_nb and do_cut:
        assert val_nb + val_cut < min(box)/2, "Half box length exceeded by neighbourlist buffer zone!"
     
    
# returns interaction function from the corresponding name 
def format_interaction_function(func_name):
    if callable(func_name):
        return func_name
    
    elif type(func_name) == str:
        import simplemd_2.interactions as smdi
        func = getattr(smdi, func_name)
        return func
    
    
def format_to_tex(text, retain_underscore=False):
    format_latex = r'$\mathrm{{ {} }}$'
    
    if retain_underscore is True:
        lst = str(text).split("_", 1)
        lst2 = lst[0] + "_{" + lst[1] + "}"
        return format_latex.format(lst2)
        
    else:
        #lst = text.replace("_",",\:").replace("-", "=")
        lst = str(text).replace(" ", "\:")
        lst2 = lst.replace("__",",\:").replace("_","\_").replace("--", "=").replace("-", "\,")
        return format_latex.format(lst2)
    
    
def get_colorlist(number, cmap1='tab20b', cmap2='tab20c', start=1):
    from matplotlib.cm import get_cmap
    
    colmap1 = get_cmap(cmap1)
    capacity = 20
    
    if number <= capacity / 4:
        colorlist = [colmap1(i) for i in range(start, capacity+1, 4)]
    elif number <= capacity / 2:
        colorlist = [colmap1(i) for i in range(start, capacity+1, 2)]
    elif number <= capacity:
        colorlist = [colmap1(i) for i in range(0, capacity, 1)]
    elif number <= 2*capacity:
        colmap2 = get_cmap('tab20c')
        colorlist = [colmap1(i) for i in range(0, capacity)]
        colorlist.append([colmap2(j) for j in range(0, capacity)])
    else:
        raise Exception("There is too many labels and not enough available colours in colormap. The maximum capacity is "+str(capacity*2))
    
    return colorlist


def boil_down_to_minmaxmean(dataframe):
    import numpy as np
    smax = dataframe.iloc[1:,:].max(0)
    smin = dataframe.iloc[1:,:].min(0)
    smean = dataframe.iloc[1:,:].mean(0)
    return np.array(smax), np.array(smin), np.array(smean)


def file_saver(path):
    import os
    number = 0
    
    while os.path.isfile(path) is True:
        pth, ext = path.rsplit('.', 1)
        
        if number == 0:
            path = "_{:d}.".format(number).join([pth, ext])
        else:
            path = "_{:d}.".format(number).join([pth[:-(len(str(number-1))+1)], ext]) 
        number += 1

    return path



def write_csv_with_comments(path, data, comments):
    import os
    
    commentf = '# {key} {value}{end}'
        
    with open(path, 'w') as file:
        for key, val in comments.items():
            file.write(commentf.format(key=key, value=val, end=os.linesep))

        data.to_csv(file)
    
    
def read_csv_comments(file):
    comments = dict()
    convert = {'box':(lambda x: [float(y) for y in x.replace('\n', '').strip('][').split(', ')]), 
               'repeats':int,
               'shape':(lambda x: [int(y) for y in x.replace('\n', '').strip(')(').split(', ')]),
               'N':int,
               'rho':float,
               'folder':str,
               'frame_key':str,
               'frame_start':int,
               'dr':float,
              }  
    
    def check_line(line, dictionary):
        if line.startswith('#'):
            splitline = line.split(" ", 2)
            key = splitline[1]
            value = splitline[2].replace('\n', '')
            dictionary.update({key: convert[key](value)})
        elif line.startswith('\n'):
            raise NameError('continue')
        else:
            raise AttributeError('break')
        
    if type(file) == str and file.endswith('.csv'):
        with open(file, 'r') as f:
            for line in f:
                try:
                    check_line(line, comments)
                except NameError:
                    continue
                except AttributeError:
                    break
                    
    else:
        for line in file:
            try:
                check_line(line, comments)
            except NameError:
                continue
            except AttributeError:
                break
                
    return comments


def format_to_pandas(data):
    import pandas as pd
    
    output = None
    for config in data:
        col_labels = list()
        col_values = list()
        
        for i,n in enumerate(config):
            if type(n) == tuple:
                a, b = n
                col_labels.append(a)
                col_values.append(b)
            else:
                data = config[i:]
                break
        indices = pd.MultiIndex.from_frame(pd.DataFrame([col_values]), names=col_labels)
        pd_config = pd.DataFrame(data=[data], index=indices)
        
        if output is None:
            output = pd_config
        else:
            output = output.append(pd_config)
    return output


def check_up_ipython():
    use_ipy_feat = True
    try:
        get_ipython().__class__.__name__
    except NameError:
        use_ipy_feat = False
        
    return use_ipy_feat


def add_points(r, function, data_x, data_y, data_dy):
    if r < np.max(data_x):
        index = np.min(np.where(data_x > r))
        a, b = function(r)
        data_x_new = np.insert(data_x, index, np.array([r]*2))
        data_y_new = np.insert(data_y, index, np.array([a, 0.0]))
        data_dy_new = np.insert(data_dy, index, np.array([b, 0.0]))
        
        return data_x_new, data_y_new, data_dy_new
    else:
        return data_x, data_y, data_dy

