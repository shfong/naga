import matplotlib.pyplot as plt
import numpy as np

def manhattan_plot(df):
    # -log_10(pvalue)
    df['minuslog10pvalue'] = -np.log10(df['TopSNP P-Value'])
    df['Chr'] = df['Chr'].astype('category')
    #df.chromosome = df.chromosome.cat.set_categories(['ch-%i' % i for i in range(12)], ordered=True)
    df = df.sort_values(['Chr','Gene Start'])

    # How to plot gene vs. -log10(pvalue) and colour it by chromosome?
    df['ind'] = range(len(df))
    df_grouped = df.groupby(('Chr'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red','green','blue', 'yellow']
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y='minuslog10pvalue',color=colors[num % len(colors)], ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, len(df)])
    ax.set_ylim([0, 10])
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('-Log10 p-value')
    plt.show()

def get_neighbors(graph, n, center):  
    """Get the neighbors of a networkx graph""" 
    
    nodes = set([center]) 
    for i in range(n): 
        for n in nodes: 
            nodes = nodes.union(set(graph.neighbors(n)))
        
    return nodes

def binarize(a, threshold=5e-6):
    """Binarize array based on threshold"""

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    binned = np.zeros(a.shape)
    binned[a < threshold] = 1

    return binned


def neg_log_val(a, floor=None, ceiling=10.0):
    """Negative log of an array

    Parameters
    ----------
    a : `numpy ndarray` or list
        Array to be transformed
    floor : float
        Threshold after transformation. Below which, the
        transformed array will be floored to 0.
    ceiling : float 
        Threshold capping the maximum value after transformation.
        Above which the transformed array will be capped to ceiling.
    """

    if not isinstance(a, np.ndarray):
        a = np.array(a)

    vals = -np.log(a)

    if floor is not None:
        vals[vals < floor] = 0
    
    if ceiling is not None:
        vals[vals > ceiling] = ceiling
        
    return vals