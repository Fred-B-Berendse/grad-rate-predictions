import numpy as np
from matplotlib import cm


def make_color_dict(labels, cmap, start=0, end=1):
    '''
    Makes a dictionary of colors for each key in keys
    '''
    colors = cmap(np.linspace(start, end, len(labels)))
    return dict(zip(labels, colors))


def get_colors(labels, color_dict):
    '''
    Extracts a list of colors from a color dictionary in the order of labels
    '''
    results = []
    [results.append(color_dict[l]) for l in labels]
    return np.array(results)


def targets_color_dict():
    labels = ['Asian', 'Black', 'Hispanic', 'Nat. Am.', 'Pac. Isl.', 'White',
              '2+ Races']
    # labels = np.array(['Graduation Rate: ' + l for l in labels]) 
    color_dict = make_color_dict(labels, cm.Accent)
    labels = ['Pell Grant', 'SSL', 'Non-Recipient']
    # labels = np.array(['Graduation Rate: ' + l for l in labels])
    color_dict.update(make_color_dict(labels, cm.brg))
    return color_dict
