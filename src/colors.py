import numpy as np
from plotting import make_color_dict
from matplotlib import cm


def targets_color_dict():
    labels = ['Asian', 'Black', 'Hispanic', 'Nat. Am.', 'Pac. Isl.', 'White',
              '2+ Races']
    labels = np.array(['Graduation Rate: ' + l for l in labels]) 
    color_dict = make_color_dict(labels, cm.Accent)
    labels = ['Pell Grant', 'SSL', 'Non-Recipient']
    labels = np.array(['Graduation Rate: ' + l for l in labels])
    color_dict.update(make_color_dict(labels, cm.brg))
    return color_dict
