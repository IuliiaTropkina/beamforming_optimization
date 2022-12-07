import os
import pickle
from itertools import repeat, zip_longest
from typing import List, Union, Callable, Tuple, Iterable, Dict

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.ticker
import mpl_toolkits
import mpl_toolkits.mplot3d
import numpy as np
from matplotlib import cycler, ticker
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.pyplot import subplots, show
from scipy.interpolate import interp1d
from scipy.signal import get_window
from numpy import ma

a = [x for x in original_groups]                                  # The original group numbers (unsorted)
idx = sorted(range(len(a)), key=lambda k: a[k])                   # Get indices of sorted group numbers
current_group = original_groups[idx[0]]                           # Set the current group to the be the first sorted group number
temp_patches = []                                                 # Create a temporary patch list

for i in idx:                                                     # iterate through the sorted indices
    if current_group == original_groups[i]:                       # Detect whether a  change in group number has occured
        temp_patches.append(original_patches[i])                  # Add patch to the temporary variable since group number didn't change
    else:
        p = PatchCollection(temp_patches, alpha=0.6)              # Add all patches belonging to the current group number to a PatchCollection
        p.set_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])  # Set all shapes belonging to this group to the same random color
        ax.add_collection(p)                                      # Add all shapes belonging this group to the axes object
        current_group = original_groups[i]                        # The group number has changed, so update the current group number
        temp_patches = [original_patches[i]]                      # Reset temp_patches, to begin collecting patches of the next group number

p = PatchCollection(temp_patches, alpha=0.6)                      # temp_patches currently contains the patches belonging to the last group. Add them to a PatchCollection
p.set_color([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
ax.add_collection(p)

ax.autoscale()                                                    # Default scale may not capture the appropriate region
plt.show()