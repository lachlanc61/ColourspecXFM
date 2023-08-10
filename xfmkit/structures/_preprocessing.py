import numpy as np
import pandas as pd

import xfmkit.structures as structures
import xfmkit.utils as utils
import xfmkit.processops as processops
import xfmkit.config as config

from math import sqrt, log

amplify_factor=config.get('preprocessing', 'amplify_factor')
suppress_factor=config.get('preprocessing', 'suppress_factor')

affected_lines=config.get('element_lists', 'affected_lines')
non_element_lines=config.get('element_lists', 'non_element_lines')
light_lines=config.get('element_lists', 'light_lines')

print("TESTING LINES")
print(affected_lines)
N_TO_AVG=5

#----------------------
#local
#----------------------
def mean_highest_lines(max_set, elements, n_to_avg=N_TO_AVG):
    """
    get the mean of the highest N lines

    excluding light, bad and non-element lines
    """

    df = pd.DataFrame(columns=elements)
    df.loc[0]=max_set

    #use a filter to avoid error if a label is not present for df.drop
    drop_filter = df.filter(light_lines+non_element_lines+affected_lines)
    df.drop(drop_filter, inplace=True, axis=1) 
    sorted = df.iloc[0].sort_values(ascending=False)
    result = sorted[0:n_to_avg].mean()

    return result


#----------------------
#for import
#----------------------
def weight_by_transform(self, transform=None):
    """
    adjust weights to correspond to transformation of data

    eg. weight so max(final) = sqrt(max(raw))

    should stack with amplify, suppress etc. 
    """

    if not self.weights.shape[0] == self.data.shape[1]:
            raise ValueError(f"shape mistmatch between weights {self.weights.shape} and data {self.data.shape}")
    
    for i in range(self.data.shape[1]):
        max_ = np.max(self.data.d[:,i])

        if transform == 'sqrt':
            self.weights[i] = self.weights[i]*sqrt(max_)/max_

        elif transform == 'log':
            self.weights[i] = self.weights[i]*log(max_)/max_
            
        elif transform == None:
            pass  
        else:
            raise ValueError(f"invalid value for transform: {transform}")       

def apply_direct_transform(self, transform=None):
    """
    modify weighted by transform
    """
    if self.weighted == None:
        raise ValueError("PixelSet self.weighted not initialised")

    if transform == 'sqrt':
        self.weighted.set_to(np.sqrt(self.weighted.d))

    elif transform == 'log':
        self.weighted.set_to(np.log(self.weighted.d))          

    elif transform == None:
        pass  
    else:
        raise ValueError(f"invalid value for transform: {transform}")

   
def process_weights(self, amplify_list=[], suppress_list=[], ignore_list=[], normalise=False,weight_transform=None, data_transform=None):
    """
    perform specified preprocessing steps, applying weights to data

    - suppress and amplify specified elements
    - normalise
    - perform data/weight transformations
    """

    if normalise:
        if weight_transform is not None or data_transform is not None:
            print("WARNING: normalise with transforms may produce unexpected results")

    if weight_transform is not None and data_transform is not None:
        raise ValueError("Can't perform both weight and data transformation")

    max_set = np.zeros(len(self.data.d[1]))

    for i, label in enumerate(self.labels):
        max_set[i] = np.max(self.data.d[:,i])

    smoothed_max = float(mean_highest_lines(max_set, self.labels, N_TO_AVG))

    #normalise non-element lines to smoothed_max/10
    for target in non_element_lines:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i])
                if max_ < smoothed_max:
                    self.weights[i] = self.weights[i]*smoothed_max/max_/10

    #normalise high affected lines to smoothed_max/10
    for target in affected_lines:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i])
                if max_ > smoothed_max/10:
                    self.weights[i] = self.weights[i]*smoothed_max/max_/10

    #amplify targets unless already > smoothed_max
    for target in amplify_list:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i])
                if max_ < smoothed_max:
                    self.weights[i] = self.weights[i]*amplify_factor

    #suppress targets
    for target in suppress_list:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i])
                if False:    #use sqrt
                    self.weights[i] = self.weights[i]*sqrt(max_)/max_
                else:       #use factor
                    self.weights[i] = self.weights[i]/suppress_factor                    

    if normalise:
        for i, label in enumerate(self.labels):
            max_=np.max(self.data.d[:,i])
            self.weights[i] = self.weights[i]/max_

    if weight_transform is not None:
        self.weight_by_transform(transform=weight_transform)

    #ignore targets
    for target in ignore_list:
        for i, label in enumerate(self.labels):
            if label == target:
                self.weights[i] = 0.0

    self.apply_weights()

    if data_transform is not None:
        self.apply_direct_transform(data_transform)

