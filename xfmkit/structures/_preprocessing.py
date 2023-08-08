import numpy as np
import pandas as pd

import xfmkit.structures as structures
import xfmkit.utils as utils
import xfmkit.processops as processops

from math import sqrt, log

IGNORE_LINES=[]
AFFECTED_LINES=['Ar', 'Mo', 'MoL']
NON_ELEMENT_LINES=['sum','Back','Compton']
LIGHT_LINES=['Mg', 'Al', 'Si', 'P', 'S']

AMP_FACTOR=10.0
SUPPRESS_FACTOR=10.0

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
    drop_filter = df.filter(LIGHT_LINES+NON_ELEMENT_LINES+AFFECTED_LINES)
    df.drop(drop_filter, inplace=True, axis=1) 
    sorted = df.iloc[0].sort_values(ascending=False)
    result = sorted[0:n_to_avg].mean()

    return result


#----------------------
#for import
#----------------------
def apply_transform_via_weights(self, transform=None):
    if not self.weights.shape[0] == self.data.shape[1]:
            raise ValueError(f"shape mistmatch between weights {self.weights.shape} and data {self.data.shape}")
    
    for i in range(self.data.shape[1]):
        max_ = np.max(self.data.d[:,i])

        if transform == 'sqrt':
            self.weights[i] = self.weights[i]*sqrt(max_)/max_
        
        if transform == 'log':
            self.weights[i] = self.weights[i]*log(max_)/max_
        
        elif transform == None:
            pass  
        else:
            raise ValueError(f"invalid value for transform: {transform}")       

def apply_transform(self, transform=None):

    if self.weighted == None:
        raise ValueError("PixelSet self.weighted not initialised")

    if transform == 'sqrt':
        self.weighted.set_to(np.sqrt(self.weighted.d))

    elif transform == 'log':
        self.weighted.set_to(np.log(self.weighted.d))          

    elif transform == None:
        pass  
    else:
        raise ValueError(f"invalue value for transform: {transform}")


def process_weights(self, amplify_list=[], suppress_list=[], normalise=False,weight_transform=None, data_transform=None):
    """
    perform preprocess steps according to args, applying weights to data

    - suppress/amplify specified elements
    - normalise if selected
    - perform data/weight transformations
    """

    if normalise:
        if not amplify_list== [] \
            or not suppress_list == [] \
            or weight_transform is not None:
            print("WARNING: normalise with transforms may produce unexpected results")

    if weight_transform is not None and data_transform is not None:
        print("WARNING: performing both weight and data transformation")

    max_set = np.zeros(len(self.data.d[1]))

    for i, label in enumerate(self.labels):
        max_set[i] = np.max(self.data.d[:,i])

    smoothed_max = float(mean_highest_lines(max_set, self.labels, N_TO_AVG))

    #normalise non-element lines to smoothed_max/10
    for target in NON_ELEMENT_LINES:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i])
                if max_ < smoothed_max:
                    self.weights[i] = self.weights[i]*smoothed_max/max_/10

    #normalise high affected lines to smoothed_max/10
    for target in AFFECTED_LINES:
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
                    self.weights[i] = self.weights[i]*AMP_FACTOR

    #suppress targets
    for target in suppress_list:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i])
                if True:    #use sqrt
                    self.weights[i] = self.weights[i]*sqrt(max_)/max_
                else:
                    self.weights[i] = self.weights[i]/SUPPRESS_FACTOR                    

    if normalise:
        for i, label in enumerate(self.labels):
            max_=np.max(self.data.d[:,i])
            self.weights[i] = self.weights[i]/max_

    if weight_transform is not None:
        self.apply_transform_via_weights(transform=weight_transform)

    self.apply_weights()

    if data_transform is not None:
        self.apply_transform(data_transform)

