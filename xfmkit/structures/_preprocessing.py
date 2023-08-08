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


def mean_highest_lines(max_set, elements, n):

    df = pd.DataFrame(columns=elements)
    df.loc[0]=max_set
    df.drop(labels=LIGHT_LINES+NON_ELEMENT_LINES+AFFECTED_LINES, axis=1) 
    sorted = df.iloc[0].sort_values(ascending=False)
    result = sorted[0:N_TO_AVG].mean()

    return result

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

