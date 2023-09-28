import numpy as np
import pandas as pd

import xfmkit.structures as structures
import xfmkit.utils as utils
import xfmkit.processops as processops
import xfmkit.imgops as imgops
import xfmkit.config as config

from math import sqrt, log

amplify_factor=config.get('preprocessing', 'amplify_factor')
suppress_factor=config.get('preprocessing', 'suppress_factor')

affected_lines=config.get('elements', 'affected_lines')
non_element_lines=config.get('elements', 'non_element_lines')
light_lines=config.get('elements', 'light_lines')

conc_sanity_threshold=float(config.get('preprocessing', 'conc_sanity_threshold'))
snr_threshold=float(config.get('preprocessing', 'snr_threshold'))
deweight_on_downsample_factor=float(config.get('preprocessing', 'deweight_on_downsample_factor'))

BASEFACTOR=1/10000 #ppm to wt%

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


def generate_weighted(self):

    print("-----------------")
    print(f"APPLYING CHANNEL WEIGHTS")            
    
    _result = np.zeros(self.data.shape)

    for i in range(self.data.shape[1]):
        _result[:,i] = self.data.d[:,i]*self.weights[i]
    
    result = structures.DataSeries(_result, self.data.dimensions)

    return result


def apply_weights(self, amplify_list=[], suppress_list=[], ignore_list=[], normalise=False,weight_transform=None, data_transform=None):
    """
    perform specified preprocessing steps, applying weights to data

    - suppress and amplify specified elements
    - normalise
    - perform data/weight transformations
    """

    print("-----------------")
    print(f"CALCULATING CHANNEL WEIGHTS")    

    if normalise:
        if weight_transform is not None or data_transform is not None:
            print("WARNING: normalise with transforms may produce unexpected results")

    if weight_transform is not None and data_transform is not None:
        raise ValueError("Can't perform both weight and data transformation")

    max_set = np.zeros(len(self.data.d[1]))
    max_set_weighted = np.zeros(len(self.data.d[1]))    

    for i, label in enumerate(self.labels):
        max_set[i] = np.max(self.data.d[:,i]*self.weights[i])   #apply weights here
        #max_set[i] = max_set[i]

    avg_max = float(mean_highest_lines(max_set, self.labels, N_TO_AVG))

    #normalise non-element lines to avg_max/10
    for target in non_element_lines:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i]*self.weights[i])
                if max_ < avg_max:
                    self.weights[i] = self.weights[i]*avg_max/max_/10

    #normalise high affected lines to avg_max/10
    for target in affected_lines:
        for i, label in enumerate(self.labels):
            if label == target:
                max_=np.max(self.data.d[:,i]*self.weights[i])
                if max_ > avg_max/10:
                    self.weights[i] = self.weights[i]*avg_max/max_/10

    #amplify targets up to average max
    for target in amplify_list:
        for i, label in enumerate(self.labels):
            if label == target:
                print(f"----amplifying {label}")
                max_=np.max(self.data.d[:,i]*self.weights[i])
                if max_ < avg_max:
                    self.weights[i] = self.weights[i]*avg_max/max_

    #suppress targets
    for target in suppress_list:
        for i, label in enumerate(self.labels):
            if label == target:
                print(f"----suppressing {label}")
                max_=np.max(self.data.d[:,i]*self.weights[i])
                if True:    #use sqrt
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

    #generate the weighted dataset
    self.weighted = self.generate_weighted()

    #apply full transforms
    if data_transform is not None:
        self.apply_direct_transform(data_transform)


def downsample_by_se(self, deweight=False):

    print("-----------------")
    print(f"SMOOTHING CHANNELS")    

    self.check()

    if not np.issubdtype(self.data.d.dtype, np.floating):
        print("WARNING: dtype changing to float")

    mapview_ = np.zeros(self.data.mapview.shape, dtype=np.float32)
    se_map_ = np.zeros(self.se.mapview.shape, dtype=np.float32)

    if deweight == True:
        deweight_factor = deweight_on_downsample_factor
    else:
        deweight_factor = 1.0

    if np.max(self.se.d) == 0:
        print("WARNING: downsampling without valid data for errors - data will be left unchanged")
    else:
        for i in range(self.data.d.shape[1]):

            try:
                label_=self.labels[i]
            except:
                label_=""

            img_ = np.ndarray.copy(self.data.mapview[:,:,i])
            se_ = np.ndarray.copy(self.se.mapview[:,:,i])

            #ratio, q2_sd, q99_data = utils.calc_se_ratio(img_, se_)

            #print(f"*****1 element {label_} ({i}),  dmax: {q99_data:.3f},  seavg: {q2_sd:.3f}, ratio: {ratio:.3f}")

            ratio, data_max, se_mean = utils.calc_simple_se_ratio(img_, se_)

            #print(f"*****2 element {label_} ({i}), dmax: {data_max:.3f},  seavg: {se_avg:.3f}, ratio: {simple_ratio:.3f}")


            j=0
            while ratio <= snr_threshold:
                print(f"---averaging element {label_} ({i}), cycle {j} -- max: {data_max:.3f}, se_avg: {se_mean:.3f}, ratio: {ratio:.3f}")
                img_, se_ = imgops.apply_gaussian(img_, 1, se_)

                #deweight channel for each gaussian applied
                self.weights[i] = self.weights[i]*deweight_factor

                ratio, data_max, se_mean = utils.calc_simple_se_ratio(img_, se_)
                j+=1

            print(f"FINISHED element {label_} ({i}), max: {data_max*BASEFACTOR:.3f} %, se_avg: {se_mean*BASEFACTOR:.3f} %, ratio: {ratio:.3f}")

            #check if value is unreasonably high and normalise back to threshold/2 if needed
            if np.max(img_) >= conc_sanity_threshold:

                print(f"**WARNING: element {label_} ({i}) max of {data_max} unexpectedly high, normalising")
                norm_factor = conc_sanity_threshold/np.max(img_)/2
                img_ = img_*norm_factor
                se_ = se_*norm_factor

            mapview_[:,:,i] = img_
            se_map_[:,:,i] = se_

    self.data.set_to(mapview_)
    self.se.set_to(se_map_)

    self.check()

    print("-----------------")
    print(f"SMOOTHING COMPLETE")    
