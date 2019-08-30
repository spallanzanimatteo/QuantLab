# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import numpy as np
import re
import os
import pickle

def readSingleFile(fname):
    with open(fname) as f:
        fileCont = f.read()
        arrs = re.findall('array\(\[(.*)\]\)', fileCont)
        arrs = [np.fromstring(a, sep=',', dtype=np.int16) for a in arrs]
#        print('fname: %s' % fname)
#        print([t.shape for t in arrs])
        arrs = [t.reshape(64,-1) for t in arrs] #shape: n_t x 64
    
    #sum of lengths: 8*60+48+52 = 580
    #'normal' size: 400 --> overlap of 10 on both sides (or 20 on one)
    arrsConcat = [arrs[0]] + [t[:,20:] for t in arrs[1:]]
    spectrogram = np.concatenate(arrsConcat, axis=1)
    return spectrogram #64 x 25600

def getClasses(rootDir):
    filelist = os.listdir(rootDir) 
    # regex for format {className}_{someNum}_{randomString}.csv to parse class
    classes = (re.findall('^(.*)\_\d*_.*.csv$', fname) for fname in filelist)
    classes = filter(lambda s: len(s) >= 1, classes)
    classes = (s[0] for s in classes)
    classes = list(set(classes)) # uniquify
    return classes

def readClassSpectrograms(cl, rootDir):
    filelist = os.listdir(rootDir)
    clFiles = (re.findall('^(%s_.*.csv)$' % cl, fname) for fname in filelist)
    clFiles = filter(lambda s: len(s) >= 1, clFiles)
    clFiles = (rootDir + s[0] for s in clFiles)
    clSpectrograms = [readSingleFile(fname) for fname in clFiles]
    return clSpectrograms


#readSingleFile('./test/car_172_offset25.csv')
#readSingleFile('./test/car_172_offset50.csv')



classes = getClasses('./train/')
print('classes: %s' % str(classes))
datasetTrain = {cl: readClassSpectrograms(cl, './train/') for cl in classes}
datasetTest = {cl: readClassSpectrograms(cl, './test/') for cl in classes}

fname = './train.pickle'
with open(fname, 'wb') as f:
    pickle.dump(datasetTrain, f)
fname = './test.pickle'
with open(fname, 'wb') as f:
    pickle.dump(datasetTest, f)


#import matplotlib.pyplot as plt
#spectrogram = datasetTrain['acoustic_guitar'][3]
#plt.imshow(spectrogram)
