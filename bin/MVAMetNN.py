#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time
#import ROOT

import csv
import numpy as np
import math
import theano
import theano.tensor as T

import lasagne

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_datasetcsv(inputfileX, inputfileY):
    #create Treevariable
    tree = ""
    
    #f = ROOT.TFile.Open(inputfile, "read")
    #f = ROOT.TFile.Open("daten.csv", "read")
    #tree = f.Get("MAPAnalyzer/t")
    
    #inputdatencsv1 = np.loadtxt(open("daten.csv","rb"),delimiter=",",skiprows=1)
    
    #Manually load x and y input
    
    """
    reader=csv.reader(open(inputfileX,"rb"),delimiter=',')
    x=list(reader)
    inputdatenX =np.array(x).astype(np.float32)
    
    reader=csv.reader(open(inputfileY,"rb"),delimiter=',')
    y=list(reader)
    inputdatenY =np.array(y).astype(np.float32)
    """
    

    reader=csv.reader(open("../data/dataMVAMet.csv","rb"),delimiter=',')
    datacsv=list(reader)
    header = np.array(datacsv[0]).astype(np.str)
    inputdatentot =np.array(datacsv[1:]).astype(np.float32)
    
    
    #print(header.shape)
    #print(inputdatentot.shape)
    
    
    
    
    meanwhole = inputdatentot.mean(axis=0)
    varwhole = inputdatentot.std(axis = 0)
    minwhole = inputdatentot.min(axis=0)
    maxwhole = inputdatentot.max(axis=0)
   
    """
    #Print out mean, var, min and max of Variables which var != 0
    validvar = 0
    for index in range(0,header.shape[0]):
	if varwhole[index] != 0:
	    print(header[index])
	    print("mean: ",meanwhole[index])
	    print("var: ",varwhole[index])
	    print("min:", minwhole[index])
	    print("max:", maxwhole[index])
	    validvar += 1
	    
    print(validvar," useful Variables")
    """
    
    #for index in range(0,header.shape[0]):
	#if varwhole[index] == 0:
	 #   print(header[index]," is not useful (var = 0)")
	    
    
    
    #targetnames = {"genMet_Pt","genMet_Phi"}
    """
    for index in range(0,header.shape[0]):
	if header[index] == "Boson_Phi":
	    boson_Phi = index
	if header[index] == "recoilpatpfMETT1_Phi":
	    recoilpatpfMETT1_Phi = index
    """
    inputnames = {"recoilpatpfMETT1_Pt",
			"recoilpatpfMETT1_Phi",
			"recoilpatpfMETT1_sumEt",
			"recoilslimmedMETsPuppi_Pt",
			"recoilslimmedMETsPuppi_Phi",
			"recoilslimmedMETsPuppi_sumEtFraction",
			"recoilpatpfTrackMET_Pt",
			"recoilpatpfTrackMET_Phi",
			"recoilpatpfTrackMET_sumEtFraction",
			"recoilpatpfPUMET_Pt",
			"recoilpatpfPUMET_Phi",
			"recoilpatpfPUMET_sumEtFraction",
			"recoilpatpfNoPUMET_Pt",
			"recoilpatpfNoPUMET_Phi",
			"recoilpatpfNoPUMET_sumEtFraction",
			"recoilpatpfPUCorrectedMET_sumEtFraction",
			"recoilpatpfPUCorrectedMET_Pt",
			"recoilpatpfPUCorrectedMET_Phi",
			"Jet0_Pt", "Jet0_Eta", "Jet0_Phi",
			"Jet1_Pt", "Jet1_Eta", "Jet1_Phi",
			"NCleanedJets",
			"NVertex",}
    
    
    #inputnames = {"targetphi"}
    targetnames = {"targetphi"}
    
    plotnames = {"targetphi","targetrecoil", "MVAMET_Phi", "Boson_Phi", "Boson_Pt","dmvamet_Phi", "dpfmet_Pt", "dpfmet_Phi", "MVAMET_Pt", "MVAMET_sumEt", "PhiTrainingResponse", "RecoilTrainingResponse", "PhiCorrectedRecoil_Pt", "PhiCorrectedRecoil_LongZ", "PhiCorrectedRecoil_PerpZ", "PhiCorrectedRecoil_Phi", "PhiCorrectedRecoil_MET", "PhiCorrectedRecoil_METPhi", "LongZCorrectedRecoil_Phi", "LongZCorrectedRecoil_LongZ"}
    
    
    for variable in plotnames:
	make_Plot(variable,inputdatentot,header)
    
    firstX, firstY, firstPlot = True, True, True
    for index in range(0,header.shape[0]):
	if header[index] in inputnames:
	    if firstX:
		inputDataX = np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)
		headerX = np.array(header[index]).reshape(1,1)
		firstX = False
	    else:
		if varwhole[index] != 0:
		    inputDataX = np.hstack((inputDataX, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
		    headerX = np.vstack((headerX, np.array(header[index]).reshape(1,1)))
		    
		      
	if header[index] in targetnames:
	    if firstY:
		inputDataY = np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)
		headerY = np.array(header[index]).reshape(1,1)
		firstY = False
	    else:
		if varwhole[index] != 0:
		    inputDataY = np.hstack((inputDataY, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
		    headerY = np.vstack((headerY, np.array(header[index]).reshape(1,1)))
		    
	if header[index] in plotnames:
	    if firstPlot:
		inputDataPlot = np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)
		headerPlot = np.array(header[index]).reshape(1,1)
		firstPlot = False
	    else:
		inputDataPlot = np.hstack((inputDataPlot, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
		headerPlot = np.vstack((headerPlot, np.array(header[index]).reshape(1,1)))
		
		    
    print("xshape ", inputDataX.shape)
    print("yshape ", inputDataY.shape)
    
    
    print("mean: ",inputDataY.mean())
    print("var: ",inputDataY.std())
    print("min:", inputDataY.min())
    print("max:", inputDataY.max())


    #x = inputDataX[1:1000,0]
    #y = inputDataY[1:1000,0]
    #print("x_plot shape ", x.shape)
    #plt.plot(x,y)
    #plt.savefig('test.png')
    #Generate x and y input
    #inputDataX = np.array(inputdatentot[:,0])
    """
    firstX, firstY = True, True
    for index in range(0,header.shape[0]):
	if header[index] in targetnames:
	    if firstY:
		inputDataY = np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)
		headerY = np.array(header[index]).reshape(1,1)
		firstY = False
	    else:
		inputDataY = np.hstack((inputDataY, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
		headerY = np.vstack((headerY, np.array(header[index]).reshape(1,1)))
	else:
	    if firstX:
		inputDataX = np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)
		headerX = np.array(header[index]).reshape(1,1)
		firstX = False
	    else:
		if varwhole[index] != 0:
		    inputDataX = np.hstack((inputDataX, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
		    headerX = np.vstack((headerX, np.array(header[index]).reshape(1,1)))
		
    
    print("inputDataX shape = ",inputDataX.shape)
    print("inputDataY shape = ",inputDataY.shape)

    """

    
    print(meanwhole.shape)
    print(minwhole.shape)
    
    """
    for index in range(0,header.shape[0]):
	if header[index] in targetnames:
	    inputDataX = np.delete(inputDataX,index,axis=1)
	else:
	    inputDataY = np.delete(inputDataY,index,axis=1)
	    if varwhole[index] == 0:
		inputDataX = np.delete(inputDataX,index,axis=1)
    """
    
	    
    meanSelectedX = inputDataX[:-inputDataX.shape[0]*3/6].mean(axis=0)
    varSelectedX = inputDataX[:-inputDataX.shape[0]*3/6].std(axis = 0)

    meanSelectedY = inputDataY[:-inputDataY.shape[0]*3/6].mean(axis=0)
    varSelectedY = inputDataY[:-inputDataY.shape[0]*3/6].std(axis = 0)

    meanSelectedPlot = inputDataPlot[:-inputDataPlot.shape[0]*3/6].mean(axis=0)
    varSelectedPlot = inputDataPlot[:-inputDataPlot.shape[0]*3/6].std(axis = 0)
    
    inputDataX = (inputDataX - meanSelectedX) / varSelectedX
    inputDataY = (inputDataY - meanSelectedY) / varSelectedY
    #inputDataPlot = (inputDataPlot - meanSelectedPlot) / varSelectedPlot

    print("headerx shape ", headerX.shape)
    print("meanselectedX shape ", meanSelectedX.shape)
    
    
    
    meanSelectedXcheck = inputDataX.mean(axis=0)
    varSelectedXcheck = inputDataX.std(axis = 0)

    meanSelectedYcheck = inputDataY.mean(axis=0)
    varSelectedYcheck = inputDataY.std(axis = 0)
    
    
    """
    for index in range(0,headerX.shape[0]):
	print(headerX[index])
	print("mean: ",meanSelectedXcheck[index])
	print("var: ",varSelectedXcheck[index])
    

    for index in range(0,headerY.shape[0]):
	print(headerY[index])
	print("mean: ",meanSelectedYcheck[index])
	print("var: ",varSelectedYcheck[index])
    """

    #print("inputDataX shape = ",inputDataX.shape)
    #print("inputDataY shape = ",inputDataY.shape)
    
    #3/6 Training, 1/6 Validierung, 1/3 Test
    X_train, X_val, X_test = inputDataX[:-inputDataX.shape[0]*3/6], inputDataX[inputDataX.shape[0]*3/6:inputDataX.shape[0]*4/6], inputDataX[-inputDataX.shape[0]*2/6:]
    y_train, y_val, y_test = inputDataY[:-inputDataY.shape[0]*3/6], inputDataY[inputDataY.shape[0]*3/6:inputDataY.shape[0]*4/6], inputDataY[-inputDataY.shape[0]*2/6:]

    inputDataPlot = inputDataPlot[-inputDataPlot.shape[0]*2/6:]
    #meanPlot = inputDataPlot.mean(axis=0)
    #varPlot = inputDataPlot.std(axis=0)

    
    """
    for index in range(0,X_train.shape[1]):
	print(header[index])
	print("mean: ",meanX[index])
    print("shape meanx: ",meanX.shape)
    
    for index in range(0,headerX.shape[0]):
	if headerX[index] == 'targetphi':
	    targetX = index
    
    for index in range(0,headerPlot.shape[0]):
	if headerPlot[index] == 'targetphi':
	    targetPlot = index
    
    targetY = 0;
    
    print('min target X:', X_test[:,targetX].min())
    print('max target X:', X_test[:,targetX].max())
    print('min target Y:', y_test[:,targetY].min())
    print('max target Y:', y_test[:,targetY].max())
    print('min target Plot:', inputDataPlot[:,targetPlot].min())
    print('max target Plot:', inputDataPlot[:,targetPlot].max())
    
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    """
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test, inputDataPlot, headerPlot, meanSelectedPlot, varSelectedPlot
    
    
    
    #print(inputdatencsv2.shape)
    #print(inputdatencsv3.shape)
    #Get entries unequal 0
    #usefulentries = 0
    #for event in tree :
	#if event.Boson_Pt != 0:
	    #usefulentries = usefulentries + 1
    
    #print(usefulentries)
    
    #branches = tree.GetListOfBranches()
    
    #w, h = branches.GetEntries(), tree.GetEntries(), 
    #inputdata = [[0 for x in range(0,w)] for y in range(0,h)] 
    
    #for branchname in leaves :
	#print(branchname)
	#for event in tree :
	    
    
    
    #for branchname in branches :
	#print(branchname)
	#for event in range(0,tree.GetEntries()) :
	#for event in tree :
	#print(branchname)
	    #inputdata[event][param] = branchname.GetLeaf(branchname.GetName()).GetValue(event)
	    #print(branchname.GetLeaf(branchname.GetName()).GetValue(event))
	#param = param + 1    
	#var = event.Boson_Pt
	#print var
    
    
    """
    float testlist[entryarray->GetEntries()];
	for (int j = 0; j < 10; j++)
	    {
		inputTree1->SetBranchAddress(entryarray->At(j)->GetName(),&testlist[j]);
	    }
	
	for (int i = 0; i < 100; i++)
	{
	    inputTree1 -> GetEntry(i);
	    //for (int j = 0; j < entryarray->GetEntries(); j++)
	    for (int j = 0; j < 10; j++)
	    {
		//inputTree1->SetBranchAddress(entryarray->At(j)->GetName(),&testlist[]);
		
		textoutput <<  testlist[j] << "  " ;
		
		
	    }
	    textoutput << endl;
	}	
    
    for i in range(0,10)
	
    
    
    
    
    
    param = 0
    for event in range(0,tree.GetEntries()) :
	tree.GetEntry(event)
	for branchname in branches :
	    inputdata[param][event] = branchname.GetLeaf(branchname.GetName()).GetValue()
	param = param + 1  

    #print(inputdata)  
    """
    
def make_Plot(variablename, inputData, inputHeader):
    for index in range(0,inputHeader.shape[0]):
	if inputHeader[index] == variablename:
	    variableIndex = index
	    
    histData = inputData[:,variableIndex]
    
    num_bins = 50
    
    n, bins, patches = plt.hist(histData, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel(variablename)
    plt.ylabel('Hits')
    plt.savefig(('../plots/'+variablename+".png"))
    plt.clf()
    return 0
    
def make_ULongOverRecoilPlot(plotData,headerPlot, targetvariable, nbins=20):
    for index in range(0,headerPlot.shape[0]):
	if headerPlot[index] == targetvariable:
	    targetindex = index
	if headerPlot[index] == "LongZCorrectedRecoil_LongZ":
	    LongZ = index
	if headerPlot[index] == "Boson_Pt":
	    BosonPt = index
    
    #XRange = np.arange(plotData[:,targetindex].min(),plotData[:,targetindex].max(),(plotData[:,targetindex].max()-plotData[:,targetindex].min())/nbins)
    XRange = np.arange(plotData[:,targetindex].min(),200,(200-plotData[:,targetindex].min())/nbins)
    YMean = np.zeros((nbins,1))
    YVar = np.zeros((nbins,1))
    #YValues 
    
    for index in range(0,nbins-1):
	firstLoop = True
	for event in range(0,plotData.shape[0]):
	    if plotData[event,targetindex] > XRange[index] and plotData[event,targetindex] < XRange[index+1]:
		currentDistri = np.array(np.absolute(-plotData[event,LongZ]/plotData[event,BosonPt])).reshape(1,1)
		firstLoop = False
	    else:
		currentDistri = np.vstack((currentDistri,np.array(np.absolute(-plotData[event,LongZ]/plotData[event,BosonPt])).reshape(1,1)))
	print('current bin ',index,' min:'currentDistri.min())
	print('current bin ',index,' max:'currentDistri.max())
	print('current bin ',index,' mean:'currentDistri.mean())
	print('current bin ',index,' var:'currentDistri.var())
	YMean[index] = currentDistri.mean()
	YVar[index] = currentDistri.var()
	
    plt.clf()
    plt.errorbar(XRange,YMean,yerr=YVar)
    plt.savefig("../plots/ULongoverRecoil.png")
    plt.clf()


def plot_results(plotData, headerPlot, meanPlot, varPlot):
    
    for index in range(0,headerPlot.shape[0]):
	if headerPlot[index] == "targetphi":
	    targetphi = index
	if headerPlot[index] == "NNoutput":
	    NNoutput = index
	if headerPlot[index] == "PhiCorrectedRecoil_Phi":
	    BDToutput = index
	if headerPlot[index] == "LongZCorrectedRecoil_LongZ":
	    LongZ = index
	if headerPlot[index] == "Boson_Pt":
	    BosonPt = index

    
    print('min target ', plotData[:,targetphi].min())
    print('max target ', plotData[:,targetphi].max())
    print('min NN output', plotData[:,NNoutput].min()*varPlot[targetphi])
    print('max NN output', plotData[:,NNoutput].max()*varPlot[targetphi])
    histDataNN = (plotData[:,targetphi]-meanPlot[targetphi])/varPlot[targetphi] - plotData[:,NNoutput]
    print('mean data-NN:', histDataNN.mean())
    print('var data-NN:', histDataNN.var())
    histDataBDT = plotData[:,targetphi]- plotData[:,BDToutput]
    histDataNNresponse = plotData[:,NNoutput]*varPlot[targetphi]+meanPlot[targetphi]
    histTarget = (plotData[:,targetphi]-meanPlot[targetphi])/varPlot[targetphi]
    
    
    #histDataLongOverZPt = (plotData[:,LongZ]/plotData[:,BosonPt])
    
    
    make_ULongOverRecoilPlot(plotData, headerPlot, 'Boson_Pt',20)
    
    
    
    print("mean target", meanPlot[targetphi])
    print("var target", varPlot[targetphi])
    print("mean BDT", meanPlot[BDToutput])
    print("var BDT", varPlot[BDToutput])
    num_bins = 50
    # the histogram of the data
    #n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
    n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('Target - NNoutput')
    plt.ylabel('Hits')
    plt.savefig("../plots/SumVarianceNN.png")
    plt.clf()
    
    n, bins, patches = plt.hist(histDataBDT, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('Deviation from Result')
    plt.ylabel('Probability')
    plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    plt.savefig("../plots/SumVarianceBDT.png")
    plt.clf()
    
    n, bins, patches = plt.hist(histDataNNresponse, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('NN output')
    plt.ylabel('Hits')
    plt.savefig("../plots/NNresponse.png")
    plt.clf()
    # add a 'best fit' line
    #y = mlab.normpdf(bins, mu, sigma)
    #plt.plot(bins, y, 'r--')
    
    
    n, bins, patches = plt.hist(histTarget, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('Targetphi')
    plt.ylabel('Hits')
    plt.savefig("../plots/TargetPhi.png")
    plt.clf()
    
    
   
    # normal distribution center at x=0 and y=5
    plt.hist2d(plotData[:,targetphi], histDataNN,bins = 80, norm=LogNorm())
    #plt.ylim([-0.25,0.25])
    plt.xlabel('Phi')
    plt.ylabel('Hits')
    plt.savefig("../plots/NNVariance")
    plt.clf()
    
    """
    plt.hist2d(plotData[:,BosonPt], histDataLongOverZPt,bins = 80, norm=LogNorm())
    #plt.ylim([-0.25,0.25])
    plt.xlabel('Boson Pt')
    plt.ylabel('u_long/Boson Pt')
    plt.savefig("../plots/ULongoverBosonPt")
    plt.clf()
    """

    return True

    

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_mlpMVA(inputcount, targetcount, input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.
    
    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, inputcount),
                                     input_var=input_var)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=800,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.1)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.linear)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.1)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=targetcount,
            nonlinearity=lasagne.nonlinearities.linear)
    """
    
    #without dropout
    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, inputcount),
                                     input_var=input_var)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(l_in,num_units=800,
            nonlinearity=lasagne.nonlinearities.linear)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(l_hid1,num_units=800,
            nonlinearity=lasagne.nonlinearities.linear)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(l_hid2,num_units=targetcount,
            nonlinearity=lasagne.nonlinearities.linear)
    """
    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network

def build_custom_mlpMVA(input_var=None, depth=2, width=800, drop_input=.2,
                     drop_hidden=.5):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
  
def build_cnnMVA(inputcount, outputcount, input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, inputcount),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    #network = lasagne.layers.MaxPool1DLayer(network, pool_size=1)

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv1DLayer(
            network, num_filters=32, filter_size=1,
            nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.MaxPool1DLayer(network, pool_size=1)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=outputcount,
            nonlinearity=lasagne.nonlinearities.linear)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', num_epochs=50):
    
    
    # Load the dataset
    print("Loading data...")
    X_trainMVA, y_trainMVA, X_valMVA, y_valMVA, X_testMVA, y_testMVA, plotData, plotHeader, meanPlot, varPlot = load_datasetcsv("xtrain.csv","ytrain.csv")
    
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    print(X_trainMVA.shape)
    print(y_trainMVA.shape)
   
    # Prepare Theano variables for inputs and targets
    input_var = T.fmatrix('inputs')
    target_var = T.fmatrix('targets')
    
    inputcount = X_trainMVA.shape[1]
    targetcount = y_trainMVA.shape[1]
    
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlpMVA(inputcount, targetcount, input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlpMVA(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnnMVA(inputcount, targetcount, input_var)
    else:
        print("Unrecognized model type %r." % model)
        return
    
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)
    
    predict_fn = theano.function([input_var], test_prediction)

    #test = predict_fn(X_testMVA)
    #print("testshape", test.shape)

    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_trainMVA, y_trainMVA, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_valMVA, y_valMVA, 500, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            #val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  validation accuracy:\t\t{:.2f} %".format(
        #    val_acc / val_batches * 100))
    
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_testMVA, y_testMVA, 500, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        #test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    #print("  test accuracy:\t\t{:.2f} %".format(
    #    test_acc / test_batches * 100))
    
    


    num_bins = 50
    
    n, bins, patches = plt.hist(y_testMVA[:,0], num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('prediction')
    plt.ylabel('Hits')
    plt.savefig(('../plots/test.png'))
    plt.clf()

    """
    firstPrediction = True
    for batch in iterate_minibatches(X_testMVA, y_testMVA, 500, shuffle=False):
	inputs, targets = batch
	if firstPrediction:
	    testSetPrediction = predict_fn(inputs).reshape(predict_fn(inputs).shape[0],1)
	    print('predict shape', predict_fn(inputs).shape)
	    print('testsetpredict shape', testSetPrediction.shape)
	    firstPrediction = False
	else:
	    tempOutput = predict_fn(inputs)
	    testSetPrediction = np.vstack((testSetPrediction, predict_fn(inputs).reshape(predict_fn(inputs).shape[0],1)))
	    
    print('testSetPrediction shape: ',testSetPrediction.shape)
    """
    testSetPrediction = predict_fn(X_testMVA)
    for index in range(0,plotHeader.shape[0]):
	if plotHeader[index] == "targetphi":
	    targetphi = index
    
    print('testsetpredictionmin: ', testSetPrediction.min())
    print('testsetpredictionmax: ', testSetPrediction.max())
    print('testsettargetmin: ', y_testMVA.min())
    print('testsettargetmax: ', y_testMVA.max())
    print('error: ', val_fn(X_testMVA,(y_testMVA/varPlot[targetphi])))

    plotData = np.hstack((plotData, np.array(testSetPrediction).reshape(testSetPrediction.shape[0],1)))
    plotHeader = np.vstack((plotHeader, np.array("NNoutput").reshape(1,1)))
    print("plotDatashape", plotData.shape)
    plot_results(plotData, plotHeader, meanPlot, varPlot)
   
    
    
    
    
    
    
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    
    for epoch in range(num_epochs):
        print('hallo')
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
        #    inputs, targets = batch
        #    train_err += train_fn(inputs, targets)
        #    train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 50, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
"""
    #Optionally, you could now dump the network weights to a file like this:
    #np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
