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

import time

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

"""
Files are referred to in variable fileName as
DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 1
DYJetsToLL_M-50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8 --> 8
DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8 --> 9
DYJetsToLL_M-10to50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 6
DYJetsToLL_M-150_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 7
DYJetsToLL_M-50_HT-100to200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 2
DYJetsToLL_M-50_HT-200to400_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 3
DYJetsToLL_M-50_HT-400to600_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 4
DYJetsToLL_M-50_HT-600toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8 --> 5
"""




def load_datasetcsv():
    #create Treevariable
    
    start = time.time()
    
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
    
    
    #header = np.vstack((header.reshape(header.shape[0],1),np.array('targetrecoilNN').reshape(1,1)))
    print(header.shape)
    
    
    
    inputnames = {"recoilslimmedMETs_Pt",
			"recoilslimmedMETs_Phi",
			"recoilslimmedMETs_sumEtFraction",
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
			"Jet2_Pt", "Jet2_Eta", "Jet2_Phi",
			"Jet3_Pt", "Jet3_Eta", "Jet3_Phi",
			"Jet4_Pt", "Jet4_Eta", "Jet4_Phi",
			"NCleanedJets",
			"NVertex"}
    
    
    #inputnames = {"targetphi"}
    #targetnames = {'targetphi','targetrecoilNN'}
    targetnames = {'Boson_Phi','Boson_Pt'}
    
    
    plotnames = {"targetphi","targetrecoil","targetrecoilNN", "MVAMET_Phi", "Boson_Phi", "Boson_Pt","dmvamet_Phi", "dpfmet_Pt", "dpfmet_Phi", "MVAMET_Pt", "MVAMET_sumEt","recoilslimmedMETs_Pt", "recoilslimmedMETs_LongZ", "recoilslimmedMETs_Phi", "PhiTrainingResponse", "RecoilTrainingResponse", "PhiCorrectedRecoil_Pt", "PhiCorrectedRecoil_LongZ", "PhiCorrectedRecoil_PerpZ", "PhiCorrectedRecoil_Phi", "PhiCorrectedRecoil_MET", "PhiCorrectedRecoil_METPhi", "LongZCorrectedRecoil_Phi", "LongZCorrectedRecoil_LongZ", "NVertex"}
    
    
    
    #Insert targetvariable for NN
    for index in range(0,header.shape[0]):
	if header[index] == 'Boson_Pt':
	    print("Boson_Pt conv:", index)
	
    #inputdatentot = np.hstack((inputdatentot,np.array(inputdatentot[:,bosonPt]/inputdatentot[:,METT1_Pt]).reshape(inputdatentot.shape[0],1)))
    
    
    inputDataX = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    inputDataY = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    inputDataPlot = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    
    dictInputX = {}
    dictInputY = {}
    dictPlot = {}
    dictInputTot = {}
    
    dt = int((time.time() - start))
    print('Elapsed time for loading dataset: ', dt)
    
    
    for index in range(0,header.shape[0]):
	dictInputTot[header[index]] = index
	if header[index] in inputnames:
	    if varwhole[index] != 0:
		dictInputX[header[index]] = inputDataX.shape[1]
		inputDataX = np.hstack((inputDataX, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
		
	if header[index] in targetnames:
	    if varwhole[index] != 0:
		dictInputY[header[index]] = inputDataY.shape[1]
		inputDataY = np.hstack((inputDataY, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))

	if header[index] in plotnames:
	    if varwhole[index] != 0:
		dictPlot[header[index]] = inputDataPlot.shape[1]
		inputDataPlot = np.hstack((inputDataPlot, np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))

    #header = np.vstack((header.reshape(header.shape[0],1),np.array('targetrecoilNN').reshape(1,1)))
    
    if 'targetrecoilNN' in inputnames:
	dictInputX['targetrecoilNN'] = inputDataX.shape[1]
	inputDataX = np.hstack((inputDataX,np.array(inputdatentot[:,dictInputTot['Boson_Pt']]/inputdatentot[:,dictInputTot['recoilslimmedMETs_Pt']]).reshape(inputdatentot.shape[0],1)))
	
    if 'targetrecoilNN' in targetnames:
	dictInputY['targetrecoilNN'] = inputDataY.shape[1]
	inputDataY = np.hstack((inputDataY,np.array(inputdatentot[:,dictInputTot['Boson_Pt']]/inputdatentot[:,dictInputTot['recoilslimmedMETs_Pt']]).reshape(inputdatentot.shape[0],1)))
	    
    if 'targetrecoilNN' in plotnames:
	dictPlot['targetrecoilNN'] = inputDataPlot.shape[1]
	inputDataPlot = np.hstack((inputDataPlot,np.array(inputdatentot[:,dictInputTot['Boson_Pt']]/inputdatentot[:,dictInputTot['recoilslimmedMETs_Pt']]).reshape(inputdatentot.shape[0],1)))
	    
		    
    #for variable in plotnames:
	#make_Plot(variable,inputDataPlot,dictPlot)
    
		    
    print("xshape ", inputDataX.shape)
    print("yshape ", inputDataY.shape)
    
    
    print("mean: ",inputDataY.mean())
    print("var: ",inputDataY.std())
    print("min:", inputDataY.min())
    print("max:", inputDataY.max())


 

    
    print(meanwhole.shape)
    print(minwhole.shape)
    
    meanSelectedX = inputDataX[:-inputDataX.shape[0]*3/6].mean(axis=0)
    varSelectedX = inputDataX[:-inputDataX.shape[0]*3/6].std(axis = 0)

    meanSelectedY = inputDataY[:-inputDataY.shape[0]*3/6].mean(axis=0)
    varSelectedY = inputDataY[:-inputDataY.shape[0]*3/6].std(axis = 0)

    meanSelectedPlot = inputDataPlot[:-inputDataPlot.shape[0]*3/6].mean(axis=0)
    varSelectedPlot = inputDataPlot[:-inputDataPlot.shape[0]*3/6].std(axis = 0)
    
    inputDataX = (inputDataX - meanSelectedX) / varSelectedX
    inputDataY = (inputDataY - meanSelectedY) / varSelectedY
    #inputDataPlot = (inputDataPlot - meanSelectedPlot) / varSelectedPlot

    
    meanSelectedXcheck = inputDataX.mean(axis=0)
    varSelectedXcheck = inputDataX.std(axis = 0)

    meanSelectedYcheck = inputDataY.mean(axis=0)
    varSelectedYcheck = inputDataY.std(axis = 0)
    

    #3/6 Training, 1/6 Validierung, 1/3 Test
    X_train, X_val, X_test = inputDataX[:-inputDataX.shape[0]*3/6], inputDataX[inputDataX.shape[0]*3/6:inputDataX.shape[0]*4/6], inputDataX[-inputDataX.shape[0]*2/6:]
    y_train, y_val, y_test = inputDataY[:-inputDataY.shape[0]*3/6], inputDataY[inputDataY.shape[0]*3/6:inputDataY.shape[0]*4/6], inputDataY[-inputDataY.shape[0]*2/6:]

    inputDataPlot = inputDataPlot[-inputDataPlot.shape[0]*2/6:]


    return X_train, y_train, X_val, y_val, X_test, y_test, dictInputY, inputDataPlot, dictPlot, meanSelectedPlot, varSelectedPlot
    
    
    
def make_Plot(variablename, inputData, dictPlot):

    histData = inputData[:,dictPlot[variablename]]
    
    num_bins = 50
    
    n, bins, patches = plt.hist(histData, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel(variablename)
    plt.ylabel('Hits')
    plt.savefig(('../plots/'+variablename+".png"))
    plt.clf()
    return 0
    
    
def make_ULongMinusRecoilPlot(plotData,dictPlot, bosonName, targetvariable, minrange=42,maxrange=0, stepwidth=0):


    #XRange = np.arange(plotData[:,targetindex].min(),plotData[:,targetindex].max(),(plotData[:,targetindex].max()-plotData[:,targetindex].min())/nbins)
    if minrange == 42:
	minrange = plotData[:,targetindex].min()
    if maxrange == 0:
	maxrange = plotData[:,targetindex].max()
    if stepwidth == 0:
	stepwidth = (maxrange-minrange)/20
    XRange = np.arange(minrange,maxrange,stepwidth)
    YMean = np.zeros((XRange.shape[0]-1,1))
    YStd = np.zeros((XRange.shape[0]-1,1))
    
    print('%s minus Boson Pt versus %s'%(bosonName,targetvariable))
    #YValues 
    for index in range(0,XRange.shape[0]-1):
	currentDistri = np.empty(shape=[0,1])
	for event in range(0,plotData.shape[0]):
	    if plotData[event,dictPlot[targetvariable]] > XRange[index] and plotData[event,dictPlot[targetvariable]] < XRange[index+1]:
		currentDistri = np.vstack((currentDistri,np.array(((plotData[event,dictPlot[bosonName]])+plotData[event,dictPlot['Boson_Pt']])).reshape(1,1)))
	
	print('current distri shape: ', currentDistri.shape)
	if currentDistri.shape == (0,1):
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	    print('current bin ',index,' entries:',currentDistri.shape[0])	
	    print('current bin ',index,' min:',currentDistri.min())
	    print('current bin ',index,' max:',currentDistri.max())
	    print('current bin ',index,' mean:',currentDistri.mean())
	    print('current bin ',index,' std:',currentDistri.std())
	    YMean[index] = currentDistri.mean()
	    YStd[index] = currentDistri.std()
	
	    if index == 0 or index == 1 or index == 2 or index == 3:
		plt.clf()
		num_bins = 50
		n, bins, patches = plt.hist(currentDistri, num_bins, facecolor='green', alpha=0.5)
		plt.savefig(('../plots/SingleDistributions/Distribution_%sminusBosonPt_vs_s%s_%i.png' %(bosonName[:10],targetvariable, index)))
	    
    plt.clf()
    plt.errorbar(XRange[:-1],YMean[:],yerr=YStd[:],fmt='--o')
    plt.savefig("../plots/%s_minusBosonRecoilvs_%s.png" %(bosonName[:15],targetvariable))
    plt.figure(5)
    plt.errorbar(XRange[:-1],YMean[:],yerr=YStd[:],ls='--',label=bosonName)
    plt.figure(0)
    plt.clf()    
    
    
def make_ULongOverRecoilPlot(plotData,dictPlot, bosonName, targetvariable, minrange=42,maxrange=0, stepwidth=0):
  
    #XRange = np.arange(plotData[:,targetindex].min(),plotData[:,targetindex].max(),(plotData[:,targetindex].max()-plotData[:,targetindex].min())/nbins)
    if minrange == 42:
	minrange = plotData[:,targetindex].min()
    if maxrange == 0:
	maxrange = plotData[:,targetindex].max()
    if stepwidth == 0:
	stepwidth = (maxrange-minrange)/20
	
    XRange = np.arange(minrange,maxrange,stepwidth)
    YMean = np.zeros((XRange.shape[0]-1,1))
    YStd = np.zeros((XRange.shape[0]-1,1))
    print('%s over Boson Pt versus %s'%(bosonName,targetvariable))
    
    #YValues 
    ignoredEntries = 0
    for index in range(0,XRange.shape[0]-1):
	ignoredEntriesLocal = 0
	currentDistri = np.empty(shape=[0,1])
	for event in range(0,plotData.shape[0]):
	    if plotData[event,dictPlot[targetvariable]] > XRange[index] and plotData[event,dictPlot[targetvariable]] < XRange[index+1]:
		#if (-plotData[event,dictPlot[bosonName]]/plotData[event,dictPlot['Boson_Pt']]) > 5:
		#    ignoredEntries += 1
		#    ignoredEntriesLocal += 1
		#else:
		currentDistri = np.vstack((currentDistri,np.array((-plotData[event,dictPlot[bosonName]]/plotData[event,dictPlot['Boson_Pt']])).reshape(1,1)))
		    
	if currentDistri.shape == [0,1]:
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	    print('current bin ',index,' entries:',currentDistri.shape[0])	
	    print('current bin ',index,' min:',currentDistri.min())
	    print('current bin ',index,' max:',currentDistri.max())
	    print('current bin ',index,' mean:',currentDistri.mean())
	    print('current bin ',index,' std:',currentDistri.std())
	    print('current bin ',index,' ignored Entries:',ignoredEntriesLocal)
	    YMean[index] = currentDistri.mean()
	    YStd[index] = currentDistri.std()
	    if index == 0 or index == 1 or index == 2 or index == 3:
		plt.clf()
		num_bins = 50
		n, bins, patches = plt.hist(currentDistri, num_bins, facecolor='green', alpha=0.5)
		plt.savefig(('../plots/SingleDistributions/Distribution_%soverBosonPt_vs_s%s_%i.png' %(bosonName[:10],targetvariable, index)))
	    
    print('Ignored Entries: ', ignoredEntries)
    plt.clf()
    plt.errorbar(XRange[:-1],YMean[:],yerr=YStd[:],fmt='--o')
    plt.figure(4)
    plt.errorbar(XRange[:-1],YMean[:],yerr=YStd[:],ls='--',label=bosonName)
    plt.figure(0)
    plt.savefig("../plots/%s_overBosonRecoilvs_%s.png" %(bosonName[:15],targetvariable))
    plt.clf()
    


def plot_results(plotData, dictPlot, meanPlot, varPlot, dictTarget):
    

    #Transform NNoutput back
    for targetname in dictTarget:
	plotData[:,dictPlot['NNOutput_%s'%targetname]] = plotData[:,dictPlot['NNOutput_%s'%targetname]]*varPlot[dictPlot[targetname]]+meanPlot[dictPlot[targetname]]


    if not os.path.exists('../plots/SingleDistributions'):
	os.makedirs('../plots/SingleDistributions')
	
    
    
    
    #Boson Pt
    #comparisonMinus.xlabel('Boson_Pt')
    #comparisonOver.xlabel('Boson_Pt')
    #comparisonMinus.ylabel('|Boson_Pt-Prediction|')
    #comparisonOver.ylabel('Prediction/Boson_Pt')
    #BDT
    make_ULongOverRecoilPlot(plotData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'Boson_Pt',10,200,10) 
    make_ULongMinusRecoilPlot(plotData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'Boson_Pt',10,200,10)
    
    
    #NN-Net
    if 'NNOutput_Boson_Pt' in dictPlot:
	make_ULongOverRecoilPlot(plotData, dictPlot, 'NNOutput_Boson_Pt', 'Boson_Pt',10,200,10)
	make_ULongMinusRecoilPlot(plotData, dictPlot, 'NNOutput_Boson_Pt', 'Boson_Pt',10,200,10)
	
    
    #Puppi-Met
    make_ULongOverRecoilPlot(plotData, dictPlot, 'recoilslimmedMETs_LongZ', 'Boson_Pt',10,200,10)
    make_ULongMinusRecoilPlot(plotData, dictPlot, 'recoilslimmedMETs_LongZ', 'Boson_Pt',10,200,10)
    
    plt.figure(4)
    plt.savefig('../plots/PredOverBoson_Compare_Boson_Pt')
    plt.clf()
    plt.figure(5)
    plt.savefig('../plots/PredMinusBoson_Compare_Boson_Pt')
    plt.clf
    plt.figure(0)
    
    
    
    #NVertex
    
    #comparisonMinus.xlabel('NVertex')
    #comparisonOver.xlabel('NVertex')
    
    #BDT
    make_ULongOverRecoilPlot(plotData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'NVertex',5,40,5)
    make_ULongMinusRecoilPlot(plotData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'NVertex',5,40,5)
    
    ##NN
    if 'NNOutput_Boson_Pt' in dictPlot:
	make_ULongOverRecoilPlot(plotData, dictPlot, 'NNOutput_Boson_Pt', 'NVertex',5,40,5)
	make_ULongMinusRecoilPlot(plotData, dictPlot, 'NNOutput_Boson_Pt', 'NVertex',5,40,5)
    
    #Puppi-Met
    make_ULongOverRecoilPlot(plotData, dictPlot, 'recoilslimmedMETs_LongZ', 'NVertex',5,40,5)
    make_ULongMinusRecoilPlot(plotData, dictPlot, 'recoilslimmedMETs_LongZ', 'NVertex',5,40,5)
    
    plt.figure(4)
    plt.savefig('../plots/PredOverBoson_Compare_NVertex')
    plt.clf()
    plt.figure(5)
    plt.savefig('../plots/PredMinusBoson_Compare_NVertex')
    plt.clf
    plt.figure(0)
    
    num_bins = 50
    # the histogram of the data
    
    for targetname in dictTarget:
	histDataNN = plotData[:,dictPlot[targetname]] - plotData[:,dictPlot['NNOutput_%s'%targetname]]
	
	n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
	plt.xlabel('Target - NNOutput1')
	plt.ylabel('Hits')
	plt.savefig("../plots/NNVariance_%s.png"%targetname)
	plt.clf()
	
	# normal distribution center at x=0 and y=5
	plt.hist2d(plotData[:,dictPlot[targetname]], histDataNN,bins = 80, norm=LogNorm())
	#plt.ylim([-0.25,0.25])
	plt.xlabel(targetname)
	plt.ylabel('Hits')
	plt.savefig("../plots/NNVariance2D_%s.png"%targetname)
	plt.clf()
    
	
      
    histDataBDT = plotData[:,dictPlot['targetphi']]- plotData[:,dictPlot['PhiCorrectedRecoil_Phi']]
    
    n, bins, patches = plt.hist(histDataBDT, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('Deviation from Result')
    plt.ylabel('Probability')
    plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    plt.savefig("../plots/SumVarianceBDT.png")
    plt.clf()
    
    for targetname in dictTarget:
	histDataNNresponse = plotData[:,dictPlot['NNOutput_%s'%targetname]]
	
	n, bins, patches = plt.hist(histDataNNresponse, num_bins, facecolor='green', alpha=0.5)
	plt.xlabel('NN output')
	plt.ylabel('Hits')
	plt.savefig("../plots/NNresponse_%s.png"%targetname)
	plt.clf()

    
    if 'NNOutput_Boson_Pt' in dictPlot and 'NNOutput_Boson_Phi' in dictPlot:
	histDataULong = plotData[:,dictPlot['recoilslimmedMETs_Pt']]*plotData[:,dictPlot['NNOutput_Boson_Pt']]*np.cos(plotData[:,dictPlot['targetphi']]-plotData[:,dictPlot['NNOutput_Boson_Pt']])

	n, bins, patches = plt.hist(histDataULong, num_bins, facecolor='green', alpha=0.5)
	plt.xlabel('U Longitudinal')
	plt.ylabel('Hits')
	plt.savefig("../plots/U_LongNN.png")
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

    
# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

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
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.1)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.tanh)

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

def main(model='mlp', num_epochs=300):
    
    
    # Load the dataset
    print("Loading data...")
    X_trainMVA, y_trainMVA, X_valMVA, y_valMVA, X_testMVA, y_testMVA, dictTarget, plotData, dictPlot, meanPlot, varPlot = load_datasetcsv()
    
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

    load = True
    
    if load:
	with np.load('model.npz') as f:
	    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
    else:
	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
	    # In each epoch, we do a full pass over the training data:
	    train_err = 0
	    train_batches = 0
	    start_time = time.time()
	    for batch in iterate_minibatches(X_trainMVA, y_trainMVA, 200, shuffle=True):
		inputs, targets = batch
		train_err += train_fn(inputs, targets)
		train_batches += 1

	    # And a full pass over the validation data:
	    val_err = 0
	    val_acc = 0
	    val_batches = 0
	    for batch in iterate_minibatches(X_valMVA, y_valMVA, 200, shuffle=False):
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
	for batch in iterate_minibatches(X_testMVA, y_testMVA, 200, shuffle=False):
	    inputs, targets = batch
	    err = val_fn(inputs, targets)
	    test_err += err
	    #test_acc += acc
	    test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	#print("  test accuracy:\t\t{:.2f} %".format(
	#    test_acc / test_batches * 100))
	#Optionally, you could now dump the network weights to a file like this:
	np.savez('modelBoson_PhiandPt.npz', *lasagne.layers.get_all_param_values(network))
    


    
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
    
    print('testsetpredictionmin: ', testSetPrediction.min())
    print('testsetpredictionmax: ', testSetPrediction.max())
    print('testsettargetmin: ', y_testMVA.min())
    print('testsettargetmax: ', y_testMVA.max())
    print('error: ', val_fn(X_testMVA,(y_testMVA/varPlot[dictPlot['targetphi']])))

    dictPlot['NNOutput2'] = 0
    for variablename in dictTarget:
	dictPlot["NNOutput_%s"%variablename] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(testSetPrediction[:,dictTarget[variablename]]).reshape(testSetPrediction.shape[0],1)))
	
    
    
    print("plotDatashape", plotData.shape)
    plot_results(plotData, dictPlot, meanPlot, varPlot, dictTarget)
   
    
    
    
    
    
    
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
