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
import os.path
import time
#import ROOT

import time
import threading
import csv
import numpy as np
import math
import theano
import theano.tensor as T
import argparse
import lasagne
import json

import matplotlib.mlab as mlab
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

def writeResult(config, resultData, dictResult):
  
    addEntry = True
    

    if os.path.exists(config['resultFile']):
	reader=csv.reader(open(config['resultFile'],"rb"),delimiter=',')
	datacsv=list(reader)
	print('datacsv: ',len(datacsv))
    
	for row in datacsv:
	    if row[0] == config['inputFile'][18:-4]:
		addEntry = False
    
	variablescsv = datacsv[0]
	
    else:
	with open(config['resultFile'], 'wb') as f:
	    writer = csv.writer(f)
	    indexrow = [['Trainingname'],[name for name in dictResult]]
	    writer.writerow(['Trainingname'] + [name for name in dictResult])
	    variablescsv = ['Trainingname']
	    for name in dictResult:
		variablescsv.append(name)
		
	    print(variablescsv)
	    for name in variablescsv:
		print(name)
		
	    for index in range(0,len(variablescsv)):
		print(variablescsv[index])
	#for name in dictResult:
	
    print('variablescsv len: ',len(variablescsv))
    
    if addEntry:
	print('Adding new file!')
	with open(config['resultFile'], 'a') as f:
	    writer = csv.writer(f)
	    writer.writerow([config['inputFile'][18:-4]] + [resultData[0,dictResult[variablescsv[index]]] for index in range(1,len(variablescsv))] )

    """
    for row in datacsv:
        print(row[0])
        resultwriter.writerow(['Spam'] * 2 + ['Baked Beans'])
    """
    #inputdatentot =np.array(datacsv[1:]).astype(np.float32)
    #print('names', names.shape)
    """
    with open(config['resultFile'], 'wb') as csvfile:
	resultwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for index in range(5):
	    resultwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    """


def handleFlatPtWeight(config, inputData, dictInput):
  
    trainingconfig = config[config['activeTraining']]

    

    if 'flatPtWeight' in dictInput and trainingconfig['useWeightsFromBDT']:
	weights = np.empty(shape=[inputData.shape[0],0]).astype(np.float32)
	if trainingconfig['adjustDimWeight']:
	    for i in range(0,trainingconfig['dimWeightIfAdjust']):
		weights = np.hstack((weights,np.array(inputData[:,dictInput['flatPtWeight']]).reshape(inputData.shape[0],1)))
	else:
	    for i in range(0,len(trainingconfig['targetVariables'])):
		weights = np.hstack((weights,np.array(inputData[:,dictInput['flatPtWeight']]).reshape(inputData.shape[0],1)))
		
    elif trainingconfig['createFlatWeights']:
	print('creating flat weights..')
	weights = np.empty(shape=[0,1]).astype(np.float32)
	
	minPt = 0
	counter = 0
	sortedData = inputData[inputData[:,dictInput['Boson_Pt']].argsort()]
	binSize = trainingconfig['EntriesPerBin']
	
	while counter < sortedData.shape[0] - binSize and sortedData[counter+binSize,dictInput['Boson_Pt']] < trainingconfig['maxPtForFlatWeights']:
	    counter += binSize
	    width = sortedData[counter,dictInput['Boson_Pt']]-minPt
	    weights = np.vstack((weights, (np.ones(binSize)*1000*width/binSize).reshape(binSize,1).astype(np.float32)))
	    minPt = sortedData[counter,dictInput['Boson_Pt']]
	    #print (counter, ' : ', minPt,'  -  ', width/binSize)
	
	if counter < sortedData.shape[0]:
	    weights = np.vstack((weights, (np.zeros(sortedData.shape[0]-counter)).reshape((sortedData.shape[0]-counter),1).astype(np.float32)))
	    
	print('sortedData: ', sortedData.shape)
	print('inputData: ', inputData.shape)
	inputData = sortedData
	
	if trainingconfig['adjustDimWeight']:
	    for i in range(1,trainingconfig['dimWeightIfAdjust']):
		weights = np.hstack((weights,weights.reshape(inputData.shape[0],1)))
	else:
	    for i in range(1,len(trainingconfig['targetVariables'])):
		weights = np.hstack((weights,weights[:,0].reshape(weights.shape[0],1)))
    else:
	weights = np.empty(shape=[inputData.shape[0],0]).astype(np.float32)
	if trainingconfig['adjustDimWeight']:
	    weights = np.hstack((weights,np.ones((inputData.shape[0],trainingconfig['dimWeightIfAdjust'])).reshape(inputData.shape[0],trainingconfig['dimWeightIfAdjust']).astype(np.float32)))
	else:
	    weights = np.hstack((weights,np.ones((inputData.shape[0],len(trainingconfig['targetVariables']))).reshape(inputData.shape[0],len(trainingconfig['targetVariables'])).astype(np.float32)))
	    print('Could not find flat Pt weights nor creating them, initializing with equal weight')
	
    if 'flatPtWeight' in dictInput and trainingconfig['createFlatWeights']:
	print('creating flatPtWeightBDT dict Entry')
	dictInput['flatPtWeightBDT'] = dictInput['flatPtWeight']
	dictInput['flatPtWeight'] = inputData.shape[1]
	inputData = np.hstack((inputData, weights[:,0].reshape(weights.shape[0],1)))
	
    if not 'flatPtWeight' in dictInput:
	dictInput['flatPtWeight'] = inputData.shape[1]
	inputData = np.hstack((inputData, weights[:,0].reshape(weights.shape[0],1)))
	
    
    print('weightsshape begin:', weights.shape)
    
    return inputData, dictInput, weights

def load_datasetcsv(config):
    #create Treevariable
    
    start = time.time()
    
    trainingconfig = config[config['activeTraining']]
    

    reader=csv.reader(open(config['inputFile'],"rb"),delimiter=',')
    datacsv=list(reader)
    header = np.array(datacsv[0]).astype(np.str)
    inputdatentot =np.array(datacsv[1:]).astype(np.float32)
    
    dictInputTot = {}
    for index in range(0,header.shape[0]):
	dictInputTot[header[index]] = index
    
    print('datentot before: ',inputdatentot.shape)
    print('dictInput length before: ', len(dictInputTot))
    
    print('shape before cut: ', inputdatentot.shape[0])
    #inputdatentot = inputdatentot[inputdatentot[:,dictInputTot['select']] == 2]
    print('shape after cut: ', inputdatentot.shape[0])
    
    inputdatentot, dictInputTot, weights = handleFlatPtWeight(config, inputdatentot, dictInputTot)
  

    print('datentot after: ',inputdatentot.shape)
    print('dictInput length after:', len(dictInputTot))
    #print(header.shape)
    #print(inputdatentot.shape)
    

    
    stupidData = inputdatentot[inputdatentot[:,dictInputTot['targetRecoilFromBDT']]>1000]
    stupidData = np.vstack((stupidData,inputdatentot[inputdatentot[:,dictInputTot['targetRecoilFromBDT']]<-1000]))
    print('stupidData shape: ',stupidData.shape)
    #inputdatentot = stupidData
    #Print out mean, var, min and max of Variables which var != 0
    meanwhole = inputdatentot.mean(axis = 0)
    varwhole = inputdatentot.std(axis = 0)
    minwhole = inputdatentot.min(axis = 0)
    maxwhole = inputdatentot.max(axis = 0)
    for index in range(0,header.shape[0]):
	if header[index] in config[config['activePlotting']]['plotVariables']:
	    print(header[index])
	    print("mean: ",meanwhole[index])
	    print("var: ",varwhole[index])
	    print("min:", minwhole[index])
	    print("max:", maxwhole[index])

    
    #for index in range(header.shape[0]):
	#print('stupidPoint ',header[index],' - ',stupidData[0,index]) 
    
    
    
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
    print('inputdatenshape: ', inputdatentot.shape)    

    
    
    
    inputnames = trainingconfig['trainingVariables']
    """
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
    """
    
    #targetnames = {'targetPhiFromSlimmed'}
    #targetnames = {'targetPhiFromSlimmed','targetRecoilFromSlimmed'}
    #targetnames = {'Boson_Phi'}
    targetnames = trainingconfig['targetVariables']
    
    plotnames = config[config['activePlotting']]['plotVariables']
    #plotnames = {"targetPhiFromSlimmed","targetRecoilFromSlimmed","targetRecoilFromBDT", "MVAMET_Phi", "Boson_Phi", "Boson_Pt","dmvamet_Phi", "dmvamet_Pt", "dpfmet_Pt", "dpfmet_Phi", "MVAMET_Pt", "recoilslimmedMETsPuppi_Pt", "recoilslimmedMETsPuppi_Phi","recoilslimmedMETsPuppi_LongZ","MVAMET_sumEt","recoilslimmedMETs_Pt", "recoilslimmedMETs_LongZ", "recoilslimmedMETs_Phi", "PhiTrainingResponse", "RecoilTrainingResponse", "PhiCorrectedRecoil_Pt", "PhiCorrectedRecoil_LongZ", "PhiCorrectedRecoil_PerpZ", "PhiCorrectedRecoil_Phi", "PhiCorrectedRecoil_MET", "PhiCorrectedRecoil_METPhi", "LongZCorrectedRecoil_Phi", "LongZCorrectedRecoil_LongZ", "NVertex"}
    
    
    
    #inputdatentot = np.hstack((inputdatentot,np.array(inputdatentot[:,bosonPt]/inputdatentot[:,METT1_Pt]).reshape(inputdatentot.shape[0],1)))
    

    inputDataX = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    inputDataY = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    inputDataPlot = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    
    
    dictInputX = {}
    dictInputY = {}
    dictPlot = {}




    dt = int((time.time() - start))
    print('Elapsed time for loading dataset: ', dt)
    
    
    for index, entry in enumerate(dictInputTot):
	if entry in inputnames:
	    dictInputX[entry] = inputDataX.shape[1]
	    inputDataX = np.hstack((inputDataX, np.array(inputdatentot[:,dictInputTot[entry]]).reshape(inputdatentot.shape[0],1)))
		
	if entry in targetnames:
	    dictInputY[entry] = inputDataY.shape[1]
	    inputDataY = np.hstack((inputDataY, np.array(inputdatentot[:,dictInputTot[entry]]).reshape(inputdatentot.shape[0],1)))

	if entry in plotnames:
	    dictPlot[entry] = inputDataPlot.shape[1]
	    inputDataPlot = np.hstack((inputDataPlot, np.array(inputdatentot[:,dictInputTot[entry]]).reshape(inputdatentot.shape[0],1)))

    

    
    print("xshape ", inputDataX.shape)
    print("yshape ", inputDataY.shape)
    

    for name in dictInputY:
	print('Target before norm %s:'%name,' mean: ', inputDataY[:,dictInputY[name]].mean())
	print('Target before norm %s:'%name,' std: ', inputDataY[:,dictInputY[name]].std())
	print('Target before norm %s:'%name,' min: ', inputDataY[:,dictInputY[name]].min())
	print('Target before norm %s:'%name,' max: ', inputDataY[:,dictInputY[name]].max())
 

    if config['performTraining']:
	x_train = np.empty(shape=[0,inputDataX.shape[1]]).astype(np.float32)
	trainweights = np.empty(shape=[0,weights.shape[1]]).astype(np.float32)
	x_val = np.empty(shape=[0,inputDataX.shape[1]]).astype(np.float32)
	x_test = np.empty(shape=[0,inputDataX.shape[1]]).astype(np.float32)
	y_train = np.empty(shape=[0,inputDataY.shape[1]]).astype(np.float32)
	y_val = np.empty(shape=[0,inputDataY.shape[1]]).astype(np.float32)
	valweights = np.empty(shape=[0,weights.shape[1]]).astype(np.float32)
	y_test = np.empty(shape=[0,inputDataY.shape[1]]).astype(np.float32)
	testweights = np.empty(shape=[0,weights.shape[1]]).astype(np.float32)
	inputDataPlotShuffled = np.empty(shape=[0,inputDataPlot.shape[1]]).astype(np.float32)
	
	print('Split and shuffle data for Training/Validation/Test:')
	countEvents = 0
	batchsize = inputDataX.shape[0]/1000
	print('batchsize: ', batchsize)
	for batch in iterate_minibatchesInput(inputDataX, inputDataY, inputDataPlot, weights, batchsize, shuffle=True):
	    x_TestTemp, y_TestTemp, plot_Temp, weightstemp = batch
	    
	    inputDataPlotShuffled = np.vstack((inputDataPlotShuffled,np.array(plot_Temp)))
	    
	    if countEvents < inputDataX.shape[0]*3/6:
		x_train = np.vstack((x_train, np.array(x_TestTemp)))
		trainweights = np.vstack((trainweights, np.array(weightstemp)))
		y_train = np.vstack((y_train, np.array(y_TestTemp)))
	    if countEvents >= inputDataX.shape[0]*3/6 and countEvents <= inputDataX.shape[0]*4/6:
		x_val = np.vstack((x_val, np.array(x_TestTemp)))
		y_val = np.vstack((y_val, np.array(y_TestTemp)))
		valweights = np.vstack((valweights, np.array(weightstemp)))
	    if countEvents > inputDataX.shape[0]*4/6:
		x_test = np.vstack((x_test, np.array(x_TestTemp)))
		y_test = np.vstack((y_test, np.array(y_TestTemp)))
		testweights = np.vstack((testweights, np.array(weightstemp)))
	    
	    countEvents += batchsize
	    if countEvents % (batchsize*100) == 0:
		print('processed Events: ', countEvents, '/', inputDataX.shape[0])
    else:
	x_train = inputDataX[:inputDataX.shape[0]*3/6,:]
	y_train = inputDataY[:inputDataX.shape[0]*3/6,:]
	trainweights = weights[:inputDataX.shape[0]*3/6,:]
	x_val = inputDataX[inputDataX.shape[0]*3/6:inputDataX.shape[0]*5/6,:]
	y_val = inputDataY[inputDataX.shape[0]*3/6:inputDataX.shape[0]*5/6,:]
	valweights = weights[inputDataX.shape[0]*3/6:inputDataX.shape[0]*5/6,:]
	x_test = inputDataX[-inputDataX.shape[0]*1/6:,:]
	y_test = inputDataY[-inputDataX.shape[0]*1/6:,:]
	testweights = weights[-inputDataX.shape[0]*1/6:,:]
	inputDataPlotShuffled = inputDataPlot
	
    
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(x_test.shape)
    print(inputDataPlotShuffled.shape)
    
    meanSelectedX = x_train.mean(axis=0)
    stdSelectedX = x_train.std(axis = 0)

    meanSelectedY = y_train.mean(axis=0)
    stdSelectedY = y_train.std(axis = 0)

    x_train = (x_train - meanSelectedX) / stdSelectedX
    x_val = (x_val - meanSelectedX) / stdSelectedX
    x_test = (x_test - meanSelectedX) / stdSelectedX
    
    y_train = (y_train - meanSelectedY) / stdSelectedY
    y_val = (y_val - meanSelectedY) / stdSelectedY
    y_test = (y_test - meanSelectedY) / stdSelectedY
    
    dt = int((time.time() - start))
    print('Elapsed time for loading whole data: ', dt)

    print('weights: ', trainweights.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, trainweights, valweights, testweights, dictInputY, inputDataPlotShuffled, dictPlot, meanSelectedY, stdSelectedY
    
    
    
def make_Plot(variablename, inputData, dictPlot, outputdir):

    histData = inputData[:,dictPlot[variablename]]
    
    if not os.path.exists(outputdir):
	os.makedirs(outputdir)
	
    
    
    num_bins = 100
    
    if variablename == 'targetRecoilFromBDT' or variablename == 'targetRecoilFromSlimmed':
	n, bins, patches = plt.hist(histData, num_bins, facecolor='green', alpha=0.5, range=[-50, 50])
    else:
	n, bins, patches = plt.hist(histData, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel(variablename)
    plt.ylabel('Hits')

	
    plt.savefig((outputdir+variablename+".svg"))
    plt.clf()
    return 0
    
    
def make_ResponseCorrectedPlot(config, XRange, YStd, YResponse, bosonName, targetvariable, resultData, dictResult, minrange,maxrange, stepwidth, ptmin,ptmax):

    plt.clf()
    ResCorr = YStd[:]/YResponse[:]
    plt.plot(XRange[:-1]+stepwidth/2.,ResCorr,'o')
    plt.xlabel(targetvariable)
    plt.ylabel('Resolution / Response')
    if ptmax == 0:
	  plt.savefig(config['outputDir'] + "ControlPlots/ResponseCorrected_%s_vs_%s.svg" %(bosonName,targetvariable))
    else:
	  plt.savefig(config['outputDir'] + "ControlPlots/ResponseCorrected(%i<Pt<%i)_%s_vs_%s.svg" %(ptmin,ptmax,bosonName,targetvariable))
    plt.figure(6)
    plt.plot(XRange[:-1]+stepwidth/2.,ResCorr,'o',label=bosonName)
    plt.figure(0)
    plt.clf()    
    
    
    if bosonName == 'LongZCorrectedRecoil_LongZ':
    #if bosonName == 'recoilslimmedMETs_LongZ':
	if ptmax > config[config['activePlotting']]['BosonCut']:
	    Upper = 'Max'
	else:
	    Upper = 'Cut'
	    
	if ptmin < config[config['activePlotting']]['BosonCut']:
	    Lower = 'Min'
	else:
	    Lower = 'Cut'
      
	if targetvariable == 'Boson_Pt':
	    resultvalue = ResCorr[np.isfinite(ResCorr)].min()
	    dictResult["ResCorr_Min_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	    resultvalue = ResCorr[np.isfinite(ResCorr)].sum()/ResCorr[np.isfinite(ResCorr)].shape[0]
	    dictResult["ResCorr_Int_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	    resultvalue = ResCorr[np.isfinite(ResCorr)].max()
	    dictResult["ResCorr_Max_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	elif targetvariable == 'NVertex':

	    resultvalue = ResCorr[np.isfinite(ResCorr)].min()
	    dictResult["ResCorr_Min_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    
	    resultvalue = ResCorr[np.isfinite(ResCorr)].sum()/ResCorr[np.isfinite(ResCorr)].shape[0]
	    dictResult["ResCorr_Int_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	
	    resultvalue = ResCorr[np.isfinite(ResCorr)].max()
	    dictResult["ResCorr_Max_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
    
    
    
    return resultData, dictResult
    
def make_ResolutionPlot(config,plotData,dictPlot, bosonName, targetvariable, resultData, dictResult, minrange=42,maxrange=0, stepwidth=0, ptmin =0,ptmax=0):


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
    
    print('Resolution %s versus %s'%(bosonName,targetvariable))
    #YValues 
    for index in range(0,XRange.shape[0]-1):

	AlternativeDistri = plotData[(XRange[index]<plotData[:,dictPlot[targetvariable]]) & (XRange[index+1]>plotData[:,dictPlot[targetvariable]])]
	currentDistri = AlternativeDistri[:,dictPlot[bosonName]]+AlternativeDistri[:,dictPlot['Boson_Pt']]

	if currentDistri.shape == (0,1):
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	    """
	    print('current bin ',index,' entries:',currentDistri.shape[0])	
	    print('current bin ',index,' min:',currentDistri.min())
	    print('current bin ',index,' max:',currentDistri.max())
	    print('current bin ',index,' mean:',currentDistri.mean())
	    print('current bin ',index,' std:',currentDistri.std())
	    """
	    YMean[index] = currentDistri.mean()
	    YStd[index] = currentDistri.std()
	
	    if index < 12:
		plt.clf()
		num_bins = 50
		n, bins, patches = plt.hist(currentDistri, num_bins, normed=1, facecolor='green', alpha=0.5)
		y = mlab.normpdf(bins, currentDistri.mean(), currentDistri.std())
		plt.xlabel('%s at %f, mean: %f'%(targetvariable,(XRange[index+1]+XRange[index])/2,currentDistri.mean()))
		plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
		plt.plot(bins, y, 'r--')
		if ptmax == 0:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Resolution_%s_vs_%s_%i.svg' %(bosonName,targetvariable, index)))
		else:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Resolution(%i<Pt<%i)_%s_vs_%s_%i.svg' %(ptmin,ptmax,bosonName,targetvariable, index)))
		    
    plt.clf()
    plt.plot(XRange[:-1]+stepwidth/2.,YStd[:],'o')
    plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
    plt.xlabel(targetvariable)
    if ptmax == 0:
	plt.savefig(config['outputDir'] + "ControlPlots/Resolution_%s_vs_%s.svg" %(bosonName,targetvariable))
    else:
	plt.savefig(config['outputDir'] + "ControlPlots/Resolution(%i<Pt<%i)_%s_vs_%s.svg" %(ptmin,ptmax,bosonName,targetvariable))
    plt.figure(5)
    plt.plot(XRange[:-1]+stepwidth/2.,YStd[:],'o',label=bosonName)
    plt.figure(0)
    plt.clf()
    
    
    if bosonName == 'LongZCorrectedRecoil_LongZ':
    #if bosonName == 'recoilslimmedMETs_LongZ':
	if ptmax > config[config['activePlotting']]['BosonCut']:
	    Upper = 'Max'
	else:
	    Upper = 'Cut'
	    
	if ptmin < config[config['activePlotting']]['BosonCut']:
	    Lower = 'Min'
	else:
	    Lower = 'Cut'
      
	if targetvariable == 'Boson_Pt':
	    resultvalue = YStd[np.isfinite(YStd)].min()
	    dictResult["Resolution_Min_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	    resultvalue = YStd[np.isfinite(YStd)].sum()/YStd[np.isfinite(YStd)].shape[0]
	    dictResult["Resolution_Int_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	elif targetvariable == 'NVertex':

	    resultvalue = YStd[np.isfinite(YStd)].min()
	    dictResult["Resolution_Min_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    
	    resultvalue = YStd[np.isfinite(YStd)].sum()/YStd[np.isfinite(YStd)].shape[0]
	    dictResult["Resolution_Int_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))

    
    
    
    return resultData, dictResult, XRange, YStd
    
def make_ResponsePlot(config, plotData,dictPlot, bosonName, targetvariable, resultData, dictResult, minrange=42,maxrange=0, stepwidth=0, ptmin=0, ptmax=0):
  
    #XRange = np.arange(plotData[:,targetindex].min(),plotData[:,targetindex].max(),(plotData[:,targetindex].max()-plotData[:,targetindex].min())/nbins)
    if minrange == 42:
	minrange = plotData[:,dictPlot[targetvariable]].min()
    if maxrange == 0:
	maxrange = plotData[:,dictPlot[targetvariable]].max()
    if stepwidth == 0:
	stepwidth = (maxrange-minrange)/20
	
    XRange = np.arange(minrange,maxrange,stepwidth)
    YMean = np.zeros((XRange.shape[0]-1,1))
    YStd = np.zeros((XRange.shape[0]-1,1))
    print('Response %s versus %s'%(bosonName,targetvariable))
    
    #YValues 
    ignoredEntries = 0
    for index in range(0,XRange.shape[0]-1):

	AlternativeDistri = plotData[(XRange[index]<plotData[:,dictPlot[targetvariable]]) & (XRange[index+1]>plotData[:,dictPlot[targetvariable]])]

	currentDistri = -AlternativeDistri[:,dictPlot[bosonName]]/AlternativeDistri[:,dictPlot['Boson_Pt']]
	"""
	if AlternativeDistri.shape[0] == 0:
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	"""
	YMean[index] = currentDistri.mean()
	YStd[index] = currentDistri.std()
	if index < 12:
	    plt.clf()
            num_bins = 50
	    n, bins, patches = plt.hist(currentDistri, num_bins, normed=1, facecolor='green', alpha=0.5)
	    y = mlab.normpdf(bins, currentDistri.mean(), currentDistri.std())
	    plt.xlabel('%s at %f, mean: %f'%(targetvariable,(XRange[index+1]+XRange[index])/2,currentDistri.mean()))
	    plt.ylabel('(MET Boson PT_Long)/(True Boson Pt)')
	    plt.plot(bins, y, 'r--')
	    if ptmax == 0:
		plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Response_%s_vs_%s_%i.svg' %(bosonName,targetvariable, index)))
	    else:
		plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Response(%i<Pt<%i)_%s_vs_%s_%i.svg' %(ptmin,ptmax,bosonName,targetvariable, index)))
    plt.clf()
    plt.plot(XRange[:-1]+stepwidth/2.,YMean[:],'o')

    plt.xlabel(targetvariable)
    if ptmax == 0:
	plt.savefig(config['outputDir'] + "ControlPlots/Response_%s_vs_%s.svg" %(bosonName,targetvariable))
    else:
	plt.savefig(config['outputDir'] + "ControlPlots/Response(%i<Pt<%i)_%s_vs_%s.svg" %(ptmin,ptmax,bosonName,targetvariable))
    plt.clf()
    plt.figure(4)
    plt.plot(XRange[:-1]+stepwidth/2.,YMean[:],'o',label=bosonName)
    plt.figure(0)
    

    if bosonName == 'LongZCorrectedRecoil_LongZ':
    #if bosonName == 'recoilslimmedMETs_LongZ':
	if ptmax > config[config['activePlotting']]['BosonCut']:
	    Upper = 'Max'
	else:
	    Upper = 'Cut'
	    
	if ptmin < config[config['activePlotting']]['BosonCut']:
	    Lower = 'Min'
	else:
	    Lower = 'Cut'
      
      
	if targetvariable == 'Boson_Pt':
	    resultvalue = ((YMean[np.isfinite(YMean)]-1)**2).sum()/YMean[np.isfinite(YMean)].shape[0]
	    dictResult["Response_Chi_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	    
	    YMeanCut = YMean[config[config['activePlotting']]['BosonCut']<XRangeTemp[:]]
	    print('shape cut:', YMeanCut.shape)
	    print('shape whole:', YMean.shape)
	    print('shape finite:', YMean[np.isfinite(YMean)].shape)
	    resultvalue = ((YMeanCut[np.isfinite(YMeanCut)]-1)**2).sum()/YMeanCut[np.isfinite(YMeanCut)].shape[0]
	    dictResult["Response_Chi_%s_CuttoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	elif targetvariable == 'NVertex':
	    
	    resultvalue = ((YMean[np.isfinite(YMean)]-1)**2).sum()/YMean[np.isfinite(YMean)].shape[0]
	    dictResult["Response_Chi_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
    
    return resultData, dictResult, YMean

    
def make_ResolutionPerpPlot(config,plotData,dictPlot, bosonName, targetvariable, resultData, dictResult, minrange=42,maxrange=0, stepwidth=0, ptmin =0,ptmax=0):


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
    
    print('Resolution Perp %s versus %s'%(bosonName,targetvariable))
    #YValues 
    for index in range(0,XRange.shape[0]-1):

	AlternativeDistri = plotData[(XRange[index]<plotData[:,dictPlot[targetvariable]]) & (XRange[index+1]>plotData[:,dictPlot[targetvariable]])]
	currentDistri = AlternativeDistri[:,dictPlot[bosonName]]

	if currentDistri.shape == (0,1):
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	    """
	    print('current bin ',index,' entries:',currentDistri.shape[0])	
	    print('current bin ',index,' min:',currentDistri.min())
	    print('current bin ',index,' max:',currentDistri.max())
	    print('current bin ',index,' mean:',currentDistri.mean())
	    print('current bin ',index,' std:',currentDistri.std())
	    """
	    YMean[index] = currentDistri.mean()
	    YStd[index] = currentDistri.std()
	
	    if index < 12:
		plt.clf()
		num_bins = 50
		n, bins, patches = plt.hist(currentDistri, num_bins, normed=1, facecolor='green', alpha=0.5)
		y = mlab.normpdf(bins, currentDistri.mean(), currentDistri.std())
		plt.xlabel('%s at %f, mean: %f'%(targetvariable,(XRange[index+1]+XRange[index])/2,currentDistri.mean()))
		plt.ylabel('MET Boson PT_Perp')
		plt.plot(bins, y, 'r--')
		if ptmax == 0:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/ResolutionPerp_%s_vs_%s_%i.svg' %(bosonName,targetvariable, index)))
		else:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/ResolutionPerp(%i<Pt<%i)_%s_vs_%s_%i.svg' %(ptmin,ptmax,bosonName,targetvariable, index)))
		    
    plt.clf()
    plt.plot(XRange[:-1]+stepwidth/2.,YStd[:],'o')
    plt.ylabel('MET Boson PT_Perp')
    plt.xlabel(targetvariable)
    if ptmax == 0:
	plt.savefig(config['outputDir'] + "ControlPlots/ResolutionPerp_%s_vs_%s.svg" %(bosonName,targetvariable))
    else:
	plt.savefig(config['outputDir'] + "ControlPlots/ResolutionPerp(%i<Pt<%i)_%s_vs_%s.svg" %(ptmin,ptmax,bosonName,targetvariable))
    plt.figure(8)
    plt.plot(XRange[:-1]+stepwidth/2.,YStd[:],'o',label=bosonName)
    plt.figure(0)
    plt.clf()
    

    if bosonName == 'LongZCorrectedRecoil_PerpZ':
    #if bosonName == 'recoilslimmedMETs_LongZ':
	if ptmax > config[config['activePlotting']]['BosonCut']:
	    Upper = 'Max'
	else:
	    Upper = 'Cut'
	    
	if ptmin < config[config['activePlotting']]['BosonCut']:
	    Lower = 'Min'
	else:
	    Lower = 'Cut'
      
	if targetvariable == 'Boson_Pt':
	    resultvalue = YStd[np.isfinite(YStd)].min()
	    dictResult["ResolutionPerp_Min_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	    resultvalue = YStd[np.isfinite(YStd)].sum()/YStd[np.isfinite(YStd)].shape[0]
	    dictResult["ResolutionPerp_Int_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	elif targetvariable == 'NVertex':

	    resultvalue = YStd[np.isfinite(YStd)].min()
	    dictResult["ResolutionPerp_Min_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    
	    resultvalue = YStd[np.isfinite(YStd)].sum()/YStd[np.isfinite(YStd)].shape[0]
	    dictResult["ResolutionPerp_Int_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))

    
    
    
    return resultData, dictResult
    
def make_ResponsePerpPlot(config, plotData,dictPlot, bosonName, targetvariable, resultData, dictResult, minrange=42,maxrange=0, stepwidth=0, ptmin=0, ptmax=0):
  
    #XRange = np.arange(plotData[:,targetindex].min(),plotData[:,targetindex].max(),(plotData[:,targetindex].max()-plotData[:,targetindex].min())/nbins)
    if minrange == 42:
	minrange = plotData[:,dictPlot[targetvariable]].min()
    if maxrange == 0:
	maxrange = plotData[:,dictPlot[targetvariable]].max()
    if stepwidth == 0:
	stepwidth = (maxrange-minrange)/20
	
    XRange = np.arange(minrange,maxrange,stepwidth)
    YMean = np.zeros((XRange.shape[0]-1,1))
    YStd = np.zeros((XRange.shape[0]-1,1))
    print('Response Perp %s versus %s'%(bosonName,targetvariable))
    
    #YValues 
    ignoredEntries = 0
    for index in range(0,XRange.shape[0]-1):

	AlternativeDistri = plotData[(XRange[index]<plotData[:,dictPlot[targetvariable]]) & (XRange[index+1]>plotData[:,dictPlot[targetvariable]])]

	currentDistri = AlternativeDistri[:,dictPlot[bosonName]]
	"""
	if AlternativeDistri.shape[0] == 0:
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	"""
	YMean[index] = currentDistri.mean()
	YStd[index] = currentDistri.std()
	if index < 12:
	    plt.clf()
            num_bins = 50
	    n, bins, patches = plt.hist(currentDistri, num_bins, normed=1, facecolor='green', alpha=0.5)
	    y = mlab.normpdf(bins, currentDistri.mean(), currentDistri.std())
	    plt.xlabel('%s at %f, mean: %f'%(targetvariable,(XRange[index+1]+XRange[index])/2,currentDistri.mean()))
	    plt.ylabel('MET Boson PT_Perp')
	    plt.plot(bins, y, 'r--')
	    if ptmax == 0:
		plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/ResponsePerp_%s_vs_%s_%i.svg' %(bosonName,targetvariable, index)))
	    else:
		plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/ResponsePerp(%i<Pt<%i)_%s_vs_%s_%i.svg' %(ptmin,ptmax,bosonName,targetvariable, index)))
    plt.clf()
    plt.plot(XRange[:-1]+stepwidth/2.,YMean[:],'o')

    plt.xlabel(targetvariable)
    if ptmax == 0:
	plt.savefig(config['outputDir'] + "ControlPlots/ResponsePerp_%s_vs_%s.svg" %(bosonName,targetvariable))
    else:
	plt.savefig(config['outputDir'] + "ControlPlots/ResponsePerp(%i<Pt<%i)_%s_vs_%s.svg" %(ptmin,ptmax,bosonName,targetvariable))
    plt.clf()
    plt.figure(7)
    plt.plot(XRange[:-1]+stepwidth/2.,YMean[:],'o',label=bosonName)
    plt.figure(0)
    

    if bosonName == 'LongZCorrectedRecoil_PerpZ':
    #if bosonName == 'recoilslimmedMETs_LongZ':
	if ptmax > config[config['activePlotting']]['BosonCut']:
	    Upper = 'Max'
	else:
	    Upper = 'Cut'
	    
	if ptmin < config[config['activePlotting']]['BosonCut']:
	    Lower = 'Min'
	else:
	    Lower = 'Cut'
      
      
	if targetvariable == 'Boson_Pt':
	    resultvalue = ((YMean[np.isfinite(YMean)]-1)**2).sum()/YMean[np.isfinite(YMean)].shape[0]
	    dictResult["ResponsePerp_Chi_%s_MintoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	    XRangeTemp = XRange[:-1]
	    
	    
	    YMeanCut = YMean[config[config['activePlotting']]['BosonCut']<XRangeTemp[:]]
	    resultvalue = ((YMeanCut[np.isfinite(YMeanCut)]-1)**2).sum()/YMeanCut[np.isfinite(YMeanCut)].shape[0]
	    dictResult["ResponsePerp_Chi_%s_CuttoMax"%(targetvariable)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
	elif targetvariable == 'NVertex':
	    
	    resultvalue = ((YMean[np.isfinite(YMean)]-1)**2).sum()/YMean[np.isfinite(YMean)].shape[0]
	    dictResult["ResponsePerp_Chi_%s_%sto%s"%(targetvariable,Lower,Upper)] = resultData.shape[1]
	    resultData = np.hstack((resultData, np.array(resultvalue.reshape(1,1))))
    
    return resultData, dictResult

    

def make_ControlPlots(config, plotData,dictPlot, bosonName, targetvariable, resultData, dictResult, minrange=42,maxrange=0, stepwidth=0, ptmin=0,ptmax=0):

    bosonNameLong = bosonName + '_LongZ'
    bosonNamePerp = bosonName + '_PerpZ'
    maxrange += stepwidth
    if not os.path.exists((config['outputDir'] + 'ControlPlots/SingleDistributions/')):
	os.makedirs((config['outputDir'] + 'ControlPlots/SingleDistributions/'))
    resultData, dictResult, XRange, YVariance = make_ResolutionPlot(config, plotData, dictPlot, bosonNameLong, targetvariable, resultData, dictResult, minrange,maxrange,stepwidth, ptmin, ptmax) 
    resultData, dictResult, YResponse = make_ResponsePlot(config, plotData, dictPlot, bosonNameLong, targetvariable, resultData, dictResult, minrange,maxrange,stepwidth, ptmin, ptmax)
    resultData, dictResult = make_ResponseCorrectedPlot(config, XRange, YVariance, YResponse, bosonNameLong, targetvariable, resultData, dictResult, minrange,maxrange, stepwidth, ptmin, ptmax)
    resultData, dictResult = make_ResolutionPerpPlot(config, plotData, dictPlot, bosonNamePerp, targetvariable, resultData, dictResult, minrange,maxrange,stepwidth, ptmin, ptmax) 
    resultData, dictResult = make_ResponsePerpPlot(config, plotData, dictPlot, bosonNamePerp, targetvariable, resultData, dictResult, minrange,maxrange,stepwidth, ptmin, ptmax)
    return resultData, dictResult

    
def make_MoreBDTPlots(config, plotData, dictPlot):
    
    plt.clf()
    plt.hist2d(plotData[:,dictPlot['Boson_Pt']], plotData[:,dictPlot['flatPtWeight']],bins = 80, norm=LogNorm())
    plt.xlabel('Boson Pt')
    plt.ylabel('flat Pt weight')
    plt.savefig(config['outputDir'] + "/ControlPlots/BosonOverWeights.svg")
    
    num_bins = 50
    
    plt.clf()
    if 'fileName' in dictPlot:
	for i in range (0,9):
	    if i in plotData[:,dictPlot['fileName']]:
		currentDistri = plotData[i==plotData[:,dictPlot['fileName']],dictPlot['fileName']]
		n, bins, patches = plt.hist(currentDistri, num_bins, range=[0,plotData[:,dictPlot['Boson_Pt']].max()], alpha=0.5, label=('File %i'%i))
	plt.xlabel('Boson Pt')
	plt.ylabel('Entries')
	plt.savefig(config['outputDir'] + "/ControlPlots/BosonPtSpectrum.svg")
	plt.clf()
      
    histBosonPtLow = plotData[plotData[:,dictPlot['Boson_Pt']]<30,dictPlot['Boson_Pt']]
    
    n, bins, patches = plt.hist(histBosonPtLow, num_bins, range=[0,30],alpha=0.5)
    plt.xlabel('Boson Pt')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/LowBosonPtSpectrum.svg")
    plt.clf()

    borders = [0,10,20,30,40,50,60,70,80,90,100]
    for index,min in enumerate(borders):
	histname = 'targetRecoilFromBDT'
	slicedData = plotData[min<=plotData[:,dictPlot['Boson_Pt']],:]
	slicedData = slicedData[min+10>=slicedData[:,dictPlot['Boson_Pt']],:]
	histDataTargetBDT = slicedData[:,dictPlot[histname]]

	histDataOutputBDT = slicedData[:,dictPlot['LongZCorrectedRecoil_LongZ']]/slicedData[:,dictPlot['PhiCorrectedRecoil_LongZ']]
		
	histDataVarianceBDT = histDataOutputBDT - histDataTargetBDT
	
	histDataAtOnce = [histDataTargetBDT,histDataOutputBDT]
		
	names = ['Target scale factor, mean: %f'%histDataTargetBDT.mean(),'BDT predicted scale factor, mean %f'%histDataOutputBDT.mean()]
	n, bins, patches = plt.hist(histDataAtOnce, num_bins, range=[-2, 4], alpha=0.5, label=names)
	plt.legend(loc='upper right')
	plt.xlabel('Comparison scalefactor BDT Output and Target. %i GeV < Boson Pt < %i'%(min,min+10))
	plt.ylabel('Entries')
	plt.savefig(config['outputDir'] + "/ControlPlots/BDT_Scalefactor_OutputAndTarget%ito%i.svg"%(min,min+10))
	plt.clf()

    
    histVarJet0AndBosonPtCut = np.abs(plotData[30<plotData[:,dictPlot['Boson_Pt']],dictPlot['Boson_Phi']] - plotData[30<plotData[:,dictPlot['Boson_Pt']],dictPlot['Jet0_Phi']])

    n, bins, patches = plt.hist(histVarJet0AndBosonPtCut, num_bins,alpha=0.5)
    plt.xlabel('Var Phi (Jet0,BosonPt), BosonPt > 30')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/VarJet0AndBosonPtCut30.svg")
    plt.clf()
    
    histVarJet1AndBosonPtCut = np.abs(plotData[30<plotData[:,dictPlot['Boson_Pt']],dictPlot['Boson_Phi']] - plotData[30<plotData[:,dictPlot['Boson_Pt']],dictPlot['Jet1_Phi']])
    

    n, bins, patches = plt.hist(histVarJet1AndBosonPtCut, num_bins,alpha=0.5)
    plt.xlabel('Var Phi (Jet1,BosonPt), BosonPt > 30')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/VarJet1AndBosonPtCut30.svg")
    plt.clf()
    
    histVarJet0AndBosonCleaned = np.abs(plotData[10<plotData[:,dictPlot['Jet0_Pt']],dictPlot['Boson_Phi']] - plotData[10<plotData[:,dictPlot['Jet0_Pt']],dictPlot['Jet0_Phi']])

    n, bins, patches = plt.hist(histVarJet0AndBosonCleaned, num_bins,alpha=0.5)
    plt.xlabel('Var Phi (Jet0,BosonPt), JetPt > 10')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/VarJet0AndBosonCleaned10.svg")
    plt.clf()
    
    histVarJet1AndBosonCleaned = np.abs(plotData[10<plotData[:,dictPlot['Jet1_Pt']],dictPlot['Boson_Phi']] - plotData[10<plotData[:,dictPlot['Jet1_Pt']],dictPlot['Jet1_Phi']])
    

    n, bins, patches = plt.hist(histVarJet1AndBosonCleaned, num_bins,alpha=0.5)
    plt.xlabel('Var Phi (Jet1,BosonPt), JetPt > 10')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/VarJet1AndBosonCleaned10.svg")
    plt.clf()
    
    
    histVarJet0AndBoson = np.abs(plotData[:,dictPlot['Boson_Phi']] - plotData[:,dictPlot['Jet0_Phi']])

    n, bins, patches = plt.hist(histVarJet0AndBoson, num_bins,alpha=0.5)
    plt.xlabel('Var Phi (Jet0,BosonPt)')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/VarJet0AndBoson.svg")
    plt.clf()
    
    histVarJet1AndBoson = np.abs(plotData[:,dictPlot['Boson_Phi']] - plotData[:,dictPlot['Jet1_Phi']])
    

    n, bins, patches = plt.hist(histVarJet1AndBoson, num_bins,alpha=0.5)
    plt.xlabel('Var Phi (Jet1,BosonPt)')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/VarJet1AndBoson.svg")
    plt.clf()
    
    muData = plotData[plotData[:,dictPlot['select']]==1]
    eData = plotData[plotData[:,dictPlot['select']]==2]
    
    plotVariable = 'Boson_Pt'
    
    histData = [muData[:,dictPlot[plotVariable]],eData[:,dictPlot[plotVariable]]]
    names = ['%s - Z to mumu'%plotVariable, '%s - Z to ee'%plotVariable]
    n, bins, patches = plt.hist(histData, num_bins, range=[0, 300], normed=1, alpha=0.5, label=names)
    plt.legend(loc='upper right')
    plt.xlabel('Boson Pt in GeV')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/BDT_MuEComparison_%s"%plotVariable)
    plt.clf()

    plotVariable = 'NVertex'
    num_bins = 35
    histData = [muData[:,dictPlot[plotVariable]],eData[:,dictPlot[plotVariable]]]
    names = ['%s - Z to mumu'%plotVariable, '%s - Z to ee'%plotVariable]
    n, bins, patches = plt.hist(histData, num_bins, range=[5, 40], normed=1, alpha=0.5, label=names)
    plt.legend(loc='upper right')
    plt.xlabel('Boson Pt in GeV')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/BDT_MuEComparison_%s"%plotVariable)
    plt.clf()
    
    
    plotVariable = 'NCleanedJets'
    
    num_bins = 15
    histData = [muData[:,dictPlot[plotVariable]],eData[:,dictPlot[plotVariable]]]
    names = ['%s - Z to mumu'%plotVariable, '%s - Z to ee'%plotVariable]
    n, bins, patches = plt.hist(histData, num_bins, range=[0, 15], normed=1, alpha=0.5, label=names)
    plt.legend(loc='upper right')
    plt.xlabel('Boson Pt in GeV')
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "/ControlPlots/BDT_MuEComparison_%s"%plotVariable)
    plt.clf()
    
    


def make_PtSpectrumPlot(config, plotData, dictPlot, maxBosonPt=0, stepwidth=0):

  
    if maxBosonPt==0:
	maxBosonPt=plotData[:,dictPlot['Boson_Pt']].max()
    if stepwidth==0:
	stepwidth = maxBosonPt/100
    
    XRange = np.arange(0,maxBosonPt,stepwidth)
    YSum = np.zeros((XRange.shape[0]-1,1))
    for index in range(0,XRange.shape[0]-1):
	AlternativeDistri = plotData[(XRange[index]<plotData[:,dictPlot['Boson_Pt']]) & (XRange[index+1]>plotData[:,dictPlot['Boson_Pt']]) & (plotData[:,dictPlot['nCombinations']]==1)]
	sumEntries = AlternativeDistri[:,dictPlot['flatPtWeight']].sum()
	YSum[index] = sumEntries
 
    plt.clf()
    plt.plot(XRange[:-1],YSum[:],'o')
    plt.xlabel('Boson Pt')
    plt.ylabel('Weighted Boson Pt')
    plt.savefig(config['outputDir'] + "WeightedBosonPt.svg")
    
    plt.clf()
    
    weightPt = plotData[:,dictPlot['flatPtWeight']]
    
    num_bins = 50
    n, bins, patches = plt.hist(plotData[:,dictPlot['Boson_Pt']], num_bins, facecolor='green', alpha=0.5, weights=weightPt)
    plt.savefig(config['outputDir'] + "WeightedBosonPtHist.svg")
    plt.clf()
    
    if 'flatPtWeightBDT' in dictPlot:
	n, bins, patches = plt.hist(plotData[:,dictPlot['Boson_Pt']], num_bins, facecolor='green', alpha=0.5, weights=plotData[:,dictPlot['flatPtWeightBDT']])
	plt.savefig(config['outputDir'] + "WeightedBosonPtHistBDT.svg")
	plt.clf()
    
 
 

def make_PhiVariancePlot(config, plotData, dictPlot, targetvariable, ptmin, ptmax, xlabelname = ''):
  
    if xlabelname == '':
	xlabelname = targetvariable
    num_bins = 50
    histDataPhi = plotData[:,dictPlot['Boson_Phi']] + math.pi - plotData[:,dictPlot[targetvariable]]
    print(targetvariable,' phi shape: ',histDataPhi.shape)
    for event in range(0,histDataPhi.shape[0]):
	if histDataPhi[event] > math.pi:
	    histDataPhi[event] -= 2*math.pi
	if histDataPhi[event] < -math.pi:
	    histDataPhi[event] += 2*math.pi
    MSE = (histDataPhi**2).mean()
    n, bins, patches = plt.hist(histDataPhi, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel('Variance %s from true Boson Phi (%i < Boson Pt < %i)GeV. MSE: %f'%(xlabelname, ptmin, ptmax, MSE))
    plt.ylabel('Entries')
    plt.savefig(config['outputDir'] + "PhiVariance%s_%ito%iPt.svg"%(xlabelname,ptmin,ptmax))
    plt.clf()
    print('MSE %s (%i<Pt<%i): '%(xlabelname,ptmin,ptmax),MSE)    

    # normal distribution center at x=0 and y=5
    plt.hist2d(plotData[:,dictPlot['Boson_Phi']], histDataPhi,bins = 80, norm=LogNorm())
    #plt.ylim([-0.25,0.25])
    plt.xlabel('Boson Phi (%i < Boson Pt < %i)GeV'%(ptmin, ptmax))
    plt.ylabel('Variance of (Prediction-Target) %s'%xlabelname)
    plt.savefig(config['outputDir'] + "Variance2D_%s(%i<Pt<%i).svg"%(xlabelname,ptmin,ptmax))
    plt.clf()

    
    
    
def plot_results(config, plotData, dictPlot, meanTarget, stdTarget, dictTarget):
    
    #plotData = plotData[0==plotData[:,dictPlot['NCleanedJets']],:]
    
    resultData = np.empty(shape=[1,0]).astype(np.float32)
    dictResult = {}
    
    
    
    
    
    
    
    plotconfig = config[config['activePlotting']]
    
    num_bins = 50
    #Transform NNoutput back
    for targetname in dictTarget:
	plotData[:,dictPlot['NNOutput_%s'%targetname]] = plotData[:,dictPlot['NNOutput_%s'%targetname]]*stdTarget[dictTarget[targetname]]+meanTarget[dictTarget[targetname]]

    
    
	
    if 'NNOutput_Boson_Phi' in dictPlot and 'NNOutput_Boson_Pt' in dictPlot:
	NN_LongZ = -plotData[:, dictPlot['NNOutput_Boson_Pt']]*np.cos(plotData[:, dictPlot['Boson_Phi']]-plotData[:, dictPlot['NNOutput_Boson_Phi']])
	dictPlot["NN_LongZ"] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(NN_LongZ.reshape(NN_LongZ.shape[0],1))))
    elif 'NNOutput_Boson_Phi' in dictPlot:
	NN_LongZ = -plotData[:, dictPlot['recoilslimmedMETs_Pt']]*np.cos(plotData[:, dictPlot['Boson_Phi']]-plotData[:, dictPlot['NNOutput_Boson_Phi']])
	dictPlot["NN_LongZ"] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(NN_LongZ.reshape(NN_LongZ.shape[0],1))))
    elif 'NNOutput_targetPhiFromSlimmed' in dictPlot:
	NN_LongZ = -plotData[:, dictPlot['recoilslimmedMETs_Pt']]*np.cos(math.pi + plotData[:, dictPlot['Boson_Phi']]-plotData[:, dictPlot['NNOutput_targetPhiFromSlimmed']]-plotData[:, dictPlot['recoilslimmedMETs_Phi']])
	dictPlot["NN_LongZ"] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(NN_LongZ.reshape(NN_LongZ.shape[0],1))))
    
    
    elif 'NNOutput_targetX' in dictPlot:
	PhiFromKart = np.arctan( plotData[:,dictPlot['NNOutput_targetY']]/plotData[:,dictPlot['NNOutput_targetX']])
	PtFromKart = np.sqrt( plotData[:,dictPlot['NNOutput_targetY']]**2+plotData[:,dictPlot['NNOutput_targetX']]**2)
	NN_LongZ = -PtFromKart*np.cos(plotData[:, dictPlot['Boson_Phi']]-PhiFromKart)
	dictPlot["NN_LongZ"] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(NN_LongZ.reshape(NN_LongZ.shape[0],1))))

    if not os.path.exists(config['outputDir']):
	os.makedirs(config['outputDir'])
	
    
    #if not os.path.exists('../plots/PlotVariables'):
	#os.makedirs('../plots/PlotVariables')
    
    if plotconfig['plotPlotVariables']:
	if plotconfig['splitNCleanedJets']:
	    plotVariableData = plotData[0<plotData[:,dictPlot['NCleanedJets']],:]
	    outputdir = config['outputDir'] + 'PlotVariables/1CleanedJet/'
	    for variable in dictPlot:
		make_Plot(variable,plotVariableData,dictPlot,outputdir)
		
	    plotVariableData = plotData[1<plotData[:,dictPlot['NCleanedJets']],:]
	    outputdir = config['outputDir'] + 'PlotVariables/2CleanedJet/'
	    for variable in dictPlot:
		make_Plot(variable,plotVariableData,dictPlot,outputdir)
		
	    plotVariableData = plotData[0==plotData[:,dictPlot['NCleanedJets']],:]
	    outputdir = config['outputDir'] + 'PlotVariables/UncleanedJet/'
	    for variable in dictPlot:
		make_Plot(variable,plotVariableData,dictPlot,outputdir)
		
	outputdir = config['outputDir'] + 'PlotVariables/'
	for variable in dictPlot:
	    make_Plot(variable,plotData,dictPlot,outputdir)
      
    #Boson Pt
    #comparisonMinus.xlabel('Boson_Pt')
    #comparisonOver.xlabel('Boson_Pt')
    #comparisonMinus.ylabel('|Boson_Pt-Prediction|')
    #comparisonOver.ylabel('Prediction/Boson_Pt')
    
    bosonmin = [0,0,plotconfig['BosonCut']]
    bosonmax = [plotData[:,dictPlot['Boson_Pt']].max(),plotconfig['BosonCut'],plotData[:,dictPlot['Boson_Pt']].max()]
    
    #BDT Performance Plots 
    #Phi ControlPlots
    if plotconfig['sliceData']:
	for i, min in enumerate(bosonmin):
	    if plotconfig['plotBDTPerformance']:
		histname = 'targetRecoilFromBDT'
		slicedData = plotData[min<=plotData[:,dictPlot['Boson_Pt']],:]
		slicedData = slicedData[bosonmax[i]>=slicedData[:,dictPlot['Boson_Pt']],:]
		histDataTargetBDT = slicedData[:,dictPlot[histname]]

		histDataOutputBDT = slicedData[:,dictPlot['LongZCorrectedRecoil_LongZ']]/slicedData[:,dictPlot['PhiCorrectedRecoil_LongZ']]
		
		histDataVarianceBDT = histDataOutputBDT - histDataTargetBDT
		
		n, bins, patches = plt.hist(histDataVarianceBDT, num_bins, range=[-2, 4], facecolor='green', alpha=0.5)
		plt.xlabel('Scalefactor variance (BDT-Target). (%i  < Boson Pt < %i)GeV, mean: %f'%(min,bosonmax[i],histDataVarianceBDT.mean()))
		plt.ylabel('Entries')
		plt.savefig(config['outputDir'] + "BDT_Scalefactor_VarianceOutputandTarget%ito%i.svg"%(min,bosonmax[i]))
		plt.clf()
		
		histDataAtOnce = [histDataTargetBDT,histDataOutputBDT]
		
		names = ['Target scale factor, mean: %f'%histDataTargetBDT.mean(),'BDT predicted scale factor, mean %f'%histDataOutputBDT.mean()]
		n, bins, patches = plt.hist(histDataAtOnce, num_bins, range=[-2, 4], alpha=0.5, label=names)
		plt.legend(loc='upper right')
		plt.xlabel('Comparison scalefactor BDT Output and Target. %i GeV < Boson Pt < %i'%(min,bosonmax[i]))
		plt.ylabel('Entries')
		plt.savefig(config['outputDir'] + "BDT_Scalefactor_OutputAndTarget%ito%i.svg"%(min,bosonmax[i]))
		plt.clf()
	
		
		make_PhiVariancePlot(config, slicedData,dictPlot,'PhiCorrectedRecoil_Phi', min, bosonmax[i], 'BDT Phi')
	        
		#BDT
		resultData, dictResult = make_ControlPlots(config, slicedData, dictPlot, 'LongZCorrectedRecoil', 'NVertex',resultData, dictResult, 5,40,5,min,bosonmax[i])
		
		resultData, dictResult = make_ControlPlots(config, slicedData, dictPlot, 'PhiCorrectedRecoil', 'NVertex',resultData, dictResult, 5,40,5,min,bosonmax[i])
		
	  
	    
	    if plotconfig['plotNNPerformance']:
		if not os.path.exists(config['outputDir'] + 'NNPlots'):
		    os.makedirs(config['outputDir'] + 'NNPlots')
		
		for targetname in dictTarget:
		    histDataNNresponse = slicedData[:,dictPlot['NNOutput_%s'%targetname]]
		    
		    n, bins, patches = plt.hist(histDataNNresponse, num_bins, facecolor='green', alpha=0.5)
		    plt.xlabel('NN output %s (%i<Pt<%i)'%(targetname,min,bosonmax[i]))
		    plt.ylabel('Entries')
		    plt.savefig(config['outputDir'] + "NNPlots/NNresponse(%i<Pt<%i)_%s.svg"%(min,bosonmax[i],targetname))
		    plt.clf()
		    
		    histDataNN = slicedData[:,dictPlot[targetname]] - slicedData[:,dictPlot['NNOutput_%s'%targetname]]

		    if 'NNOutput_Boson_Phi' in dictPlot:
			
			for event in range(0,histDataNN.shape[0]):
			    if histDataNN[event] > math.pi:
				histDataNN[event] -= 2*math.pi
			    if histDataNN[event] < -math.pi:
				histDataNN[event] += 2*math.pi
			
			MSE = (histDataNN**2).mean()
			print('MSE NN Net: ',MSE)
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Variance %s from true Boson Phi (%i < Boson Pt < %i). MSE: %f'%('NN_OutputPhi', min, bosonmax[i], MSE))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/PhiVarianceNNOutputPhi(%i<Pt<%i).svg"%(min,bosonmax[i]))
			plt.clf()
			
			
			
			
		
		    elif 'NNOutput_targetPhiFromSlimmed' in dictPlot:
			histDataNN = slicedData[:,dictPlot['Boson_Phi']] + math.pi - (slicedData[:,dictPlot['NNOutput_targetPhiFromSlimmed']] + slicedData[:,dictPlot['recoilslimmedMETs_Phi']])

			for event in range(0,histDataNN.shape[0]):
			    if histDataNN[event] > math.pi:
				histDataNN[event] -= 2*math.pi
			    if histDataNN[event] < -math.pi:
				histDataNN[event] += 2*math.pi
			    #if histDataNN[event] > math.pi:
				#histDataNN[event] -= 2*math.pi
			
			MSE = (histDataNN**2).mean()
			print('MSE NN Net: ',MSE)
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Variance %s from true Boson Phi (%i < Boson Pt < %i). MSE: %f'%('NNOutput_targetPhiFromSlimmed', min, bosonmax[i], MSE))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/PhiVariance%s(%i<Pt<%i).svg"%('NNOutput_targetPhiFromSlimmed',min,bosonmax[i]))
			plt.clf()
		    elif 'NNOutput_targetX' in dictPlot:
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Target - %s (%i < Boson Pt < %i)'%(targetname,min,bosonmax[i]))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/NNVariance_%s(%i<Pt<%i).svg"%(targetname,min,bosonmax[i]))
			plt.clf()
				
		    else:
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Target - %s (%i < Boson Pt < %i)'%(targetname,min,bosonmax[i]))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/NNVariance_%s(%i<Pt<%i).svg"%(targetname,min,bosonmax[i]))
			plt.clf()
		    
		    
		    # normal distribution center at x=0 and y=5
		    plt.hist2d(slicedData[:,dictPlot[targetname]], histDataNN,bins = 80, norm=LogNorm())
		    #plt.ylim([-0.25,0.25])
		    plt.xlabel(targetname)
		    plt.ylabel('Variance of (Prediction-Target)')
		    plt.savefig(config['outputDir'] + "NNPlots/NNVariance2D_%s(%i<Pt<%i).svg"%(targetname,min,bosonmax[i]))
		    plt.clf()
		
		if 'NN_LongZ' in dictPlot:
			resultData, dictResult = make_ControlPlots(config, slicedData, dictPlot, 'NN', 'NVertex',resultData, dictResult, 5,40,5,min,bosonmax[i])
      
	    
	    #slimmedMet
	    resultData, dictResult = make_ControlPlots(config, slicedData, dictPlot, 'recoilslimmedMETs', 'NVertex',resultData, dictResult, 5,40,5,min,bosonmax[i])
	    
	    #Puppi-Met
	    resultData, dictResult = make_ControlPlots(config, slicedData, dictPlot, 'recoilslimmedMETsPuppi', 'NVertex',resultData, dictResult, 5,40,5,min,bosonmax[i])
	    
	    plt.figure(4)
	    legend = plt.legend(loc='lower right', shadow=True)
	    legend.get_frame().set_alpha(0.5)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('(MET Boson PT_Long)/(True Boson Pt)')
	    plt.savefig(config['outputDir'] + 'Response_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(5)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
	    legend = plt.legend(loc='lower right', shadow=True)
	    legend.get_frame().set_alpha(0.5)
	    plt.savefig(config['outputDir'] + 'Resolution_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(6)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('Resolution / Response')
	    legend = plt.legend(loc='lower right', shadow=True)
	    legend.get_frame().set_alpha(0.5)
	    plt.savefig(config['outputDir'] + 'ResponseCorrected_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(7)
	    legend = plt.legend(loc='lower right', shadow=True)
	    legend.get_frame().set_alpha(0.5)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('MET Boson PT_Perp')
	    plt.savefig(config['outputDir'] + 'ResponsePerp_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(8)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('MET Boson PT_Perp')
	    legend = plt.legend(loc='lower right', shadow=True)
	    legend.get_frame().set_alpha(0.5)
	    plt.savefig(config['outputDir'] + 'ResolutionPerp_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(0)
	    

	    
	    make_PhiVariancePlot(config, slicedData,dictPlot,'recoilslimmedMETs_Phi', min, bosonmax[i], 'PF Phi')
	    
	    make_PhiVariancePlot(config, slicedData,dictPlot,'recoilslimmedMETsPuppi_Phi', min, bosonmax[i],'PUPPI Phi')
	  
    
    
    #Boson PT
    
    if plotconfig['plotBDTPerformance']:
	resultData, dictResult = make_ControlPlots(config, plotData, dictPlot, 'LongZCorrectedRecoil', 'Boson_Pt',resultData, dictResult, 10,200,10,0,0)
	resultData, dictResult = make_ControlPlots(config, plotData, dictPlot, 'PhiCorrectedRecoil', 'Boson_Pt',resultData, dictResult, 10,200,10,0,0)
	make_MoreBDTPlots(config, plotData, dictPlot)

    resultData, dictResult = make_ControlPlots(config, plotData, dictPlot, 'recoilslimmedMETs', 'Boson_Pt',resultData, dictResult, 10,200,10,0,0)
    
    resultData, dictResult = make_ControlPlots(config, plotData, dictPlot, 'recoilslimmedMETsPuppi', 'Boson_Pt',resultData, dictResult, 10,200,10,0,0)

    if plotconfig['plotNNPerformance']:
	if 'NN_LongZ' in dictPlot:
	    resultData, dictResult = make_ControlPlots(config, plotData, dictPlot, 'NN', 'Boson_Pt',resultData, dictResult, 10,200,10,0,0)
	    
	if 'NNOutput_targetX' in dictPlot:
	    DPhiFromKart = PhiFromKart - plotData[:, dictPlot['Boson_Phi']]
	    DPtFromKart = PtFromKart - plotData[:, dictPlot['Boson_Pt']]
	    n, bins, patches = plt.hist(DPhiFromKart, num_bins, facecolor='green', alpha=0.5)
	    plt.xlabel('Delta Phi from X,Y Prediction')
	    plt.ylabel('Entries')
	    plt.savefig(config['outputDir'] + "NNPlots/NNVariancePhiXY.svg")
	    plt.clf()
	    
	    n, bins, patches = plt.hist(DPtFromKart, num_bins, facecolor='green', alpha=0.5)
	    plt.xlabel('Delta Pt from X,Y Prediction')
	    plt.ylabel('Entries')
	    plt.savefig(config['outputDir'] + "NNPlots/NNVariancePtXY.svg")
	    plt.clf()

    plt.figure(4)
    legend = plt.legend(loc='lower right', shadow=True)
    legend.get_frame().set_alpha(0.5)
    plt.xlabel('Boson Pt')
    plt.ylabel('(MET Boson PT_Long)/(True Boson Pt)')
    plt.savefig(config['outputDir'] + 'Response_vs_BosonPt')
    plt.clf()
    plt.figure(5)
    plt.xlabel('Boson Pt')
    plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
    legend = plt.legend(loc='lower right', shadow=True)
    legend.get_frame().set_alpha(0.5)
    plt.savefig(config['outputDir'] + 'Resolution_vs_BosonPt')
    plt.clf()
    plt.figure(6)
    plt.xlabel('Boson Pt')
    plt.ylabel('Resolution / Response')
    legend = plt.legend(loc='lower right', shadow=True)
    legend.get_frame().set_alpha(0.5)
    plt.savefig(config['outputDir'] + 'ResponseCorrected_vs_BosonPt')
    plt.clf()
    plt.figure(7)
    legend = plt.legend(loc='lower right', shadow=True)
    legend.get_frame().set_alpha(0.5)
    plt.xlabel('Boson Pt')
    plt.ylabel('MET Boson PT_Perp')
    plt.savefig(config['outputDir'] + 'ResponsePerp_vs_BosonPt')
    plt.clf()
    plt.figure(8)
    plt.xlabel('Boson Pt')
    plt.ylabel('MET Boson PT_Perp')
    legend = plt.legend(loc='lower right', shadow=True)
    legend.get_frame().set_alpha(0.5)
    plt.savefig(config['outputDir'] + 'ResolutionPerp_vs_BosonPt')
    plt.clf()
    plt.figure(0)
	    
    
    print('resultDatashape: ',resultData.shape)
    for name in dictResult:
	print(name)
	print(resultData[0,dictResult[name]])
    
    
    
    #make Boson Pt Spectrum with weights
    make_PtSpectrumPlot(config, plotData, dictPlot, 650, 5)
    
    if config['writeResults']:
	writeResult(config, resultData, dictResult)
    
    print('result written')
    return True

    
# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.



def build_custom_mlpMVA(inputcount, targetcount, trainingconfig, input_var=None):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    exec("weightshidden = {}".format(trainingconfig['weightInitHidden']))
    exec("weightsout = {}".format(trainingconfig['weightInitOutput'])) 
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, inputcount),
                                     input_var=input_var)
    
    #network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    if trainingconfig['dropoutInput']:
        network = lasagne.layers.dropout(network, p=trainingconfig['dropoutInput'])
    # Hidden layers and dropout:
    #nonlin = lasagne.nonlinearities.tanh
    exec("nonlin = {}".format(trainingconfig['nonlinearActivationFuncHidden'])) 
    for _ in range(trainingconfig['depthHiddenLayers']):
        network = lasagne.layers.DenseLayer(
                network, trainingconfig['sizeHiddenLayers'], nonlinearity=nonlin, W=weightshidden)
        if trainingconfig['dropoutHidden']:
            network = lasagne.layers.dropout(network, p=trainingconfig['dropoutHidden'])
    # Output layer:
    exec("nonlinout = {}".format(trainingconfig['nonlinearActivationFuncOutput'])) 
    network = lasagne.layers.DenseLayer(network, targetcount, nonlinearity=nonlinout, W=weightsout)
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

def iterate_minibatchesInput(inputs, targets, plotData, weights, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(inputs) + batchsize -1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if start_idx + batchsize > len(targets):
	    excerpt = slice(start_idx, len(targets))
	if start_idx < len(targets):  
	    yield inputs[excerpt], targets[excerpt], plotData[excerpt], weights[excerpt]
        

def iterate_minibatchesWeights(inputs, targets, weights, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) + batchsize -1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if start_idx + batchsize > len(targets):
            excerpt = slice(start_idx, len(targets))
        if start_idx < len(targets):
            yield inputs[excerpt], targets[excerpt], weights[excerpt]


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(inputs) + batchsize -1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if start_idx + batchsize > len(targets):
	    excerpt = slice(start_idx, len(targets))
	if start_idx < len(targets):  
	    yield inputs[excerpt], targets[excerpt]
        
        """
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
	"""

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(config):
    
    trainingconfig = config[config['activeTraining']]
    # Load the dataset
    print("Loading data...")
    X_trainMVA, y_trainMVA, X_valMVA, y_valMVA, X_testMVA, y_testMVA, weights, valweights, testweights, dictTarget, plotData, dictPlot, meanPlot, stdPlot = load_datasetcsv(config)
    
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    print(X_trainMVA.shape)
    print(y_trainMVA.shape)
    print(X_valMVA.shape)
    print(X_testMVA.shape)
   
    print('weights:')
    print(weights.shape) 
    print('min: ',weights.min())
    print('max: ',weights.max())
    print('mean: ',weights.mean())
    print('std: ',weights.std())
    weightcounter = 0
    for i in range(0,weights.shape[0]):
        if weights[i,0] == 0:
	    weightcounter += 1

    for name in dictTarget:
	print('Target %s:'%name,' mean: ', y_trainMVA[:,dictTarget[name]].mean())
	print('Target %s:'%name,' std: ', y_trainMVA[:,dictTarget[name]].std())
	print('Target %s:'%name,' min: ', y_trainMVA[:,dictTarget[name]].min())
	print('Target %s:'%name,' max: ', y_trainMVA[:,dictTarget[name]].max())
	print('Zeros: ', weightcounter)   
    # Prepare Theano variables for inputs and targets
    input_var = T.fmatrix('inputs')
    target_var = T.fmatrix('targets')
    weight_var = T.fmatrix('weightsbatch')
    testweight_var = T.fmatrix('weightsbatch')
 
    inputcount = X_trainMVA.shape[1]
    targetcount = y_trainMVA.shape[1]
    
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    

    network = build_custom_mlpMVA(inputcount,targetcount, trainingconfig, input_var)
    """
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
    """
    
  
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    
    prediction = lasagne.layers.get_output(network)
    
    """
    if 'Boson_Phi' in dictTarget and 'Boson_Pt' in dictTarget:
	rangePhi = (y_trainMVA[:,dictTarget['Boson_Phi']].max() - y_trainMVA[:,dictTarget['Boson_Phi']].min())/2
	print("Range Phi: ",rangePhi)
	custom_loss = T.sqr((T.minimum(prediction[:,dictTarget['Boson_Phi']]-target_var[:,dictTarget['Boson_Phi']],rangePhi-abs(prediction[:,dictTarget['Boson_Phi']]-target_var[:,dictTarget['Boson_Phi']])))**2+(prediction[:,dictTarget['Boson_Pt']]-target_var[:,dictTarget['Boson_Pt']])**2)
	
	custom_loss = T.sqr(target_var[:,dictTarget['Boson_Pt']]-(prediction[:,dictTarget['Boson_Phi']]*np.cos(target_var[:,dictTarget['Boson_Phi']]-prediction[:,dictTarget['Boson_Phi']])))
	
    elif 'Boson_Phi' in dictTarget:
	rangePhi = (y_trainMVA[:,dictTarget['Boson_Phi']].max() - y_trainMVA[:,dictTarget['Boson_Phi']].min())/2
	print("Range Phi: ",rangePhi)
	#custom_loss = T.sqr(T.sin((prediction-target_var)*2*rangePhi/rangePhi))
	#custom_loss = T.sqr(T.minimum(prediction-target_var,T.minimum(rangePhi-(prediction-target_var),rangePhi+(prediction-target_var))))
	#custom_loss = 1./30.*(prediction-target_var)**2*((prediction-target_var)-2*rangePhi)**2*((prediction-target_var)+2*rangePhi)**2
	#custom_loss = T.sin((prediction-target_var)*math.pi/(rangePhi*2))**2*((prediction-target_var)*math.pi/(rangePhi*2)>-5./4*math.pi)*((prediction-target_var)*math.pi/(rangePhi*2)<5./4*math.pi) + ((prediction-target_var)*math.pi/(rangePhi*2)+(0.5-5./4*math.pi))*((prediction-target_var)*math.pi/(rangePhi*2)>5./4*math.pi)+ (-(prediction-target_var)*math.pi/(rangePhi*2)+(0.5-5./4*math.pi))*((prediction-target_var)*math.pi/(rangePhi*2)<-5./4*math.pi)
	custom_loss = lasagne.objectives.squared_error(prediction, target_var)
    elif 'targetPhiFromSlimmed' in dictTarget:
	rangePhi = (y_trainMVA[:,dictTarget['targetPhiFromSlimmed']].max() - y_trainMVA[:,dictTarget['targetPhiFromSlimmed']].min())/2
	custom_loss = 1./30.*(prediction-target_var)**2*((prediction-target_var)-2*rangePhi)**2*((prediction-target_var)+2*rangePhi)**2
    else:
	custom_loss = lasagne.objectives.squared_error(prediction, target_var)
    """
    

    exec("{}".format(trainingconfig['trainingLossFunctionHelper']))

    exec("loss = {}".format(trainingconfig['trainingLossFunction'])) 
    if trainingconfig['transposeWeight']:
	loss = loss * weight_var.transpose()
    else:
	loss = loss * weight_var
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(
            #loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adam(loss,params)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.squared_error(prediction, target_var)
    
    
    """
    if 'Boson_Phi' in dictTarget and 'Boson_Pt' in dictTarget:
	rangePhi = (y_trainMVA[:,dictTarget['Boson_Phi']].max() - y_trainMVA[:,dictTarget['Boson_Phi']].min())/2
	print("Range Phi: ",rangePhi)
	custom_lossTest = T.sqr((T.minimum(prediction[:,dictTarget['Boson_Phi']]-target_var[:,dictTarget['Boson_Phi']],rangePhi-abs(prediction[:,dictTarget['Boson_Phi']]-target_var[:,dictTarget['Boson_Phi']])))**2+(prediction[:,dictTarget['Boson_Pt']]-target_var[:,dictTarget['Boson_Pt']])**2)
    elif 'Boson_Phi' in dictTarget:
	rangePhi = (y_trainMVA[:,dictTarget['Boson_Phi']].max() - y_trainMVA[:,dictTarget['Boson_Phi']].min())/2
	print("Range Phi: ",rangePhi)
	#custom_lossTest = T.sqr(T.sin((prediction-target_var)*2*rangePhi/rangePhi))
	#custom_lossTest = T.sqr(T.minimum((prediction-target_var)**2,(2*rangePhi)**2-(prediction-target_var)**2))

	#custom_lossTest = 1./30.*(prediction-target_var)**2*((prediction-target_var)-2*rangePhi)**2*((prediction-target_var)+2*rangePhi)**2
	#custom_lossTest = T.sin((prediction-target_var)*math.pi/(rangePhi*2))**2*((prediction-target_var)*math.pi/(rangePhi*2)>-5./4*math.pi)*((prediction-target_var)*math.pi/(rangePhi*2)<5./4*math.pi) + ((prediction-target_var)*math.pi/(rangePhi*2)+(0.5-5./4*math.pi))*((prediction-target_var)*math.pi/(rangePhi*2)>5./4*math.pi)+ (-(prediction-target_var)*math.pi/(rangePhi*2)+(0.5-5./4*math.pi))*((prediction-target_var)*math.pi/(rangePhi*2)<-5./4*math.pi)
	custom_lossTest = lasagne.objectives.squared_error(prediction, target_var)
    elif 'targetPhiFromSlimmed' in dictTarget:
	rangePhi = (y_trainMVA[:,dictTarget['targetPhiFromSlimmed']].max() - y_trainMVA[:,dictTarget['targetPhiFromSlimmed']].min())/2
	custom_lossTest = 1./30.*(prediction-target_var)**2*((prediction-target_var)-2*rangePhi)**2*((prediction-target_var)+2*rangePhi)**2
    else:
	custom_lossTest = lasagne.objectives.squared_error(prediction, target_var)
    """
    exec("{}".format(trainingconfig['testLossFunctionHelper']))
    exec("test_loss = {}".format(trainingconfig['testLossFunction'])) 
    if trainingconfig['transposeWeight']:
	test_loss = test_loss * testweight_var.transpose()
    else:
	test_loss = test_loss * testweight_var
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var, weight_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var, testweight_var], test_loss)
    
    predict_fn = theano.function([input_var], prediction)

    #test = predict_fn(X_testMVA)
    #print("testshape", test.shape)

    load = config['loadExistingModel']
    Train = config['performTraining']
    
    if load:
	with np.load(config['importFileNetworkModel']) as f:
	    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
	
    if Train:
	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	epoch = 0
	valDelta = True
	minError = 0
	minErrorEpoch = 0
	#for epoch in range(trainingconfig['trainingEpochs']):
	while epoch < trainingconfig['trainingEpochs'] and valDelta:
	    # In each epoch, we do a full pass over the training data:
	    train_err = 0
	    train_batches = 0
	    start_time = time.time()
	    for batch in iterate_minibatchesWeights(X_trainMVA, y_trainMVA, weights, trainingconfig['batchSize'], shuffle=True):
		inputs, targets, weightsbatch = batch
		train_err += train_fn(inputs, targets, weightsbatch)
		train_batches += 1

	    # And a full pass over the validation data:
	    val_err = 0
	    val_acc = 0
	    val_batches = 0
	    for batch in iterate_minibatchesWeights(X_valMVA, y_valMVA, valweights, trainingconfig['batchSize'], shuffle=False):
		inputs, targets, weightsbatch = batch
		err = val_fn(inputs, targets, weightsbatch)
		val_err += err
		#val_acc += acc
		val_batches += 1

	    # Then we print the results for this epoch:
	    print("Epoch {} of {} took {:.3f}s".format(
		epoch + 1, trainingconfig['trainingEpochs'], time.time() - start_time))
	    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
	    
	    if epoch == 0 or val_err/val_batches < minError:
		minError = val_err/val_batches
		minErrorEpoch = epoch
		bestparams = lasagne.layers.get_all_param_values(network)
	    
	    if epoch - minErrorEpoch > trainingconfig['cancelAfterUselessEpochs']:
		valDelta = False
	    epoch += 1
	
	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatchesWeights(X_testMVA, y_testMVA, testweights, trainingconfig['batchSize'], shuffle=False):
	    inputs, targets, weightsbatch = batch
	    err = val_fn(inputs, targets, weightsbatch)
	    test_err += err
	    #test_acc += acc
	    test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

	#Optionally, you could now dump the network weights to a file like this:
	print('Minimal Error: ', minError)
	print('Setting network to min Error params')
	lasagne.layers.set_all_param_values(network, bestparams)
	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatchesWeights(X_testMVA, y_testMVA, testweights, trainingconfig['batchSize'], shuffle=False):
	    inputs, targets, weightsbatch = batch
	    err = val_fn(inputs, targets, weightsbatch)
	    test_err += err
	    #test_acc += acc
	    test_batches += 1
	print("Final results with min Error params:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	
	np.savez(config['exportFileNetworkModel'], *lasagne.layers.get_all_param_values(network))
    

    testSetPrediction = np.empty(shape=[0,y_testMVA.shape[1]]).astype(np.float32)
    
    for batch in iterate_minibatches(X_trainMVA, y_trainMVA, trainingconfig['batchSize'], shuffle=False):
	X_TestTemp, Y_TestTemp = batch
	#print('ztesttempshape: ',X_TestTemp.shape)
	#print('predictshape: ', predict_fn(X_TestTemp).shape)
	testSetPrediction = np.vstack((testSetPrediction, np.array(predict_fn(X_TestTemp))))
    
    for batch in iterate_minibatches(X_valMVA, y_valMVA, trainingconfig['batchSize'], shuffle=False):
	X_TestTemp, Y_TestTemp = batch
	#print('ztesttempshape: ',X_TestTemp.shape)
	#print('predictshape: ', predict_fn(X_TestTemp).shape)
	testSetPrediction = np.vstack((testSetPrediction, np.array(predict_fn(X_TestTemp))))
	
    for batch in iterate_minibatches(X_testMVA, y_testMVA, trainingconfig['batchSize'], shuffle=False):
	X_TestTemp, Y_TestTemp = batch
	#print('ztesttempshape: ',X_TestTemp.shape)
	#print('predictshape: ', predict_fn(X_TestTemp).shape)
	testSetPrediction = np.vstack((testSetPrediction, np.array(predict_fn(X_TestTemp))))

    
    print('testsetprediction shape: ', testSetPrediction.shape)
    print('plotData shape: ', plotData.shape)
    print('testsetpredictionmin: ', testSetPrediction.min())
    print('testsetpredictionmax: ', testSetPrediction.max())
    print('testsettargetmin: ', y_testMVA.min())
    print('testsettargetmax: ', y_testMVA.max())

    for variablename in dictTarget:
	dictPlot["NNOutput_%s"%variablename] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(testSetPrediction[:,dictTarget[variablename]]).reshape(testSetPrediction.shape[0],1)))
	
    
    
    print("plotDatashape", plotData.shape)
    plot_results(config, plotData, dictPlot, meanPlot, stdPlot, dictTarget)
   
    
    
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load YAML config files from directory and create a sqlite3 database out of them.')
    parser.add_argument('-config', '--inputconfig', default='../configs/config.json', help='Path to configurations file')
    args = parser.parse_args()
  
    print('Used Configfile: ',args.inputconfig)



    with open(args.inputconfig, 'r') as f:
	config = json.load(f)

    """
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
    """
    main(config)
