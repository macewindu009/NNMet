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
"""
def createNewWeightfile(config):

    with open(config['weightFilename'], 'wb') as csvfile:
    weightwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    weightwriter.writerow(['Spam'] * 5 + ['Baked Beans'])

"""  


def load_datasetcsv(config):
    #create Treevariable
    
    start = time.time()
    

    

    reader=csv.reader(open(config['inputFile'],"rb"),delimiter=',')
    datacsv=list(reader)
    header = np.array(datacsv[0]).astype(np.str)
    inputdatentot =np.array(datacsv[1:]).astype(np.float32)
    
    
    #print(header.shape)
    #print(inputdatentot.shape)
    
    
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
    

    trainingconfig = config[config['activeTraining']]
    
    
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
    
    varwhole = inputdatentot.std(axis = 0)
    print('varwhole: ',varwhole.shape)
    
    inputDataX = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    inputDataY = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    inputDataPlot = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    weights = np.empty(shape=[inputdatentot.shape[0],0]).astype(np.float32)
    
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

    if 'flatPtWeight' in dictInputTot and trainingconfig['useWeightsFromBDT']:
	weights = np.hstack((weights,np.array(inputdatentot[:,index]).reshape(inputdatentot.shape[0],1)))
    else:
	weights = np.hstack((weights,np.ones(inputdatentot.shape[0]).reshape(inputdatentot.shape[0],1)))
	print('Could not find flat Pt weights, initalizing with equal weight')
	
    if not 'flatPtWeight' in dictPlot:
	dictPlot['flatPtWeight'] = inputDataPlot.shape[1]
	inputDataPlot = np.hstack((inputDataPlot, weights))
    
    print("xshape ", inputDataX.shape)
    print("yshape ", inputDataY.shape)
    

    for name in dictInputY:
	print('Target before norm %s:'%name,' mean: ', inputDataY[:,dictInputY[name]].mean())
	print('Target before norm %s:'%name,' std: ', inputDataY[:,dictInputY[name]].std())
	print('Target before norm %s:'%name,' min: ', inputDataY[:,dictInputY[name]].min())
	print('Target before norm %s:'%name,' max: ', inputDataY[:,dictInputY[name]].max())
 

    
    x_train = np.empty(shape=[0,inputDataX.shape[1]]).astype(np.float32)
    x_trainweights = np.empty(shape=[0,1]).astype(np.float32)
    x_val = np.empty(shape=[0,inputDataX.shape[1]]).astype(np.float32)
    x_test = np.empty(shape=[0,inputDataX.shape[1]]).astype(np.float32)
    y_train = np.empty(shape=[0,inputDataY.shape[1]]).astype(np.float32)
    y_val = np.empty(shape=[0,inputDataY.shape[1]]).astype(np.float32)
    y_test = np.empty(shape=[0,inputDataY.shape[1]]).astype(np.float32)
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
	    x_trainweights = np.vstack((x_trainweights, np.array(weightstemp)))
	    y_train = np.vstack((y_train, np.array(y_TestTemp)))
	if countEvents >= inputDataX.shape[0]*3/6 and countEvents <= inputDataX.shape[0]*4/6:
	    x_val = np.vstack((x_val, np.array(x_TestTemp)))
	    y_val = np.vstack((y_val, np.array(y_TestTemp)))
	if countEvents > inputDataX.shape[0]*4/6:
	    x_test = np.vstack((x_test, np.array(x_TestTemp)))
	    y_test = np.vstack((y_test, np.array(y_TestTemp)))
	
	countEvents += batchsize
	if countEvents % (batchsize*100) == 0:
	    print('processed Events: ', countEvents, '/', inputDataX.shape[0])
    
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

    print('weights: ', x_trainweights.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test, x_trainweights, dictInputY, inputDataPlotShuffled, dictPlot, meanSelectedY, stdSelectedY
    
    
    
def make_Plot(variablename, inputData, dictPlot, outputdir):

    histData = inputData[:,dictPlot[variablename]]
    
    if not os.path.exists(outputdir+'PlotVariables/'):
	os.makedirs(outputdir+'PlotVariables/')
	
    
    
    num_bins = 50
    
    n, bins, patches = plt.hist(histData, num_bins, facecolor='green', alpha=0.5)
    plt.xlabel(variablename)
    plt.ylabel('Hits')
    plt.savefig((outputdir+'PlotVariables/'+variablename+".png"))
    plt.clf()
    return 0
    
    
def make_ResponseCorrectedPlot(config, XRange, YVariance, YResponse, bosonName, targetvariable,ptmin,ptmax):

    plt.clf()
    plt.plot(XRange[:-1],YVariance[:]/YResponse[:],'o')
    plt.xlabel(targetvariable)
    plt.ylabel('Resolution / Response')
    if ptmax == 0:
	  plt.savefig(config['outputDir'] + "ControlPlots/ResponseCorrected_%s_vs_%s.png" %(bosonName,targetvariable))
    else:
	  plt.savefig(config['outputDir'] + "ControlPlots/ResponseCorrected(%i<Pt<%i)_%s_vs_%s.png" %(ptmin,ptmax,bosonName,targetvariable))
    plt.figure(6)
    plt.plot(XRange[:-1],YVariance[:]/YResponse[:],'o',label=bosonName)
    plt.figure(0)
    plt.clf()    
    
    
def make_ResolutionPlot(config,plotData,dictPlot, bosonName, targetvariable, minrange=42,maxrange=0, stepwidth=0, ptmin =0,ptmax=0):


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
	
	    if index < 6:
		plt.clf()
		num_bins = 50
		n, bins, patches = plt.hist(currentDistri, num_bins, normed=1, facecolor='green', alpha=0.5)
		y = mlab.normpdf(bins, currentDistri.mean(), currentDistri.std())
		plt.xlabel('%s at %f'%(targetvariable,(XRange[index+1]+XRange[index])/2))
		plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
		plt.plot(bins, y, 'r--')
		if ptmax == 0:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Resolution_%s_vs_%s_%i.png' %(bosonName,targetvariable, index)))
		else:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Resolution(%i<Pt<%i)_%s_vs_%s_%i.png' %(ptmin,ptmax,bosonName,targetvariable, index)))
		    
    plt.clf()
    plt.plot(XRange[:-1],YStd[:],'o')
    plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
    plt.xlabel(targetvariable)
    if ptmax == 0:
	plt.savefig(config['outputDir'] + "ControlPlots/Resolution_%s_vs_%s.png" %(bosonName,targetvariable))
    else:
	plt.savefig(config['outputDir'] + "ControlPlots/Resolution(%i<Pt<%i)_%s_vs_%s.png" %(ptmin,ptmax,bosonName,targetvariable))
    plt.figure(5)
    plt.plot(XRange[:-1],YStd[:],'o',label=bosonName)
    plt.figure(0)
    plt.clf()
    
    return XRange, YStd
    

def make_PtSpectrumPlot(config, plotData, dictPlot, maxBosonPt=0, stepwidth=0):
    if maxBosonPt==0:
	maxBosonPt=plotData[:,dictPlot['Boson_Pt']].max()
    if stepwidth==0:
	stepwidth = maxBosonPt/100
    
    XRange = np.arange(0,maxBosonPt,stepwidth)
    YSum = np.zeros((XRange.shape[0]-1,1))
    for index in range(0,XRange.shape[0]-1):
	AlternativeDistri = plotData[(XRange[index]<plotData[:,dictPlot['Boson_Pt']]) & (XRange[index+1]>plotData[:,dictPlot['Boson_Pt']])]
	sumEntries = AlternativeDistri[:,dictPlot['flatPtWeight']].sum()
	YSum[index] = sumEntries
 
    plt.clf()
    plt.plot(XRange[:-1],YSum[:],'o')
    plt.xlabel('Boson Pt')
    plt.ylabel('Weighted Boson Pt')
    plt.savefig(config['outputDir'] + "WeightedBosonPt.png")
 
 
def make_ResponsePlot(config, plotData,dictPlot, bosonName, targetvariable, minrange=42,maxrange=0, stepwidth=0, ptmin=0, ptmax=0):
  
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

	if currentDistri.shape == [0,1]:
	    YMean[index] = 0
	    YStd[index] = 0
	else:
	    YMean[index] = currentDistri.mean()
	    YStd[index] = currentDistri.std()
	    if index < 6:
		plt.clf()
		num_bins = 50
		n, bins, patches = plt.hist(currentDistri, num_bins, normed=1, facecolor='green', alpha=0.5)
		y = mlab.normpdf(bins, currentDistri.mean(), currentDistri.std())
		plt.xlabel('%s at %f'%(targetvariable,(XRange[index+1]+XRange[index])/2))
		plt.ylabel('(MET Boson PT_Long)/(True Boson Pt)')
		plt.plot(bins, y, 'r--')
		if ptmax == 0:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Response_%s_vs_%s_%i.png' %(bosonName,targetvariable, index)))
		else:
		    plt.savefig((config['outputDir'] + 'ControlPlots/SingleDistributions/Response(%i<Pt<%i)_%s_vs_%s_%i.png' %(ptmin,ptmax,bosonName,targetvariable, index)))
    plt.clf()
    plt.plot(XRange[:-1],YMean[:],'o')

    plt.xlabel(targetvariable)
    if ptmax == 0:
	plt.savefig(config['outputDir'] + "ControlPlots/Response_%s_vs_%s.png" %(bosonName,targetvariable))
    else:
	plt.savefig(config['outputDir'] + "ControlPlots/Response(%i<Pt<%i)_%s_vs_%s.png" %(ptmin,ptmax,bosonName,targetvariable))
    plt.clf()
    plt.figure(4)
    plt.plot(XRange[:-1],YMean[:],'o',label=bosonName)
    plt.figure(0)
    
    
    
    return YMean

def make_ControlPlots(config, plotData,dictPlot, bosonName, targetvariable, minrange=42,maxrange=0, stepwidth=0, ptmin=0,ptmax=0):

    if not os.path.exists((config['outputDir'] + 'ControlPlots/SingleDistributions/')):
	os.makedirs((config['outputDir'] + 'ControlPlots/SingleDistributions/'))
    XRange, YVariance = make_ResolutionPlot(config, plotData, dictPlot, bosonName, targetvariable,minrange,maxrange,stepwidth, ptmin, ptmax) 
    YResponse = make_ResponsePlot(config, plotData, dictPlot, bosonName, targetvariable,minrange,maxrange,stepwidth, ptmin, ptmax)
    make_ResponseCorrectedPlot(config, XRange, YVariance, YResponse, bosonName, targetvariable, ptmin, ptmax)


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
    plt.savefig(config['outputDir'] + "PhiVariance%s_%ito%iPt.png"%(xlabelname,ptmin,ptmax))
    plt.clf()
    
    # normal distribution center at x=0 and y=5
    plt.hist2d(plotData[:,dictPlot['Boson_Phi']], histDataPhi,bins = 80, norm=LogNorm())
    #plt.ylim([-0.25,0.25])
    plt.xlabel('Boson Phi (%i < Boson Pt < %i)GeV'%(ptmin, ptmax))
    plt.ylabel('Variance of (Prediction-Target) %s'%xlabelname)
    plt.savefig(config['outputDir'] + "Variance2D_%s(%i<Pt<%i).png"%(xlabelname,ptmin,ptmax))
    plt.clf()

    
    
    
def plot_results(config, plotData, dictPlot, meanTarget, stdTarget, dictTarget):
    
    plotconfig = config[config['activePlotting']]
    
    num_bins = 50
    #Transform NNoutput back
    for targetname in dictTarget:
	plotData[:,dictPlot['NNOutput_%s'%targetname]] = plotData[:,dictPlot['NNOutput_%s'%targetname]]*stdTarget[dictTarget[targetname]]+meanTarget[dictTarget[targetname]]

    

    if not os.path.exists(config['outputDir']):
	os.makedirs(config['outputDir'])
	
    
    #if not os.path.exists('../plots/PlotVariables'):
	#os.makedirs('../plots/PlotVariables')
    
    if plotconfig['plotPlotVariables']:
	for variable in dictPlot:
	    make_Plot(variable,plotData,dictPlot,config['outputDir'])
      
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

		histDataOutputBDT = -slicedData[:,dictPlot['Boson_Pt']]/slicedData[:,dictPlot['LongZCorrectedRecoil_LongZ']]
		
		histDataVarianceBDT = histDataTargetBDT - histDataOutputBDT
		
		n, bins, patches = plt.hist(histDataVarianceBDT, num_bins, range=[-2, 4], facecolor='green', alpha=0.5)
		plt.xlabel('Scalefactor variance (BDT,Target). %i GeV < Boson Pt < %i'%(min,bosonmax[i]))
		plt.ylabel('Entries')
		plt.savefig(config['outputDir'] + "BDT_Scalefactor_VarianceOutputandTarget%ito%i.png"%(min,bosonmax[i]))
		plt.clf()
		
		histDataAtOnce = [histDataTargetBDT,histDataOutputBDT]
		
		names = ['Target scale factor','BDT predicted scale factor']
		n, bins, patches = plt.hist(histDataAtOnce, num_bins, range=[-2, 4], alpha=0.5, label=names)
		plt.legend(loc='upper right')
		plt.xlabel('Comparison scalefactor BDT Output and Target. %i GeV < Boson Pt < %i'%(min,bosonmax[i]))
		plt.ylabel('Entries')
		plt.savefig(config['outputDir'] + "BDT_Scalefactor_OutputAndTarget%ito%i.png"%(min,bosonmax[i]))
		plt.clf()
	
		
		make_PhiVariancePlot(config, slicedData,dictPlot,'PhiCorrectedRecoil_Phi', min, bosonmax[i], 'BDT Phi')
	        
		#BDT
		make_ControlPlots(config, slicedData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'NVertex',5,40,5,min,bosonmax[i])
	   
	  
	    
	    if plotconfig['plotNNPerformance']:
		if not os.path.exists(config['outputDir'] + 'NNPlots'):
		    os.makedirs(config['outputDir'] + 'NNPlots')
		"""
		##NN from Boson_Pt
		if 'NNOutput_Boson_Pt' in dictPlot and 'NNOutput_Boson_Phi' in dictPlot:
		    make_ControlPlots(config, plotData, dictPlot, 'NNOutput_Boson_Pt', 'NVertex',5,40,5,min,bosonmax[i])
		    
		##NN-Correction on slimedMets_Pt/Phi
		if 'NNOutput_targetPhiFromSlimmed' and 'NNOutput_targetRecoilFromSlimmed' in dictPlot:
		    make_ControlPlots(config, plotData, dictPlot, 'NNOutput_LongZ', 'NVertex',5,40,5,min,bosonmax[i])
		"""
		for targetname in dictTarget:
		    histDataNNresponse = slicedData[:,dictPlot['NNOutput_%s'%targetname]]
		    
		    n, bins, patches = plt.hist(histDataNNresponse, num_bins, facecolor='green', alpha=0.5)
		    plt.xlabel('NN output %s (%i<Pt<%i)'%(targetname,min,bosonmax[i]))
		    plt.ylabel('Entries')
		    plt.savefig(config['outputDir'] + "NNPlots/NNresponse(%i<Pt<%i)_%s.png"%(min,bosonmax[i],targetname))
		    plt.clf()
		    
		    histDataNN = slicedData[:,dictPlot[targetname]] - slicedData[:,dictPlot['NNOutput_%s'%targetname]]

		    if 'NNOutput_Boson_Phi' in dictPlot:
			
			for event in range(0,histDataNN.shape[0]):
			    if histDataNN[event] > math.pi:
				histDataNN[event] -= 2*math.pi
			    if histDataNN[event] < -math.pi:
				histDataNN[event] += 2*math.pi
			
			MSE = (histDataNN**2).mean()
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Variance %s from true Boson Phi (%i < Boson Pt < %i). MSE: %f'%('NN_OutputPhi', min, bosonmax[i], MSE))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/PhiVarianceNNOutputPhi(%i<Pt<%i).png"%(min,bosonmax[i]))
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
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Variance %s from true Boson Phi (%i < Boson Pt < %i). MSE: %f'%('NNOutput_targetPhiFromSlimmed', min, bosonmax[i], MSE))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/PhiVariance%s(%i<Pt<%i).png"%('NNOutput_targetPhiFromSlimmed',min,bosonmax[i]))
			plt.clf()
		    else:
			n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
			plt.xlabel('Target - %s (%i < Boson Pt < %i)'%(targetname,min,bosonmax[i]))
			plt.ylabel('Entries')
			plt.savefig(config['outputDir'] + "NNPlots/NNVariance_%s(%i<Pt<%i).png"%(targetname,min,bosonmax[i]))
			plt.clf()
		    
		    # normal distribution center at x=0 and y=5
		    plt.hist2d(slicedData[:,dictPlot[targetname]], histDataNN,bins = 80, norm=LogNorm())
		    #plt.ylim([-0.25,0.25])
		    plt.xlabel(targetname)
		    plt.ylabel('Variance of (Prediction-Target)')
		    plt.savefig(config['outputDir'] + "NNPlots/NNVariance2D_%s(%i<Pt<%i).png"%(targetname,min,bosonmax[i]))
		    plt.clf()
      
	    
	    #slimmedMet
	    make_ControlPlots(config, slicedData, dictPlot, 'recoilslimmedMETs_LongZ', 'NVertex',5,40,5,min,bosonmax[i])
	    
	    #Puppi-Met
	    make_ControlPlots(config, slicedData, dictPlot, 'recoilslimmedMETsPuppi_LongZ', 'NVertex',5,40,5,min,bosonmax[i])
	    
	    plt.figure(4)
	    legend = plt.legend(loc='lower right', shadow=True)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('(MET Boson PT_Long)/(True Boson Pt)')
	    plt.savefig(config['outputDir'] + 'Response_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(5)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
	    legend = plt.legend(loc='lower right', shadow=True)
	    plt.savefig(config['outputDir'] + 'Resolution_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(6)
	    plt.xlabel('N Vertex (%i < Boson Pt < %i)'%(min,bosonmax[i]))
	    plt.ylabel('Resolution / Response')
	    legend = plt.legend(loc='lower right', shadow=True)
	    plt.savefig(config['outputDir'] + 'ResponseCorrected_(%i<Pt<%i)_vs_NVertex'%(min,bosonmax[i]))
	    plt.clf()
	    plt.figure(0)
	    

	    
	    make_PhiVariancePlot(config, slicedData,dictPlot,'recoilslimmedMETs_Phi', min, bosonmax[i], 'PF Phi')
	    
	    make_PhiVariancePlot(config, slicedData,dictPlot,'recoilslimmedMETsPuppi_Phi', min, bosonmax[i],'PUPPI Phi')
	  
    
    
    #Boson PT
    
    if plotconfig['plotBDTPerformance']:
	make_ControlPlots(config, plotData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'Boson_Pt',10,200,10,0,0)

    make_ControlPlots(config, plotData, dictPlot, 'recoilslimmedMETs_LongZ', 'Boson_Pt',10,200,10,0,0)
    
    make_ControlPlots(config, plotData, dictPlot, 'recoilslimmedMETsPuppi_LongZ', 'Boson_Pt',10,200,10,0,0)

    plt.figure(4)
    legend = plt.legend(loc='lower right', shadow=True)
    plt.xlabel('Boson Pt')
    plt.ylabel('(MET Boson PT_Long)/(True Boson Pt)')
    plt.savefig(config['outputDir'] + 'Response_vs_BosonPt')
    plt.clf()
    plt.figure(5)
    plt.xlabel('Boson Pt')
    plt.ylabel('(MET Boson PT_Long) - (True Boson Pt)')
    legend = plt.legend(loc='lower right', shadow=True)
    plt.savefig(config['outputDir'] + 'Resolution_vs_BosonPt')
    plt.clf()
    plt.figure(6)
    plt.xlabel('Boson Pt')
    plt.ylabel('Resolution / Response')
    legend = plt.legend(loc='lower right', shadow=True)
    plt.savefig(config['outputDir'] + 'ResponseCorrected_vs_BosonPt')
    plt.clf()
    plt.figure(0)
	    
    
    
    #make Boson Pt Spectrum with weights
    make_PtSpectrumPlot(config, plotData, dictPlot, 650, 5)

    """
    if 'NNOutput_Boson_Pt' in dictPlot and 'NNOutput_Boson_Phi' in dictPlot:
	histDataULong = plotData[:,dictPlot['NNOutput_Boson_Pt']]*np.cos(plotData[:,dictPlot['NNOutput_Boson_Phi']])

	n, bins, patches = plt.hist(histDataULong, num_bins, facecolor='green', alpha=0.5)
	plt.xlabel('U Longitudinal from BosonPt and BosonPhi')
	plt.ylabel('Entries')
	plt.savefig("../plots/U_LongNN.png")
	plt.clf()
    """
    
   
    
    """
    for targetname in dictTarget:
	histDataNN = plotData[bosonthreshold<plotData[:,dictPlot['Boson_Pt']],dictPlot[targetname]] - plotData[bosonthreshold<plotData[:,dictPlot['Boson_Pt']],dictPlot['NNOutput_%s'%targetname]]
	print('NN Phi shape: ',histDataNN.shape)
	if 'NNOutput_Boson_Phi' in dictPlot:
	    
	    for event in range(0,histDataNN.shape[0]):
		if histDataNN[event] > math.pi:
		    histDataNN[event] -= 2*math.pi
		if histDataNN[event] < -math.pi:
		    histDataNN[event] += 2*math.pi
	    
	    MSE = (histDataNN**2).mean()
	    n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
	    plt.xlabel('Variance %s from true Boson Phi. MSE: %f'%('NN_OutputPhi', MSE))
	    plt.ylabel('Entries')
	    plt.savefig("../plots/PhiVarianceNNOutputPhi.png")
	    plt.clf()
    
	elif 'NNOutput_targetPhiFromSlimmed' in dictPlot:
	    histDataNN = plotData[bosonthreshold<plotData[:,dictPlot['Boson_Pt']],dictPlot['Boson_Phi']] + math.pi - (plotData[bosonthreshold<plotData[:,dictPlot['Boson_Pt']],dictPlot['NNOutput_targetPhiFromSlimmed']] + plotData[bosonthreshold<plotData[:,dictPlot['Boson_Pt']],dictPlot['recoilslimmedMETs_Phi']])
	    print('NNOutput_targetPhiFromSlimmed',' phi shape: ',histDataNN.shape)
	    
	    for event in range(0,histDataNN.shape[0]):
		if histDataNN[event] > math.pi:
		    histDataNN[event] -= 2*math.pi
		if histDataNN[event] < -math.pi:
		    histDataNN[event] += 2*math.pi
		#if histDataNN[event] > math.pi:
		    #histDataNN[event] -= 2*math.pi
	    
	    MSE = (histDataNN**2).mean()
	    n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
	    plt.xlabel('Variance %s from true Boson Phi. MSE: %f'%('NNOutput_targetPhiFromSlimmed', MSE))
	    plt.ylabel('Entries')
	    plt.savefig("../plots/PhiVariance%s.png"%'NNOutput_targetPhiFromSlimmed')
	    plt.clf()
	else:
	    n, bins, patches = plt.hist(histDataNN, num_bins, facecolor='green', alpha=0.5)
	    plt.xlabel('Target - %s'%targetname)
	    plt.ylabel('Entries')
	    plt.savefig("../plots/NNVariance_%s.png"%targetname)
	    plt.clf()
	
	# normal distribution center at x=0 and y=5
	plt.hist2d(plotData[bosonthreshold<plotData[:,dictPlot['Boson_Pt']],dictPlot[targetname]], histDataNN,bins = 80, norm=LogNorm())
	#plt.ylim([-0.25,0.25])
	plt.xlabel(targetname)
	plt.ylabel('Variance of (Prediction-Target)')
	plt.savefig("../plots/NNVariance2D_%s.png"%targetname)
	plt.clf()
    
    
    
    
    

    make_ControlPlots(plotData, dictPlot, 'LongZCorrectedRecoil_LongZ', 'Boson_Pt',10,200,10,)

    make_ControlPlots(plotData, dictPlot, 'recoilslimmedMETs_LongZ', 'Boson_Pt',10,200,10,)
    
    make_ControlPlots(plotData, dictPlot, 'recoilslimmedMETsPuppi_LongZ', 'Boson_Pt',10,200,10,)

    if 'NNOutput_Boson_Pt' in dictPlot and 'NNOutput_Boson_Phi' in dictPlot:
	make_ControlPlots(plotData, dictPlot, 'NNOutput_Boson_Pt', 'Boson_Pt',10,200,10,)

    if 'NNOutput_targetPhiFromSlimmed' and 'NNOutput_targetRecoilFromSlimmed' in dictPlot:
	dictPlot['NNOutput_LongZ'] = plotData.shape[1]
	plotData = np.hstack((plotData, np.array(plotData[:,dictPlot['NNOutput_targetRecoilFromSlimmed']]*plotData[:,dictPlot['recoilslimmedMETs_Pt']]*np.cos(plotData[:,dictPlot['recoilslimmedMETs_Phi']]+plotData[:,dictPlot['NNOutput_targetPhiFromSlimmed']])).reshape(plotData.shape[0],1)))
	make_ControlPlots(plotData, dictPlot, 'NNOutput_LongZ', 'Boson_Pt',10,200,10,)
	

    plt.figure(4)
    legend = plt.legend(loc='lower right', shadow=True)
    plt.savefig('../plots/Response_ALL_vs_Boson_Pt')
    plt.clf()
    plt.figure(5)
    legend = plt.legend(loc='lower right', shadow=True)
    plt.savefig('../plots/Variance_ALL_vs_Boson_Pt')
    plt.clf()
    plt.figure(6)
    legend = plt.legend(loc='lower right', shadow=True)
    plt.savefig('../plots/Resolution_ALL_vs_Boson_Pt')
    plt.clf()
    plt.figure(0)
    
    
    
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
            l_in, num_units=500,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotNormal())

    # We'll now add dropout of 50%:
    #l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.1)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=500,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotNormal())


    l_hid3 = lasagne.layers.DenseLayer(
            l_hid2, num_units=500,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotNormal())
    
    # 50% dropout again:
    #l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.1)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid3, num_units=targetcount,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())
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
    X_trainMVA, y_trainMVA, X_valMVA, y_valMVA, X_testMVA, y_testMVA, weights, dictTarget, plotData, dictPlot, meanPlot, stdPlot = load_datasetcsv(config)
    
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    
    print(X_trainMVA.shape)
    print(y_trainMVA.shape)
    print(X_valMVA.shape)
    print(X_testMVA.shape)
    
    for name in dictTarget:
	print('Target %s:'%name,' mean: ', y_trainMVA[:,dictTarget[name]].mean())
	print('Target %s:'%name,' std: ', y_trainMVA[:,dictTarget[name]].std())
	print('Target %s:'%name,' min: ', y_trainMVA[:,dictTarget[name]].min())
	print('Target %s:'%name,' max: ', y_trainMVA[:,dictTarget[name]].max())
   
    # Prepare Theano variables for inputs and targets
    input_var = T.fmatrix('inputs')
    target_var = T.fmatrix('targets')
    
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
    

    
    exec("loss = {}".format(trainingconfig['trainingLossFunction'])) 
    loss = loss * weights
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
    prediction = lasagne.layers.get_output(network, deterministic=True)
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

    exec("test_loss = {}".format(trainingconfig['testLossFunction'])) 
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)
    
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
	for epoch in range(trainingconfig['trainingEpochs']):
	    # In each epoch, we do a full pass over the training data:
	    train_err = 0
	    train_batches = 0
	    start_time = time.time()
	    for batch in iterate_minibatches(X_trainMVA, y_trainMVA, trainingconfig['batchSize'], shuffle=True):
		inputs, targets = batch
		train_err += train_fn(inputs, targets)
		train_batches += 1

	    # And a full pass over the validation data:
	    val_err = 0
	    val_acc = 0
	    val_batches = 0
	    for batch in iterate_minibatches(X_valMVA, y_valMVA, trainingconfig['batchSize'], shuffle=False):
		inputs, targets = batch
		err = val_fn(inputs, targets)
		val_err += err
		#val_acc += acc
		val_batches += 1

	    # Then we print the results for this epoch:
	    print("Epoch {} of {} took {:.3f}s".format(
		epoch + 1, trainingconfig['trainingEpochs'], time.time() - start_time))
	    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

	
	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_testMVA, y_testMVA, trainingconfig['batchSize'], shuffle=False):
	    inputs, targets = batch
	    err = val_fn(inputs, targets)
	    test_err += err
	    #test_acc += acc
	    test_batches += 1
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

	#Optionally, you could now dump the network weights to a file like this:
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
    parser.add_argument('-i', '--inputconfig', default='../configs/config.json', help='Path to configurations file')
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
