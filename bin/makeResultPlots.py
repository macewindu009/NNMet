import csv
import numpy as np
import matplotlib.pyplot as plt


"""
Resolution_Int_NVertex_MintoMax,ResCorr_Max_Boson_Pt_MintoMax,ResCorr_Int_Boson_Pt_MintoMax,Resolution_Int_NVertex_MintoCut,ResCorr_Min_NVertex_MintoCut,Resolution_Min_NVertex_MintoCut,Resolution_Int_NVertex_CuttoMax,ResCorr_Int_NVertex_MintoCut,ResCorr_Int_NVertex_CuttoMax,Resolution_Min_NVertex_MintoMax,ResCorr_Max_NVertex_MintoCut,Resolution_Min_Boson_Pt_MintoMax,Resolution_Int_Boson_Pt_MintoMax,Response_Chi_NVertex_CuttoMax,ResCorr_Int_NVertex_MintoMax,Response_Chi_Boson_Pt_MintoMax,ResCorr_Max_NVertex_CuttoMax,Resolution_Min_NVertex_CuttoMax,ResCorr_Max_NVertex_MintoMax,ResCorr_Min_Boson_Pt_MintoMax,ResCorr_Min_NVertex_MintoMax,Response_Chi_NVertex_MintoCut,ResCorr_Min_NVertex_CuttoMax,Response_Chi_Boson_Pt_CuttoMax,Response_Chi_NVertex_MintoMax
"""


reader=csv.reader(open('results.csv',"rb"),delimiter=',')
datacsv=list(reader)
print('csv', len(datacsv[1]))
header = np.array(datacsv[0]).astype(np.str)
header = header[1:]

print('header: ', header.shape)
#print(header)
#data =np.array(datacsv[1:][1:]).astype(np.float32)
data = datacsv[1:]
#print(datacsv[:][][1:])
print(len(data[0][1:]))
#print(data.shape)

#Only print several variables
excludeVariables = False 
exclusiveVariables = ['SlimmedMet',
'WithoutJets',
'NewWithoutJets',
'NewWithEverything',
'NewWithoutJetsWithElec'
]
"""
exclusiveVariables = ['NewOnlySlimmedAndPuppi',
'NewWithoutJets',
'NewWithoutJetsAlsoPhi',
'NewWithoutJetsWithElec',
'NewWithoutPuppiAndJets',
'NewWithEverything',
'NewWithoutJetsAndNVertex',
'NewWithoutJetsAndNVertexAlsoPhi',
'NewWithoutJetsAndNVertexWithElec',
'NewWithoutJetsAndNVertexWithElecAlsoPhi',
'WithoutJets',
'WithoutPuppiAndJets',
'WithEverything',
'SlimmedMet',
'OnlySlimmedAndPuppi'
]
"""



inputData = np.empty(shape=[0,len(data[0][1:])]).astype(np.float32)
#dictInput = np.empty(shape=[0,1]).astype(np.str)
dictInput = []
for index, row in enumerate(data):
    #dictInput = np.vstack((dictInput, np.array(row[0]).reshape(1,1)))
    
    if excludeVariables:
	if row[0] in exclusiveVariables:
	    print(row[0],index)
	    dictInput.append(row[0])
	    inputData = np.vstack((inputData, np.array(row[1:]).reshape(1,len(data[0][1:]))))
    else:
	print(row[0],index)
	dictInput.append(row[0])
	inputData = np.vstack((inputData, np.array(row[1:]).reshape(1,len(data[0][1:]))))
    
for index in range(0,len(header)):
    print(header[index],' : ',index)
    
#print(inputData.shape)
#print(dictInput)



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(35.0, 20.0))

datasets = dictInput
y_pos = np.arange(inputData.shape[0])
for index in range(0,len(header)):
    if header[index] == 'Resolution_Int_NVertex_MintoMax':
	entry = index
dataBar = inputData[:,entry]
plt.setp(axes, yticks=y_pos,yticklabels=datasets)
axes[0,0].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[0,0].set_yticks(y_pos, datasets)
axes[0,0].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'Resolution_Int_Boson_Pt_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[0,1].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[0,1].set_yticks(y_pos, datasets)
axes[0,1].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'Resolution_Min_NVertex_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[1,0].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[1,0].set_yticks(y_pos, datasets)
axes[1,0].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'Resolution_Min_Boson_Pt_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[1,1].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[1,1].set_yticks(y_pos, datasets)
axes[1,1].set_title(header[entry])

fig.suptitle("Resolution Plots")

plt.savefig(("ResolutionComparison.png"))
plt.clf()
#plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(35.0, 20.0))

datasets = dictInput
y_pos = np.arange(inputData.shape[0])
for index in range(0,len(header)):
    if header[index] == 'Response_Chi_Boson_Pt_CuttoMax':
	entry = index
dataBar = inputData[:,entry]
plt.setp(axes, yticks=y_pos,yticklabels=datasets)
axes[0,0].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[0,0].set_yticks(y_pos, datasets)
axes[0,0].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'Response_Chi_NVertex_CuttoMax':
	entry = index
dataBar = inputData[:,entry]
axes[1,0].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[1,0].set_yticks(y_pos, datasets)
axes[1,0].set_title(header[entry])
"""
for index in range(0,len(header)):
    if header[index] == 'Resolution_Min_Boson_Pt_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[1,1].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[1,1].set_yticks(y_pos, datasets)
axes[1,1].set_title(header[entry])
"""
for index in range(0,len(header)):
    if header[index] == 'Response_Chi_NVertex_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[0,1].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[0,1].set_yticks(y_pos, datasets)
axes[0,1].set_title(header[entry])
fig.suptitle("Response Plots")

plt.savefig("ResponseComparison.png")
plt.clf()



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(35.0, 20.0))

datasets = dictInput
y_pos = np.arange(inputData.shape[0])
for index in range(0,len(header)):
    if header[index] == 'ResCorr_Int_Boson_Pt_MintoMax':
	entry = index
dataBar = inputData[:,entry]
plt.setp(axes, yticks=y_pos,yticklabels=datasets)
axes[0,0].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[0,0].set_yticks(y_pos, datasets)
axes[0,0].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'ResCorr_Int_NVertex_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[1,0].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[1,0].set_yticks(y_pos, datasets)
axes[1,0].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'ResCorr_Min_Boson_Pt_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[0,1].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[0,1].set_yticks(y_pos, datasets)
axes[0,1].set_title(header[entry])

for index in range(0,len(header)):
    if header[index] == 'ResCorr_Min_NVertex_MintoMax':
	entry = index
dataBar = inputData[:,entry]
axes[1,1].barh(y_pos, dataBar, align='center', alpha=0.4)
axes[1,1].set_yticks(y_pos, datasets)
axes[1,1].set_title(header[entry])

fig.suptitle("ResponseCorrected Plots")

plt.savefig(("RespCorrComparison.png"))
plt.clf()



datasets = dictInput
y_pos = np.arange(inputData.shape[0])
entry = 23
dataBar = inputData[:,entry]
#plt.setp(axes, yticks=y_pos,yticklabels=datasets)
plt.barh(y_pos, dataBar, align='center', alpha=0.4)
plt.yticks(y_pos, datasets)
plt.title(header[entry])

plt.savefig(("ResponseBosonInt.svg"))
plt.clf()

datasets = dictInput
y_pos = np.arange(inputData.shape[0])
entry = 12
dataBar = inputData[:,entry]
#plt.setp(axes, yticks=y_pos,yticklabels=datasets)
plt.barh(y_pos, dataBar, align='center', alpha=0.4)
plt.yticks(y_pos, datasets)
plt.title(header[entry])

plt.savefig(("ResolutionBosonInt.svg"))
plt.clf()



