import os

x = [
'NewWithoutJetsAlsoPhi',
'NewWithoutJetsWithElecAlsoPhi',
'NewWithoutPuppiAndJetsAlsoPhi',
'NewWithoutPuppiAndJetsWithElecAlsoPhi',
'NewWithoutPuppiAlsoPhi',
'NewWithoutPuppiWithElecAlsoPhi',
'NewWithEverythingWithElec'
'NewWithEverything'
]




for path in x:
    if not os.path.exists(path):
	os.makedirs(path)

