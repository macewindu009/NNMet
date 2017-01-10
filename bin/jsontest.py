import json


with open('../configs/config.json', 'r') as f:
    config = json.load(f)



config2 = config['trainingdefault']

print(config2['targetVariables'])

print(config[config['activetraining']]['sizeHiddenLayers'])
