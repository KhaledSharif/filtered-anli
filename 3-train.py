import torch
import yaml

config = {}
with open('config.yaml') as f:
    config = yaml.load(f)['3-train']

input_file = config['input']
filter = torch.load(input_file)
print(filter.shape, filter)
