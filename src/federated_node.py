# Copyright (c) 2017, 2020 ADLINK Technology Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
# which is available at https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
#
# Contributors:
#   ADLINK zenoh team, <zenoh@adlink-labs.tech>

import json
import sys
import time
import argparse
import zenoh
from zenoh import Zenoh, Value

# --- Command line argument parsing --- --- --- --- --- ---
parser = argparse.ArgumentParser(
    prog='z_put',
    description='zenoh put example')
parser.add_argument('--mode', '-m', dest='mode',
                    default='peer',
                    choices=['peer', 'client'],
                    type=str,
                    help='The zenoh session mode.')
parser.add_argument('--peer', '-e', dest='peer',
                    metavar='LOCATOR',
                    action='append',
                    type=str,
                    help='Peer locators used to initiate the zenoh session.')
parser.add_argument('--listener', '-l', dest='listener',
                    metavar='LOCATOR',
                    action='append',
                    type=str,
                    help='Locators to listen on.')
parser.add_argument('--path', '-p', dest='path',
                    default='/demo/example/zenoh-python-put',
                    type=str,
                    help='The name of the resource to put.')
parser.add_argument('--value', '-v', dest='value',
                    default='Put from Python!',
                    type=str,
                    help='The value of the resource to put.')

args = parser.parse_args()
conf = { "mode": args.mode }
if args.peer is not None:
    conf["peer"] = ",".join(args.peer)
if args.listener is not None:
    conf["listener"] = ",".join(args.listener)
path = args.path
value = args.value

# --- torch code for training --- --- --- --- --- --- --- --- -
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
#import helper

print('Libraries loaded...')

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

print('Downloading dataset...')

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
#testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

print('Defining the model, criterion and optimizer...')

# TODO: Define your network architecture here
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
    
#Create the network, define the criterion and optimizer
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# TODO: Train the network here
epochs = 1

print("Start training with {} epochs...".format(epochs))

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
        
print('Model trained!')

torch.save(model.state_dict(), 'parameters.pt')

print('Model saved')

f = open('parameters.pt', 'rb')
bytes = f.read()
f.close
value = Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, bytes)


# --- zenoh-net code --- --- --- --- --- --- --- --- --- --- ---

# initiate logging
zenoh.init_logger()

print("Openning session...")
z = Zenoh(conf)

print("New workspace...")
workspace = z.workspace()

torch.save(model.state_dict(), './last.pt')
f = open('./last.pt')

print("Put Data ('{}': '{}')...".format(path, value))
workspace.put(path, value)


# --- Examples of put with other types:

# - Integer
# workspace.put('/demo/example/Integer', 3)

# - Float
# workspace.put('/demo/example/Float', 3.14)

# - Properties (as a Dictionary with str only)
# workspace.put('/demo/example/Properties', {'p1': 'v1', 'p2': 'v2'})

# - Json (str format)
# workspace.put('/demo/example/Json',
#               Value.Json(json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])))

# - Raw ('application/octet-stream' encoding by default)
# workspace.put('/demo/example/Raw', b'\x48\x69\x33'))

# - Custom
# workspace.put('/demo/example/Custom',
#               Value.Custom('my_encoding', b'\x48\x69\x33'))

z.close()
