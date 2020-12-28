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

# import json
# import sys
import time
import argparse
import zenoh
from zenoh import Zenoh, Value
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import uuid

# import helper

print('Libraries loaded...')

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
                    default='/federated/nodes',
                    type=str,
                    help='The name of the resource to put.')
parser.add_argument('--value', '-v', dest='value',
                    default='Put from Python!',
                    type=str,
                    help='The value of the resource to put.')

args = parser.parse_args()
conf = {"mode": args.mode}
if args.peer is not None:
    conf["peer"] = ",".join(args.peer)
if args.listener is not None:
    conf["listener"] = ",".join(args.listener)
path = args.path
value = args.value

# use UUID module to create a custom folder
path = path + "/" + str(uuid.getnode())
print("Using path: {}".format(path))

# --- zenoh-net code --- --- --- --- --- --- --- --- --- --- ---

# initiate logging
zenoh.init_logger()

print("Opening session...")
z = Zenoh(conf)

print("New workspace...")
workspace = z.workspace()

# --- Define network architecture, criterion and optimizer --- --- --- --- ---

print('Defining the model...')


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


# --- torch code for training --- --- --- --- --- --- --- --- -
def download_and_train():
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    print('Downloading dataset...')

    # Download and load the training data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    # testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Train the network here
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
            print(f"Training loss: {running_loss / len(trainloader)}")

    print('Model trained!')


# --- save the parameters and wrap them into a zenoh.Value --- ---
def save_and_send_parameters():
    torch.save(model.state_dict(), 'my_parameters.pt')
    f = open('my_parameters.pt', 'rb')
    binary = f.read()
    f.close()
    value = Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, binary)
    print('Model saved - zenoh.Value created')

    # --- send parameters with zenoh --- --- --- --- --- --- --- ---

    print("Put Data into {}".format(path))
    workspace.put(path, value)


# --- Listen for "messages" from the server --- --- --- --- --- ---
def listener(change):
    print(">> [Subscription listener] received {:?} for {} : {} with timestamp {}"
          .format(change.kind, change.path, '' if change.value is None else change.value.encoding_descr(),
                  change.timestamp))
    if change.value.encoding_descr() == 'application/octet-stream':
        f = open('global_parameters.pt', 'wb')
        f.write(bytearray(change.value.get_content()))
        f.close()
        print(">> File saved")
        global federated_round_permitted
        federated_round_permitted = True

    else:
        print(">> Content: {}".format(change.value))

# --- Federated Protocol --- --- --- --- --- --- --- --- --- ---


# 1 - Node asks to join a federated round --- --- --- --- --- ---
federated_round_permitted = False
print("I have enough data to train a model. Let's ask to the server if I can get into a federated learning session!")
workspace.put(path, "join-round-request")

# 2 - Node waits a response from the server. If the server accepts the request, it will write the parameters in /federated/nodes/<node_id> --- ---
subscriber = workspace.subscribe('/federated/global', listener)  # listen what the server has to say

# 3 - Node waits 60 second, then if no parameters are written, it considers his request rejected by the server
time.sleep(10)  # wait a minute a response, then unsubscribe the listener
subscriber.close()
print("Waited for 10 seconds. Listener unsubscribed...")

# 4 - if the server responded with the parameters, the training begin
if federated_round_permitted:
    # 5 - local training
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model.load_state_dict(torch.load('global_parameters.pt'))
    input("Press enter to train the model")
    download_and_train()
    # 6 - computed parameters are sent to the server for the aggregation
    input("Press enter to send parameters to the server")
    save_and_send_parameters()

z.close()
