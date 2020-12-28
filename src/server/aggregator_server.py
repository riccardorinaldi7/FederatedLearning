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

import sys
# import time
import argparse
import zenoh
from zenoh import Zenoh, Value
import torch
import torch.nn.functional as F
from torch import nn

# --- Command line argument parsing --- --- --- --- --- ---
parser = argparse.ArgumentParser(
    prog='z_sub',
    description='zenoh sub example')
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
parser.add_argument('--selector', '-s', dest='selector',
                    default='/federated/nodes/**',
                    type=str,
                    help='The selection of resources to subscribe.')

args = parser.parse_args()
conf = {"mode": args.mode}
if args.peer is not None:
    conf["peer"] = ",".join(args.peer)
if args.listener is not None:
    conf["listener"] = ",".join(args.listener)
selector = args.selector

# Hyperparameters for federated learning --- --- --- --- --- ---

num_clients = 2
# num_selected = 6
# num_rounds = 150
# epochs = 5
# batch_size = 32

trained_parameters = list()  # contains list of .pt filename


# -- function definitions  --- --- --- --- --- --- --- --- --- --- ---


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


def federated_averaging():
    if len(trained_parameters) == num_clients:
        print(">> [Federated averaging] begin averaging between {} models".format(len(trained_parameters)))
        client_models = list()
        for filename in trained_parameters:
            model = Classifier()
            model.load_state_dict(torch.load(filename))
            # model.eval() # not necessary if no inference must be processed
            client_models.append(model)
        print(">> [Federated averaging] {} model loaded".format(len(client_models)))
        # This will take simple mean of the weights of models #
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        # for model in client_models:
        #     model.load_state_dict(global_model.state_dict())
        torch.save(global_model.state_dict(), 'global_parameters.pt')
    else:
        print(">> [Federated averaging] other {} trained models required".format(num_clients-len(trained_parameters)))
    

def send_parameters_to_path(path):
    f = open('global_parameters.pt', 'rb')
    binary = f.read()
    f.close()
    value = Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, binary)
    # print('Model saved - zenoh.Value created')

    # --- send parameters with zenoh --- --- --- --- --- --- --- ---
    workspace.put(path, value)


def listener(change):
    print(">> [Subscription listener] received {:?} for {} : {} with timestamp {}"
          .format(change.kind, change.path, '' if change.value is None else change.value.encoding_descr(), change.timestamp))
    if change.value.encoding_descr() == 'application/octet-stream':
        node_id = change.path.split('/')[3]  # path = /federated/nodes/<node_id>
        filename = node_id + '.pt'
        f = open(filename, 'wb')
        f.write(bytearray(change.value.get_content()))
        f.close()
        print(">> [Subscription listener] Parameters file saved")
        if filename in trained_parameters:
            print(">> [Subscription listener] something gone wrong. Received twice from a client")
            return
        trained_parameters.append(filename)
        federated_averaging()
        return
        
    if change.value.encoding_descr() == 'text/plain':
        # print("String message arrived. Content: {}".format(change.value))
        if change.value.get_content() == 'join-round-request':
            print(">> [Subscription listener] received a request to join a round.")
            print("Yes")
            send_parameters_to_path('/federated/global')
            print("Parameters sent to /federated/global")
        else:
            print(">> [Subscription listener] Message content unknown")
        return

    else:
        print(">> Content: {}".format(change.value))
        return


global_model = Classifier()
torch.save(global_model.state_dict(), 'global_parameters.pt')

# initiate logging
zenoh.init_logger()

print("Opening session...")
z = Zenoh(conf)

print("New workspace...")
workspace = z.workspace()

print("Subscribe to '{}'...".format(selector))
sub = workspace.subscribe(selector, listener)

print("Press q to stop...")
c = '\0'
while c != 'q':
    c = sys.stdin.read(1)

sub.close()
z.close()
