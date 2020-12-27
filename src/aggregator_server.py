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
from zenoh import Zenoh
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

trained_parameters = list()


# zenoh-net code  --- --- --- --- --- --- --- --- --- --- ---


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


global_model = Classifier()


def federated_averaging():
    if len(trained_parameters) == num_clients:
        print(">> [Federated averaging] begin averaging between {} models".format(len(trained_parameters)))
        client_models = list()
        for filename in trained_parameters:
            model = Classifier()
            model.load_state_dict(torch.load(filename))
            model.eval()
            client_models.append(model)
        print(">> [Federated averaging] {} model loaded".format(len(client_models)))
        # This will take simple mean of the weights of models #
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        # for model in client_models:
        #     model.load_state_dict(global_model.state_dict())
        torch.save(global_model.state_dict(), 'global_model.pt')
    else:
        return


def listener(change):
    print(">> [Subscription listener] received {:?} for {} : {} with timestamp {}"
          .format(change.kind, change.path, '' if change.value is None else change.value.encoding_descr(), change.timestamp))
    if change.value.encoding_descr() == 'application/octet-stream':
        node_id = change.path.split('/')[-1]
        filename = node_id + '.pt'
        f = open(filename, 'wb')
        f.write(bytearray(change.value.get_content()))
        f.close()
        print(">> File saved")
        if filename in trained_parameters:
            print(">> [Subscription listener] something gone wrong. Received twice from a client")
            return
        trained_parameters.append(filename)
        federated_averaging()

    else:
        print(">> Content: {}".format(change.value))


# initiate logging
zenoh.init_logger()

print("Openning session...")
zenoh = Zenoh(conf)

print("New workspace...")
workspace = zenoh.workspace()

print("Subscribe to '{}'...".format(selector))
sub = workspace.subscribe(selector, listener)

print("Press q to stop...")
c = '\0'
while c != 'q':
    c = sys.stdin.read(1)

sub.close()
zenoh.close()
