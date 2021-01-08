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
import time
import argparse
import zenoh
from zenoh import Zenoh, Value
import torch
import torch.nn.functional as F
from torch import nn
import os

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
                    default='/federated/nodes',
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
participants = list()  # contains uuid of the participant nodes
federated_round_in_progress = False


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


# -- SAVE AND PUT THE MODEL ON /federated/global/parameters
def save_and_put_global_parameters(dictionary):
    torch.save(dictionary, 'global_parameters.pt')
    print('>> [Save and Put] global_parameters.pt saved')

    f = open('global_parameters.pt', 'rb')
    binary = f.read()
    f.close()
    file_value = Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, binary)

    # --- send parameters with zenoh --- --- --- --- --- --- --- ---
    global_parameters_path = '/federated/global/parameters'
    print("Put global parameters into {}".format(global_parameters_path))
    workspace.put(global_parameters_path, file_value)


def clean_protocol():
    participants.clear()
    trained_parameters.clear()
    global federated_round_in_progress
    federated_round_in_progress = False
    print("Ready for another round...")


def federated_averaging():
    if len(trained_parameters) == num_clients:
        print(">> [Federated averaging] begin averaging between {} models".format(num_clients))
        client_models = list()
        for filename in trained_parameters:
            model = Classifier()
            model.load_state_dict(torch.load(filename))
            # model.eval() # not necessary if no inference must be processed
            client_models.append(model)
        print(">> [Federated averaging] {} model loaded".format(len(client_models)))

        # This will take simple mean of the weights of models. Code from towardsdatascience.com
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)

        # global_model.load_state_dict(global_dict)
        save_and_put_global_parameters(dictionary=global_dict)

        clean_protocol()  # keep ready for another round
    else:
        print(">> [Federated averaging] other {} trained models required".format(num_clients-len(trained_parameters)))
    

def local_param_listener(change):
    print(">> [Local param listener] received local parameter")
    global federated_round_in_progress
    if not federated_round_in_progress:
        print(">> [Local param listener] no round in progress")

    if change.value.encoding_descr() == 'application/octet-stream':
        node_id = change.path.split('/')[3]  # path = /federated/nodes/<node_id>

        if node_id not in participants:
            print(">> [Local param listener] node not in this round: parameters discarded")
            return

        filename = node_id + '.pt'
        f = open(filename, 'wb')
        f.write(bytearray(change.value.get_content()))
        f.close()
        print(">> [Local param listener] Parameters file saved")
        if filename in trained_parameters:
            print(">> [Local param listener] something gone wrong. Received twice from a client")
            return
        trained_parameters.append(filename)
        federated_averaging()

    else:
        print(">> [Local param listener] Not a parameter file. Content: {}".format(change.value))


# for debug purposes
def simple_listener(change):
    print("Hey! Something happened at {}".format(change.path))


def send_parameters_to_all():
    global workspace

    f = open('global_parameters.pt', 'rb')
    binary = f.read()
    f.close()
    value = Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, binary)
    # print('Parameters file loaded - zenoh.Value created and ready to be sent')

    # --- send parameters with zenoh --- --- --- --- --- --- --- ---
    for node_id in participants:
        global_node_path = selector + '/' + node_id + '/global_params'
        workspace.put(global_node_path, value)  # /federated/nodes/<node_id>/global_params
        print(">> [Global params sender] global_params sent to {}".format(global_node_path))

    # every node is logged in, let's wait for the updated_parameters
    # local_params_selector = selector + '/*/local'
    # local_sub = workspace.subscribe('/federated/test', simple_listener)
    # print(">> [Global params sender] subscribed to '{}'...".format(local_params_selector))


# -- LISTEN ON /federated/nodes/*/messages PATH-- -- -- -- -- -- -- -- -- -- -- --
def message_listener(change):
    print(">> [Message listener] received {} on {} : {}".format(change.value.encoding_descr(), change.path, '' if change.value is None else change.value.get_content()))
    global federated_round_in_progress
    node_id = change.path.split('/')[3]  # sender's id
    if change.value.encoding_descr() == 'text/plain':

        # 2 - Request to join a federated session
        if change.value.get_content() == 'join-round-request':
            # check here whether accept the request or not
            print(">> [Message listener] received a request to join a round by {}.".format(node_id))

            if federated_round_in_progress:
                print(">> [Message listener] Request rejected: a federated round is already running")
                return

            participants.append(node_id)
            if len(participants) == num_clients:
                federated_round_in_progress = True
                send_parameters_to_all()           # all the clients are in, send global params to everyone
            else:
                print(">> [Message listener] {} participants missing".format(num_clients-len(participants)))

        # --- Message unknown --- --- --- --- --- --- --- --- --- --- --- ---
        else:
            print(">> [Message listener] Message content unknown")

    else:
        print(">> [Message Listener] The message from {} is not a string".format(node_id))


#  At server startup create a base global_params file as long as another file exists. In that case, the user decide what to do
if os.path.isfile("global_parameters.pt"):
    res = input("A global_parameters file already exists. Overwrite the file?\nThis action can overwrite a smarter model. (y/N) ")
    if len(res) > 0 and res[0] == 'y':
        global_model = Classifier()
        torch.save(global_model.state_dict(), 'global_parameters.pt')  # be aware that this can overwrite a smarter model if this application is stopped and restarted
else:
    global_model = Classifier()
    torch.save(global_model.state_dict(), 'global_parameters.pt')  # be aware that this can overwrite a smarter model if this application is stopped and restarted

# initiate logging
zenoh.init_logger()

print("Opening session...")
z = Zenoh(conf)

print("New workspace...")
workspace = z.workspace()


# 1 - Listen for messages
msg_selector = selector + '/*/messages'
print("Subscribed to '{}'...".format(msg_selector))
msg_subscriber = workspace.subscribe(msg_selector, message_listener)

# n - Listen for pt file from nodes
local_params_selector = selector + '/*/local'
local_sub = workspace.subscribe(local_params_selector, local_param_listener)
print("subscribed to '{}'...".format(local_params_selector))

print("Press q to stop...")
c = '\0'
while c != 'q':
    c = sys.stdin.read(1)

time.sleep(1)

z.close()
