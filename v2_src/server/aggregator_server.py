# Copyright (c) 2020 Riccardo Rinaldi
#
# v2.0
#
# This program and the accompanying materials are made available under the
# terms of the GNU License 2.0 

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
selector = args.selector                        # /federated/nodes

# Hyperparameters for federated learning --- --- --- --- --- ---

num_clients = 2
# num_selected = 6
# num_rounds = 150
# epochs = 5
# batch_size = 32

trained_parameters = list()  # contains list of .pt filename
participants = list()  # contains uuid of the participant nodes
federated_round_in_progress = False


# --- --- --- --- --- --- --- --- --- --- --- ---


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


def save_and_put_global_parameters(dictionary):
    torch.save(dictionary, 'global.pt')
    print('>> [Save and Put] global.pt saved')

    f = open('global.pt', 'rb')
    binary = f.read()
    f.close()
    file_value = Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, binary)

    # --- send parameters with zenoh --- --- --- --- --- --- --- ---
    global_parameters_path = '/federated/nodes/global'
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
        global global_model
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)

        # global_model.load_state_dict(global_dict)
        save_and_put_global_parameters(dictionary=global_dict)

        clean_protocol()  # keep ready for another round
    else:
        print(">> [Federated averaging] other {} trained models required".format(num_clients-len(trained_parameters)))


# for debug purpouses
def simple_listener(change):
    print("Hey! Something happened at {}".format(change.path))
    
    
def get_thread_function(uuid):
    print(">> [Thread] Thread {}: starting".format(uuid))
    selector = '/federated/nodes/' + uuid + '?(path=/federated/nodes/global)'
    print(">> [Thread] Thread {} getting on selector {}".format(uuid, selector))
    data = workspace.get(selector)
    filename = uuid + '.pt'
    f = open(filename, 'wb')
    f.write(bytearray(data[0].value.get_content()))
    f.close()
    trained_parameters.append(filename)
    print(">> [Thread] Thread {}: finishing".format(uuid))


def send_parameters_to_all():
    # here all the gets to the clients --- --- --- --- --- --- --- ---
    # selector = '/federated/nodes/new_uuid?(path=/federated/nodes/global)
    threads = list()
    for uuid in participants:
        print(">> [Send Parameters] create and start thread %s.".format(uuid))
        x = threading.Thread(target=get_thread_function, args=(uuid,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        print("Main    : before joining thread {}.".format(index))
        thread.join()
        print("Main    : thread {} done".format(index))
    
    federated_averaging()


# -- listen on /federated/nodes/notifications for ready clients -- -- -- -- -- -- -- -- -- -- -- --
def notification_listener(change):
    print(">> [Notification listener] received {} on {} : {}".format(change.value.encoding_descr(), change.path, '' if change.value is None else change.value.get_content()))
    global federated_round_in_progress
    
    if change.value.encoding_descr() == 'text/plain':

        # 2 - A client notified that it's ready
        node_id = change.value.get_content()
        print(">> [Message listener] {} is ready for training".format(node_id))

        if federated_round_in_progress:
            print(">> [Message listener] Request rejected: federated round already running")
            return

        participants.append(node_id)
        if len(participants) == num_clients:
            federated_round_in_progress = True
            send_parameters_to_all()           # all the clients are in, send global params to everyone
        else:
            print(">> [Message listener] {} participants missing".format(num_clients-len(participants)))

    else:
        print(">> [Message Listener] Forrbidden value of type {}".format(change.value.encoding_descr()))

# --- INIT FEDERATED LEARNING -------------------------

global_model = Classifier()

#  At server startup create a base global_params file as long as another file exists. In that case, the user decide what to do
if os.path.isfile("global.pt"):
    res = input("A global_parameters file already exists. Overwrite the file?\nThis action can overwrite a smarter model. (y/N) ")
    if len(res) > 0 and res[0] == 'y':
        torch.save(global_model.state_dict(), 'global.pt')  # be aware that this can overwrite a smarter model if this application is stopped and restarted
    else:
        global_model.load_state_dict(torch.load('global.pt'))
else:
    torch.save(global_model.state_dict(), 'global.pt')  # be aware that this can overwrite a smarter model if this application is stopped and restarted

# initiate logging
zenoh.init_logger()

print("Opening session...")
z = Zenoh(conf)

print("New workspace...")
workspace = z.workspace()

# 0 - Put global model to a path where clients can get it
print(">> [Send Parameters] Put global model in /federated/nodes/global")
save_and_put_global_parameters(global_model.state_dict())

# 1 - Listen for notifications
notification_selector = selector + '/notifications'
print("Subscribed to '{}'...".format(noitification_selector))
notification_subscriber = workspace.subscribe(notification_selector, notification_listener)


print("Press q to stop...")
c = '\0'
while c != 'q':
    c = sys.stdin.read(1)

time.sleep(1)

z.close()
