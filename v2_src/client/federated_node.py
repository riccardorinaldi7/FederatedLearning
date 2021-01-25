# Copyright (c) 2020 Riccardo Rinaldi
#
# v2.0
#
# This program and the accompanying materials are made available under the
# terms of the GNU License 2.0 

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
import random

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
base_path = args.path                             # /federated/nodes
value = args.value
notification_path = base_path + '/notifications'  # /federated/nodes/notifications

# ADDED RANDOM INT FOR LOCAL USAGE
new_uuid = str(uuid.getnode()) + str(random.randint(1, 100))   # uuid + randint

# use UUID module to create a custom folder
node_path = base_path + "/" + new_uuid            # /federated/nodes/new_uuid
print("Using path: {}".format(node_path))

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
            log_ps = globals()['model'](images)
            loss = globals()['criterion'](log_ps, labels)

            globals()['optimizer'].zero_grad()
            loss.backward()
            globals()['optimizer'].step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")

    print('Model trained!')
    

def setup_training():
    # 4 - setup training and load parameters
    input("Parameters received. Press enter to load them into the model")  
    globals()['model'] = Classifier()
    globals()['criterion'] = nn.NLLLoss()
    globals()['optimizer'] = optim.Adam(model.parameters(), lr=0.003)
    globals()['model'].load_state_dict(torch.load('global.pt'))
    

def save_model():
    # input("Press enter to send parameters to the server")
    torch.save(globals()['model'].state_dict(), 'local.pt')
    f = open('local.pt', 'rb')
    binary = f.read()
    f.close()
    file_value = zenoh.Value.Raw(zenoh.net.encoding.APP_OCTET_STREAM, binary)
    print('Model saved - zenoh.Value created')
    return file_value


# --- Reply to get from server --- --- --- --- --- ---
def eval_callback(get_request):
    print(">> [Eval listener] received get with selector: {}".format(get_request.selector))
    global round_started
    round_started = True
    
    # The returned Value is a StringPath that identify the path where the client will find 
    # the global parameters to use for the training. For example:
    # - "/federated/nodes/new_uuid?(path=/federated/global)" : the Eval function does a GET
    #      on "/federated/global" and uses the 1st result to regenerate the global_param.pt file
    global_param_path = get_request.selector.properties.get('path')
    if global_param_path.startswith('/'):
        print('>> [Eval listener] Try to get the model at path: {}'.format(global_param_path))
        result = workspace.get(global_param_path)
        print('>> [Eval listener] checking result...')
        if len(result) == 1 and result[0].value.encoding_descr() == 'application/octet-stream':
            file_bytes = result[0].value.get_content()
            f = open('global.pt', 'wb')
            f.write(bytearray(file_bytes))
            f.close()
            print(">> [Eval listener] global parameters received")
            setup_training()
            download_and_train()
            file_value = save_model()
            print(">> [Eval listener] Model trained. Sending back the update...")
            get_request.reply(node_path, file_value)
            global training_done
            training_done = True
    else:
        print(">> [Eval listener] Selector property is not a path")
        
        
# --- zenoh-net code --- --- --- --- --- --- --- --- --- --- ---

# initiate logging
zenoh.init_logger()

print("Opening session...")
z = Zenoh(conf)

print("New workspace...")
workspace = z.workspace()


# --- Federated Protocol --- --- --- --- --- --- --- --- --- ---

# 1 - Node tells the server it has enough data --- --- --- --- --- ---
training_done = False
round_started = False
print("I have enough data to train a model")
input("Press enter to send the notification")
workspace.put(notification_path, new_uuid)                   # /federated/nodes/notifications

# 2 - The client waits for server's eval with model passed as parameter
z_eval = workspace.register_eval(node_path, eval_callback)     # /federated/nodes/new_uuid

# 3 - Wait 90 seconds for a get then close the eval registration
time.sleep(90)
print('Closing eval...')
z_eval.close()

# 6 - Ending message
time.sleep(1)
if round_started:
    print('Training at work...')
    while not training_done:
        time.sleep(3)
else:
    print("Waited 90 seconds but there's been no word from the server")
    
input('Press enter to terminate...')

time.sleep(1)

z.close()
