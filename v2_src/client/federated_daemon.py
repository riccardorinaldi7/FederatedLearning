# Set in crontab
# 00  2  *  *  *  /usr/bin/python3 ~/federated_daemon.py >> federated_daemon.log
#
# remember to configure the load requirements

from datetime import datetime
import subprocess
import os
import json
from pathlib import Path

# which info?
# 1. system info
# 2. loadavg file
# 3. uptime cmd

choice = 1
train_data = False

# current date and time
timestamp = datetime.timestamp(datetime.now())
print("timestamp =", timestamp)

if choice == 1:
    # generate json file with system info
    subprocess.call("./sysinfo-to-json.sh", shell=True)

    # read json file
    home = str(Path.home())
    f = open(home + '/sysinfo.json')
    data = json.loads(f.read())[0]

    mem_size = data['children'][0]['children'][1]['size']/1024/1024
    print('mem_size: {} GB'.format(mem_size))

    cpu_count = data['children'][0]['children'][2]['physid']
    print('cpu_count: {}'.format(cpu_count))

    net_ext = data['children'][0]['children'][3]['children'][3]['logicalname']
    net_ext_cap = data['children'][0]['children'][3]['children'][3]['capacity']/1000000000
    print('interface {}: {} Gb/s'.format(net_ext, net_ext_cap))

    net_int = data['children'][0]['children'][3]['children'][8]['logicalname']
    net_int_cap = data['children'][0]['children'][3]['children'][8]['capacity']/1000000000
    print('interface {}: {} Gb/s'.format(net_ext, net_ext_cap))
    
    # TODO: check here whether to run the federated-client or not
    global train_data
    train_data = True

elif choice == 2:
    res = subprocess.run(['cat', '/proc/loadavg'], stdout=subprocess.PIPE)
    load = res.stdout.decode('UTF-8').split()
    print('System load: {}'.format(loads))
    
    # TODO: check here whether to run the federated-client or not
    global train_data
    train_data = True

elif choice == 3:
    res = subprocess.run(['uptime'], stdout=subprocess.PIPE)
    load = res.stdout.decode('UTF-8').split()
    print('System load: {}'.format(loads))
    
    # TODO: check here whether to run the federated-client or not
    global train_data
    train_data = True
    
else:
    print("Invalid choice")
    exit(1)
    
if train_data:
    exit_code = os.system("/usr/bin/python3 ~/federated_node.py > ~/federated.log")
else:
    print("It's not time for training yet")
    
exit(0)
    
