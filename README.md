# Federated Learning and Edge Computing

Download fog-image file running `wget https://github.com/riccardorinaldi7/FederatedLearning/releases/download/v1.0-alpha/fog-node-image.tar.xz`

## Install Fog05 from deb packages

```
sudo apt update
sudo apt upgrade
wget https://github.com/eclipse-fog05/fog05/releases/download/v0.2.2/zenoh_0.3.0-1_amd64.deb
wget https://github.com/eclipse-fog05/fog05/releases/download/v0.2.2/fog05_0.2.1-1_amd64.deb
wget https://github.com/eclipse-fog05/fog05/releases/download/v0.2.2/libzenoh-0.3.0-amd64.deb
wget https://github.com/eclipse-fog05/fog05/releases/download/v0.2.2/fog05-plugin-os-linux_0.2.0-1_amd64.deb
wget https://github.com/eclipse-fog05/fog05/releases/download/v0.2.2/fog05-plugin-net-linuxbridge_0.2.2-1_amd64.deb
wget https://github.com/eclipse-fog05/fog05/releases/download/v0.2.2/fog05-plugin-fdu-lxd_0.2.1-1_amd64.deb
```

then `sudo apt install` all the downloaded packages.

Finally, install the python libraries:
```
sudo apt install python3-pip
sudo pip3 install fog05-sdk==0.2.1 zenoh==0.3.0 yaks==0.3.0.post1
sudo pip3 install fog05
```

### Install LXD

Run `sudo snap install lxd`, then `lxd init`.
Set all the configurations to default except for 'Would you like to make lxd available over network?'. The deault is 'no' but you need to answer 'yes'. 
Choose default settings for ip addresses and ports and write a password when requested.

## Testing intallation

To test the installation run:
```
sudo systemctl start zenoh
sudo systemctl start fos_agent
sudo systemctl start fos_linux
sudo systemctl start fos_linuxbridge
sudo systemctl start fos_lxd
```

Then open a Python shell and execute:
```
>>> from fog05 import FIMAPI
>>> api = FIMAPI()
>>> api.node.list()
[xxxx-xxxx-xxxx-xxxx-xxxx]  # this uuid should match the content of /etc/machine-id
>>> api.node.plugins(api.node.list()[0])
[xxxx-xxxx-xxxx-xxxx-xxxx, xxxx-xxxx-xxxx-xxxx-xxxx, xxxx-xxxx-xxxx-xxxx-xxxx]  # these are the plugins' uuids running on the given node
```
When you call FIMAPI() without parameters, it tries to connect to yaks/zenoh server on localhost. To use another server run FIMAPI(remote_IP).
Be aware that in order to use the plugins with zenoh on the network, you need to change the ylocator attribute in every plugin's configuration.
The config files are located at /etc/fos/agent.json and /etc/fos/plugins/<plugin_name>/<plugin_name.json>.
Moreover, if you are on a node where plugins are using a zenoh server on the network, running `sudo systemctl start zenoh` in that machine is not needed. That service become useless since the plugins will use the server on remote_IP.

## Cold Migration

Copy and paste the code below inside LXD_Plugin at line 745. This script will check whether the training is running inside the container or not. The script running in the container must be called "federated-node.py".
```
# WARNING: MY CODE
code, out, err = cont.execute(['ps', 'aux'])
while code == 0 and out.find('federated_node.py') > 0:
    self.logger.info('migrate_fdu()', ' LXD Plugin - Instance has training process running. Wait until the end...')
    time.sleep(5)
    code, out, err = cont.execute(['ps', 'aux'])

self.logger.info('migrate_fdu()', ' LXD Plugin - Instance is ready to migrate')
# END OF MY CODE
```

## Live Migration

To test live migration in Fog05 you have to:
- comment lines 747 and 748
- add the parameter "live=True" at line 749
- start the container in your python script with fog05


## DEBUG AND TESTING

When you work with Fog05 and you want to solve some issues in you script, it is crucial to open a terminal for each plugin we want to investigate and run `journalctl -t <plugin_name> -f` in order to see the errors.