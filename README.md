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
When you call FIMAPI() without parameters, it tries to connect to yaks/zenoh server on localhost. To use another server run FIMAPI(server_IP).
But be aware that in order to use the plugins with zenoh on the network, you need to change the ylocator attribute in every plugin's configuration.
The config files are located at /etc/fos/agent.json and /etc/fos/plugins/<plugin_name>/<plugin_name.json>
