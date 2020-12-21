#!/bin/bash

replace="\"ylocator\": \"tcp/$1:7447\""

#echo "s,\"ylocator\":.*,$replace,"

sed -i "s,\"yaks\":.*7447\",\"yaks\": \"tcp/$1:7447\"," /etc/fos/agent.json

sed -i "s,\"ylocator\":.*7447\",$replace," /etc/fos/plugins/plugin-os-linux/linux_plugin.json

sed -i "s,\"ylocator\":.*7447\",$replace," /etc/fos/plugins/plugin-net-linuxbridge/linuxbridge_plugin.json

sed -i "s,\"ylocator\":.*7447\",$replace," /etc/fos/plugins/plugin-fdu-containerd/containerd_plugin.json

sed -i "s,\"ylocator\":.*7447\",$replace," /etc/fos/plugins/plugin-fdu-lxd/LXD_plugin.json

sed -i "s,\"ylocator\":.*7447\",$replace," /etc/fos/plugins/plugin-fdu-native/native_plugin.json
