from fog05 import FIMAPI
from fog05_sdk.interfaces.FDU import FDU
import uuid
import json
import sys
import os
import code

def read_file(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return data


def main(ip, fdufile):
    a = FIMAPI(ip)

    nodes = a.node.list()
    if len(nodes) == 0:
        print('No nodes')
        exit(-1)

    print('Nodes:')
    for n in nodes:
        print('UUID: {}'.format(n))

    fdu_d = FDU(json.loads(read_file(fdufile)))

    input('press enter to onboard descriptor')
    res = a.fdu.onboard(fdu_d)
    print(res.to_json())
    e_uuid = res.get_uuid()
    # code.interact(local=locals())

    # display choice
    print('Choose at which node you want to define the fdu:')
    for idx, n in enumerate(nodes):
        print('{}: {}'.format(idx, n))
    index = input('')
    
    inst_info = a.fdu.define(e_uuid, nodes[int(index)])
    print(inst_info.to_json())
    instid = inst_info.get_uuid()

    input('Press enter to configure')
    a.fdu.configure(instid)

    input('Press enter to start')
    a.fdu.start(instid)

    input('Press get info')
    info = a.fdu.instance_info(instid)
    print(info.to_json())

    input('Press enter to stop')
    a.fdu.stop(instid)

    input('Press enter to clean')
    a.fdu.clean(instid)

    input('Press enter to remove')

    a.fdu.undefine(instid)
    a.fdu.offload(e_uuid)

    exit(0)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[Usage] {} <yaks ip:port> <path to fdu descriptor>'.format(
            sys.argv[0]))
        exit(0)
    main(sys.argv[1], sys.argv[2])
