from fog05 import FIMAPI
from fog05_sdk.interfaces.FDU import FDU
import uuid
import json
import sys
import os
import code
import time
import pylxd


def read_file(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    return data


def main(ip, fdufile):
    t1 = time.time()
    a = FIMAPI(ip)
    t2 = time.time()
    t_conn = t2-t1

    nodes = a.node.list()
    if len(nodes) == 0:
        print('No nodes')
        exit(-1)

    print('Nodes:')
    for n in nodes:
        print('UUID: {}'.format(n))

    fdu_d = FDU(json.loads(read_file(fdufile)))

    input('press enter to onboard descriptor')
    t1 = time.time()
    res = a.fdu.onboard(fdu_d)
    t2 = time.time()
    t_fdu = t2-t1
    print(res.to_json())
    e_uuid = res.get_uuid()
    # code.interact(local=locals())

    # display choice
    print('Choose at which node you want to define the fdu:')
    for idx, n in enumerate(nodes):
        print('{}: {}'.format(idx, n))
    index = input('')
    
    t1 = time.time()
    inst_info = a.fdu.define(e_uuid, nodes[int(index)])
    t2 = time.time()
    t_define = t2-t1
    print(inst_info.to_json())
    instid = inst_info.get_uuid()

    input('Press enter to configure')
    t1 = time.time()
    a.fdu.configure(instid)
    t2 = time.time()
    t_config = t2-t1

    input('Press enter to start')
    # cl = pylxd.Client(endpoint='https://192.168.56.112:8443', verify=False)
    cl = pylxd.Client()
    inst = cl.instances.get("c{}".format(instid))
    t1 = time.time()
    a.fdu.start(instid)
    
    while inst.state().status != 'Running':
    	time.sleep(0.01)
    t2 = time.time()
    t_start = t2-t1

    input('Press get info')
    info = a.fdu.instance_info(instid)
    print(info.to_json())

    input('Press enter to stop')
    t1 = time.time()
    a.fdu.stop(instid)
    t2 = time.time()
    t_stop = t2-t1

    input('Press enter to clean')
    t1 = time.time()
    a.fdu.clean(instid)
    t2 = time.time()
    t_clean = t2-t1

    input('Press enter to remove')
    t1 = time.time()
    a.fdu.undefine(instid)
    t2 = time.time()
    t_undef = t2-t1
    t1 = time.time()
    a.fdu.offload(e_uuid)
    t2 = time.time()
    t_delfdu = t2-t1
    
    print("connection: {} \n Fdu: {} \n Define: {} \n Config: {} \n Start: {} \n Stop: {} \n Clean: {} \n Undef: {} \n DelFdu: {}".format(t_conn, t_fdu, t_define, t_config, t_start, t_stop, t_clean, t_undef, t_delfdu))

    exit(0)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('[Usage] {} <yaks ip:port> <path to fdu descriptor>'.format(
            sys.argv[0]))
        exit(0)
    main(sys.argv[1], sys.argv[2])
