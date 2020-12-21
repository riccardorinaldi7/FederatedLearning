from zenoh import Zenoh

if __name__ == "__main__":
    z = Zenoh({})
    w = z.workspace('/')
    results = w.get('/myhome/kitcken/temp')
    key, value = results[0].path, results[0].value
    print('  {} : {}'.format(key, value))
