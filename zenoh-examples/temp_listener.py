from zenoh import Zenoh, ChangeKind
import time

def listener(change):
    if change.kind == ChangeKind.PUT:
        print('Publication received: "{}" = "{}"'
                .format(change.path, change.value))

if __name__ == "__main__":
    z = Zenoh({})
    w = z.workspace('/')
    results = w.subscribe('/myhome/kitcken/temp', listener)
    time.sleep(60)
