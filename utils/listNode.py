from fog05 import FIMAPI
import sys

api = FIMAPI(sys.argv[1])
nodes=api.node.list()
print("\n{0} nodes found".format(len(nodes)))
for node in nodes:
	print('\n\n---------NODE-------------\n')
	print(node)
	print('\n\n--------PLUGINS-----------\n')
	print(api.node.plugins(node))
