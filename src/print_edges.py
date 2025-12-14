import sumolib

net = sumolib.net.readNet("version.net.xml")

print("All edges in the network:")
for e in net.getEdges():
    print(e.getID())
