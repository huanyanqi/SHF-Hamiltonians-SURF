import numpy as np

class Atom:
    def __init__(self, name, position):
        self.name = name
        self.position = np.array(position)

    def __str__(self):
        return "Atom: {0.name}, Position: {0.position}".format(self)

    def __repr__(self):
        return "{0.name}, ".format(self) + str(tuple(self.position))


    def distance(self, other):
        return np.around(np.linalg.norm(self.position - other.position), 5)

# filename = "y2sio5-huge.xyz"
# source = ("Y2")
# target = ("Y1", "Y2")

filename = "yvo4-huge88.xyz"
source = ("Y",)
target = ("Y", "V") 

with open(filename, "r") as f:
    data = f.readlines()

# 1st line is the number of atoms
# 2nd line is a comment, 3rd line onwards is data
n = int(data[0])
data = data[2:]

# Iterate through the data file and parse the name and positions
atoms = []
for line in data:
    name, x, y, z = line.split()
    atoms.append(Atom(name, tuple(map(float, (x, y, z)))))

# We now compute the average distance from each source atom to every other target atom
avg_distance_list = []

for idx, atom in enumerate(atoms):
    print("{}/{} atoms".format(idx, len(atoms)))
    num_targets = 0
    total_distance = 0

    # Skip over this atom if it's not the correct source type
    # if atom.name not in source:
    if atom.name not in source or not -5 < atom.position[0] < 5 or not -5 < atom.position[1] < 5 or not -5 < atom.position[2] < 5:
        continue
    # Iterate over all possible other atoms 
    for other in atoms:
        # Ignore if it's the atom itself or if it's the wrong type
        if other is atom or other.name not in target:
            continue
        num_targets += 1
        total_distance += atom.distance(other)

    avg_distance_list.append((atom, total_distance/num_targets))

# Sort the list by the average distance to all target atoms
avg_distance_list.sort(key=lambda x: x[1])
for i in avg_distance_list[:10]: 
    print(i)

# Choose the atom with the smallest average distance to neighbours, for this
# indicates that it is in the middle and thus has a full "shell" of neighbours 
# around it.
chosen_atom = avg_distance_list[1][0]
print(chosen_atom)

# We now compute the separation vector + distance to each target atom 
vectors_list = []

for other in atoms:
    if other is chosen_atom or other.name not in target:
        continue
    vectors_list.append((other, tuple(np.around(other.position-chosen_atom.position,5)), chosen_atom.distance(other)))

# We sort by the distance to get the separation vector to the closest neighbours.
vectors_list.sort(key=lambda x: x[2])
for i in vectors_list[:15]:
    print(i)

with open(filename[:-4]+"_output.txt", "w") as f:
    f.write("Chosen atom: {0}\n\n".format(chosen_atom))
    for i in vectors_list[:]:
        # f.write(str(i) + "\n")
        if i[0].name == "V":
            properties = "7/2, 1.4711"
        elif i[0].name in ("Y", "Y1", "Y2"):
            properties = "1/2, -0.2748308"
        else:
            raise Exception
        f.write("({}, {}, '{}')\n".format(str(i[1]), properties, i[0].name))

    f.write("\n\n")

    for i in vectors_list[:30]:
        f.write(str(i) + "\n")