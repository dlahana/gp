import numpy as np
import sys

def get_energy(path: str, method: str) -> float:

    f = open(path)
    lines = f.readlines()
    for line in lines:
        if ("FINAL ENERGY:" in line):
            energy = float(line.strip().split()[2])
    
    return energy


def get_energy_gradient(n: int, path: str, method: str):

    f = open(path)
    lines = f.readlines()
    data = np.zeros(3 * n + 1)
    for index, line in enumerate(lines):
        if ("FINAL ENERGY:" in line):
            data[0] = float(line.strip().split()[2])
        if ("Gradient units are Hartree/Bohr" in line):
            for i in range(n):
                parsed = lines[index + 3 + i].strip().split()
                for j in range(3):
                    data[3 * i + j + 1] = float(parsed[j])
    
    return data


def write_geom(n: int, path:str, name: str):

    return


if __name__ == "__main__":
    print("import, dont execute this directly\n")