import numpy as np
import subprocess
import os

def read_energy(path: str, method: str) -> float:

    f = open(path)
    lines = f.readlines()
    for line in lines:
        if ("FINAL ENERGY:" in line):
            energy = float(line.strip().split()[2])
    f.close()
    
    return energy


def read_energy_gradient(n: int, path: str, method: str = "hf"):

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
    f.close()

    return data


def read_geom(n: int, geom_file: str):
    infile = open(geom_file, "r")
    x = np.zeros(3 * n)
    infile.readline()
    infile.readline()
    for i in range(n):
        parsed = infile.readline().strip().split()
        x[3 * i + 0] = float(parsed[1])
        x[3 * i + 1] = float(parsed[2])
        x[3 * i + 2] = float(parsed[3])
    return x


def write_geom(n: int, atoms, x, geom_file: str, mode: str="w"):
    f = open(geom_file, mode)
    f.write(str(n) + "\n\n")
    for i in range(n):
        f.write("%s %f %f %f\n" % (atoms[i], x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]))
    f.close()
    
    return


def launch_job(path):
    command = ["terachem", "start", ">", "out"]
    print("started job")
    out = open("out","w")
    subprocess.run(command, stdout=out)
    out.close()
    print("ended job")
    return


if __name__ == "__main__":
    print("import, dont execute this directly\n")