import multiprocessing as mp
import os
import subprocess
import sys

index_lock = mp.Lock()
shell_file = sys.argv[1]


def read_shell(filename):
    temp_list = list()
    with open(filename, 'r') as fin:
        for line in fin:
            temp_list.append(line.strip())
    return temp_list


def run_command(command):
    print os.getpid(), command
    # status = os.system(command)
    status = subprocess.call([command], shell=True)
    if status > 0:
        print "Run Err:", command
        return 1
    else:
        print "Finished:", command
        return 0


if __name__ == "__main__":
    shell_list = read_shell(shell_file)
    pool = mp.Pool(int(sys.argv[2]))
    print pool.map(run_command, [path for path in shell_list])
    pool.close()
    pool.join()

