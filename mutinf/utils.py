#!/usr/bin/python

import os.path, time, os, sys

# flush the stdout buffer
def flush():
    sys.stdout.flush()
    return ""

# make a directory and return the absolute path to it
def mkdir(dir):
    d = os.path.abspath(dir)
    run("mkdir -p %s" % d)
    return d

# make directory and cd into it
def mkdir_cd(dir):
    mkdir(dir)
    cd(dir)

def cd(dir):
    os.chdir(dir)

# parse the pdb file name from file paths
def parse_pdbnames(fns):
    pdbnames = []
    for fn in fns:
        pdbnames += [os.path.basename(fn).replace('.pdb', '').replace('.gz','')]
    return pdbnames
def parse_pdbname(fn): return parse_pdbnames([fn])[0]

# convert an array to a pretty string
def arr2str2(a, precision=2, length=5, col_names=None, row_names=None):
    fmt = "%" + str(length) + "s"
    shape = a.shape
    assert(len(shape) == 2)
    #s = "\n".join([fmt_floats(a[row,:], digits=precision) for row in range(shape[0])]) + "\n"
    str_arr = []
    if col_names != None: str_arr += [" "*(length+1) + " ".join([fmt%str(col) for col in col_names])]
    for i in range(shape[0]):
        row_header = ""
        if row_names != None: row_header = fmt % str(row_names[i]) + " "
        str_arr += [row_header + fmt_floats(a[i,:], digits=precision, length=length)]

    return "\n".join(str_arr) + "\n"

# convert an array to a pretty string
def arr1str1(a, precision=2, length=5, col_names=None, row_names=None):
    fmt = "%" + str(length) + "s"
    shape = a.shape
    assert(len(shape) == 1)
    #s = "\n".join([fmt_floats(a[row,:], digits=precision) for row in range(shape[0])]) + "\n"
    str_arr = []
    if col_names != None: str_arr += [" "*(length+1) + " ".join([fmt%str(col) for col in col_names])]
    for i in range(shape[0]):
        row_header = ""
        if row_names != None: row_header = fmt % str(row_names[i]) + " "
        str_arr += [row_header + fmt_floats( [a[i]], digits=precision, length=length)]

    return "\n".join(str_arr) + "\n"

# run a shell cmd
def run(cmd):
    import commands
    status, output = commands.getstatusoutput(cmd)
    if status != 0: raise Exception("run('%s') exited with status %d; Output:\n     %s" % (cmd, status, output))
    return output

# track elapsed time in seconds
class Timer:
    def __init__(self): self._start = time.time()
    def elapsed(self):  return time.time() - self._start
    def __str__(self):  return "%0d:%02d" % (self.elapsed()/60, self.elapsed()%60)

# pretty print a list of floats
def fmt_floats(fs, digits=2, length="", sep=" "):
    fmt = "%"+str(length)+"."+str(digits)+"f"
    return sep.join([fmt%f for f in fs])

