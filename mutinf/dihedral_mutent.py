#!/usr/bin/env python

# MutInf program 
# Copyright 2010 Christopher McClendon and Gregory Friedland
# Released under GNU Lesser Public License

from numpy import *
#import mdp
import re, os, sys, os.path, time, shelve
from optparse import OptionParser
import weave
from scipy import stats as stats
from scipy import special as special
from scipy import integrate as integrate
from scipy import misc as misc
from weave import converters
import time
import PDBlite, utils
from triplet import *
from constants import *
from input_output import *
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from scipy.integrate import quad
try:
       import MDAnalysis 
except: pass
try:
       from pymbar.timeseries import * #from PyMBAR by Shirts and Chodera
except: 
       print "couldn't import timeseries from PyMBAR by Shirts and Chodera: autocorrelation not available"
       pass
set_printoptions(linewidth=120)




########################################################
##  INSTRUCTIONS:
##  
##
############################################################


##  INSTALLATION:
##  You will need to have NumPy and SciPy installed.
##  I recommend building LAPACK then ATLAS using gcc and gfortran
##  with the -fPIC option. These will provide optimized linear algebra
##  routines for your particular architecture. These plus the "weave" statements with pointer arithmetic yield performance that can approach that of C or FORTRAN code while allowing the code to be written for the most part in higher-level expressions that read more like math. 
#
##  Note: by default this code uses Numpy built with the IntelEM64-T C-compiler (command-line option "-g intelem").
##  This will use 'icc' to compile optimized inline C-code. 
##  If you instead would like to use 'gcc', indicate this with command-line option "-g gcc" 
##  PDBlite.py and utils.py should reside in the same directory as
##  dihedral_mutent.py

##  DIHEDRAL (TORSION) ANGLES FROM SIMULATIONS
##  This program uses an undersampling correction based on correlations
##  between torsions in different simulations or different equal-sized blocks of the same
##  long simulation.
##  Ideally, six or more simulations or simulation blocks are ideal.
##  This program currently assumes that all simulations or blocks are of equal length. (While in principle the code is general enough to hanlde differences in lengths of simulations/blocks, this feature is not currently enabled as a bug manifests itself when multiple sub-samples were used (i.e. when the "-o" argument is less than the "-n" argument)
##
##
##  Next, you will need dihedral angle files for each torsion, containing two space-delimited fields:
#   1) time or timestep or any number, and 2) dihedral angle in degrees.
##  These files can be named .xvg or .xvg.gz, for each torsion for each residue. for example:
##
## chi1PHE42.xvg.gz
## chi2PHE42.xvg.gz
## phiPHE42.xvg.gz
## psiPHE42.xvg.gz
####  I use the GROMACS utility "g_chi" to produce these files. 
##

##
## RESIDUE LIST:
## You then need a residue list file (i.e. test_new3_adaptive.reslist) with three space-delimited fields:
## 1: torsion number (i.e. 42 for above) 2. residue name  3. Number Chain (i.e. 42A for residue number 42, chain A)
## 
##
## For proteins with more than one chain, the torsion numbers can just be sequential (i.e. add 100 for each new chain), but must be unique.
##
## The residue list is used to map torsion numbers to biologically meaningful strings (i.e. "42A")
##
##  You can create a residue list using a shell command like:
## cat ${mypdb}.pdb | grep CA | awk '{print NR, substr($0,18,3), substr($0,23,4) substr($0,22,1)}' > temp

##
##  DIRECTORY STRUCTURE THE PROGRAM LOOKS FOR:
##  The -x option indicates the base directory for your dihedral data for each run.
##  Under this directory you should have directories run1, run2, run3, run4, run5, run6, for example.
##  Place the .xvg files under directories named /dihedrals/g_chi/ under each run, unless the -d option is used. For example, if "-d /" is used then the torsion angle files would be in the run1 to run6 directories.
##
##
##  RUNNING THE PROGRAM:
##  run the program like this:
##  python ~/trajtools/dihedral_mutent.py -x ~/IL2_Flex/${pdb}_62GLU_SS/ -a "yes" -o 6 -w 30 -p 0 -c "no" -n 6  test_new3_adaptive.reslist > test_new3_adaptive_out.txt
##  More options are listed in the __main__ part of the code.
##  For the purposes of this excercise, we will just aggregate all the data together as per McClendon, Friedland, et. al. 2009. 
##  To do this, the -o option (number of runs to take at a time)  needs to be the same as the -n option (the number of runs)
##  
## For this example, the directory structure would look like:
## ~/IL2_Flex/${pdb}_62GLU_SS/
##      run1/  
##           dihedrals/g_chi/*.xvg.gz
##      run2/
##           dihedrals/g_chi/*.xvg.gz
##      run3/
##           dihedrals/g_chi/*.xvg.gz
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##

















#SPEED = "vfast" # set to "vfast" to use the vfast_cov(); "fast" to use fast_cov(); "*" to use cov()
#print "Running at speed: ", SPEED
#################################################################################################################################
## Needs fixing if it is to be used
## Use Abel 2009 JCTC eq. 27 weighting for KNN estimates for k=1,2,3
#QKNN = zeros((3),float64)
#WEIGHTS_KNN = zeros((3),float64)
# Qk = sum_{j=k}^{inf} 1/(j^2)
#for i in range(3):
#    for j in range(999):
#        QKNN[i] += 1/((i+1+j)*(i+1+j))
#WEIGHTS_KNN = (1/QKNN) / sum(1/QKNN)


#########################################################################################################################################


#########################################################################################################################################
#### Class for two-variable vonMises distribution, used for comparing calculated/analytical entropies, mutual information ###############
#########################################################################################################################################

class vonMises:
    C = 0
    entropy = 0
    entropy_marginal1 = 0
    entropy_marginal2 = 0
    mutinf = 0
    u1 = u2 = k1 = k2 = l1 = l2 = 0
    def __k1_dC_dk1__(self):
            value = 0
            for i in range(100):
                value += misc.comb(2*i,i,exact=1) * (((self.lambd ** 2)/(4*self.k1*self.k2)) ** i) * special.iv(i+1,self.k1) * special.iv(i, self.k2)
            return value * 4 * PI * PI * self.k1
    def __k2_dC_dk2__(self):
            value = 0
            for i in range(100):
                value += misc.comb(2*i,i,exact=1) * (((self.lambd ** 2)/(4*self.k1*self.k2)) ** i) * special.iv(i+1,self.k2) * special.iv(i, self.k1)
            return value * 4 * PI * PI * self.k2
    def  __lambd_dC_dlambd__(self):
            value = 0
            for i in range(100):
                value += misc.comb(2*i,i,exact=1) * (((self.lambd ** 2)/(4*self.k1*self.k2)) ** i) * special.iv(i,self.k1) * special.iv(i, self.k2)
            return value * 4 * PI * PI
    def p(self,phi1,phi2):
        return (1.0 / self.C) * exp(self.k1 * cos(self.l1 * (phi1 - self.u1)) +
                                    self.k2 * cos(self.l2 * (phi2 - self.u2)) +
                        self.lambd * sin(self.l1 * (phi1 - self.u1)) * sin(self.l2 * (phi2 - self.u2)))
    def p1(self,phi1):
        return (2*PI/self.C)*special.i0(sqrt(self.k2*self.k2+self.lambd*self.lambd*((sin(self.l1 * (phi1 - self.u1))) ** 2))) * exp(self.k1 * cos(self.l1 * (phi1 - self.u1)))
    def p2(self,phi2):
        return (2*PI/self.C)*special.i0(sqrt(self.k1*self.k1+self.lambd*self.lambd*((sin(self.l2 * (phi2 - self.u2)) ** 2)))) * exp(self.k2 * cos(self.l2 * (phi2 - self.u2)))
    def randarray1(self,mylen):
        myoutput = stats.uniform.rvs(size=mylen, scale=2*PI)
        for i in range(mylen):
            myoutput[i]= self.p1(myoutput[i])
        return myoutput
    def randarray2(self,mylen):
        myoutput = stats.uniform.rvs(size=mylen, scale=2*PI)
        for i in range(mylen):
            myoutput[i]= self.p2(myoutput[i])
        return myoutput
    def __init__(self,u1,u2,k1,k2,l1,l2,lambd):
        self.u1 = u1
        self.u2 = u2
        self.k1 = k1
        self.k2 = k2
        self.l1 = l1
        self.l2 = l2
        self.lambd = lambd; #lambda coupling parameter
        self.C = 0 #normalization constant 
        self.k1_dC_dk1 = self.__k1_dC_dk1__()
        self.k2_dC_dk2 = self.__k2_dC_dk2__()
        self.lambd_dC_dlambd = self.__lambd_dC_dlambd__()
        for i in range(100): #compute C using rapdily-converging infinite series
            self.C += 4*PI*PI * misc.comb(2*i,i,exact=1) * (((self.lambd ** 2)/(4*self.k1*self.k2)) ** i) * special.iv(i,self.k1) * special.iv(i, self.k2)
        print "normalization constant: "+str(self.C)+"\n"
        print "marginal p1(0):"+str(self.p1(0))
        #calculate entropies for joint and marginal distributions, and mutual information
        #for mytestval in arange( 0, 2*PI, 0.1 ):
        #       print "p1 ("+str(mytestval)+"): "+str(self.p1(mytestval))+"+\n"
        self.entropy = log(self.C) + (1 / self.C) * (- self.k1_dC_dk1 - self.k2_dC_dk2 - self.lambd_dC_dlambd)
        plnp1 = lambda x: self.p1(x) * log(self.p1(x))
        plnp2 = lambda x: self.p2(x) * log(self.p2(x))
        self.entropy_marginal1 = -1.0 * (integrate.quad(plnp1, 0, 2*PI,epsabs=1e-9,epsrel=1e-9,limit=10000))[0] - log(2*PI/self.C) - self.k1_dC_dk1/self.C
        self.entropy_marginal2 = -1.0 * (integrate.quad(plnp2, 0, 2*PI,epsabs=1e-9,epsrel=1e-9,limit=10000))[0] - log(2*PI/self.C) - self.k2_dC_dk2/self.C
        self.mutinf = self.entropy_marginal1 + self.entropy_marginal2 - self.entropy
        #Scipy documentation on scipy functions used here:
        #scipy.misc.comb(N, k, exact=0)
        #Combinations of N things taken k at a time.
        #If exact==0, then floating point precision is used, otherwise exact long integer is computed.
        #scipy.special.iv(x1, x2[, out])
        #y=iv(v,z) returns the modified Bessel function of real order v of z. If z is of real type and negative, v must be integer valued.

#test1 = vonMises(3.141592654, 3.141592654, 10, 15, 3, 2, 10)

#########################################################################################################################################
#####  Memory and CPU information-gathering routines  ###################################################################################
#########################################################################################################################################


"""Get some of the info from /proc on Linux
from  http://www.dcs.warwick.ac.uk/~csueda/proc.py
and http://www.velocityreviews.com/forums/t587425-python-system-information.html
'In meminfo, you're probably most interested in the fields MemTotal and
MemFree (and possibly Buffers and Cached if you want to see how much
of the used memory is in fact cache rather than user programs). You
might also want to look at SwapTotal and SwapFree.'
"""



import re
re_meminfo_parser = re.compile(r'^(?P<key>\S*):\s*(?P<value>\d*)\s*kB')


def meminfo():
    """-> dict of data from meminfo (str:int).
    Values are in kilobytes.
    """
    result = dict()
    for line in open('/proc/meminfo'):
        match = re_meminfo_parser.match(line)
        if not match:
            continue  # skip lines that don't parse
        key, value = match.groups(['key', 'value'])
        result[key] = int(value)
    #close('/proc/meminfo')
    return result



def loadavg():
    """-> 5-tuple containing the following numbers in order:
     - 1-minute load average (float)
     - 5-minute load average (float)
     - 15-minute load average (float)
     - Number of threads/processes currently executing (<= number of CPUs)
       (int)
     - Number of threads/processes that exist on the system (int)
     - The PID of the most recently-created process on the system (int)
    """
    loadavgstr = open('/proc/loadavg', 'r').readline().strip()
    data = loadavgstr.split()
    avg1, avg5, avg15 = map(float, data[:3])
    threads_and_procs_running, threads_and_procs_total = \
        map(int, data[3].split('/'))
    most_recent_pid = int(data[4])
    return avg1, avg5, avg15, threads_and_procs_running, \
             threads_and_procs_total, most_recent_pid



def cpuusage():
    """-> dict of cpuid : (usertime, nicetime, systemtime, idletime)
    cpuid "cpu" means the total for all CPUs.
    cpuid "cpuN" means the value for CPU N.
    """
    wanted_records = [line for line in open('/proc/stat') if line.startswith('cpu')]
    result = {}
    for cpuline in wanted_records:
        fields = cpuline.split()[:5]
        data = map(int, fields[1:])
        result[fields[0]] = tuple(data)
    return result



 #check for free memory at least 15%
def check_for_free_mem():
      mymemory = meminfo()
      free_memory = mymemory["MemFree"] #/ float(mymemory["SwapTotal"])
      free_memory = mymemory["MemFree"]
      print "Free Swap: "+str(mymemory["SwapFree"])+" Free memory: "+str(mymemory["MemFree"])+" percent free: "+str(free_memory*100)+" \% \n"
      assert(free_memory > 5*1024)


#########################################################################################################################################
### Routines for Combinatorics  #########################################################################################################
#########################################################################################################################################

# from: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/190465
def xcombinations(items, n): # akes n distinct elements from the sequence, order matters
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xcombinations(items[:i]+items[i+1:],n-1):
                yield [items[i]]+cc
def xuniqueCombinations(items, n): # takes n distinct elements from the sequence, order is irrelevant
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xuniqueCombinations(items[i+1:],n-1):
                yield [items[i]]+cc
def xuniqueCombinations2(items, n): # takes n distinct elements from the sequence, order is irrelevant
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for cc in xuniqueCombinations2(items[i+1:],n-1):
                yield [items[i]]+cc

def xselections(items, n): # takes n elements (not necessarily distinct) from the sequence, order matters.
    if n==0: yield []
    else:
        for i in xrange(len(items)):
            for ss in xselections(items, n-1):
                yield [items[i]]+ss

#########################################################################################################################################
####  Covariance calculations: different routines for different-sized arrays so allocation happens only once ############################
#########################################################################################################################################

# calculates the covariance matrix over the rows
cov_mat = None
def fast_cov(m):
   global cov_mat

   nx, ny = m.shape
   X = array(m, ndmin=2, dtype=float64)
   X -= X.mean(axis=0)

   if cov_mat is None: cov_mat = zeros((ny, ny), float64)
   cov_mat[:,:] = 0

   code = """
      // weave1   float64
      double inv_nx, cov;

      inv_nx = 1./double(nx-1); // unbiased estimated
      for (int y1=0; y1<ny; ++y1) {
         for (int y2=y1; y2<ny; ++y2) {
            cov=0;
            for (int x=0; x<nx; ++x) {
               cov += X(x,y1) * X(x,y2);
            }
            cov_mat(y1,y2) = cov*inv_nx;
            cov_mat(y2,y1) = cov_mat(y1,y2);
         }
      }
      //printf("fast_cov max=%f\\n", max);
   """
   weave.inline(code, ['X', 'cov_mat', 'nx', 'ny'],
                type_converters = converters.blitz, compiler = mycompiler, runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"] )
   return cov_mat


# calculates the covariance matrix over the rows
def vfast_cov(m):
   global cov_mat

   nx, ny = m.shape
   X = array(m, ndmin=2, dtype=float64)
   X -= X.mean(axis=0)

   if cov_mat is None: cov_mat = zeros((ny, ny), float64)
   cov_mat[:,:] = 0
   
   code = """
      // weave2 float64
      double inv_nx, cov;
      double *pCovy1, *pXy1, *pXy2;
      int tmp;

      inv_nx = 1./double(nx-1); // unbiased estimated
      for (int y1=0; y1<ny; ++y1) {
         pCovy1 = cov_mat+y1*ny;
         pXy1 = X+y1;
         for (int y2=y1; y2<ny; ++y2) {
            cov=0;
            pXy2 = X+y2;
            for (int x=0; x<nx; ++x) {
               //cov += X(x,y1) * X(x,y2);
               //cov += *(X+x*ny+y1) * *(X+x*ny+y2);               
               tmp=x*ny;
               cov += *(pXy1+tmp) * *(pXy2+tmp);
            }
            //cov_mat(y1,y2) = cov*inv_nx;
            //cov_mat(y2,y1) = cov_mat(y1,y2);
            *(pCovy1+y2) = cov*inv_nx;
            *(cov_mat+y2*ny+y1) = *(pCovy1+y2);
         }
      }
   """
   weave.inline(code, ['X', 'cov_mat', 'nx', 'ny'],
                #type_converters = converters.blitz,
                compiler = mycompiler,runtime_library_dirs="/usr/lib/x86_64-linux-gnu/", library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
   return cov_mat

#cov_mat_for1D = None
def vfast_cov_for1D(m):
   global cov_mat_for1D

   nx, ny = m.shape
   X = array(m, ndmin=2, dtype=float64)
   X -= X.mean(axis=0)

   cov_mat_for1D = zeros((ny, ny), float64)
   cov_mat_for1D[:,:] = 0
   
   code = """
      // weave3  float64
      double inv_nx, cov;
      double *pCovy1, *pXy1, *pXy2;
      int tmp;

      inv_nx = 1./double(nx-1); // unbiased estimated
      for (int y1=0; y1<ny; ++y1) {
         pCovy1 = cov_mat_for1D+y1*ny;
         pXy1 = X+y1;
         for (int y2=y1; y2<ny; ++y2) {
            cov=0;
            pXy2 = X+y2;
            for (int x=0; x<nx; ++x) {
               //cov += X(x,y1) * X(x,y2);
               //cov += *(X+x*ny+y1) * *(X+x*ny+y2);               
               tmp=x*ny;
               cov += *(pXy1+tmp) * *(pXy2+tmp);
            }
            //cov_mat_for1D(y1,y2) = cov*inv_nx;
            //cov_mat_for1D(y2,y1) = cov_mat_for1D(y1,y2);
            *(pCovy1+y2) = cov*inv_nx;
            *(cov_mat_for1D+y2*ny+y1) = *(pCovy1+y2);
         }
      }
   """
   weave.inline(code, ['X', 'cov_mat_for1D', 'nx', 'ny'],
                #type_converters = converters.blitz,
                compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
   return cov_mat_for1D

cov_mat_for1D_ind = None
def vfast_cov_for1D_ind(m):
   global cov_mat_for1D_ind

   nx, ny = m.shape
   X = array(m, ndmin=2, dtype=float64)
   X -= X.mean(axis=0)

   if cov_mat_for1D_ind is None: cov_mat_for1D_ind = zeros((ny, ny), float64)
   cov_mat_for1D_ind[:,:] = 0
   
   code = """
      // weave4 float64
      double inv_nx, cov;
      double *pCovy1, *pXy1, *pXy2;
      int tmp;

      inv_nx = 1./double(nx-1); // unbiased estimated
      for (int y1=0; y1<ny; ++y1) {
         pCovy1 = cov_mat_for1D_ind+y1*ny;
         pXy1 = X+y1;
         for (int y2=y1; y2<ny; ++y2) {
            cov=0;
            pXy2 = X+y2;
            for (int x=0; x<nx; ++x) {
               //cov += X(x,y1) * X(x,y2);
               //cov += *(X+x*ny+y1) * *(X+x*ny+y2);               
               tmp=x*ny;
               cov += *(pXy1+tmp) * *(pXy2+tmp);
            }
            //cov_mat_for1D_ind(y1,y2) = cov*inv_nx;
            //cov_mat_for1D_ind(y2,y1) = cov_mat_for1D_ind(y1,y2);
            *(pCovy1+y2) = cov*inv_nx;
            *(cov_mat_for1D_ind+y2*ny+y1) = *(pCovy1+y2);
         }
      }
   """
   weave.inline(code, ['X', 'cov_mat_for1D_ind', 'nx', 'ny'],
                #type_converters = converters.blitz,
                compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
   return cov_mat_for1D_ind

cov_mat_for1D_boot = None
def vfast_cov_for1D_boot(m):
   global cov_mat_for1D_boot

   nx, ny = m.shape
   X = array(m, ndmin=2, dtype=float64)
   X -= X.mean(axis=0)

   if cov_mat_for1D_boot is None: cov_mat_for1D_boot = zeros((ny, ny), float64)
   cov_mat_for1D_boot[:,:] = 0
   
   code = """
      // weave5 float64
      double inv_nx, cov;
      double *pCovy1, *pXy1, *pXy2;
      int tmp;

      inv_nx = 1./double(nx-1); // unbiased estimated
      for (int y1=0; y1<ny; ++y1) {
         pCovy1 = cov_mat_for1D_boot+y1*ny;
         pXy1 = X+y1;
         for (int y2=y1; y2<ny; ++y2) {
            cov=0;
            pXy2 = X+y2;
            for (int x=0; x<nx; ++x) {
               //cov += X(x,y1) * X(x,y2);
               //cov += *(X+x*ny+y1) * *(X+x*ny+y2);               
               tmp=x*ny;
               cov += *(pXy1+tmp) * *(pXy2+tmp);
            }
            //cov_mat_for1D_boot(y1,y2) = cov*inv_nx;
            //cov_mat_for1D_boot(y2,y1) = cov_mat_for1D_boot(y1,y2);
            *(pCovy1+y2) = cov*inv_nx;
            *(cov_mat_for1D_boot+y2*ny+y1) = *(pCovy1+y2);
         }
      }
   """
   weave.inline(code, ['X', 'cov_mat_for1D_boot', 'nx', 'ny'],
                #type_converters = converters.blitz,
                compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
   return cov_mat_for1D_boot




def vfast_cov_for1D_boot_multinomial(m):
   

   nx, ny = m.shape
   X = array(m, ndmin=2, dtype=float64)
   X -= X.mean(axis=0)

   cov_mat_for1D_boot_multinomial = zeros((ny, ny), float64)
   cov_mat_for1D_boot_multinomial[:,:] = 0
   
   code = """
      // weave5 float64
      double inv_nx, cov;
      double *pCovy1, *pXy1, *pXy2;
      int tmp;

      inv_nx = 1./double(nx-1); // unbiased estimated
      for (int y1=0; y1<ny; ++y1) {
         pCovy1 = cov_mat_for1D_boot_multinomial+y1*ny;
         pXy1 = X+y1;
         for (int y2=y1; y2<ny; ++y2) {
            cov=0;
            pXy2 = X+y2;
            for (int x=0; x<nx; ++x) {
               //cov += X(x,y1) * X(x,y2);
               //cov += *(X+x*ny+y1) * *(X+x*ny+y2);               
               tmp=x*ny;
               cov += *(pXy1+tmp) * *(pXy2+tmp);
            }
            //cov_mat_for1D_boot_multinomial(y1,y2) = cov*inv_nx;
            //cov_mat_for1D_boot_multinomial(y2,y1) = cov_mat_for1D_boot(y1,y2);
            *(pCovy1+y2) = cov*inv_nx;
            *(cov_mat_for1D_boot_multinomial+y2*ny+y1) = *(pCovy1+y2);
         }
      }
   """
   weave.inline(code, ['X', 'cov_mat_for1D_boot_multinomial', 'nx', 'ny'],
                #type_converters = converters.blitz,
                compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
   return cov_mat_for1D_boot_multinomial


#note: here, the pairs of runs  are in the first dimension, and the permutations in the second dimension
#need to fix this routine here
cov_mat_for1Dpairs_ind = None
def vfast_var_for1Dpairs_ind(m):
   global cov_mat_for1Dpairs_ind

   nboot, nx, ny = m.shape
   #print "matrix for var: "+str(nx)+" x "+str(ny)
   X = array(m, ndmin=3, dtype=float64)
   #print "shape of matrix for var: "+str(shape(X))
   X = swapaxes(X,0,2)
   #print mean(X, axis=0)
   X -= mean(X,axis=0)
   X = swapaxes(X,0,2)

   if cov_mat_for1Dpairs_ind is None: cov_mat_for1Dpairs_ind = zeros((nboot,nx), float64)
   cov_mat_for1Dpairs_ind[:,:] = 0
   
   code = """
      // weave6 float64
      double inv_ny, var;
      double *pXy1;
      int tmp;

      inv_ny = 1./double(ny-1-1); // unbiased estimated, extra -1 to remove orig data, just want var of permuted data
      for(int bootstrap=0; bootstrap < nboot; bootstrap++) {
         for (int x1=0; x1<nx; ++x1) {
            pXy1 = X+bootstrap*ny*nx + ny*x1;
         
            var=0;
         
            for (int y=0; y<ny-1; ++y) {
               //var += X(x1,y) * X(x1,y);
               //var += *(X+x1*ny+y) * *(X+x1*ny+y);               
               //tmp=x*ny;
               var += *(pXy1+y) * *(pXy1+y);
            }
            *(cov_mat_for1Dpairs_ind + bootstrap*nx + x1) = var*inv_ny;
         
      }
     }
   """
   weave.inline(code, ['X', 'cov_mat_for1Dpairs_ind', 'nx', 'ny', 'nboot'],
                #type_converters = converters.blitz,
                compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
   return cov_mat_for1Dpairs_ind



#########################################################################################################################################
##### Wolfram Fit Exponential to Data                                                   #################################################
#########################################################################################################################################

### http://mathworld.wolfram.com/LeastSquaresFittingExponential.html
# y = A * exp(B*x)
def exponential_fit(x, y):
       sum_y = sum(y)
       sum_xylogy = sum(x * y * log(y+SMALL))
       sum_ylogy = sum( y * log(y+SMALL))
       sum_xy = sum(x * y)
       sum_x2y = sum( x * x * y)
       sum_xy_sq = (sum( x * y) * sum( x * y) )
       print "exponential fitting"
       print "sum_y:      "+str(sum_y)
       print "sum_xylogy: "+str(sum_xylogy)
       print "sum_ylogy:  "+str(sum_ylogy)
       print "sum_xy:     "+str(sum_xy)
       print "sum_x2y:    "+str(sum_x2y)
       print "sum_xy_sq:  "+str(sum_xy_sq)
       A = (sum_x2y*sum_ylogy - sum_xy*sum_xylogy) / (sum_y * sum_x2y - sum_xy_sq)
       B = (sum_y * sum_xylogy - sum_xy*sum_ylogy) / (sum_y * sum_x2y - sum_xy_sq)
       print "exponential fit: A*exp(Bx), A: "+str(A)+" B: "+str(B)
       return A, B       





#########################################################################################################################################
#####  Python histogram and MI routines for testing
#########################################################################################################################################

def entropy(counts):
    '''Compute entropy.'''
    ps = counts/float(sum(counts))  # coerce to float and normalize
    ps = ps[nonzero(ps)]            # toss out zeros
    H = -sum(ps * log(ps))   # compute entropy
    
    return H

def python_mi_bins(x, y, bins):
    '''Compute mutual information'''
    counts_xy = histogram2d(x, y, bins=bins, range=[[0, 311335], [0, 311335]])[0]
    counts_x  = histogram(  x,    bins=bins, range=[0, 311335])[0]
    counts_y  = histogram(  y,    bins=bins, range=[0, 311335])[0]
    
    H_xy = entropy(counts_xy)
    H_x  = entropy(counts_x)
    H_y  = entropy(counts_y)
    
    return H_x + H_y - H_xy


def python_mi(x, y):
    '''Compute mutual information'''
    counts_xy = histogram2d(x, y, bins=bins, range=[[min(x), max(x)], [min(y), max(y)]])[0]
    counts_x  = histogram(  x,    bins=bins, range=[min(x), max(x)])[0]
    counts_y  = histogram(  y,    bins=bins, range=[min(y), max(y)])[0]
    
    H_xy = entropy(counts_xy)
    H_x  = entropy(counts_x)
    H_y  = entropy(counts_y)
    
    return H_x + H_y - H_xy

def kde_python_entropy(x):
       # Constants
       MIN_DOUBLE = 4.9406564584124654e-324 
                    # The minimum size of a Float64; used here to prevent the
                    #  logarithmic function from hitting its undefined region
                    #  at its asymptote of 0.
       INF = float('inf')  # The floating-point representation for "infinity"

       # x and y are previously defined as collections of 
       # floating point values with the same length

       # Kernel estimation
       gkde_x = gaussian_kde(x)
       

       #if len(binned_x) != len(binned_y) and len(binned_x) != len(x):
       #       x.append(x[0])
       #       y.append(y[0])

       #gkde_xy = gaussian_kde([x,y])
       entropy_lambda = lambda a,b: gkde_x([a]) * math.log((gkde_x([a])))

       
       (entropy_x, err_x) = \
           quad(entropy_lambda, -INF, INF )

       print 'debug: KDE entropy_x = ', entropy_x
       return entropy_x

def kde_python_mi(x,y):

       # Constants
       MIN_DOUBLE = 4.9406564584124654e-324 
                    # The minimum size of a Float64; used here to prevent the
                    #  logarithmic function from hitting its undefined region
                    #  at its asymptote of 0.
       INF = float('inf')  # The floating-point representation for "infinity"

       # x and y are previously defined as collections of 
       # floating point values with the same length

       # Kernel estimation
       gkde_x = gaussian_kde(x)
       gkde_y = gaussian_kde(y)

       #if len(binned_x) != len(binned_y) and len(binned_x) != len(x):
       #       x.append(x[0])
       #       y.append(y[0])

       gkde_xy = gaussian_kde([x,y])
       mutual_info = lambda a,b: gkde_xy([a,b]) * \
           math.log((gkde_xy([a,b]) / ((gkde_x(a) * gkde_y(b)) + MIN_DOUBLE)) \
        + MIN_DOUBLE)

       # Compute MI(X,Y)
       (minfo_xy, err_xy) = \
           dblquad(mutual_info, -INF, INF, lambda a: -INF, lambda a: INF)

       print 'debug: KDE mutual info minfo_xy = ', minfo_xy
       return minfo_xy

#########################################################################################################################################
##### RunParameters class:  Stores the parameters for an Entropy/Mutinf calculation run #################################################
#########################################################################################################################################

# e.g. Access with runparams.permutations
class RunParameters(object):
    def __init__(self, **kwds):
        self.__dict__ = kwds
        # print self.__dict__

    # allow get access e.g. o.myattribute
    def __getattr__(self, name):
        try: return self.__dict__[name]
        except KeyError:
            print "FATAL ERROR: can't find key '%s' in RunParamaters object" % name
            sys.exit(1)

    # allow set access e.g. o.myattribute = 2
    def __setattr_(self, name, val):
        self.__dict__[name] = val

    def __str__(self):
        d = self.__dict__
        return "Num sims = %d; Runs being used = %s; Binwidth = %.1f; Permutations = %d; Num Structs = %s" % \
               (d['num_sims'], " ".join(map(str, d['which_runs'])), d['binwidth'], d['permutations'], d['num_structs'])

    def get_logfile_prefix(self):
        return "%s-nsims%d-structs%d-bin%d" % (os.path.basename(self.resfile_fn), self.num_sims, self.num_structs, int(self.binwidth))

#########################################################################################################################################
##### AllAngleInfo class: stores dihedral angles from pdb-format trajectories  ##########################################################
#########################################################################################################################################

# Class to hold all the dihedral angle info for a set of trajectories
class AllAngleInfo:
   CACHE_FN = os.path.expanduser("~/dihed_mutent.cache")

   def __init__(self, run_params):
      rp = run_params
      self.all_chis = None
      self.num_sims, self.num_structs, self.num_res = rp.num_sims, rp.num_structs, rp.num_res
      self.calc_variance = False
   # load dihedrals from a pdb trajectory
   def load_angles_from_traj(self, sequential_sim_num, pdb_traj, run_params, cache_to_disk=True):
      sim_chis, found_in_cache = None, False

      # check if the simulations is cached
      if cache_to_disk:
          key = pdb_traj.traj_fn
          if os.path.exists(self.CACHE_FN):
              try:
                  angleInfo = None
                  cache = shelve.open(self.CACHE_FN)
                  self.num_res, sim_chis = cache[key]
                  cache.close()
                  found_in_cache = True
                  print "Loaded '%s' from the cache" % key, utils.flush()
              except Exception,e:
                  print e
                  pass # if the data isn't found in the cache, then lod it

      # if it wasn't found in the cache, load it
      if not found_in_cache:
          print "Loading trajectory '%s'" % pdb_traj.traj_fn, utils.flush()

          pdb_num = 0
          for pdb in pdb_traj.get_next_pdb():
              if pdb_num >= self.num_structs: continue
              
              if run_params.phipsi == -3: #calphas
                 pdb_phipsiomega = pdb.calc_phi_psi_omega()
                 self.num_res = pdb_phipsiomega.shape[0]
                 if sim_chis is None:
                        sim_chis = zeros((self.num_structs, self.num_res,6), float64)
                 pdb_calpha_coords = pdb.get_ca_xyz_matrix() #nres x 3 matrix
                 sim_chis[pdb_num, :, :3] = pdb_calpha_coords[:,:]
                 #print "pdb calpha coords: "+str(pdb_num)
                 #print pdb_calpha_coords
                 if (pdb_num+1) % 100 == 0:
                     print "Loaded pdb #%d" % (pdb_num), utils.flush()
               
              else:
                 pdb_chis = pdb.calc_chis()
                 pdb_phipsiomega = pdb.calc_phi_psi_omega()
                 if (pdb_num+1) % 100 == 0:
                     print "Loaded pdb #%d" % (pdb_num), utils.flush()

                 if sim_chis is None:
                     self.num_res = pdb_chis.shape[0]
                     sim_chis = zeros((self.num_structs, self.num_res,6), float64)
                 sim_chis[pdb_num, :, 0:2] = pdb_phipsiomega[:,0:2]
                 sim_chis[pdb_num, :, 2:] = pdb_chis
                 #print " shape phipsiomega:"+str(shape(pdb_phipsiomega))+" shape chis: "+str(pdb_chis.shape[0])
              
              pdb_num += 1
          print "sim chis", sim_chis

      # Store the angle data into the cache
      if cache_to_disk and not found_in_cache:
          cache = shelve.open(self.CACHE_FN)
          cache[key] = [self.num_res, sim_chis]
          cache.close()

      if self.all_chis is None:  
          self.all_chis = zeros((self.num_sims, self.num_structs , self.num_res,6), float64)
      self.all_chis[sequential_sim_num, :, : , :] = sim_chis[:, :, :]
      print "all chis:"
      print self.all_chis

   def get_angles(self, sequential_sim_num, sequential_res_num, res_name, backbone_only, phipsi, max_num_chis):
       chi_nums = []
       if phipsi == 2: chi_nums += [0,1] # which angles to use
       if backbone_only == 0: chi_nums += range(2, min(max_num_chis,NumChis[res_name]) + 2)
       if phipsi == -3: chi_nums += [0,1,2] #C-alpha x,y,z
       #print chi_nums
       #print self.all_chis

       curr_angles = zeros((len(chi_nums), self.all_chis.shape[1]))
       for sequential_chi_num, chi_num in zip(range(len(chi_nums)), chi_nums):
           curr_angles[sequential_chi_num, :] = self.all_chis[sequential_sim_num, :, int(sequential_res_num)-1, chi_num]
       return curr_angles, curr_angles.shape[1]




def make_name_num_list(reslist):
    name_num_list=[]
    
    for res in reslist: 
           chain = ''
           try:
                  if(str(res.chain)==None):
                         res.chain = ''
                  chain = res.chain
           except:
                  chain = ''
           name_num_list.append(res.name + str(res.num) + str(chain))
    return name_num_list



def compute_CA_dist_matrix(reslist,pdb_obj):  #pdb_obj is from PDBlite.py 
    CA_dist_matrix = zeros((len(reslist),len(reslist)),float32)
    
    for res_ind1, myres1 in zip(range(len(reslist)), reslist):
        for res_ind2, myres2 in zip(range(res_ind1, len(reslist)), reslist[res_ind1:]):
            print "\n#### Working on residues %s and %s (%s and %s):" % (myres1.num, myres2.num, myres1.name, myres2.name) 
            BBi = pdb_obj.get(myres1.chain,myres1.num)
            BBj = pdb_obj.get(myres2.chain,myres2.num)
            CAi = BBi.get_atom("CA") #CA alpha carbon from res i
            CAj = BBj.get_atom("CA") #CA alpha carbon from res j
            CA_dist_matrix[res_ind1,res_ind2] = sqrt(CAi.calc_dist2(CAj))
            CA_dist_matrix[res_ind2,res_ind1] = CA_dist_matrix[res_ind1,res_ind2] # for symmetry
            
    return CA_dist_matrix


#main


#########################################################################################################################################
##### Calc_entropy: calculates 1-dimensional entropy  ##########################################################
#########################################################################################################################################

   
# counts has shape [bootstrap_sets x nchi x nbins*MULTIPLE_1D_BINS]
def calc_entropy(counts, nchi, numangles_bootstrap, calc_variance=False, entropy = None, var_ent = None, symmetry=None, expansion_factors=None):
    #assert(sum(counts[5,2,:]) >= 0.9999 and sum(counts[5,2,:]) <= 1.0001)
    nbins = counts.shape[-1] #this will be nbins*MULTIPLE_1D_BINS
    bootstrap_sets = counts.shape[0]
    nchi = counts.shape[1]
    #this normalization converts discrete space sampled to continuous domain of integral
    #using number of total bins instead of number of nonzero bins overcorrects when not all bins populated
    #consider this as the coarse-grain entropy, like the number of minima, as opposed to the shape of the minima
    #think Boltzman's law: S = k log W 
    #normalization = sum(counts > 0, axis=-1) * 1.0 / TWOPI #number of nonzero bins divided by 2pi, foreeach boot/chi
    
    expansion_factors =  ones((bootstrap_sets, nchi),float64)
    print "expansion factors:" 
    print expansion_factors
    if expansion_factors is None:
           expansion_factors *= TWOPI 
    else:
           for i in range(bootstrap_sets):
                  expansion_factors[i,:] = expansion_factors[i,:] / TWOPI
    #normalization = nbins / (expansion_factors)
    normalization = nbins / TWOPI
    print "log normalization"
    print log(normalization)
    #need to expand the numangles_bootstrap over nchi and nbins for use below
    print "counts shape: "
    print shape(counts)
    numangles_bootstrap_chi_vector = repeat(repeat(reshape(numangles_bootstrap.copy(),(-1, 1, 1)),nbins,axis=2),nchi,axis=1)
    print " numangles_bootstrap_chi_vector shape: "
    print shape(numangles_bootstrap_chi_vector)
    #now replicate array of symmtery for chis, over bootstraps 
    #symmetry_bootstraps = resize(symmetry.copy(),(bootstrap_sets,nchi))
    if(VERBOSE >= 2):
        print "Counts"
        print counts
        print "entropy elements before sum and normalization:"
        print (counts * 1.0 / numangles_bootstrap_chi_vector) * (log(numangles_bootstrap_chi_vector) - special.psi(counts + SMALL) - (1 - 2*int16(counts % 2)) / (counts + 1.0))
    #print "Counts mod 2"
    #print counts % 2
    #print "Psi"
    #print special.psi(counts+SMALL)
    #print "Correction"
    #print (-(1 - 2*(int16(counts % 2))) / (counts + 1.0))
    print "entij"
    print ((counts * 1.0 / numangles_bootstrap_chi_vector) * (log(numangles_bootstrap_chi_vector) - special.psi(counts + SMALL) - (1 - 2*int16(counts % 2)) / (counts + 1.0)))
    
    if(NO_GRASSBERGER == False):
           entropy[:,:] = sum((counts * 1.0 / numangles_bootstrap_chi_vector) * (log(numangles_bootstrap_chi_vector) - special.psi(counts + SMALL) - (1 - 2*int16(counts % 2)) / (counts + 1.0)),axis=2) - log(normalization) 
    else:
           entropy[:,:] = sum(((counts * 1.0) / (numangles_bootstrap_chi_vector * 1.0)) * (log((counts + SMALL)* 1.0 / (numangles_bootstrap_chi_vector * 1.0))), axis=2) - log(normalization)
    #the -log(normalization) is to convert from discrete space to continuous space
    #symmetry is taken into acount as a contraction of the radial space over which integration occurs
    #the +log(symmetry_bootstraps) would be to correct for torsions with symmtery,
    #but in my case I use fewer bins for the data-rich part of symmetric torsions
    #so the symmetry correction cancels out and thus does not appear here
    
         
    return entropy, var_ent


def calc_entropy_adaptive(counts_adaptive, num_sims, nchi, bootstrap_choose=0, calc_variance=False, entropy = None, var_ent = None, binwidths=None):
    #assert(entropy != None and var_ent != None)
    #if(symmetry is None): symmetry = ones((nchi),int8)
    #print counts[5,0,:]
    #print counts[5,1,:]
    #print counts[5,2,:]   
    #print sum(counts[5,0,:])
    #assert(sum(counts[5,2,:]) >= 0.9999 and sum(counts[5,2,:]) <= 1.0001)
    #nbins = counts.shape[-1] #convert historgram to normalized pdf
    #inv_normalization = 1 / normalization
    #counts *= normalization  #normalization constant (arrayed to multiply with counts elementwise) so that integral of pi ln pi is unity over [0, 2pi], symmetry already accounted for in counts
    # note: symmetry affects range and nbins of nonzero value equally
    log_counts = log(counts_adaptive + SMALL)
    print "Counts_Adaptive:\n"
    print shape(counts_adaptive)
    print "Log Counts_Adaptive:\n"
    print shape(log_counts)
    print "Binwidths:\n"
    print shape(binwidths)
    #print log_counts.shape
    #print binwidths.shape
    #print shape(counts_adaptive * ( log_counts) * binwidths)
    #print shape(sum(counts_adaptive * ( log_counts) * binwidths ,axis=2))
    print  counts_adaptive * ( log_counts) * binwidths
    entropy[:,:] = -sum(counts_adaptive * ( log_counts) * binwidths ,axis=2)
    
    #print -sum(av_counts * ( log_av_counts),axis=1)
    print "Entropy:\n"
    print entropy
    
    if(calc_variance):
        #deriv_vector = - (1 + log_counts)
        #print "chi: "+str(mychi)+" bin: "+str(b)+" av_counts: " +str(av_counts)+"\n"+" entropy: "+str(-av_counts * ( log_av_counts))+" unc_av: "+str(unc_counts)+" var_ent this chi: "+str((- (1 + log_av_counts)) * (- (1 + log_av_counts)) * (unc_counts * unc_counts))
        if(num_sims > 1 and calc_variance):
            for mychi in range(nchi):
                var_ent[mychi] = vfast_cov_for1D(entropy[:,mychi])
            #bins_cov = vfast_cov_for1D(counts[:,mychi,:]) #observations are in the rows, conditions(variables) are in the columns
            #print bins_cov.shape, deriv_vector.shape
            #now we will gather all terms, diagonal and cross-terms, in the variance of joint entropy using matrix multiplication instead of explicit loop
            # var = A * Vij * T(A), where A={dy/dxj}|x=mu
            #if VERBOSE >= 2:
            #    print av_counts.shape, deriv_vector.shape, bins_cov.shape, transpose(deriv_vector).shape
            #var_ent[mychi] = inner(deriv_vector,inner(bins_cov,transpose(deriv_vector)))

         
    return entropy, var_ent



# Calculate the MI between two angle distributions, x & y.
# First calculate the vector (over all bins) of means of the joint probabilities, Pij.
# Then calculate the vectors (over all bins) of Pi*Pj and Pi+Pj.
# Now MI(x,y) = sum over all bins [ Pij * log(Pij/(Pi*Pj)) ].
# The variance of MI = d*C*d'    where d={dy/dxj}|x=mu = 1 + log(Pij/(Pi*Pj)) - (Pi+Pj) * Pij/(Pi*Pj)
#     and C is the matrix of the covariances Cij = sum over angles i [ (xi-mu(x)) * (yi-mu(y)) ]/(N-1)
pop_matrix = U = logU = pxipyj_flat = pop_matrix_sequential = U_sequential = logU_sequential = pxipyj_flat_sequential = None

    

#needed for entropy of small data sets
#### WARNING: this routine is broken and has not been tested

sumstuff_lookup = None
def sumstuff(ni_vector,numangles_bootstrap,permutations):
    nbins = ni_vector.shape[-1]
    bootstrap_sets = numangles_bootstrap.shape[0]
    mysum = zeros((bootstrap_sets,permutations+1,nbins),float64)
    maxnumangles = int(max(numangles_bootstrap))
    global sumstuff_lookup
    
    
    code_create_lookup = """
    int ni, N;
    for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      N = *(numangles_bootstrap + mybootstrap);
      for(ni = 0; ni < N; ni ++) {
        for(int j=ni+2;j<=N+2;j++) {
            *(sumstuff_lookup + mybootstrap*maxnumangles + ni) += 1.0/(double(j)) ;
          }
      }
    }
    """
    
    code_do_lookup = """
            //weave 6b
            int ni, N;
            for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                N = *(numangles_bootstrap + mybootstrap);
                for (int permut=0; permut < permutations + 1; permut++) {
                    for(int bin=0; bin < nbins; bin++) {
                      ni = *(ni_vector  +  mybootstrap*(0 + 1)*permut*nbins  +  permut*nbins  +  bin);
                        *(mysum + mybootstrap*(0 + 1)*permut*nbins  +  permut*nbins  +  bin) =
                             *(sumstuff_lookup + mybootstrap*maxnumangles + ni);
                      
                    }
                }
            }
            """
    
    if(sumstuff_lookup is None):
        sumstuff_lookup = zeros((bootstrap_sets,maxnumangles),float64)
        weave.inline(code_create_lookup,['mysum','bootstrap_sets','permutations','ni_vector','nbins','numangles_bootstrap','maxnumangles','sumstuff_lookup'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])


    weave.inline(code_do_lookup,['mysum','bootstrap_sets','permutations','ni_vector','nbins','numangles_bootstrap','maxnumangles','sumstuff_lookup'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
    return mysum


#########################################################################################################################################
##### Mutual Information Under the Null Hypothesis via Transition Matrix Master Equation  ##########################################################
#########################################################################################################################################


def calc_ind_mutinf_from_transition_matrix(chi_counts1, chi_counts2, bins1, bins2, chi_counts_sequential1, chi_counts_sequential2, bins1_sequential, bins2_sequential, num_sims, nbins, numangles_bootstrap, numangles, calc_variance=False,bootstrap_choose=0,permutations=0,which_runs=None):
       count_matrix = zeros((bootstrap_sets, permutations + 1 , nbins*nbins), float64)
       bootstrap_sets = len(which_runs)
       assert(all(chi_counts1 >= 0))
       assert(all(chi_counts2 >= 0))
       permutations_sequential = permutations
       #permutations_sequential = 0 #don't use permutations for mutinf between sims
       #max_num_angles = int(max(numangles))
       #if(numangles_star != None): 
       #       max_num_angles_star = int(max(numangles_star))
       max_num_angles = int(min(numangles))
       if(numangles_star != None): 
              max_num_angles_star = int(min(numangles_star))
       num_pair_runs = pair_runs.shape[1]
       print pair_runs
       print "bootstrap_sets: "+str(bootstrap_sets)+"num_pair_runs: "+str(num_pair_runs)+"\n"
       nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
       print "nbins_cor: "+str(nbins_cor)
       permutations = 0
       code = """

       ### BUILD 2-D COUNTS MATRICES ###  ## Could try CUDA histogram256, modify to 1024 if possible

    // weave6a
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     int mybootstrap, mynumangles, permut, anglenum; 
     unsigned long long myoffset;
     #pragma omp parallel for private(mybootstrap, permut, mynumangles, anglenum, myoffset)
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      for (permut=0; permut < permutations + 1; permut++) {
          mynumangles = *(numangles_bootstrap + mybootstrap);
          myoffset = mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins;
          for (anglenum=1; anglenum< mynumangles; anglenum++) {
          if(mybootstrap == bootstrap_sets - 1) {
            //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
            }
             *(count_matrix  + myoffset   +  (*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))) += 1;
             
          }
        }
       }

       """

       if(VERBOSE >= 2): print "about to populate count_matrix"
       weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','offset'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler] )

       # now build singlet transition matrix
       code2= """
       // weave_transition_matrix
       // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
       //#include <math.h>
       int mynumangles, mybootstrap, anglenum, lagtime, ibin, jbin = 0;
       //#pragma parallel for private (mybootstrap, mynumangles, lagtime, anglenum, ibin, jbin)
       for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {        
         mynumangles = *(numangles_bootstrap + mybootstrap);
         for (lagtime = 1; lagtime < 50; lagtime++) { 
            for (anglenum=lagtime_interval*lagtime; anglenum< mynumangles; anglenum++) {
              // compute "from" bin and "to" bin for T(bin1(t-dt),bin1(t)) , using naive reversible symmetrization
              *(transition_matrix1 + mybootstrap*nbins*nbins*nlagtimes + lagtime*nbins*nbins + (*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles + anglenum - lagtime_interval*lagtime))*nbins +   (*(bins1 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + anglenum))) += 1;
              *(transition_matrix2 + mybootstrap*nbins*nbins*nlagtimes + lagtime*nbins*nbins + (*(bins2  +  mybootstrap*bootstrap_choose*max_num_angles + anglenum - lagtime_interval*lagtime))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + anglenum))) +=1;
              // symmetrize
              *(transition_matrix1 + mybootstrap*nbins*nbins*nlagtimes + lagtime*nbins*nbins + (*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles + anglenum))*nbins +   (*(bins1 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + anglenum - lagtime_interval*lagtime))) += 1;
              *(transition_matrix2 + mybootstrap*nbins*nbins*nlagtimes + lagtime*nbins*nbins + (*(bins2  +  mybootstrap*bootstrap_choose*max_num_angles + anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + anglenum  - lagtime_interval*lagtime))) +=1;
            }
            // transition matrix should be normalized -- need to see how to convert eigenvalues of this to implied timescales ... 
            for(ibin = 0; ibin < nbins; ibin++)
            {
               for(jbin = 0; jbin < nbins; jbin++)
               {
                 *(transition_matrix1 + mybootstrap*nbins*nbins*nlagtimes + lagtime*nbins*nbins + ibin*nbins + jbin) /= (mynumangles / (lagtime_interval * lagtime))
                 *(transition_matrix1 + mybootstrap*nbins*nbins*nlagtimes + lagtime*nbins*nbins + ibin*nbins + jbin) /= (mynumangles / (lagtime_interval * lagtime));
                   
         }
          
       }
       """

       weave.inline(code2, ['num_sims', 'numangles_bootstrap', 'lagtime_interval','nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"], extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler])
       
       # next, determine 

       ### GET INITIAL 2D PDF ###


       ### MATRIX EXPONENTIAL TO PROPAGATE TO GET INDEPENDENT PDF
       ### expm(A, q = 7)  Compute the matrix exponential using Pade approximation of order q.
       ### expm2(A) Compute the matrix exponential using eigenvalue decomposition.
       ### expm3(A, q = 20) Compute the matrix exponential using a Taylor series.of order q.

       ### we may want to get eigenvectors and eigenvalues of the transition matrix on our own using MAGMA GPU eigenvalue solver instead


       ### CALCULATE MUTIN FOR TARGET PDF
       

       




#########################################################################################################################################
##### Mutual Information Under the Null Hypothesis via Monte Carlo Simulation  ##########################################################
#########################################################################################################################################

   
        
def calc_mutinf_multinomial_constrained( nbins, counts1, counts2, adaptive_partitioning, bootstraps = None ):
    
    # allocate the matrices the first time only for speed
    
    
    
    #ninj_prior = 1.0 / float64(nbins*nbins) #Perks' Dirichlet prior
    #ni_prior = nbins * ninj_prior           #Perks' Dirichlet prior
    ninj_prior = 1.0                        #Uniform prior
    ni_prior = nbins * ninj_prior           #Uniform prior
    
    
    #if(VERBOSE >=2 ):
    #       print "shape of counts1 for mu:" +str(shape(counts1))

    #bootstrap_sets = 1000 # generate this many random matrices
    #if(adaptive_partitioning == 0): bootstrap_sets = 100
    permutations = permutations_multinomial = 1000     # just fake here
    
    print "counts for multinomial"
    print counts1[0,:]
    
    #numangles_bootstrap = ones((bootstrap_sets),int32) * sum(counts1[0,:])  #make a local version of this variable
    if(bootstraps is None):
           bootstrap_sets = (shape(counts1))[0] #infer bootstrap sets from counts
    else:
           bootstrap_sets = bootstraps
    numangles_bootstrap = sum(counts1[:,:], axis=1) 

    ref_counts1 = zeros((bootstrap_sets, nbins),int32)
    ref_counts2 = zeros((bootstrap_sets, nbins),int32)

   

    chi_counts1 = resize(ref_counts1,(bootstrap_sets,permutations_multinomial,nbins))
    chi_counts2 = resize(ref_counts2,(bootstrap_sets,permutations_multinomial,nbins))
    chi_countdown1 = zeros((shape(chi_counts1)),float64)
    chi_countdown2 = zeros((shape(chi_counts2)),float64)

    #create reference distribution
    code_create_ref = """
    //weave6a0
    int bin = 0;
    int mynumangles = 0;
    int numangles = 0;
    int mybootstrap = 0;
    int permut = 0;
    for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) 
    {
       mynumangles = int(*(numangles_bootstrap + mybootstrap)); //assume only one real bootstrap set of data
       for (int anglenum=0; anglenum< mynumangles; anglenum++) {
          if(bin >= nbins) bin=0;
          *(ref_counts1 + mybootstrap*nbins + bin) += 1;
          *(ref_counts2 + mybootstrap*nbins + bin) += 1;
          bin++;
       }
      for(permut=0; permut < permutations_multinomial; permut++ )
      {
        for(bin=0; bin < nbins; bin++)
        {
           *(chi_counts1 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin) = *(ref_counts1 +  mybootstrap*nbins + bin) ;
           *(chi_countdown1 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin) = *(ref_counts1 +  mybootstrap*nbins + bin) ;
           *(chi_counts2 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin) = *(ref_counts2 +  mybootstrap*nbins + bin) ; 
           *(chi_countdown2 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin) = *(ref_counts2 +  mybootstrap*nbins + bin) ; 
        }
      }
    }
    """
    if(adaptive_partitioning != 0):
        print "preparing multinomial distribution for two-D independent histograms, given marginal distributions"
        weave.inline(code_create_ref,['nbins','ref_counts1','ref_counts2','numangles_bootstrap','bootstrap_sets','permutations_multinomial','chi_counts1','chi_counts2','chi_countdown1','chi_countdown2'], compiler=mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
    else:
        ref_counts1 = counts1.copy()
        ref_counts2 = counts2.copy()
    
    
                  
    print "chi_counts1 trial 1 :"
    print chi_counts1[0,:]
    print "chi_counts2 trial 1:"
    print chi_counts2[0,:]
    #chi_countdown1 = zeros((bootstrap_sets,permutations_multinomial,nbins),int32)
    #chi_countdown2 = zeros((bootstrap_sets,permutations_multinomial,nbins),int32)
    chi_countdown1 = chi_counts1.copy()
    chi_countdown2 = chi_counts2.copy()
    #chi_countdown1[:,:] = resize(chi_counts1[0,:].copy(),(bootstrap_sets,nbins)) #replicate up to bootstrap_sets
    #chi_countdown2[:,:] = resize(chi_counts2[0,:].copy(),(bootstrap_sets,nbins)) #replicate up to bootstrap_sets
    
    print "counts1 to pick from without replacement, first sample"
    print chi_countdown1[:,0]
    print "counts1 to pick from without replacement, last sample"
    print chi_countdown1[:,-1]
    
    
    if(True):
       
       # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
       #U = zeros((bootstrap_sets,0 + 1, nbins*nbins), float64)
       #logU = zeros((bootstrap_sets,0 + 1, nbins*nbins),float64)

       count_matrix_multi = zeros((bootstrap_sets, permutations_multinomial , nbins*nbins), int32)
       
       ninj_flat_Bayes = zeros((bootstrap_sets, permutations_multinomial ,nbins*nbins),float64)
           
       
       
       numangles_bootstrap_matrix = zeros((bootstrap_sets,permutations_multinomial,nbins*nbins),float64)
       for bootstrap in range(bootstrap_sets):
           for permut in range(permutations_multinomial):
               numangles_bootstrap_matrix[bootstrap,permut,:]=numangles_bootstrap[bootstrap]

    chi_counts1_vector = chi_counts1.copy() #reshape(chi_counts1,(bootstrap_sets,permutations_multinomial,nbins)) 
    chi_counts2_vector = chi_counts2.copy() #reshape(chi_counts2,(bootstrap_sets,permutations_multinomial,nbins)) 
    numangles_bootstrap_vector = numangles_bootstrap_matrix[:,:,:nbins]    
    count_matrix_multi[:,:,:] = 0
    
    
        
    code_multi = """
    // weave6a
     #include <math.h>
     int bin1, bin2, bin1_found, bin2_found = 0;
     int mybootstrap =0;
     int permut = 0;
     int  anglenum = 0;
     int  mynumangles = 0;
     // #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,bin1_found,bin2_found,bin1,bin2)
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      printf("sampling independent distribution -- bootstrap: %i\\n",mybootstrap);
      mynumangles = 0; // init
      for (permut=0; permut < permutations_multinomial; permut++) {
          mynumangles = *(numangles_bootstrap + mybootstrap); 
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
             bin1_found = 0;
             bin2_found = 0;
             while(bin1_found == 0) {       //sampling without replacement
                bin1 = int(drand48() * int(nbins));
                if( *(chi_countdown1 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin1) > 0.999) {  // the 0.999 is for fractional counts, which come into with weights
                  bin1_found = 1;
                  // #pragma omp atomic
                  *(chi_countdown1 + mybootstrap*nbins*permutations_multinomial + permut*nbins + bin1) -= 1;
                }
             }
             while(bin2_found == 0) {      //sampling without replacement
                bin2 = int(drand48() * int(nbins));
                if( *(chi_countdown2 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin2) > 0.999) { // the 0.999 is for fractional counts, which come into play with weights
                  bin2_found = 1;
                  // #pragma omp atomic
                  *(chi_countdown2 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin2) -= 1;
                }
             }
          //printf("bin1 %d bin2 %d\\n", bin1, bin2);
          // #pragma omp atomic
          *(count_matrix_multi  +  mybootstrap*permutations_multinomial*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2) += 1;
             
          }
        }
       }
      """
    weave.inline(code_multi, ['numangles_bootstrap', 'nbins', 'count_matrix_multi','chi_counts1','chi_counts2','chi_countdown1','chi_countdown2','bootstrap_sets','permutations_multinomial'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                 #extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler]
                 #extra_compile_args =['  -fopenmp'],
                 #extra_link_args=['-lgomp'])

    del chi_countdown1 #free up mem
    del chi_countdown2 #free up mem
    #print "done with count matrix setup for multinomial\n"
    for bootstrap in range(bootstrap_sets):
           for permut in range(permutations_multinomial):
              my_flat = outer(chi_counts1[bootstrap,permut] + ni_prior,chi_counts2[bootstrap,permut] + ni_prior).flatten() 
              #my_flat = resize(my_flat,(permutations_multinomial,(my_flat.shape)[0]))
              ninj_flat_Bayes[bootstrap,permut,:] = my_flat[:] #[:,:]

    if(numangles_bootstrap[0] > 0): #small sample stuff turned off for now cause it's broken
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):
      if(NO_GRASSBERGER == False):     
        ent1_boots = sum((chi_counts1_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_vector + SMALL) - (1 - 2*(chi_counts1_vector % 2)) / (chi_counts1_vector + 1.0)),axis=2)

        ent2_boots = sum((chi_counts2_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_vector + SMALL) - (1 - 2*(chi_counts2_vector % 2)) / (chi_counts2_vector + 1.0)),axis=2)
        
    
        # MI = H(1)+H(2)-H(1,2)
        # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
        #print "Numangles Bootstrap Matrix:\n"+str(numangles_bootstrap_matrix)
        
        mutinf_thisdof = ent1_boots + ent2_boots  \
                     - (sum((count_matrix_multi * 1.0 /numangles_bootstrap_matrix)  \
                           * ( log(numangles_bootstrap_matrix) - \
                               special.psi(count_matrix_multi + SMALL)  \
                               - (1 - 2*(count_matrix_multi % 2)) / (count_matrix_multi + 1.0) \
                               ),axis=2))
      else:  #using naive histogram p log p estimate
        mutinf_thisdof =  sum((chi_counts1_vector * 1.0 / numangles_bootstrap_vector) * (log((chi_counts1_vector * 1.0 / numangles_bootstrap_vector + SMALLER))),axis=2) \
            +     sum((chi_counts2_vector * 1.0 / numangles_bootstrap_vector) * (log((chi_counts2_vector * 1.0 / numangles_bootstrap_vector + SMALLER))),axis=2) \
            -     sum((count_matrix_multi * 1.0 /numangles_bootstrap_matrix)  \
                           * ( log((count_matrix_multi * 1.0 /numangles_bootstrap_matrix) + SMALLER)), axis=2)

        
    else:
        ent1_boots = sum((chi_counts1_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts1_vector,numangles_bootstrap_vector,permutations_multinomial),axis=2)

        ent2_boots = sum((chi_counts2_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts2_vector,numangles_bootstrap_vector,permutations_multinomial),axis=2)

        #print "shapes:"+str(ent1_boots)+" , "+str(ent2_boots)+" , "+str(sum((count_matrix_multi + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix_multi,numangles_bootstrap_matrix,permutations_multinomial),axis=2))
        
        mutinf_thisdof = ent1_boots + ent2_boots \
                         -sum((count_matrix_multi + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix_multi,numangles_bootstrap_matrix,permutations_multinomial),axis=2)
    
    
    
            
    
    
    
    #Now, since permutations==0 as we are doing Monte Carlo sampling instead of permutations, filter according to Bayesian estimate of distribution
    # of mutual information, M. Hutter and M. Zaffalon 2004 (or 2005).
    # Here, we will discard those MI values with p(I | data < I*) > 0.05.
    # Alternatively, we could use the permutation method or a more advanced monte carlo simulation
    # over a Dirichlet distribution to empirically determine the distribution of mutual information of the uniform
    # distribution.  The greater variances of the MI in nonuniform distributions suggest this approach
    # rather than a statistical test against the null hypothesis that the MI is the same as that of the uniform distribution.
    # The uniform distribution or sampling from a Dirichlet would be appropriate since we're using adaptive partitioning.

    #First, compute  ln(nij*n/(ni*nj) = logU, as we will need it and its powers shortly.
    #Here, use Perks' prior nij'' = 1/(nbins*nbins)
    
    if(True):  #then use Bayesian approach to approximate distribution of mutual information given data,prior
        count_matrix_multi_wprior = count_matrix_multi + ninj_prior
        numangles_bootstrap_matrix_wprior = numangles_bootstrap_matrix + ninj_prior*nbins*nbins
        numangles_bootstrap_vector_wprior = numangles_bootstrap_vector + ninj_prior*nbins*nbins
        Uij = (numangles_bootstrap_matrix_wprior) * (count_matrix_multi_wprior) / (ninj_flat_Bayes)
        logUij = log(Uij) # guaranteed to not have a zero denominator for non-zero prior (non-Haldane prior)

        Jij=zeros((bootstrap_sets, permutations_multinomial, nbins*nbins),float64)
        Jij = (count_matrix_multi_wprior / (numangles_bootstrap_matrix_wprior)) * logUij

        #you will see alot of "[:,0]" following. This means to take the 0th permutation, in case we're permuting data
    
        J = (sum(Jij, axis=-1))#[:,0] #sum over bins ij
        K = (sum((count_matrix_multi_wprior / (numangles_bootstrap_matrix_wprior)) * logUij * logUij, axis=-1))#[:,0] #sum over bins ij
        L = (sum((count_matrix_multi_wprior / (numangles_bootstrap_matrix_wprior)) * logUij * logUij * logUij, axis=-1))#[:,0] #sum over bins ij
    
        #we will need to allocate Ji and Jj for row and column sums over matrix elemenst Jij:

        Ji=zeros((bootstrap_sets, permutations_multinomial, nbins),float64)
        Jj=zeros((bootstrap_sets, permutations_multinomial, nbins),float64)
        chi_counts_bayes_flat1 = zeros((bootstrap_sets,permutations_multinomial, nbins*nbins),int32)
        chi_counts_bayes_flat2 = zeros((bootstrap_sets,permutations_multinomial, nbins*nbins),int32)
    
    
        
        #repeat(chi_counts2[bootstrap] + ni_prior,permutations_multinomial +1,axis=0)
        for bootstrap in range(bootstrap_sets):
            chi_counts_matrix1 = reshape(resize(chi_counts1[bootstrap] + ni_prior, bootstrap_sets*(permutations_multinomial)*nbins),(bootstrap_sets,permutations_multinomial,nbins))
            chi_counts_matrix2 = reshape(resize(chi_counts2[bootstrap] + ni_prior, bootstrap_sets*(permutations_multinomial)*nbins),(bootstrap_sets,permutations_multinomial,nbins))
        
            #print "chi counts 1:" + str(chi_counts1[bootstrap])
            #print "chi counts 2:" + str(chi_counts2[bootstrap])
            
            #print "counts:\n" + str(count_matrix_multi[bootstrap])
            #now we need to reshape the marginal counts into a flat "matrix" compatible with count_matrix_multi
            # including counts from the prior
            #print "chi_counts2 shape:"+str(shape(chi_counts2))
            #print "chi_counts2[bootstrap]+ni_prior shape:"+str((chi_counts2[bootstrap] + ni_prior).shape)
            
            #chi_counts_bayes_flat1[bootstrap] = chi_counts1[bootstrap] + ni_prior
            #chi_counts_bayes_flat2[bootstrap] = chi_counts2[bootstrap] + ni_prior
            for permut in range(permutations):
                   chi_counts_bayes_flat2[bootstrap,permut,:] = repeat(chi_counts2[bootstrap,permut] + ni_prior, nbins, axis=0)
                   chi_counts_bayes_flat1[bootstrap,permut,:] = (transpose(reshape(resize(chi_counts1[bootstrap,permut] + ni_prior, nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)    
                   #chi_counts_bayes_flat2[bootstrap,0,:] = repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
                   ##print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
                   ##handling the slower-varying index will be a little bit more tricky
                   #chi_counts_bayes_flat1[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1[bootstrap] + ni_prior, nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
                   ##we will also need to calculate row and column sums Ji and Jj:
                   Jij_2D_boot = reshape(Jij[bootstrap,permut,:],(nbins,nbins))
                   #Ji is the sum over j for row i, Jj is the sum over i for column j, j is fastest varying index
                   #do Ji first
                   Ji[bootstrap,permut,:] = sum(Jij_2D_boot, axis=1)
                   Jj[bootstrap,permut,:] = sum(Jij_2D_boot, axis=0)


        #ok, now we calculated the desired quantities using the matrices we just set up
    
    
        numangles_bootstrap_wprior = transpose(resize(numangles_bootstrap + ninj_prior*nbins*nbins,(permutations_multinomial, bootstrap_sets)))
        
        M = (sum((1.0/(count_matrix_multi_wprior + SMALL) - 1.0/chi_counts_bayes_flat1 -1.0/chi_counts_bayes_flat2 \
                  + 1.0/numangles_bootstrap_matrix_wprior) \
                 * count_matrix_multi_wprior * logUij, axis=2))#[:,0]

        Q = (1 - sum(count_matrix_multi_wprior * count_matrix_multi_wprior / ninj_flat_Bayes, axis=2))#[:,0]

        if(VERBOSE >= 2):
            print "shapes"
            print "Ji:   "+str(shape(Ji))
            print "Jj:   "+str(shape(Jj))
            print "chi counts matrix 1:   "+str(chi_counts_matrix1)
            print "chi counts matrix 2:   "+str(chi_counts_matrix2)
            print "numangles bootstrap wprior:   "+str(shape(numangles_bootstrap_wprior))
            print "count_matrix_multi_wprior:" +str(shape(count_matrix_multi_wprior))
            print "chi_counts_bayes_flat1:"+str(shape(chi_counts_bayes_flat1))
            print "chi_counts_bayes_flat2:"+str(shape(chi_counts_bayes_flat2))
            
        P = (sum((numangles_bootstrap_vector_wprior) * Ji * Ji / chi_counts_matrix1, axis=2) \
             + sum(numangles_bootstrap_vector_wprior * Jj * Jj / chi_counts_matrix2, axis=2))#[:,0]

        #Finally, we are ready to calculate moment approximations for p(I | Data)
        E_I_mat = ((count_matrix_multi_wprior) / (numangles_bootstrap_matrix_wprior * 1.0)) \
                  * (  special.psi(count_matrix_multi_wprior + 1.0) \
                     - special.psi(chi_counts_bayes_flat1 + 1.0) \
                     - special.psi(chi_counts_bayes_flat2 + 1.0) \
                     + special.psi(numangles_bootstrap_matrix_wprior + 1.0))
        #E_I = average(sum(E_I_mat,axis = 2)[:,0]) # to get rid of permutations dimension and to average over bootstrap samples
        E_I = average(sum(E_I_mat,axis = 2),axis=1) # average over permutations dimension 
        #Var_I_runs = var(sum(E_I_mat,axis = 2)[:,0]) #average over bootstraps is needed here
        Var_I_runs = var(sum(E_I_mat,axis = 2),axis=1) #average over permutations not bootstraps
        
        #Var_I = abs(average( \
        #     ((K - J*J))/(numangles_bootstrap_wprior + 1) + (M + (nbins-1)*(nbins-1)*(0.5 - J) - Q)/((numangles_bootstrap_wprior + 1)*(numangles_bootstrap_wprior + 2)), axis=0)) # all that remains here is to average over bootstrap samples
        Var_I = abs(average( \
             ((K - J*J))/(numangles_bootstrap_wprior + 1) + (M + (nbins-1)*(nbins-1)*(0.5 - J) - Q)/((numangles_bootstrap_wprior + 1)*(numangles_bootstrap_wprior + 2)), axis=1)) # all that remains here is to average over permutations

        #now for higher moments, leading order terms

        #E_I3 = average((1.0 / (numangles_bootstrap_wprior * numangles_bootstrap_wprior) ) \
        #       * (2.0 * (2 * J**3 -3*K*J + L) + 3.0 * (K + J*J - P)))

        #E_I4 = average((3.0 / (numangles_bootstrap_wprior * numangles_bootstrap_wprior)) * ((K - J*J) ** 2))

        #convert to skewness and kurtosis (not excess kurtosis)

        
        print "Descriptive Mutinf Multinomial:"
        print average(mutinf_thisdof[:,0])
        #print "Moments for Bayesian p(I|Data) for Multinomial Dist, w/constrained marginal counts:"
        print "E_I      Multinomial: "+str(E_I)
        #print "Var_I    Multinomial: "+str(Var_I)
        #print "Stdev_I  Multinomial:"+str(sqrt(Var_I))
        #print "skewness Multinomial: "+str(E_I3/ (Var_I ** (3/2)) )
        #print "kurtosis Multinomial: "+str(E_I4/(Var_I ** (4/2)) )
        
        #print "mutinf multinomial shape:"
        #print mutinf_thisdof.shape
        print
    

    else:  #do nothing
        print 
    var_mi_thisdof = zeros((bootstrap_sets), float64)
    for bootstrap in range(bootstrap_sets):
           var_mi_thisdof[bootstrap] = vfast_cov_for1D_boot_multinomial(reshape(mutinf_thisdof[bootstrap,:].copy(),(mutinf_thisdof.shape[1],1)))[0,0]
    
    E_I3 = E_I * 0.0 #zeroing now for speed since we're not using these
    E_I4 = E_I * 0.0 #zeroing now for speed since we're not using these
    del count_matrix_multi_wprior, count_matrix_multi,  chi_counts_matrix1, chi_counts_matrix2, chi_counts1_vector, chi_counts2_vector
    return E_I, Var_I , E_I3, E_I4, Var_I_runs, mutinf_thisdof, var_mi_thisdof



#########################################################################################################################################
count_matrix_markov = None
numangles_bootstrap_matrix_markov = None
def calc_mutinf_markov_independent( nbins, chi_counts1_markov, chi_counts2_markov, ent1_markov_boots, ent2_markov_boots, bins1, bins2, bootstrap_sets, bootstrap_choose, markov_samples, max_num_angles, numangles_bootstrap, bins1_slowest_lagtime, bins2_slowest_lagtime):
    
    global count_matrix_markov
    global numangles_bootstrap_matrix_markov
    global OUTPUT_INDEPENDENT_MUTINF_VALUES
    # allocate the matrices the first time only for speed
    
    markov_interval = zeros((bootstrap_sets),int16)
    # allocate the matrices the first time only for speed
    if(markov_samples > 0 and bins1_slowest_lagtime != None and bins2_slowest_lagtime != None ):
           for bootstrap in range(bootstrap_sets):
                  max_lagtime = max(bins1_slowest_lagtime[bootstrap], bins2_slowest_lagtime[bootstrap] )
                  markov_interval[bootstrap] = int(max_num_angles / max_lagtime) #interval for final mutual information calc, based on max lagtime
    else:
           markov_interval[:] = 1

    print "markov_interval :"
    print markov_interval
    #ninj_prior = 1.0 / float64(nbins*nbins) #Perks' Dirichlet prior
    #ni_prior = nbins * ninj_prior           #Perks' Dirichlet prior
    ninj_prior = 1.0                        #Uniform prior
    ni_prior = nbins * ninj_prior           #Uniform prior
    
    #if count_matrix_markov is None:    

    count_matrix_markov = zeros((bootstrap_sets, markov_samples , nbins*nbins), float64)
       
    print "shape of chi counts1_markov:"
    print shape(chi_counts1_markov)
       
    
       
    #if(numangles_bootstrap_matrix_markov is None):
    print "numangles bootstrap: "+str(numangles_bootstrap)
    numangles_bootstrap_matrix_markov = zeros((bootstrap_sets,markov_samples,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
           for markov_chain in range(markov_samples):
                  numangles_bootstrap_matrix_markov[bootstrap,markov_chain,:]=numangles_bootstrap[bootstrap]
    numangles_bootstrap_vector = numangles_bootstrap_matrix_markov[:,:,:nbins]
        #print "Numangles Bootstrap Vector:\n"+str(numangles_bootstrap_vector)
    
    #if(chi_counts2_markov_matrix is None):
    #chi_counts1_markov_matrix = zeros((bootstrap_sets, markov_samples, nbins*nbins),float64)        
    #chi_counts2_markov_matrix = zeros((bootstrap_sets, markov_samples, nbins*nbins),float64)       
    

    
    count_matrix_markov[:,:,:] = 0
    ent1_boots = zeros((bootstrap_sets,markov_samples),float64)
    ent2_boots = zeros((bootstrap_sets,markov_samples),float64)
    ent_1_2_boots = zeros((bootstrap_sets,markov_samples),float64)

    ## 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
    chi_counts1_markov_vector = chi_counts1_markov #reshape(chi_counts1_markov.copy(),(bootstrap_sets, markov_samples ,nbins)) #no permutations for marginal distributions...
    chi_counts2_markov_vector = chi_counts2_markov #reshape(chi_counts2_markov.copy(),(bootstrap_sets, markov_samples ,nbins)) #no permutations for marginal distributions...
    ## already contains data along markov_samples axis 
    ##chi_counts1_markov_vector = repeat(chi_counts1_markov_vector, markov_samples, axis=1)     #but we need to repeat along markov_samples axis
    ##chi_counts2_markov_vector = repeat(chi_counts2_markov_vector, markov_samples, axis=1)     #but we need to repeat along markov_samples axis
    
    #for markov_chain in range(markov_samples):
    #       for bootstrap in range(bootstrap_sets): #copy here is critical to not change original arrays
    #              chi_counts2_markov_matrix[bootstrap,markov_chain,:] = repeat(chi_counts2_markov[bootstrap,markov_chain].copy(),nbins,axis=-1) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
    #              #print repeat(chi_counts2_markov[bootstrap] + ni_prior,nbins,axis=0)
    #              #handling the slower-varying index will be a little bit more tricky
    #              chi_counts1_markov_matrix[bootstrap,markov_chain,:] = (transpose(reshape(resize(chi_counts1_markov[bootstrap,markov_chain].copy(), nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
    
    ## Getting too many 0 values for bin12, need to see what is up with this in terms of interating over arrays ... make sure no trailing zeros from bins arrays being too long

    #no_boot_weights = False
    #if (boot_weights is None):
    #       boot_weights = ones((bootstrap_sets, max_num_angles * bootstrap_choose), float64)
    #       no_boot_weights = True





    code_ent2 = """
     int bin1, bin2, bin3 = 0;
     int mybootstrap, mynumangles,markov_chain,anglenum;
     //long  offset1, offset2, offset3, offset4;
     long  counts2 ;
     double counts2d ;
     double dig2;
     double mysign2 = 0 ;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
 
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, mysign1, mysign2, bin1, bin2, dig1, dig2, counts1d, counts2d, counts12, counts12d)  
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
       for(bin2=0; bin2 < nbins; bin2++)
          {
           //printf("bin2: %i counts2 index: %i \\n",  bin2, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin2);
           counts2 = *(chi_counts2_markov  +  mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  bin2 );
          

           mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
 
           if(counts2 > 0)
           {
            counts2d = 1.0 * counts2;
            dig2 = DiGamma_Function(counts2d );
           

           
           *(ent2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ( (counts2 + 0.0) / mynumangles)*(log(mynumangles) - dig2 - (mysign2 / ((double)(counts2d + 1.0L)))); 

           printf("bin2: %i counts2 index: %i \\n",  bin2, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin2);
           printf("counts to numangles ratio %f \\n", (( (counts2d) / mynumangles)));
           printf("log numangles             %f \\n", (( (log(mynumangles)))));
           printf("mysign1                   %f \\n",  mysign2 );
           printf("mysign1 / counts+1        %e \\n",  mysign2 / ((double)(counts2d + 1.0L)));
           printf("log numangles minus dig1  %f \\n", (( (log(mynumangles) - dig2))));
           printf("corr                      %e \\n",  (mysign2 / ((double)((counts2d + 1.0L)))));
           printf("log numangles minus corr. %e \\n", (( (log(mynumangles) - dig2 - ((double)mysign1 / ((double)(counts2d + 1.0L)))))));
           
           printf("ent2 boots term counts2d:%f, dig:%f, term:%f sum:%e \\n",counts2d, dig2,(double)( (counts2d ) / mynumangles)*(log(mynumangles) - dig2 - (mysign2 / (double)(counts2d + 1.0L))), (double)(*(ent2_boots + mybootstrap*markov_samples + markov_chain))); 
           }
           printf("\\n");
          }
      } 
      
    """

    code = """
    // weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     #include <math.h>

  
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mybootstrap, mynumangles,markov_chain,anglenum;
     //long  offset1, offset2, offset3, offset4;
     long  counts1, counts2, counts12;
     double counts1d, counts2d, counts12d;
     double dig1, dig2;
     double mysign1, mysign2 = 0 ;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
 
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, mysign1, mysign2, bin1, bin2, dig1, dig2, counts1d, counts2d, counts12, counts12d) 
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
        
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
 
       
             // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8)  --- but with chi already dereferenced :  the bin for each dihedral
             // python: self.chi_counts_markov=zeros((self.nchi, bootstrap_sets, self.markov_samples, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
             //if(anglenum % markov_interval[mybootstrap] == 0) {

              angle1_bin = *(bins1  +  (long)(mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum));
              angle2_bin = *(bins2  +  (long)(mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum));
        
              
              *(count_matrix_markov  + (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle2_bin )) += 1.0 ;  //* weight;
             //}
             
          }

          // now actually compute the entropies for each order to be combined later
 
          
         for(bin1=0; bin1 < nbins; bin1++) 
          { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
         
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           counts12 = *(count_matrix_markov  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
 
           if(counts12 > 0)
           {
            mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts12);
           
           
           counts12d = 1.0 * counts12;
           *(ent_1_2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts12d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts12 + 1.0)))); 
           }
          }
         }
       
    

        }
       }
      """

    code_no_grassberger = """
    <// weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     #include <math.h>

  
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mybootstrap, mynumangles,markov_chain,anglenum;
     //long  offset1, offset2, offset3, offset4;
     long  counts1, counts2, counts12;
     double counts1d, counts2d, counts12d;
     double dig1, dig2;
     double mysign1, mysign2 = 0 ;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
 
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, mysign1, mysign2, bin1, bin2, dig1, dig2, counts1d, counts2d, counts12, counts12d) 
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
        
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
 
       
             // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8)  --- but with chi already dereferenced :  the bin for each dihedral
             // python: self.chi_counts_markov=zeros((self.nchi, bootstrap_sets, self.markov_samples, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
             //if(anglenum % markov_interval[mybootstrap] == 0) {

              angle1_bin = *(bins1  +  (long)(mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum));
              angle2_bin = *(bins2  +  (long)(mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum));
        
              
              *(count_matrix_markov  + (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle2_bin )) += 1.0 ;  //* weight;
             //}
             
          }

          // now actually compute the entropies for each order to be combined later
 
          
         for(bin1=0; bin1 < nbins; bin1++) 
          { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
         
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           counts12 = *(count_matrix_markov  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
 
           if(counts12 > 0)
           {
            mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts12);
           
           
           counts12d = 1.0 * counts12;
           *(ent_1_2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += -1.0 * ((double)counts12d / mynumangles)*(log((double)counts12d / mynumangles + SMALLER)); 
           }
          }
         }
       
    

        }
       }
      """

    code_markov_interval = """
    // weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     #include <math.h>

  
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mybootstrap, mynumangles,markov_chain,anglenum;
     //long  offset1, offset2, offset3, offset4;
     long  counts1, counts2, counts12;
     double counts1d, counts2d, counts12d;
     double dig1, dig2;
     double mysign1, mysign2 = 0 ;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
 
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, mysign1, mysign2, bin1, bin2, dig1, dig2, counts1d, counts2d, counts12, counts12d) 
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
        
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
             //if(anglenum == mynumangles - 1) {
             //  printf("");  // just to make sure memory writes complete before we proceed
             //}  
       
             // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8)  --- but with chi already dereferenced :  the bin for each dihedral
             // python: self.chi_counts_markov=zeros((self.nchi, bootstrap_sets, self.markov_samples, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
             if(anglenum % markov_interval[mybootstrap] == 0) {

              angle1_bin = *(bins1  +  (long)(mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum));
              angle2_bin = *(bins2  +  (long)(mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum));
        
              
              *(count_matrix_markov  + (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle2_bin )) += 1.0 ;  //* weight;
             }
             
          }
          

         for(bin1=0; bin1 < nbins; bin1++) 
          { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
         
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           counts12 = *(count_matrix_markov  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
 
           if(counts12 > 0)
           {
            mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts12);
           
           
           counts12d = 1.0 * counts12;
           *(ent_1_2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts12d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts12 + 1.0)))); 
           }
          }
         }
       
    

        }
       }
      """


    code_markov_interval_singlets = """
    // weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     #include <math.h>

  
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mybootstrap, mynumangles,markov_chain,anglenum;
     //long  offset1, offset2, offset3, offset4;
     long  counts1, counts2, counts12;
     double counts1d, counts2d, counts12d;
     double dig1, dig2;
     double mysign1, mysign2 = 0 ;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
 
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, mysign1, mysign2, bin1, bin2, dig1, dig2, counts1d, counts2d, counts12, counts12d) 
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {

          // now actually compute the entropies for each order to be combined later
            
          for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1 = int(*(chi_counts1_markov  + (long)( mybootstrap*markov_samples*nbins   +  markov_chain*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1 += *(count_matrix_markov  + (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins    +  bin1*nbins + bin2));
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1);
                
                
                counts1d = 1.0 * counts1;
                *(ent1_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts1 + 1.0)))); 
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2 = int(*(chi_counts2_markov  + (long)( mybootstrap*markov_samples*nbins   +  markov_chain*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix_markov  + (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins +  bin2*nbins + bin1));
              }
             }

 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts2d / mynumangles)*(log(mynumangles) - dig2 - ((double)mysign2 / ((double)(counts2 + 1.0)))); 
              }
         }
      } 
     }
     """


    #this old code used to be under the loop over markov_samples in the above code snippet, but the functionality has been migrated to markov 1D sampling
    old_code = """
       for(bin1=0; bin1 < nbins; bin1++) 
          {
              printf("markov_chain: %i bin1: %i counts1 index: %i counts:%i \\n",  markov_chain, bin1, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1,*(chi_counts1_markov  +   mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1 )) ;
          }
          for(bin1=0; bin1 < nbins; bin1++) 
          {
              printf("markov_chain: %i bin1: %i counts2 index: %i counts:%i \\n",  markov_chain, bin1, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1,*(chi_counts2_markov  +   mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1 )) ;
          }               
          printf("\\n ");
          printf("\\n ");
          for(bin1=0; bin1 < nbins; bin1++) 
          {
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           // python: self.chi_counts_markov=zeros((self.nchi, bootstrap_sets, self.markov_samples, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
           
           counts1 = *(chi_counts1_markov  +   mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1 );
           mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
           if(counts1 > 0)
           {
 
           
           
           
           counts1d = 1.0 * counts1;
           dig1 = DiGamma_Function(counts1d );
           *(ent1_boots + (long)(mybootstrap*markov_samples + markov_chain)) += (double)( (counts1d ) / mynumangles)*(log(mynumangles) - dig1 - (mysign1 / ((double)(counts1d + 1.0L)))); 
           
           
           printf("bin1: %i counts1 index: %i \\n",  bin1, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1);
           printf("counts to numangles ratio %f \\n", (( (counts1d) / mynumangles)));
           printf("log numangles             %f \\n", (( (log(mynumangles)))));
           printf("mysign1                   %f \\n",  mysign1 );
           printf("mysign1 / counts+1        %e \\n",  mysign1 / ((double)(counts1d + 1.0L)));
           printf("log numangles minus dig1  %f \\n", (( (log(mynumangles) - dig1))));
           printf("corr                      %e \\n",  (mysign1 / ((double)((counts1d + 1.0L)))));
           printf("log numangles minus corr. %e \\n", (( (log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts1d + 1.0L)))))));
           
           printf("ent1 boots term counts1d:%f, dig:%f, term:%f sum:%e \\n",counts1d, dig1,(double)( (counts1d ) / mynumangles)*(log(mynumangles) - dig1 - (mysign1 / (double)(counts1d + 1.0L))), (double)(*(ent1_boots + mybootstrap*markov_samples + markov_chain))); 
           }
          }

          for(bin2=0; bin2 < nbins; bin2++)
          {
           //printf("bin2: %i counts2 index: %i \\n",  bin2, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin2);
           counts2 = *(chi_counts2_markov  +  mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  bin2 );
          

           mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
 
           if(counts2 > 0)
           {
            counts2d = 1.0 * counts2;
            dig2 = DiGamma_Function(counts2d );
           

           
           *(ent2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ( (counts2 + 0.0) / mynumangles)*(log(mynumangles) - dig2 - (mysign2 / ((double)(counts2d + 1.0L)))); 

           printf("bin2: %i counts2 index: %i \\n",  bin2, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin2);
           printf("counts to numangles ratio %f \\n", (( (counts2d) / mynumangles)));
           printf("log numangles             %f \\n", (( (log(mynumangles)))));
           printf("mysign1                   %f \\n",  mysign2 );
           printf("mysign1 / counts+1        %e \\n",  mysign2 / ((double)(counts2d + 1.0L)));
           printf("log numangles minus dig1  %f \\n", (( (log(mynumangles) - dig2))));
           printf("corr                      %e \\n",  (mysign2 / ((double)((counts2d + 1.0L)))));
           printf("log numangles minus corr. %e \\n", (( (log(mynumangles) - dig2 - ((double)mysign1 / ((double)(counts2d + 1.0L)))))));
           
           printf("ent2 boots term counts2d:%f, dig:%f, term:%f sum:%e \\n",counts2d, dig2,(double)( (counts2d ) / mynumangles)*(log(mynumangles) - dig2 - (mysign2 / (double)(counts2d + 1.0L))), (double)(*(ent2_boots + mybootstrap*markov_samples + markov_chain))); 
           }
           printf("\\n");
          }
         
          printf("\\n ");
     """
     # //weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights


    if(VERBOSE >= 2): print "about to populate triplet  count_matrix_triplet_markov"
    print "chi counts1_markov:"
    print chi_counts1_markov
    print "chi counts2_markov:"
    print chi_counts2_markov
    if(NO_GRASSBERGER == False):
           weave.inline(code, ['numangles_bootstrap', 'nbins', 'bins1', 'bins2',  'count_matrix_markov','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset','ent1_boots','ent2_boots','ent_1_2_boots','SMALL','chi_counts1_markov','chi_counts2_markov','markov_interval'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler] ,
                 support_code=my_support_code )
    else:
           weave.inline(code_no_grassberger, ['numangles_bootstrap', 'nbins', 'bins1', 'bins2',  'count_matrix_markov','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset','ent1_boots','ent2_boots','ent_1_2_boots','SMALL','SMALLER','chi_counts1_markov','chi_counts2_markov','markov_interval'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler] ,
                 support_code=my_support_code )
    #weave.inline(code_markov_interval, ['numangles_bootstrap', 'nbins', 'bins1', 'bins2',  'count_matrix_markov','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset','ent1_boots','ent2_boots','ent_1_2_boots','SMALL','chi_counts1_markov','chi_counts2_markov','markov_interval'],
    #             #type_converters = converters.blitz,
    #             compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler] ,
    #             support_code=my_support_code )

    #weave.inline(code_markov_interval_singlets, ['numangles_bootstrap', 'nbins', 'bins1', 'bins2',  'count_matrix_markov','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset','ent1_boots','ent2_boots','ent_1_2_boots','SMALL','chi_counts1_markov','chi_counts2_markov','markov_interval'],
    #             #type_converters = converters.blitz,
    #             compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler] ,
    #             support_code=my_support_code )

    code2 = """
    // weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int markov_chain;
     int anglenum;
     for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      int mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      printf("mynumangles: %i ", mynumangles);
      #pragma omp parallel for private(markov_chain, anglenum, angle1_bin, angle2_bin)
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
          if(mybootstrap == bootstrap_sets - 1) {
            //printf("bin12 %i \\n",  nbins* (*(bins1  +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum)) + *(bins2  +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum))  ;
            }
             // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8)  --- but with chi already dereferenced :  the bin for each dihedral
             // python: self.chi_counts_markov=zeros((self.nchi, bootstrap_sets, self.markov_samples, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
             angle1_bin = *(bins1  +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
             angle2_bin = *(bins2  +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
             
             *(count_matrix_markov  +  mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 ;  //* weight;

             
          }
        }
       }
      """
     # //weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights


    if(VERBOSE >= 2): print "about to populate count_matrix_markov"
    #weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix_markov','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset'],
    #             #type_converters = converters.blitz,
    #             compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler])
  
    ## Why did I comment out the section below? Make sure it is correct to leave it out
    #if (no_boot_weights != False ):
    #       count_matrix_markov /= (bootstrap_choose*max_num_angles * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be bootstrap_choose*min(numangles)

    #Counts_ij markov here
    Counts_ij = zeros((bootstrap_sets, nbins, nbins),float64)
    for mybootstrap in range(bootstrap_sets):
           #print "Counts_ij"
           #print average(count_matrix_markov[mybootstrap,:,:],axis=0)
           #print "shape of Counts_ij before reshape"
           #print shape(average(count_matrix_markov[mybootstrap,:,:],axis=1))
           Counts_ij[mybootstrap,:,:] = (average(count_matrix_markov[mybootstrap,:,:],axis=0)).reshape(nbins,nbins)

    print "count matrix markov first pass:"
    print count_matrix_markov
    print "count matrix markov shape: "
    print shape(count_matrix_markov)
    print "chi count markov shape: "
    print shape(chi_counts1_markov)
    print "sum of count_matrix_markov for bootstrap zero and markov chain last one:"
    print sum(count_matrix_markov[0,-1,:])
    #just check markov chain 0
    ninj_flat = zeros((bootstrap_sets, 0 + 1,nbins*nbins),float64)
    
    for markov_chain in range(markov_samples):
       #print "chi counts1 markov bootstrap zero markov chain: "+str(markov_chain)+" : "
       #print chi_counts1_markov[0,markov_chain]
       for bootstrap in range(bootstrap_sets):
           my_flat = outer(chi_counts1_markov[bootstrap,markov_chain] + 0.0 ,chi_counts2_markov[bootstrap,markov_chain] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
           if(VERBOSE >=1):
                  assert(all(my_flat >= 0))
           my_flat = resize(my_flat,(1 ,(my_flat.shape)[0]))
           ninj_flat[bootstrap,:,:] = my_flat[:,:]
           #now without the Bayes prior added into the marginal distribution
           #my_flat_Bayes = outer(chi_counts1_markov[bootstrap] + ni_prior,chi_counts2_markov[bootstrap] + ni_prior).flatten() 
           #my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
           #ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
           #nbins_cor = int(nbins * FEWER_COR_BTW_BINS)

           ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just use outer product to give zero MI
           if(all(count_matrix_markov[:,:,:] == 0)) and (sum(chi_counts1_markov[bootstrap,markov_chain]) > 0) and (sum(chi_counts2_markov[bootstrap,markov_chain]) > 0):
              count_matrix_markov[bootstrap,markov_chain,:] =  (outer(chi_counts1_markov[bootstrap,markov_chain] ,chi_counts2_markov[bootstrap,markov_chain] ).flatten()  ) / (numangles_bootstrap[0] * 1.0)

       if(VERBOSE >=1):
              assert(all(ninj_flat >= 0))
       Pij, PiPj = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
       Pij[1:,1:]  = (count_matrix_markov[0,markov_chain,:]).reshape((nbins,nbins)) 
       PiPj[1:,1:] = (ninj_flat[0,0,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0) #this markov chain, bootstrap 0

       if(VERBOSE >= 2):
           print "First Pass: markov chain: "+str(markov_chain)
           print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
           print "Marginal Pij, summed over j:\n"
           print sum(Pij[1:,1:],axis=1)
           print "Marginal PiPj, summed over j:\n"
           print sum(PiPj[1:,1:],axis=1)   
           print "Marginal Pij, summed over i:\n"
           print sum(Pij[1:,1:],axis=0)
           print "Marginal PiPj, summed over i:\n"
           print sum(PiPj[1:,1:],axis=0)
    ### end redundant sanity checks


       #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
       if(VERBOSE >=1):
           assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
           assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
           assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

           
    
    #print "done with count matrix setup for multinomial\n"
    for bootstrap in range(bootstrap_sets):
           my_flat = outer(chi_counts1_markov[bootstrap] + ni_prior,chi_counts2_markov[bootstrap] + ni_prior).flatten() 
           my_flat = resize(my_flat,(0 + 1,(my_flat.shape)[0]))
           #ninj_flat_Bayes[bootstrap,:,:] = my_flat[:,:]

    if(numangles_bootstrap[0] > 0): #small sample stuff turned off for now cause it's broken
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):
        #ent1_boots = sum((chi_counts1_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_markov_vector + SMALL) - (1 - 2*(chi_counts1_markov_vector % 2)) / (chi_counts1_markov_vector + 1.0)),axis=2)

        #ent2_boots = sum((chi_counts2_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_markov_vector + SMALL) - (1 - 2*(chi_counts2_markov_vector % 2)) / (chi_counts2_markov_vector + 1.0)),axis=2)
        print "ent1_boots:         : "+str(ent1_boots)
        print "ent1_boots from init: "+str(ent1_markov_boots)
        #print "chi_counts1_markov_vector bootstrap 0: "+str(chi_counts1_markov_vector[0])
        #print "ent1 old  : "+str(sum((chi_counts1_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_markov_vector + SMALL) - (1.0 - 2*(int64(chi_counts1_markov_vector) % 2)) / (chi_counts1_markov_vector + 1.0)),axis=2))
        print "ent2_boots: "+str(ent2_boots)
        print "ent2_boots from init: "+str(ent2_markov_boots)
        #print "chi_counts2_markov_vector bootstrap 0: "+str(chi_counts2_markov_vector[0])
        #print "ent2 old  : "+str(sum((chi_counts2_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_markov_vector + SMALL) - (1.0 - 2*(int64(chi_counts2_markov_vector) % 2)) / (chi_counts2_markov_vector + 1.0)),axis=2))
        #print "ent1 terms: "
        #print (chi_counts1_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_markov_vector + SMALL) - (1.0 - 2*(chi_counts1_markov_vector % 2)) / (chi_counts1_markov_vector + 1.0))
        #assert(all(abs(ent1_markov_boots - sum((chi_counts1_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_markov_vector + SMALL) - (1.0 - 2*(chi_counts1_markov_vector % 2)) / (chi_counts1_markov_vector + 1.0)),axis=2) < 0.01)))

        #assert(all(abs(ent2_markov_boots - sum((chi_counts2_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_markov_vector + SMALL) - (1.0 - 2*(chi_counts2_markov_vector % 2)) / (chi_counts2_markov_vector + 1.0)),axis=2) < 0.01)))

        #assert(all(abs(ent_1_2_boots - sum((count_matrix_markov * 1.0 /numangles_bootstrap_matrix_markov)  \
        #                   * ( log(numangles_bootstrap_matrix_markov) - \
        #                       special.psi(count_matrix_markov + SMALL)  \
        #                       - (1 - 2*(count_matrix_markov % 2)) / (count_matrix_markov + 1.0) \
        #                       ),axis=2) < 0.01)))
        
        
        
        # MI = H(1)+H(2)-H(1,2)
        # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
        #print "Numangles Bootstrap Matrix:\n"+str(numangles_bootstrap_matrix_markov)
        
        #mutinf_thisdof = ent1_boots + ent2_boots  \
        #             - (sum((count_matrix_markov * 1.0 /numangles_bootstrap_matrix_markov)  \
        #                   * ( log(numangles_bootstrap_matrix_markov) - \
        #                       special.psi(count_matrix_markov + SMALL)  \
        #                       - (1 - 2*(count_matrix_markov % 2)) / (count_matrix_markov + 1.0) \
        #                       ),axis=2))
        print "ent1_markov_boots:  "+str(ent1_markov_boots)
        print "ent2_markov_boots:  "+str(ent2_markov_boots)
        print "ent12_markov_boots: "+str(ent_1_2_boots)
        
        #if(sum(markov_interval) > bootstrap_sets): #if using markov_interval to subsample data
        #       mutinf_thisdof = ent1_boots + ent2_boots - ent_1_2_boots
        #else:
        mutinf_thisdof = ent1_markov_boots + ent2_markov_boots - ent_1_2_boots
    
    E_I = average(mutinf_thisdof)
    Var_I = 0.0  #dummy value
    E_I3 = 0.0   #dummy value
    E_I4 = 0.0  #dummy value
    Var_I_runs = 0.0  #dummy value
    var_mi_thisdof = 0.0 #dummy value
    #average over 
    #mutinf_thisdof = reshape(mutinf_thisdof, (bootstrap_sets * markov_samples)) #total number of samples is bootstrap_sets * markov_samples, make this the same as mutinf_multinomial
    #del chi_counts1_markov_vector, chi_counts2_markov_vector
    
    for mybootstrap in range(bootstrap_sets):
           #multiply by two since symmetrized transition counts are equivalent to reading trajectory forward then backward 
           num_effective_snapshots = int((1.0 * numangles_bootstrap[0]) / (1.0 + 1.0 * max(bins1_slowest_lagtime[mybootstrap],bins2_slowest_lagtime[mybootstrap]))) + 1 # take the longer of the two lagtimes as the effective lagtime, add one to ensure no divide by zero failure
           mutinf_per_order_expected_variance =  ((nbins*nbins - 1) ** (2)) * 1.0/(2.0 * num_effective_snapshots * num_effective_snapshots)
           mutinf_per_order_expected          =  (1 - 2*2) * ( (int(nbins*nbins - 1) ** 2) / (2.0*num_effective_snapshots))
           print "num effective snapshots: "+str(num_effective_snapshots)
           print "ind mutinf average: "+str(average(mutinf_thisdof[mybootstrap],axis=0))+" ind mutinf expected from num effective snapshots: "+str(mutinf_per_order_expected)
           print "ind mutinf variance: "+str(var(mutinf_thisdof))+" ind mutinf variance expected from num effective snapshots: "+str(mutinf_per_order_expected_variance)
           if (mybootstrap == 0 and OUTPUT_INDEPENDENT_MUTINF_VALUES != 0): #global var set by options.output_independent
                  OUTPUT_INDEPENDENT_MUTINF_VALUES = 1 # in case it's something else
                  myfile = open("markov_independent_mutinf_"+str(OUTPUT_INDEPENDENT_MUTINF_VALUES)+"_bootstrap_"+str(mybootstrap)+".txt",'w')
                  for i in range((shape(mutinf_thisdof))[0]):
                         myfile.write(str(mutinf_thisdof[i])+"\n")
                  myfile.close()
    print "shape of mutinf markov thisdof: "+str(shape(mutinf_thisdof))
    return E_I, Var_I , E_I3, E_I4, Var_I_runs, mutinf_thisdof, var_mi_thisdof, Counts_ij



#########################################################################################################################################
##### Mutual Information From Joint Histograms, with calculation of MI under Null Hypothesis ############################################
#########################################################################################################################################

   

count_matrix = None
count_matrix_star = None
E_I_multinomial = None
Var_I_multinomial = None
E_I3_multinomial = None
E_I4_multinomial = None
Var_I_runs_multinomial = None
mutinf_multinomial = None
var_mutinf_multinomial = None
mutinf_multinomial_sequential = None
var_mutinf_multinomial_sequential = None
E_I_uniform = None
numangles_bootstrap_matrix = None
numangles_bootstrap_vector = None
min_angles_boot_pair_runs_matrix = None
min_angles_boot_pair_runs_vector = None
chi_counts1_matrix = None
chi_counts2_matrix = None
last_good_numangles = None
#ninj_flat = None
#ninj_flat_Bayes = None
#here the _star terms are for alternative ensemble weighting, such as when termsl like p* ln p are desired
#NOTE: it is assumed here that both the regular arrays and the star arrays have the same number of bootstraps, it is the job of the calling routine to ensure this
def calc_mutinf_corrected(chi_counts1, chi_counts2, bins1, bins2, chi_counts_sequential1, chi_counts_sequential2, bins1_sequential, bins2_sequential, num_sims, nbins, numangles_bootstrap, numangles, calc_variance=False,bootstrap_choose=0,permutations=0,which_runs=None,pair_runs=None, calc_mutinf_between_sims="yes", markov_samples = 0, chi_counts1_markov=None, chi_counts2_markov=None, ent1_markov_boots=None, ent2_markov_boots=None, bins1_markov=None, bins2_markov=None, file_prefix=None, plot_2d_histograms=False, adaptive_partitioning = 0, lagtime_interval=None, bins1_slowest_timescale = None, bins2_slowest_timescale = None, bins1_slowest_lagtime = None, bins2_slowest_lagtime = None, boot_weights = None, weights = None, chi_counts1_star=None, chi_counts2_star=None, bins1_star=None, bins2_star=None,chi_counts_sequential1_star=None, chi_counts_sequential2_star=None, bins1_sequential_star=None, bins2_sequential_star=None, numangles_star=None, numangles_bootstrap_star=None, bootstrap_choose_star=None ):
    global count_matrix, count_matrix_sequential, ninj_flat_Bayes, ninj_flat_Bayes_sequential # , ninj_flat
    global nbins_cor, min_angles_boot_pair_runs, numangles_bootstrap_matrix, numangles_boot_pair_runs_matrix, numangles_bootstrap_vector
    global min_angles_boot_pair_runs_matrix, min_angles_boot_pair_runs_vector
    global count_matrix_star, count_matrix_sequential_star, ninj_flat_Bayes_star, ninj_flat_Bayes_sequential_star, numangles_bootstrap_matrix_star
    global E_I_multinomial, Var_I_multinomial, E_I3_multinomial, E_I4_multinomial, Var_I_runs_multinomial 
    global mutinf_multinomial, var_mutinf_multinomial, mutinf_multinomial_sequential, var_mutinf_multinomial_sequential
    global E_I_uniform
    
    #print "bins1 lagtime: "+str(bins1_slowest_lagtime)
    #print "bins2 lagtime: "+str(bins2_slowest_lagtime)

    # allocate the matrices the first time only for speed
    if(bootstrap_choose == 0):
        bootstrap_choose = num_sims
    bootstrap_sets = len(which_runs)
    if(VERBOSE >=1):
           assert(all(chi_counts1 >= 0))
           assert(all(chi_counts2 >= 0))
    permutations_sequential = permutations
    #permutations_sequential = 0 #don't use permutations for mutinf between sims
    max_num_angles = int(min(numangles)) # int(max(numangles))
    if(numangles_star != None): 
        max_num_angles_star = int(min(numangles_star)) # int(max(numangles_star))
    
    num_pair_runs = pair_runs.shape[1]
    nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    if(VERBOSE >=2):
           print "nbins_cor: "+str(nbins_cor)
           print pair_runs
           print "bootstrap_sets: "+str(bootstrap_sets)+"num_pair_runs: "+str(num_pair_runs)+"\n"

    #print numangles_bootstrap
    #print bins1.shape
    
    #assert(bootstrap_sets == pair_runs.shape[0] == chi_counts1.shape[0] == chi_counts2.shape[0] == bins1.shape[0] == bins2.shape[0])
    #ninj_prior = 1.0 / float64(nbins*nbins) #Perks' Dirichlet prior
    #ni_prior = nbins * ninj_prior           #Perks' Dirichlet prior
    ninj_prior = 1.0                        #Uniform prior
    ni_prior = nbins * ninj_prior           #Uniform prior
    pvalue = zeros((bootstrap_sets),float64)

    markov_interval = zeros((bootstrap_sets),int16)
    # allocate the matrices the first time only for speed
    if(markov_samples > 0 and bins1_slowest_lagtime != None and bins2_slowest_lagtime != None ):
           for bootstrap in range(bootstrap_sets):
                  max_lagtime = max(bins1_slowest_lagtime[bootstrap], bins2_slowest_lagtime[bootstrap])
                  markov_interval[bootstrap] = int(max_num_angles / max_lagtime) #interval for final mutual information calc, based on max lagtime
    else:
           markov_interval[:] = 1

    markov_interval[:] = 1
    #must be careful to zero discrete histograms that we'll put data in from weaves
    if VERBOSE >= 2:
           print "markov interval: "
           print markov_interval
           print "chi counts 1 before multinomial:"
           print chi_counts1
           print "chi counts 2 before multinomial:"
           print chi_counts2
    #initialize if permutations > 0
    if(permutations > 0):
           mutinf_multinomial = zeros((bootstrap_sets,1),float64)
           mutinf_multinomial_sequential = zeros((bootstrap_sets,1),float64)
           var_mutinf_multinomial_sequential = zeros((bootstrap_sets,1), float64)
    #only do multinomial if not doing permutations
    Counts_ij_ind = zeros((bootstrap_sets, nbins, nbins),float64)
    if(E_I_multinomial is None and permutations == 0 and markov_samples == 0 ):
        E_I_multinomial, Var_I_multinomial, E_I3_multinomial, E_I4_multinomial, Var_I_runs_multinomial, \
                         mutinf_multinomial, var_mutinf_multinomial = \
                         calc_mutinf_multinomial_constrained(nbins,chi_counts1,chi_counts2,adaptive_partitioning )
    if(permutations == 0 and markov_samples > 0 ): #run mutinf for independent markov samples for every dihedral, mutinf_multinomial now refers to markov-based independent mutinf
        E_I_multinomial, Var_I_multinomial, E_I3_multinomial, E_I4_multinomial, Var_I_runs_multinomial, \
                         mutinf_multinomial, var_mutinf_multinomial, Counts_ij_ind = \
                         calc_mutinf_markov_independent(nbins,chi_counts1_markov, chi_counts2_markov, ent1_markov_boots, ent2_markov_boots, bins1_markov,bins2_markov, bootstrap_sets, bootstrap_choose, markov_samples, max_num_angles, numangles_bootstrap, bins1_slowest_lagtime, bins2_slowest_lagtime)
    
    #NOTE: If markov_samples == 0, shape of mutinf_multinomial is (bootstrap_sets, 1). If markov_samples > 0, shape of mutinf_multinomial is (bootstrap_sets, markov_samples). 

    if count_matrix is None:    

       count_matrix = zeros((bootstrap_sets, permutations + 1 , nbins*nbins), float64)
       count_matrix_sequential = zeros((bootstrap_sets, num_pair_runs,permutations_sequential + 1 , nbins_cor*nbins_cor), float64)
       min_angles_boot_pair_runs_matrix = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor*nbins_cor),int32)
       min_angles_boot_pair_runs_vector = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor),int32)
       for bootstrap in range(bootstrap_sets):
            for which_pair in range(num_pair_runs):
                if VERBOSE >= 2:
                       print "run1 "+str(pair_runs[bootstrap,which_pair,0])+" run2 "+str(pair_runs[bootstrap,which_pair,1])
                       print "numangles shape:" +str(numangles.shape)
                #my_flat = outer(chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]],chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]]).flatten()
                #my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0])) #replicate original data over n permutations
                min_angles_boot_pair_runs_matrix[bootstrap,which_pair,:,:] = resize(array(min(numangles[pair_runs[bootstrap,which_pair,0]],numangles[pair_runs[bootstrap,which_pair,1]]),int32),(permutations_sequential + 1, nbins_cor*nbins_cor)) #replicate original data over permutations and nbins_cor*nbins_cor
                min_angles_boot_pair_runs_vector[bootstrap,which_pair,:,:] = resize(array(min(numangles[pair_runs[bootstrap,which_pair,0]],numangles[pair_runs[bootstrap,which_pair,1]]),int32),(nbins_cor)) #replicate original data over nbins_cor, no permutations for marginal dist.            
                #ninj_flat_Bayes_sequential[bootstrap,which_pair,:,:] = my_flat[:,:]

       
           
       
    if(numangles_bootstrap_matrix is None):
        if(VERBOSE >= 2):
               print "numangles bootstrap: "+str(numangles_bootstrap)
        numangles_bootstrap_matrix = zeros((bootstrap_sets,permutations+1,nbins*nbins),float64)
        for bootstrap in range(bootstrap_sets):
            for permut in range(permutations+1):
                numangles_bootstrap_matrix[bootstrap,permut,:]=numangles_bootstrap[bootstrap]
        numangles_bootstrap_vector = numangles_bootstrap_matrix[:,:,:nbins]
        #print "Numangles Bootstrap Vector:\n"+str(numangles_bootstrap_vector)
    
    #if(chi_counts2_matrix is None):
    chi_counts1_matrix = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions 
    chi_counts2_matrix = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions
    

    
    count_matrix[:,:,:] = 0
    count_matrix_sequential[:,:,:,:] =0
    # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
    chi_counts1_vector = reshape(chi_counts1.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    chi_counts2_vector = reshape(chi_counts2.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    chi_counts1_vector = repeat(chi_counts1_vector, permutations + 1, axis=1)     #but we need to repeat along permutations axis
    chi_counts2_vector = repeat(chi_counts2_vector, permutations + 1, axis=1)     #but we need to repeat along permutations axis

    ent_1_boots = zeros((bootstrap_sets,permutations + 1),float64)
    ent_2_boots = zeros((bootstrap_sets,permutations + 1),float64)
    ent_1_2_boots = zeros((bootstrap_sets,permutations + 1),float64)
    
    
    for bootstrap in range(bootstrap_sets): #copy here is critical to not change original arrays
        chi_counts2_matrix[bootstrap,0,:] = repeat(chi_counts2[bootstrap].copy(),nbins,axis=-1) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
        #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
        #handling the slower-varying index will be a little bit more tricky
        chi_counts1_matrix[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1[bootstrap].copy(), nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
    
    no_boot_weights = False
    if (boot_weights is None):
           boot_weights = ones((bootstrap_sets, max_num_angles * bootstrap_choose), float64)
           no_boot_weights = True

    code = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int bin1;
     int bin2;
     int mybootstrap, permut;
     long anglenum, mynumangles, counts1, counts2, counts12 ; 
     double mysign1, mysign2, mysign12, counts1d, counts2d, counts12d, dig1, dig2, dig12;
     
     #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,angle1_bin,angle2_bin,weight,bin1, bin2, mysign1, mysign2, mysign12, counts1, counts1d, counts2, counts2d, counts12, counts12d, dig1, dig2, dig12 )
     for( mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);
      for (permut=0; permut < permutations + 1; permut++) {          
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          //if(mybootstrap == bootstrap_sets - 1) {
          //  //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
          //  }
           if(anglenum == mynumangles - 1) {
             printf(""); //just to make sure count matrix values are written to memory before the next loop
           }
           //if(anglenum % markov_interval[mybootstrap] == 0) { 
              angle1_bin = *(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum);
              angle2_bin = *(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight;
           //} 
          }
           // do singlet entropies here
           for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1 = long(*(chi_counts1_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1);
                
                
                counts1d = 1.0 * counts1;
                *(ent_1_boots + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts1 + 1.0)))); 
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2 = long(*(chi_counts2_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin1);
              }
             }
 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts2d / mynumangles)*(log(mynumangles) - dig2 - ((double)mysign2 / ((double)(counts2 + 1.0)))); 
              }



            // do doublet entropy here
            for(bin2=0; bin2 < nbins; bin2++)
             {
         
             // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
             counts12 = long(*(count_matrix  + (long)( mybootstrap*(permutations + 1)*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 )));
 
              if(counts12 > 0)
              {
                mysign12 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                dig12 = xDiGamma_Function(counts12);
                
                
                counts12d = 1.0 * counts12;
                *(ent_1_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts12d / mynumangles)*(log(mynumangles) - dig12 - ((double)mysign1 / ((double)(counts12 + 1.0)))); 
              }
            }
         }
       
    
        }
       }
      """

    code_no_grassberger = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int bin1;
     int bin2;
     int mybootstrap, permut;
     long anglenum, mynumangles, counts1, counts2, counts12 ; 
     double mysign1, mysign2, mysign12, counts1d, counts2d, counts12d, dig1, dig2, dig12;
     
     #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,angle1_bin,angle2_bin,weight,bin1, bin2, mysign1, mysign2, mysign12, counts1, counts1d, counts2, counts2d, counts12, counts12d, dig1, dig2, dig12 )
     for( mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);
      for (permut=0; permut < permutations + 1; permut++) {          
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          //if(mybootstrap == bootstrap_sets - 1) {
          //  //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
          //  }
           if(anglenum == mynumangles - 1) {
             printf(""); //just to make sure count matrix values are written to disk before the next loop
           }
           //if(anglenum % markov_interval[mybootstrap] == 0) { 
              angle1_bin = *(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum);
              angle2_bin = *(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight;
           //} 
          }
           // do singlet entropies here
           for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1 = long(*(chi_counts1_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1);
                
                
                counts1d = 1.0 * counts1;
                *(ent_1_boots + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALLER));
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2 = long(*(chi_counts2_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin1);
              }
             }
 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts2d / mynumangles)*(log((double)counts2d / mynumangles + SMALLER)); 
              }



            // do doublet entropy here
            for(bin2=0; bin2 < nbins; bin2++)
             {
         
             // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
             counts12 = long(*(count_matrix  + (long)( mybootstrap*(permutations + 1)*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 )));
 
              if(counts12 > 0)
              {
                mysign12 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                dig12 = xDiGamma_Function(counts12);
                
                
                counts12d = 1.0 * counts12;
                *(ent_1_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts12d / mynumangles)*(log((double)counts12d / mynumangles + SMALLER)); 
              }
            }
         }
       
    
        }
       }
      """

 

    if(VERBOSE >= 2): 
           print "about to populate count_matrix"
           print "chi counts 1"
           print chi_counts1[bootstrap]
           print "chi counts 2"
           print chi_counts2[bootstrap]

    if(NO_GRASSBERGER == False):
           weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots','markov_interval','SMALL'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    else:
           weave.inline(code_no_grassberger, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots','markov_interval','SMALLER'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    if (no_boot_weights != False ):
           count_matrix /= (bootstrap_choose*max_num_angles * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be bootstrap_choose*min(numangles)

    if(VERBOSE >=2):
           print "count matrix first pass:"
           print count_matrix
           print "chi counts 1"
           print chi_counts1[bootstrap]
           print "chi counts 2"
           print chi_counts2[bootstrap]
    

    ### Redundant Sanity checks
    ninj_flat = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
        my_flat = outer(chi_counts1[bootstrap] + 0.0 ,chi_counts2[bootstrap] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
        if(VERBOSE >=1):
               assert(all(my_flat >= 0))
        my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0]))
        ninj_flat[bootstrap,:,:] = my_flat[:,:]
        #now without the Bayes prior added into the marginal distribution
        my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
        my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
        ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
        nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix[:,:,:] == 0)) and (sum(chi_counts1) > 0) and (sum(chi_counts2) > 0):
           count_matrix[:,:] =  (outer(chi_counts1[bootstrap] ,chi_counts2[bootstrap] ).flatten()  ) / (numangles_bootstrap[0] * 1.0)
    
    if(VERBOSE >=1):
           assert(all(ninj_flat >= 0))
    Pij, PiPj = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
    permutation = 0
    Pij[1:,1:]  = (count_matrix[0,permutation,:]).reshape((nbins,nbins)) 
    PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    
    if(VERBOSE >= 1 and permutation == 0):
        print "First Pass:"
        print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
        print "Marginal Pij, summed over j:\n"
        print sum(Pij[1:,1:],axis=1)
        print "Marginal PiPj, summed over j:\n"
        print sum(PiPj[1:,1:],axis=1)   
        print "Marginal Pij, summed over i:\n"
        print sum(Pij[1:,1:],axis=0)
        print "Marginal PiPj, summed over i:\n"
        print sum(PiPj[1:,1:],axis=0)
    ### end redundant sanity checks
        
        
    #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
    if(VERBOSE >=1):
           assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
           assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
           assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    if(permutations > 0):
           permutation = 1
           Pij[1:,1:]  = (count_matrix[0,permutation,:]).reshape((nbins,nbins)) 
           PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    

           if(VERBOSE >=1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))



    #######################################################################################################
    ## Now for the star terms, if they are present, otherwise set the same as the normal ones
    ########################################################################################################
    if(chi_counts1_star!=None):
         if count_matrix_star is None:    

             count_matrix_star= zeros((bootstrap_sets, permutations + 1 , nbins*nbins), float64)
             #count_matrix_sequential_star = zeros((bootstrap_sets, num_pair_runs,permutations_sequential + 1 , nbins_cor*nbins_cor), int32)
         
         if(numangles_bootstrap_matrix_star is None):
             numangles_bootstrap_matrix_star = zeros((bootstrap_sets,permutations+1,nbins*nbins),float64)
             for bootstrap in range(bootstrap_sets):
                 for permut in range(permutations+1):
                     numangles_bootstrap_matrix_star[bootstrap,permut,:]=numangles_bootstrap_star[bootstrap]
         
         count_matrix_star[:,:,:] = 0
         numangles_bootstrap_vector_star = numangles_bootstrap_matrix[:,:,:nbins]
         #count_matrix_sequential_star[:,:,:,:] =0
         # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
         chi_counts1_vector_star = reshape(chi_counts1_star.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
         chi_counts2_vector_star = reshape(chi_counts2_star.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
         chi_counts1_vector_star = repeat(chi_counts1_vector_star, permutations + 1, axis=1)     #but we need to repeat along permutations axis
         chi_counts2_vector_star = repeat(chi_counts2_vector_star, permutations + 1, axis=1)     #but we need to repeat along permutations axis
         chi_counts1_matrix_star = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions 
         chi_counts2_matrix_star = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions
    
         for bootstrap in range(bootstrap_sets): #copy here is critical to not change original arrays
             chi_counts2_matrix_star[bootstrap,0,:] = repeat(chi_counts2_star[bootstrap].copy(),nbins,axis=-1) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
             #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
             #handling the slower-varying index will be a little bit more tricky
             chi_counts1_matrix_star[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1_star[bootstrap].copy(), nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)

        
         code_star = """
         // weave6_star
         // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose_star * max_num_angles_star
          //#include <math.h>
          int mybootstrap = 0;
          int mynumangles = 0;
          int permut = 0;
          int anglenum = 0;
          
          for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
           mynumangles = 0;
           for (permut=0; permut < permutations + 1; permut++) {
               mynumangles = *(numangles_bootstrap_star + mybootstrap);
               // #pragma omp parallel for private(anglenum)
               for (anglenum=offset; anglenum< mynumangles; anglenum++) {
               //if(mybootstrap == bootstrap_sets - 1) {
               //  printf("bin12 %i \\n",(*(bins1_star  +  mybootstrap*bootstrap_choose_star*max_num_angles_star  +  anglenum))*nbins +   (*(bins2_star + permut*bootstrap_sets*bootstrap_choose_star*max_num_angles_star + mybootstrap*bootstrap_choose_star*max_num_angles_star  +  anglenum)));
               //  }
                   // #pragma omp atomic
                   *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  (*(bins1_star  +  mybootstrap*bootstrap_choose_star*max_num_angles_star  +  anglenum))*nbins +   (*(bins2_star + permut*bootstrap_sets*bootstrap_choose_star*max_num_angles_star + mybootstrap*bootstrap_choose_star*max_num_angles_star  +  anglenum - offset))) += 1;
             
               }
             }
             }
         """
         if(VERBOSE >= 2): print "about to populate count_matrix_star. max_num_angles_star: "+str(max_num_angles_star)+" bootstrap_choose_star:"+str(bootstrap_choose_star)+" numangles_bootstrap_star: "+str(numangles_bootstrap_star)+" bootstrap sets: "+str(bootstrap_sets)
         weave.inline(code_star, ['num_sims', 'numangles_bootstrap_star', 'nbins', 'bins1_star', 'bins2_star', 'count_matrix_star','bootstrap_sets','permutations','max_num_angles_star','bootstrap_choose_star'],
                      #type_converters = converters.blitz,
                       extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler], 
                      compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
         
         
         
         ninj_flat_star = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
         #ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
         for bootstrap in range(bootstrap_sets):
             my_flat = outer(chi_counts1_star[bootstrap] + 0.0 ,chi_counts2_star[bootstrap] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
             if(VERBOSE >=1):
                    assert(all(my_flat >= 0))
             my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0]))
             ninj_flat_star[bootstrap,:,:] = my_flat[:,:]
             #now without the Bayes prior added into the marginal distribution
             #my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
             #my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
             #ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
             #nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
         if(VERBOSE >=1):
                assert(all(ninj_flat_star >= 0))
    
         #print "chi counts 1 after bin filling:"
         #print chi_counts1
    
         # print out the 2D population histograms to disk
         # erase data matrix files after creation to save space; put files in sub-directories so we don't end up with too many files in a directory for linux to handle
         Pij_star, PiPj_star = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
         mypermutation = 0
         Pij_star[1:,1:]  = (count_matrix_star[0,mypermutation,:]).reshape((nbins,nbins))
         PiPj_star[1:,1:] = (ninj_flat_star[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap_star[0] * 1.0)
         #debug
         if(VERBOSE >= 2):
             print "Second Pass:"
             print "Sum Pij_star: "+str(sum(Pij_star[1:,1:]))+" Sum PiPj_star: "+str(sum(PiPj_star[1:,1:]))
             print "Marginal Pij_star, summed over j:\n"
             print sum(Pij_star[1:,1:],axis=1)
             print "Marginal PiPj_star, summed over j:\n"
             print sum(PiPj_star[1:,1:],axis=1)   
             print "Marginal Pij_star, summed over i:\n"
             print sum(Pij_star[1:,1:],axis=0)
             print "Marginal PiPj_star, summed over i:\n"
             print sum(PiPj_star[1:,1:],axis=0)


    if(chi_counts1_star==None):
            count_matrix_star = count_matrix
            chi_counts1_vector_star = chi_counts1_vector
            chi_counts1_matrix_star = chi_counts1_matrix
            chi_counts2_vector_star = chi_counts2_vector
            chi_counts2_matrix_star = chi_counts2_matrix
            numangles_bootstrap_vector_star = numangles_bootstrap_vector
            numangles_bootstrap_matrix_star = numangles_bootstrap_matrix
        
    
    #######################################################################################################
    
    dKLtot_dKL1_dKL2 = zeros((bootstrap_sets), float64)
    
    #if(VERBOSE >= 2):
    #    print "nonzero bins"
    #    print nonzero_bins * 1.0
    #    print (count_matrix[bootstrap,0])[nonzero_bins]
    #    print (chi_counts1_matrix[bootstrap,0])[nonzero_bins]
    #    print (chi_counts2_matrix[bootstrap,0])[nonzero_bins]
        

    #######################################################################################################
    
    if(VERBOSE >=1):
           assert(all(chi_counts1 >= 0))
           assert(all(chi_counts2 >= 0))
    
    ##### to work around bug in _star code, in the meantime fill count_matrix again
    if(chi_counts1_star!=None):
           count_matrix[:,:,:] = 0
           #weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose'],
           #      #type_converters = converters.blitz,
           #      compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
           if(NO_GRASSBERGER == False):
              weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
           else:
              weave.inline(code_no_grassberger, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    #print "count matrix:"
    #print count_matrix
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix[:,:,:] == 0)) and (sum(chi_counts1) > 0) and (sum(chi_counts2) > 0):
           count_matrix[:,:] =  (outer(chi_counts1[bootstrap] ,chi_counts2[bootstrap] ).flatten()  ) / (numangles_bootstrap[0] * 1.0)

    ##################################################################

    ninj_flat = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
        my_flat = outer(chi_counts1[bootstrap] + 0.0 ,chi_counts2[bootstrap] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
        if(VERBOSE >=1):
               assert(all(my_flat >= 0))
        my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0]))
        ninj_flat[bootstrap,:,:] = my_flat[:,:]
        #now without the Bayes prior added into the marginal distribution
        my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
        my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
        ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
        nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    if(VERBOSE >=1):
           assert(all(ninj_flat >= 0))
    
    #print "chi counts 1 after bin filling:"
    #print chi_counts1
    
    # print out the 2D population histograms to disk
    # erase data matrix files after creation to save space; put files in sub-directories so we don't end up with too many files in a directory for linux to handle
    Pij, PiPj = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
    mypermutation = 0
    Pij[1:,1:]  = (count_matrix[0,mypermutation,:]).reshape((nbins,nbins))
    PiPj[1:,1:] = (ninj_flat[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    #sanity checks
    if(VERBOSE >= 2):
        print "Third Pass:"
        print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
        print "Marginal Pij, summed over j:\n"
        print sum(Pij[1:,1:],axis=1)
        print "Marginal PiPj, summed over j:\n"
        print sum(PiPj[1:,1:],axis=1)   
        print "Marginal Pij, summed over i:\n"
        print sum(Pij[1:,1:],axis=0)
        print "Marginal PiPj, summed over i:\n"
        print sum(PiPj[1:,1:],axis=0)
    #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
    if(VERBOSE >= 1):
           assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
           assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
           assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    if permutations > 0:
           mypermutation = 1
           Pij[1:,1:]  = (count_matrix[0,mypermutation,:]).reshape((nbins,nbins))
           PiPj[1:,1:] = (ninj_flat[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
           #sanity checks
           if(VERBOSE >= 1):
                  print "Third Pass: permutation 1"
                  print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
                  print "Marginal Pij, summed over j:\n"
                  print sum(Pij[1:,1:],axis=1)
                  print "Marginal PiPj, summed over j:\n"
                  print sum(PiPj[1:,1:],axis=1)   
                  print "Marginal Pij, summed over i:\n"
                  print sum(Pij[1:,1:],axis=0)
                  print "Marginal PiPj, summed over i:\n"
                  print sum(PiPj[1:,1:],axis=0)
                  #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
           if(VERBOSE >= 1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

           mypermutation = -1
           Pij[1:,1:]  = (count_matrix[0,mypermutation,:]).reshape((nbins,nbins))
           PiPj[1:,1:] = (ninj_flat[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
           #sanity checks
           if(VERBOSE >= 1):
                  print "Third Pass: permutation -1"
                  print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
                  print "Marginal Pij, summed over j:\n"
                  print sum(Pij[1:,1:],axis=1)
                  print "Marginal PiPj, summed over j:\n"
                  print sum(PiPj[1:,1:],axis=1)   
                  print "Marginal Pij, summed over i:\n"
                  print sum(Pij[1:,1:],axis=0)
                  print "Marginal PiPj, summed over i:\n"
                  print sum(PiPj[1:,1:],axis=0)
                  #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
           if(VERBOSE >= 1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    
    #now sum over bootstraps for 
    Counts_ij = zeros((bootstrap_sets, nbins, nbins),float64)
    for mybootstrap in range(bootstrap_sets):
           mypermutation = 0
           Counts_ij[mybootstrap,:,:] = (count_matrix[mybootstrap,mypermutation,:]).reshape(nbins,nbins)
           #mypermutation = 1 #first independent dataset
           #Counts_ij_ind[mybootstrap,:,:] = (count_matrix[mybootstrap,mypermutation,:]).reshape(nbins,nbins)

    ### Old plot 2d histograms -- now a file with the twoD histogram counts is made if plot_2d_histograms == True  for use by another script
    #if plot_2d_histograms and file_prefix != None:
    #    file_prefix = file_prefix.replace(":", "_").replace(" ", "_")
    #    print file_prefix
    # 
    #    Pij[1:, 0] = PiPj[1:,0] = bins #i.e. bin cut points 0 to 360, nbins in length
    #    Pij[0, 1:] = PiPj[0,1:] = bins #i.e. bin cut points 0 to 360, nbins in length
    #
    #    #Pij[1:,1:]  = average(count_matrix[:,permutation,:], axis=0).reshape((nbins,nbins))
    #    #PiPj[1:,1:] = average(ninj_flat_Bayes[:,permutation,:], axis=0).reshape((nbins,nbins))
    #    Pij[1:,1:] /= sum(Pij[1:,1:])
    #    PiPj[1:,1:] /= sum(PiPj[1:,1:])
    #
    #    res1_str = "_".join(file_prefix.split("_")[:2])
    #    dirname = "plots_of_Pij_PiPj_nsims%d_nstructs%d_p%d_a%s/%s" % (num_sims, numangles_bootstrap[0]/len(which_runs[0]), 0, adaptive_partitioning, res1_str)
    #    utils.mkdir_cd(dirname)
    #    
    #    open("%s_Pij.dat"%file_prefix, "w").write(utils.arr2str2(Pij, precision=8))
    #    open("%s_PiPj.dat"%file_prefix, "w").write(utils.arr2str2(PiPj, precision=8))
    #    #open("%s_Pij_div_PiPj.dat"%file_prefix, "w").write(utils.arr2str2(Pij_div_PiPj, precision=8))
    #    utils.run("R --no-restore --no-save --no-readline %s_Pij.dat < ~/bin/dihedral_2dhist_plots.R" % (file_prefix))
    #    utils.run("R --no-restore --no-save --no-readline %s_PiPj.dat < ~/bin/dihedral_2dhist_plots.R" % (file_prefix))
    #    #utils.run("rm -f %s_*.dat" % file_prefix)
    #
    #    utils.cd("../..")
    


    ####
    #### ***  Important Note: don't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
    #### as we're using the same number of bins in each calculation *****
    ####

    #if(numangles_bootstrap[0] > 0 or nbins >= 6):
    if(True): #small sample stuff turned off for now because it's broken
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):  
        #ent1_boots = sum((chi_counts1_vector_star * 1.0 / numangles_bootstrap_vector_star) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_vector + SMALL) - (1 - 2*(chi_counts1_vector % 2)) / (chi_counts1_vector + 1.0)),axis=2) 
        
        #ent2_boots = sum((chi_counts2_vector_star * 1.0 / numangles_bootstrap_vector_star) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_vector + SMALL) - (1 - 2*(chi_counts2_vector % 2)) / (chi_counts2_vector + 1.0)),axis=2)
        
        #ent12_boots = sum((count_matrix_star * 1.0 /numangles_bootstrap_matrix_star)  \
        #                   * ( log(numangles_bootstrap_matrix) - \
        #                       special.psi(count_matrix + SMALL)  \
        #                       - (1 - 2*(count_matrix % 2)) / (count_matrix + 1.0) \
        #                       ),axis=2)
        # MI = H(1)+H(2)-H(1,2)
        # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
        #print "Numangles Bootstrap Matrix:\n"+str(numangles_bootstrap_matrix)
        
        mutinf_thisdof = ent_1_boots + ent_2_boots  \
                     - ent_1_2_boots

        ent12_boots_for_MI_norm = ent_1_2_boots[:,0] #value to return of MI_norm
    else:
        ent1_boots = sum((chi_counts1_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts1_vector,numangles_bootstrap_vector,permutations),axis=2)

        ent2_boots = sum((chi_counts2_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts2_vector,numangles_bootstrap_vector,permutations),axis=2)

        if(VERBOSE >= 2):
            print "shapes:"+str(ent1_boots)+" , "+str(ent2_boots)+" , "+str(sum((count_matrix + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix,numangles_bootstrap_matrix,permutations_multinomial),axis=2))
        
        mutinf_thisdof = ent1_boots + ent2_boots \
                         -sum((count_matrix + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix,numangles_bootstrap_matrix,permutations),axis=2)

    
    if (VERBOSE >=2): 
           print "Avg Descriptive Mutinf:    "+str(average(mutinf_thisdof[:,0]))
    if(permutations == 0):
           if(VERBOSE >= 2):
                  print "Avg Descriptive MI ind:    "+str(average(mutinf_multinomial,axis=1))
    else:
           if(VERBOSE >=2):
                  print "Avg Descriptive MI ind:    "+str(average(mutinf_thisdof[:,1:]))
                  print "Number of permutations:    "+str(mutinf_thisdof.shape[1] -1)
                  print "distribution of MI ind:    "+str((mutinf_thisdof[:,1:]))
    #Now, if permutations==0, filter according to Bayesian estimate of distribution
    # of mutual information, M. Hutter and M. Zaffalon 2004 (or 2005).
    # Here, we will discard those MI values with p(I | data < I*) > 0.05.
    # Alternatively, we could use the permutation method or a more advanced monte carlo simulation
    # over a Dirichlet distribution to empirically determine the distribution of mutual information of the uniform
    # distribution.  The greater variances of the MI in nonuniform distributions suggest this approach
    # rather than a statistical test against the null hypothesis that the MI is the same as that of the uniform distribution.
    # The uniform distribution or sampling from a Dirichlet would be appropriate since we're using adaptive partitioning.

    #First, compute  ln(nij*n/(ni*nj) = logU, as we will need it and its powers shortly.
    #Here, use Perks' prior nij'' = 1/(nbins*nbins)
    
    if (markov_samples > 0): #if using markov model to get distribution under null hypothesis of independence           
           for bootstrap in range(bootstrap_sets):
                mutinf_multinomial_this_bootstrap = mutinf_multinomial[bootstrap]  # since our markov samples are in axis 1 , and we aren't averaging over bootstraps since their transition matrices are different
                num_greater_than_obs_MI = sum(1.0 * (mutinf_multinomial_this_bootstrap > mutinf_thisdof[bootstrap,0]))

                pvalue_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_multinomial_this_bootstrap.shape[0])
                
                pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_multinomial)
                if(VERBOSE >=1):
                       print "Descriptive P(I=I_markov):"+str(pvalue_multinomial)
                       print "number of markov samples with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later
    elif (permutations == 0):  #then use Bayesian approach to approximate distribution of mutual information given data,prior
        count_matrix_wprior = count_matrix + ninj_prior
        count_matrix_wprior_star = count_matrix_star + ninj_prior #for alternative ensemble, weighting, as in cross terms like p* ln p.
        numangles_bootstrap_matrix_wprior = numangles_bootstrap_matrix + ninj_prior*nbins*nbins
        numangles_bootstrap_vector_wprior = numangles_bootstrap_vector + ninj_prior*nbins*nbins
        numangles_bootstrap_matrix_wprior_star = numangles_bootstrap_matrix_star + ninj_prior*nbins*nbins
        numangles_bootstrap_vector_wprior_star = numangles_bootstrap_vector_star + ninj_prior*nbins*nbins
        Uij = (numangles_bootstrap_matrix_wprior) * (count_matrix_wprior) / (ninj_flat_Bayes)
        logUij = log(Uij) # guaranteed to not have a zero denominator for non-zero prior (non-Haldane prior)

        Jij=zeros((bootstrap_sets, permutations + 1, nbins*nbins),float64)
        Jij = (count_matrix_wprior_star / (numangles_bootstrap_matrix_wprior)) * logUij

        #you will see alot of "[:,0]" following. This means to take the 0th permutation, in case we're permuting data
    
        J = (sum(Jij, axis=-1))[:,0] #sum over bins ij
        K = (sum((count_matrix_wprior_star / (numangles_bootstrap_matrix_wprior_star)) * logUij * logUij, axis=-1))[:,0] #sum over bins ij
        L = (sum((count_matrix_wprior_star / (numangles_bootstrap_matrix_wprior_star)) * logUij * logUij * logUij, axis=-1))[:,0] #sum over bins ij
    
        #we will need to allocate Ji and Jj for row and column sums over matrix elemenst Jij:

        Ji=zeros((bootstrap_sets, permutations + 1, nbins),float64)
        Jj=zeros((bootstrap_sets, permutations + 1, nbins),float64)
        chi_counts_bayes_flat1 = zeros((bootstrap_sets,permutations + 1, nbins*nbins),float64)
        chi_counts_bayes_flat2 = zeros((bootstrap_sets,permutations + 1, nbins*nbins),float64)
    
    
        
        #repeat(chi_counts2[bootstrap] + ni_prior,permutations +1,axis=0)
        for bootstrap in range(bootstrap_sets):
            chi_counts_matrix1 = reshape(resize(chi_counts1[bootstrap] + ni_prior, bootstrap_sets*(permutations+1)*nbins),(bootstrap_sets,permutations+1,nbins))
            chi_counts_matrix2 = reshape(resize(chi_counts2[bootstrap] + ni_prior, bootstrap_sets*(permutations+1)*nbins),(bootstrap_sets,permutations+1,nbins))
        
            if(VERBOSE >= 2):
                print "chi counts 1:" + str(chi_counts1[bootstrap])
                print "chi counts 2:" + str(chi_counts2[bootstrap])

            mycounts_mat = reshape(count_matrix[bootstrap],(nbins,nbins))
            if(VERBOSE >= 2):
                print "counts:\n" + str(mycounts_mat)
            if(VERBOSE >=1):
                   if(VERBOSE >= 2):
                          print sum(mycounts_mat,axis=1)
                          print sum(mycounts_mat,axis=0)
                          print chi_counts1[bootstrap]
                          print chi_counts2[bootstrap]
                   assert(all(abs(chi_counts1[bootstrap] - sum(mycounts_mat,axis=1)) < nbins))
                   assert(all(abs(chi_counts2[bootstrap] - sum(mycounts_mat,axis=0)) < nbins))

            #now we need to reshape the marginal counts into a flat "matrix" compatible with count_matrix
            # including counts from the prior
            if(VERBOSE >= 2):
                print "chi_counts2 shape:"+str(shape(chi_counts2))
            #print "chi_counts2[bootstrap]+ni_prior shape:"+str((chi_counts2[bootstrap] + ni_prior).shape)
            
            chi_counts_bayes_flat2[bootstrap,0,:] = repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
            #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
            #handling the slower-varying index will be a little bit more tricky
            chi_counts_bayes_flat1[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1[bootstrap] + ni_prior, nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
            #we will also need to calculate row and column sums Ji and Jj:
            Jij_2D_boot = reshape(Jij[bootstrap,0,:],(nbins,nbins))
            #Ji is the sum over j for row i, Jj is the sum over i for column j, j is fastest varying index
            #do Ji first
            Ji[bootstrap,0,:] = sum(Jij_2D_boot, axis=1)
            Jj[bootstrap,0,:] = sum(Jij_2D_boot, axis=0)


        #ok, now we calculated the desired quantities using the matrices we just set up
    
    
        numangles_bootstrap_wprior = numangles_bootstrap + ninj_prior*nbins*nbins
        
        M = (sum((1.0/(count_matrix_wprior + SMALL) - 1.0/chi_counts_bayes_flat1 -1.0/chi_counts_bayes_flat2 \
                  + 1.0/numangles_bootstrap_matrix_wprior) \
                 * count_matrix_wprior * logUij, axis=2))[:,0]

        Q = (1 - sum(count_matrix_wprior * count_matrix_wprior / ninj_flat_Bayes, axis=2))[:,0]

        #####DEBUGGING STATEMENTS
        #print "shapes"
        #print "Ji:   "+str(shape(Ji))
        #print "Jj:   "+str(shape(Jj))
        #print "chi counts matrix 1:   "+str(chi_counts_matrix1)
        #print "chi counts matrix 2:   "+str(chi_counts_matrix2)
        #print "numangles bootstrap wprior:   "+str(shape(numangles_bootstrap_wprior))

        P = (sum((numangles_bootstrap_vector_wprior) * Ji * Ji / chi_counts_matrix1, axis=2) \
             + sum(numangles_bootstrap_vector_wprior * Jj * Jj / chi_counts_matrix2, axis=2))[:,0]

        #####DEBUGGING STATEMENTS
        #print "ni prior:\n"+str(ni_prior)
        #print "numangles bootstrap wprior:\n"+str(numangles_bootstrap_wprior)
        #print "chi_counts_bayes_flat1:\n"+str(chi_counts_bayes_flat1)
        #print "chi_counts_bayes_flat2:\n"+str(chi_counts_bayes_flat2)

        #print intermediate values

        #print "J:"+str(J)
        #print "K:"+str(K)
        #print "L:"+str(L)
        #print "M:"+str(M)
        #print "P:"+str(P)
        #print "Q:"+str(Q)

        #Finally, we are ready to calculate moment approximations for p(I | Data)
        E_I_mat = ((count_matrix_wprior_star) / (numangles_bootstrap_matrix_wprior_star * 1.0)) \
                  * (  special.psi(count_matrix_wprior + 1.0) \
                     - special.psi(chi_counts_bayes_flat1 + 1.0) \
                     - special.psi(chi_counts_bayes_flat2 + 1.0) \
                     + special.psi(numangles_bootstrap_matrix_wprior + 1.0))
        #print "\n"
        if(VERBOSE >= 2):
            print "E_I_mat:\n"
            print E_I_mat
        E_I = (sum(E_I_mat,axis = 2))[:,0] # to get rid of permutations dimension 
        #print "Stdev of E_I over bootstraps:\n"
        #print stats.std(average(sum(E_I_mat,axis = 2), axis = 1))
        #print "estimated pop std of E_I over bootstraps, dividing by sqrt n"
        #print stats.std(average(sum(E_I_mat,axis = 2), axis = 1)) / sqrt(num_sims)
        #print "Stdev_mat:\n"
        #print sqrt(((K - J*J))/(numangles_bootstrap_wprior + 1) + (M + (nbins-1)*(nbins-1)*(0.5 - J) - Q)/((numangles_bootstrap_wprior + 1)*(numangles_bootstrap_wprior + 2)))

        Var_I = abs( \
             ((K - J*J))/(numangles_bootstrap_wprior + 1) + (M + (nbins-1)*(nbins-1)*(0.5 - J) - Q)/((numangles_bootstrap_wprior + 1)*(numangles_bootstrap_wprior + 2))) # a different variance for each bootstrap sample

        #now for higher moments, leading order terms

        #E_I3 = (1.0 / (numangles_bootstrap_wprior * numangles_bootstrap_wprior) ) \
        #       * (2.0 * (2 * J**3 -3*K*J + L) + 3.0 * (K + J*J - P))

        #E_I4 = (3.0 / (numangles_bootstrap_wprior * numangles_bootstrap_wprior)) * ((K - J*J) ** 2)

        #convert to skewness and kurtosis (not excess kurtosis)

        
        if(VERBOSE >= 2):
               print "Moments for Bayesian p(I|Data):"
               print "E_I:                   "+str(E_I)
               print "Var_I:                 "+str(Var_I)
        #print "Stdev_I:               "+str(sqrt(Var_I))
        #print "skewness:              "+str(E_I3/ (Var_I ** (3/2)) )
        #print "kurtosis:              "+str(E_I4/(Var_I ** (4/2)) )

        
        
        def Edgeworth_pdf(u1,u2,u3,u4):
            #convert moments to cumulants for u0=1, u1=0, u2=1, central moments normalized to zero mean and unit variance
            skewness = u3 / (u2 ** (3/2))
            excess_kurtosis = u4 / (u2 ** (4/2)) - 3
            s = sqrt(u2)
            k3 = skewness
            k4 = excess_kurtosis
            
            return lambda x: stats.norm.pdf((x-u1)/s) * (1.0 + (1.0/6)*k3*(special.hermitenorm(3)((x-u1)/s)) \
                              + (1.0/24)*k4*(special.hermitenorm(4)((x-u1)/s)))

        def Edgeworth_quantile(crit_value,u1,u2):
            #only Gaussian for now for speed #,u3,u4):
            #convert critical value to z-score
            #func = Edgeworth_pdf(u1,u2,u3,u4)
            #print func
            #normalization = integrate.quad( func, -100*sqrt(u2), 100*sqrt(u2) )[0]
            #integral = integrate.quad( func, -integrate.inf, crit_value)[0] #output just the definite integral
            #print "integral:              "+str(integral)
            #print "normalization:         "+str(normalization)
            #pval = abs(integral)
            #print "plusminus_sigma_check:"+str(integrate.quad( func, u1-sqrt(u2), u1+sqrt(u2))[0]/normalization)
            pval_gauss = stats.norm.cdf((crit_value - u1)/sqrt(u2))
            #print "p-value Bayes Edgeworth: "+str(pval)
            if(VERBOSE>=2):
                   print "p-value Bayes Gaussian:  "+str(pval_gauss)
            return pval_gauss

        #if(bootstrap_sets > 1):
        #    numangles_bootstrap_avg_wprior = average(numangles_bootstrap_wprior)
        #else:
        #    numangles_bootstrap_avg_wprior = numangles_bootstrap_wprior[0]

        if E_I_uniform is None:
            E_I_uniform = average(special.psi(numangles_bootstrap_wprior / (nbins * nbins) + 1) \
                                  - special.psi(numangles_bootstrap_wprior / nbins + 1) \
                                  - special.psi(numangles_bootstrap_wprior / nbins + 1) \
                                  + special.psi(numangles_bootstrap_wprior + 1))

        #####DEBUGGING STATEMENTS
        #print "Edgeworth pdf for E_I and E_I_uniform:"


        
        #print Edgeworth_pdf( E_I, Var_I, E_I3, E_I4)(E_I)
        #print Edgeworth_pdf( E_I, Var_I, E_I3, E_I4)(E_I_uniform)

        
        
        #print "E_I_uniform            :"+str(E_I_uniform)
        #print "E_I_multinomial_constr :"+str(E_I_multinomial)
        #now, determine the probability given the data that the true mutual information is
        #greater than that expected for the uniform distribution plus three sigma

        #pvalue for false positive
        #lower pvalue is better
        ##############################
        
        
        for bootstrap in range(bootstrap_sets):
            #for speed, use shortcuts for obviously significant or obviously insignificant mutual information
            if (E_I[bootstrap] < E_I_multinomial[bootstrap]):
                pvalue[bootstrap] = 1.0
            else:
                if(E_I[bootstrap] > E_I_multinomial[bootstrap] + 10 * sqrt(Var_I[bootstrap])):
                    pvalue[bootstrap] = 0.0
                else:
                    pvalue[bootstrap] = Edgeworth_quantile(E_I_multinomial[bootstrap], E_I[bootstrap], Var_I[bootstrap]) # , E_I3[bootstrap], E_I4[bootstrap])
            if(VERBOSE >=2 ):
                   print "Bayesian P(I<E[I]mult) bootstrap sample:"+str(bootstrap)+" = "+str(pvalue[bootstrap])
            num_greater_than_obs_MI = sum(1.0 * (mutinf_multinomial[bootstrap] > mutinf_thisdof[bootstrap,0]))
            if num_greater_than_obs_MI < 1:
                num_greater_than_obs_MI = 0
            pvalue_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_multinomial[bootstrap].shape[0])
            if (VERBOSE >= 2):
                   #print "Mutinf Multinomial Shape:"+str(mutinf_multinomial.shape)
                   print "Num Ind Greater than Obs:"+str(num_greater_than_obs_MI)
                   print "bootstrap: "+str(bootstrap)+" Descriptive P(I=I_mult):"+str(pvalue_multinomial)
            pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_multinomial)
            
        if(VERBOSE >= 2):
               print "Max pvalue             :"+str(pvalue[bootstrap])
               #print "integrate check: "+str(Edgeworth_quantile(integrate.inf,  E_I, Var_I)) #, E_I3, E_I4))
        #could use a monte carlo simulation to generate distribution of MI of uniform distribution over adaptive partitioning
        #this would be MI of independent variables
        #for non-Bayesian significance test against null hypothesis of independence
        #but not at the present time
           
    else:  #use permutation test to filter out true negatives, possibly in addition to the Bayesian filter above
        
        #pvalue is the fraction of mutinf values from samples of permuted data that are greater than the observed MI
        #pvalue for false negative
        #lower pvalue is better
        if(permutations > 0): #otherwise, keep pvalue as 0 for now, use wilcoxon signed ranks test at the end
            for bootstrap in range(bootstrap_sets):
                num_greater_than_obs_MI = sum(mutinf_thisdof[bootstrap,1:] > mutinf_thisdof[bootstrap,0])
                pvalue[bootstrap] += num_greater_than_obs_MI * 1.0 / permutations
                if(VERBOSE >=2):
                       print "number of permutations with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later
        else:
            for bootstrap in range(bootstrap_sets):
                num_greater_than_obs_MI = sum(1.0 * (mutinf_multinomial[bootstrap] > mutinf_thisdof[bootstrap,0]))

                pvalue_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_multinomial[bootstrap].shape[0])
                
                pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_multinomial)
                if(VERBOSE >=2):
                       print "Descriptive P(I=I_mult):"+str(pvalue_multinomial)
                       print "number of permutations with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later

    
    #print mutinf_thisdof
    if(bootstrap_sets > 1 and calc_variance == True):
       if(permutations > 0):
           var_mi_thisdof = (vfast_cov_for1D_boot(reshape(mutinf_thisdof[:,0],(mutinf_thisdof.shape[0],1))) - average(mutinf_thisdof[:,1:],axis=1))[0,0]
       else:
           var_mi_thisdof = (vfast_cov_for1D_boot(reshape(mutinf_thisdof[:,0],(mutinf_thisdof.shape[0],1))))[0,0]
    else:
       var_mi_thisdof = sum(Var_I)
    if(VERBOSE >=2):
           print "var_mi_thisdof: "+str(var_mi_thisdof)+"\n"
    if(calc_mutinf_between_sims == "no" or num_pair_runs <= 1):
        mutinf_thisdof_different_sims= zeros((bootstrap_sets,permutations_sequential + 1),float64)
        return mutinf_thisdof, var_mi_thisdof , mutinf_thisdof_different_sims, 0, average(mutinf_multinomial, axis=1), zeros((bootstrap_sets),float64), pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots_for_MI_norm
    #########################################################################################################
    ##
    ##  Now we will calculate mutinf between torsions between different sims for the undersampling correction 
    ##
    ##
    ##  For now, we will use the same undersampling correction for both regular and "_star" terms
    ##  
    ## 
    #########################################################################################################

    count_matrix_sequential[:,:,:,:] = 0
    
    #max_num_angles = int(max(numangles))
    max_num_angles = int(min(numangles))

    if(permutations == 0):
        if(mutinf_multinomial_sequential is None or adaptive_partitioning == 0):
            E_I_multinomial_sequential, Var_I_multinomial_sequential, E_I3_multinomial_sequential, E_I4_multinomial_sequential, Var_I_runs_multinomial_sequential, mutinf_multinomial_sequential, var_mutinf_multinomial_sequential = \
                                        calc_mutinf_multinomial_constrained(nbins_cor,chi_counts_sequential1.copy(),chi_counts_sequential2.copy(),adaptive_partitioning, bootstraps = bootstrap_sets )
            if VERBOSE >= 2:
                   print "shape of mutinf_multinomial_sequential: "+str(shape(mutinf_multinomial_sequential))
    else:
        mutinf_multinomial_sequential = var_mutinf_multinomial_sequential = zeros((bootstrap_sets,1),float64)
    
    #no_weights = False
    if (weights is None):
           weights = ones((num_sims, max_num_angles), float64)
    #       no_weights = True
    if VERBOSE >= 3:       
           print "weights"
           print weights
           print "bins1_sequential"
           print bins1_sequential
           print "bins2_sequential"
           print bins2_sequential
    code = """
     // weave7
     #include <math.h>
     
     int mynumangles, mynumangles1, mynumangles2, run1, run2, fetch1, fetch2, bin1, bin2, permut, anglenum = 0;
     double weight, weights_sum = 0 ;
     for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
       weights_sum = 0; // accumulator for normalization
       for(int which_run_pair=0; which_run_pair < num_pair_runs; which_run_pair++) {
         mynumangles1 = *(numangles + (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2 + 0)));
         mynumangles2 = *(numangles + (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2 + 1)));
         if(mynumangles1 <= mynumangles2) mynumangles = mynumangles1; else mynumangles = mynumangles2;
         run1 = (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2  + 0));
         run2 = (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2 + 1));
         for (permut=0; permut < permutations_sequential + 1; permut++) {
           for (anglenum=0; anglenum< mynumangles; anglenum++) {
             bin1 = *(bins1_sequential  +  run1*max_num_angles  +  anglenum);
             bin2 = *(bins2_sequential  + permut*num_sims*max_num_angles + run2*max_num_angles  +  anglenum);
             //printf("bin1: %i bin2: %i \\n",bin1,bin2); 
             // take effective weight as product of the individual runs weights, will later renormalize
             weight = *(weights + run1*max_num_angles + anglenum) * (*(weights + run2*max_num_angles + anglenum));  
             weights_sum += weight;
             *(count_matrix_sequential  +  mybootstrap*num_pair_runs*(permutations_sequential + 1)*nbins_cor*nbins_cor + which_run_pair*(permutations_sequential + 1)*nbins_cor*nbins_cor  +  permut*nbins_cor*nbins_cor  +  bin1*nbins_cor +   bin2) += 1.0 * weight ;
           }
           // now normalize so that sum of snapshots weights equals numangles
           for(bin1 = 0; bin1 < nbins_cor; bin1++) {
             for(bin2 = 0; bin2 < nbins_cor; bin2++) {
                *(count_matrix_sequential  +  mybootstrap*num_pair_runs*(permutations_sequential + 1)*nbins_cor*nbins_cor + which_run_pair*(permutations_sequential + 1)*nbins_cor*nbins_cor  +  permut*nbins_cor*nbins_cor  +  bin1*nbins_cor +   bin2) *= (mynumangles / weights_sum);
                }
           }
           weights_sum = 0; //reset 
         }
        }
       }
    """                                           
                                          
                                           
    weave.inline(code, ['num_sims', 'numangles','max_num_angles', 'nbins_cor', 'bins1_sequential', 'bins2_sequential', 'count_matrix_sequential','pair_runs','num_pair_runs','bootstrap_sets','permutations_sequential','weights'],
                 #type_converters = converters.blitz,
                  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
    #if (no_weights != False):
    #count_matrix_sequential /= (min(numangles) * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be min(numangles)
    #for bootstrap in range(bootstrap_sets):
    #    for which_run_pair
    #    count_matrix_sequential[bootstrap,:,:] /= min((numangles[pair_runs[bootstrap,0]], numangles[pair_runs[bootstrap,1]]))
    

                                                                              
                                     
     
    
    #logU_sequential[:,:,:] = 0
    #U_sequential = count_matrix_sequential / (ninj_flat_Bayes_sequential + SMALL)
    if(VERBOSE >= 2):
        print "count_matrix_sequential\n"
        print count_matrix_sequential[0,0,0:]
        print "chi counts sequential1"
        print chi_counts_sequential1[0]
        print "chi counts sequential2"
        print chi_counts_sequential2[0]
    ##### DEBUGGING STUFF ##############################
    
    #print "shape of count_matrix_sequential\n"
    #print shape(count_matrix_sequential)
    #print "chi pop hist sequential1\n"
    #print chi_pop_hist_sequential1[pair_runs[0,0,0]]*numangles[pair_runs[0,0,0]]
    #print "chi pop hist sequential1\n"
    #print chi_pop_hist_sequential1[0]*numangles[0]
    #print "sum"
    #print sum(chi_pop_hist_sequential1[pair_runs[0,0,0]]*numangles[pair_runs[0,0,0]])
    #print "chi pop hist sequential2\n"
    #print chi_pop_hist_sequential2[pair_runs[0,0,1]]*numangles[pair_runs[0,0,1]]
    #print "chi pop hist sequential2\n"
    #print chi_pop_hist_sequential2[1]*numangles[1]
    #print "sum"
    #print sum(chi_pop_hist_sequential2[1]*numangles[1])
    #print "before log\n"
    #logU_sequential = log(U_sequential + SMALL)
                                                                              
    #print "min angles boot pair runs vector shape:"
    #print  shape(min_angles_boot_pair_runs_vector)
    ######################################################


    #take entropy for sims in chi counts sequential, summing over sims in each bootstrap
    ent1_boots_sims = zeros((bootstrap_sets, num_pair_runs, permutations_sequential + 1),float64)
    ent2_boots_sims = zeros((bootstrap_sets, num_pair_runs, permutations_sequential + 1),float64)

    if(NO_GRASSBERGER == False):
      for bootstrap in range(bootstrap_sets):
        for which_pair in range(num_pair_runs):
            #pick index 0 for axis=2 at the end because the arrays are over 
            
            myent1                                \
                                                  =\
                                                  sum((chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) \
                                                      * (log(min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) \
                                                         - special.psi( \
                chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] + SMALL) \
                                                         - (1 - 2*( \
                int32(chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]]) % 2)) / (chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] + 1.0)),axis=0) #sum over bins
            
            #print "ent1 boots sims thispair shape: "
            #print myent1.shape
            #print "ent1 boots: "+str(myent1)

            ent1_boots_sims[bootstrap,which_pair,:] = myent1
            
            myent2                                \
                                                  =\
                                                  sum((chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) \
                                                      * (log(min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) \
                                                         - special.psi( \
                chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] + SMALL) \
                                                         - (1 - 2*( \
                int32(chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]]) % 2)) / (chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] + 1.0)),axis=0) #sum over bins


            #print "ent1 boots sims thispair shape: "
            #print myent2.shape
            #print "ent1 boots: "+str(myent2)

            ent2_boots_sims[bootstrap,which_pair,:] = myent2
            #ent1_boots = average(ent1_boots_sims,axis=1) # avg over sims in each bootstrap
            #ent2_boots = average(ent2_boots_sims,axis=1) # avg over sims in each bootstrap


    else:  #do not use Grassberger corrected entropies
      for bootstrap in range(bootstrap_sets):
        for which_pair in range(num_pair_runs):
            #pick index 0 for axis=2 at the end because the arrays are over 
            
            myent1                                \
                                                  =\
                                                  sum((chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) \
                                                      * (log((chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) + SMALLER)),axis=0) #sum over bins
            
            #print "ent1 boots sims thispair shape: "
            #print myent1.shape
            #print "ent1 boots: "+str(myent1)

            ent1_boots_sims[bootstrap,which_pair,:] = myent1
            
            myent2                                \
                                                  =\
                                                  sum((chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) \
                                                      * (log((chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) + SMALLER )),axis=0) #sum over bins


            #print "ent1 boots sims thispair shape: "
            #print myent2.shape
            #print "ent1 boots: "+str(myent2)

            ent2_boots_sims[bootstrap,which_pair,:] = myent2
            #ent1_boots = average(ent1_boots_sims,axis=1) # avg over sims in each bootstrap
            #ent2_boots = average(ent2_boots_sims,axis=1) # avg over sims in each bootstrap



    #print "ent1 boots sims shape:"
    #print shape(ent1_boots_sims)
    #print "ent1 boots sims :"
    #print ent1_boots_sims
    #print "ent2 boots sims :"
    #print ent2_boots_sims
    ent1_boots = ent1_boots_sims
    ent2_boots = ent2_boots_sims
    
    
    
    # MI = H(1)+H(2)-H(1,2)
    # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI

    ent12_boots = sum((count_matrix_sequential * 1.0 /min_angles_boot_pair_runs_matrix)  \
                           * ( log(min_angles_boot_pair_runs_matrix) - \
                               special.psi(count_matrix_sequential + SMALL)  \
                               - (1 - 2*(int16(count_matrix_sequential) % 2)) / (count_matrix_sequential + 1.0) \
                               ),axis=3)

    #print "ent12_boots shape:"
    #print ent12_boots.shape
    #print "ent12_boots:      "
    #print ent12_boots
    
    mutinf_thisdof_different_sims_bootstrap_pairs = ent1_boots + ent2_boots - ent12_boots
                                                                                          
    #print "bootstrap sets:"+str(bootstrap_sets)
    #print "permutations:  "+str(permutations)
    #print "num pair runs: "+str(num_pair_runs)
    #print "mutinf sim1 sim2 shape:"

    #print mutinf_thisdof_different_sims_bootstrap_pairs.shape
    if(permutations == 0):
           if(VERBOSE >=2):
                  print "mutinf_multinomial_difsm:"+str(average(mutinf_multinomial_sequential))
                  print "stdev mutinf multi difsm:"+str(sqrt(average(var_mutinf_multinomial_sequential)))
    if(VERBOSE >=2):
           print "avg mutinf sim1 sim2        :"+str(average(mutinf_thisdof_different_sims_bootstrap_pairs[0,:,0]))
    mutinf_thisdof_different_sims = average(mutinf_thisdof_different_sims_bootstrap_pairs, axis=1) #average over pairs of runs in each bootstrap sample
    #now the nbins_cor*nbins_cor dimensions and num_pair_runs dimensions have been removed, leaving only bootstraps and permutations
    #print "mutinf values between sims for over original data for bootstrap=0 and permuted data\n"
    #print mutinf_thisdof_different_sims[0,0]
    #print mutinf_thisdof_different_sims[0,1:]
    if(VERBOSE >= 2):
        if bootstrap_sets > 1:
            print "mutinf values between sims for over original data for bootstrap=1 and permuted data\n"
            print mutinf_thisdof_different_sims[1,0]
            print mutinf_thisdof_different_sims[1,1:]
    #just first pair for debugging 
    #mutinf_thisdof_different_sims = mutinf_thisdof_different_sims_bootstrap_pairs[:,0]
    if(permutations > 0):
        var_ind_different_sims_pairs=zeros((bootstrap_sets,num_pair_runs),float64)
        #for bootstrap in range(bootstrap_sets):
        #   var_ind_different_sims_pairs[bootstrap,:] = vfast_var_for1Dpairs_ind(mutinf_thisdof_different_sims_bootstrap_pairs[bootstrap,:,1:])
        var_ind_different_sims_pairs = vfast_var_for1Dpairs_ind(mutinf_thisdof_different_sims_bootstrap_pairs)
        if(VERBOSE >= 2):
               print "variance of mutinf between sims for orig data and permuts"
               print var_ind_different_sims_pairs
        var_ind_different_sims = average(var_ind_different_sims_pairs, axis=1) #average variance over pairs of runs
        #NOTE: var_ind_different_sims is of shape = (bootstrap_sets), while var_mutinf_multinomial_sequential is only 1 number
        # But this will be used in array context -- for permutations == 0, it will be implicity repeated in calc_excess_mutinf
        # for permutations > 0, it is already of shape = (bootstrap_sets)
        var_mutinf_multinomial_sequential = var_ind_different_sims 
        #print "variance of mutinf between sims for orig data and permuts"     
        
    if calc_variance == False:
       #if VERBOSE: print "   mutinf = %.5f," % (mutinf_thisdof),
       return mutinf_thisdof, Var_I , mutinf_thisdof_different_sims, var_mutinf_multinomial_sequential, average(mutinf_multinomial,axis=1),average(mutinf_multinomial_sequential,axis=1), pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots_for_MI_norm
    # print out the number of nonzero bins
    if VERBOSE >=2:
     #  num_nonzero_jointpop, num_nonzero_pxipyj = len(nonzero(av_joint_pop)[0]), len(nonzero(pxipyj_flat)[0]), 
     #  print "   nonzero bins (tot=%d): jointpop=%d, pxipyj=%d, combined=%d" % (nbins*nbins, num_nonzero_jointpop, num_nonzero_pxipyj, len(rnonzero_bins[nonzero_bins==True]))
       print "   mutinf this degree of freedom "+str(mutinf_thisdof)
    ##### No longer used #######################################################################
    # calculate the variance of the mutual information using its derivative and error-propagation
    #pxi_plus_pyj_flat = add.outer(chi_pop_hist1, chi_pop_hist2).flatten()
    #deriv_vector = 1 + logU[0,:] - (pxi_plus_pyj_flat[0,:]) * U
    ##############################################################################################
    
    return mutinf_thisdof, var_mi_thisdof, mutinf_thisdof_different_sims, var_mutinf_multinomial_sequential, average(mutinf_multinomial,axis=1), average(mutinf_multinomial_sequential,axis=1), pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots_for_MI_norm


#####################################################################################################################


###############
###############  The code below is duplicated to work with second order KLdiv -- might be able to be merged in with the above code
###############
###############
def calc_mutinf_corrected_star(chi_counts1, chi_counts2, bins1, bins2, chi_counts_sequential1, chi_counts_sequential2, bins1_sequential, bins2_sequential, num_sims, nbins, numangles_bootstrap, numangles, calc_variance=False,bootstrap_choose=0,permutations=0,which_runs=None,pair_runs=None, calc_mutinf_between_sims="yes", markov_samples = 0, chi_counts1_markov=None, chi_counts2_markov=None, ent1_markov_boots=None, ent2_markov_boots=None, bins1_markov=None, bins2_markov=None, file_prefix=None, plot_2d_histograms=False, adaptive_partitioning = 0, lagtime_interval=None, bins1_slowest_timescale = None, bins2_slowest_timescale = None, bins1_slowest_lagtime = None, bins2_slowest_lagtime = None, boot_weights = None, weights = None, chi_counts1_star=None, chi_counts2_star=None, bins1_star=None, bins2_star=None,chi_counts_sequential1_star=None, chi_counts_sequential2_star=None, bins1_sequential_star=None, bins2_sequential_star=None, numangles_star=None, numangles_bootstrap_star=None, bootstrap_choose_star=None ):
    global count_matrix, count_matrix_sequential, ninj_flat_Bayes, ninj_flat_Bayes_sequential # , ninj_flat
    global nbins_cor, min_angles_boot_pair_runs, numangles_bootstrap_matrix, numangles_boot_pair_runs_matrix, numangles_bootstrap_vector
    global min_angles_boot_pair_runs_matrix, min_angles_boot_pair_runs_vector
    global count_matrix_star, count_matrix_sequential_star, ninj_flat_Bayes_star, ninj_flat_Bayes_sequential_star, numangles_bootstrap_matrix_star
    global E_I_multinomial, Var_I_multinomial, E_I3_multinomial, E_I4_multinomial, Var_I_runs_multinomial 
    global mutinf_multinomial, var_mutinf_multinomial, mutinf_multinomial_sequential, var_mutinf_multinomial_sequential
    global E_I_uniform
    
    #print "bins1 lagtime: "+str(bins1_slowest_lagtime)
    #print "bins2 lagtime: "+str(bins2_slowest_lagtime)

    # allocate the matrices the first time only for speed
    if(bootstrap_choose == 0):
        bootstrap_choose = num_sims
    bootstrap_sets = len(which_runs)
    if(VERBOSE >=1):
           assert(all(chi_counts1 >= 0))
           assert(all(chi_counts2 >= 0))
    permutations_sequential = permutations
    #permutations_sequential = 0 #don't use permutations for mutinf between sims
    max_num_angles = int(min(numangles)) # int(max(numangles))
    if(numangles_star != None): 
        max_num_angles_star = int(min(numangles_star)) # int(max(numangles_star))
    
    num_pair_runs = pair_runs.shape[1]
    nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    if(VERBOSE >=2):
           print "nbins_cor: "+str(nbins_cor)
           print pair_runs
           print "bootstrap_sets: "+str(bootstrap_sets)+"num_pair_runs: "+str(num_pair_runs)+"\n"

    #print numangles_bootstrap
    #print bins1.shape
    
    #assert(bootstrap_sets == pair_runs.shape[0] == chi_counts1.shape[0] == chi_counts2.shape[0] == bins1.shape[0] == bins2.shape[0])
    #ninj_prior = 1.0 / float64(nbins*nbins) #Perks' Dirichlet prior
    #ni_prior = nbins * ninj_prior           #Perks' Dirichlet prior
    ninj_prior = 1.0                        #Uniform prior
    ni_prior = nbins * ninj_prior           #Uniform prior
    pvalue = zeros((bootstrap_sets),float64)

    markov_interval = zeros((bootstrap_sets),int16)
    # allocate the matrices the first time only for speed
    if(markov_samples > 0 and bins1_slowest_lagtime != None and bins2_slowest_lagtime != None ):
           for bootstrap in range(bootstrap_sets):
                  max_lagtime = max(bins1_slowest_lagtime[bootstrap], bins2_slowest_lagtime[bootstrap])
                  markov_interval[bootstrap] = int(max_num_angles / max_lagtime) #interval for final mutual information calc, based on max lagtime
    else:
           markov_interval[:] = 1

    markov_interval[:] = 1
    #must be careful to zero discrete histograms that we'll put data in from weaves
    if VERBOSE >= 2:
           print "markov interval: "
           print markov_interval
           print "chi counts 1 before multinomial:"
           print chi_counts1
           print "chi counts 2 before multinomial:"
           print chi_counts2
    #initialize if permutations > 0
    if(permutations > 0):
           mutinf_multinomial = zeros((bootstrap_sets,1),float64)
           mutinf_multinomial_sequential = zeros((bootstrap_sets,1),float64)
           var_mutinf_multinomial_sequential = zeros((bootstrap_sets,1), float64)
    #only do multinomial if not doing permutations
    Counts_ij_ind = zeros((bootstrap_sets, nbins, nbins),float64)
    if(E_I_multinomial is None and permutations == 0 and markov_samples == 0 ):
        E_I_multinomial, Var_I_multinomial, E_I3_multinomial, E_I4_multinomial, Var_I_runs_multinomial, \
                         mutinf_multinomial, var_mutinf_multinomial = \
                         calc_mutinf_multinomial_constrained(nbins,chi_counts1,chi_counts2,adaptive_partitioning )
    if(permutations == 0 and markov_samples > 0 ): #run mutinf for independent markov samples for every dihedral, mutinf_multinomial now refers to markov-based independent mutinf
        E_I_multinomial, Var_I_multinomial, E_I3_multinomial, E_I4_multinomial, Var_I_runs_multinomial, \
                         mutinf_multinomial, var_mutinf_multinomial, Counts_ij_ind = \
                         calc_mutinf_markov_independent(nbins,chi_counts1_markov, chi_counts2_markov, ent1_markov_boots, ent2_markov_boots, bins1_markov,bins2_markov, bootstrap_sets, bootstrap_choose, markov_samples, max_num_angles, numangles_bootstrap, bins1_slowest_lagtime, bins2_slowest_lagtime)
    
    #NOTE: If markov_samples == 0, shape of mutinf_multinomial is (bootstrap_sets, 1). If markov_samples > 0, shape of mutinf_multinomial is (bootstrap_sets, markov_samples). 

    if count_matrix is None:    

       count_matrix = zeros((bootstrap_sets, permutations + 1 , nbins*nbins), float64)
       count_matrix_sequential = zeros((bootstrap_sets, num_pair_runs,permutations_sequential + 1 , nbins_cor*nbins_cor), float64)
       min_angles_boot_pair_runs_matrix = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor*nbins_cor),int32)
       min_angles_boot_pair_runs_vector = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor),int32)
       for bootstrap in range(bootstrap_sets):
            for which_pair in range(num_pair_runs):
                if VERBOSE >= 2:
                       print "run1 "+str(pair_runs[bootstrap,which_pair,0])+" run2 "+str(pair_runs[bootstrap,which_pair,1])
                       print "numangles shape:" +str(numangles.shape)
                #my_flat = outer(chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]],chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]]).flatten()
                #my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0])) #replicate original data over n permutations
                min_angles_boot_pair_runs_matrix[bootstrap,which_pair,:,:] = resize(array(min(numangles[pair_runs[bootstrap,which_pair,0]],numangles[pair_runs[bootstrap,which_pair,1]]),int32),(permutations_sequential + 1, nbins_cor*nbins_cor)) #replicate original data over permutations and nbins_cor*nbins_cor
                min_angles_boot_pair_runs_vector[bootstrap,which_pair,:,:] = resize(array(min(numangles[pair_runs[bootstrap,which_pair,0]],numangles[pair_runs[bootstrap,which_pair,1]]),int32),(nbins_cor)) #replicate original data over nbins_cor, no permutations for marginal dist.            
                #ninj_flat_Bayes_sequential[bootstrap,which_pair,:,:] = my_flat[:,:]

       
           
       
    if(numangles_bootstrap_matrix is None):
        if(VERBOSE >= 2):
               print "numangles bootstrap: "+str(numangles_bootstrap)
        numangles_bootstrap_matrix = zeros((bootstrap_sets,permutations+1,nbins*nbins),float64)
        for bootstrap in range(bootstrap_sets):
            for permut in range(permutations+1):
                numangles_bootstrap_matrix[bootstrap,permut,:]=numangles_bootstrap[bootstrap]
        numangles_bootstrap_vector = numangles_bootstrap_matrix[:,:,:nbins]
        #print "Numangles Bootstrap Vector:\n"+str(numangles_bootstrap_vector)
    
    #if(chi_counts2_matrix is None):
    chi_counts1_matrix = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions 
    chi_counts2_matrix = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions
    

    
    count_matrix[:,:,:] = 0
    count_matrix_sequential[:,:,:,:] =0
    # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
    chi_counts1_vector = reshape(chi_counts1.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    chi_counts2_vector = reshape(chi_counts2.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    chi_counts1_vector = repeat(chi_counts1_vector, permutations + 1, axis=1)     #but we need to repeat along permutations axis
    chi_counts2_vector = repeat(chi_counts2_vector, permutations + 1, axis=1)     #but we need to repeat along permutations axis

    ent_1_boots = zeros((bootstrap_sets,permutations + 1),float64)
    ent_2_boots = zeros((bootstrap_sets,permutations + 1),float64)
    ent_1_2_boots = zeros((bootstrap_sets,permutations + 1),float64)
    
    
    for bootstrap in range(bootstrap_sets): #copy here is critical to not change original arrays
        chi_counts2_matrix[bootstrap,0,:] = repeat(chi_counts2[bootstrap].copy(),nbins,axis=-1) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
        #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
        #handling the slower-varying index will be a little bit more tricky
        chi_counts1_matrix[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1[bootstrap].copy(), nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
    
    no_boot_weights = False
    if (boot_weights is None):
           boot_weights = ones((bootstrap_sets, max_num_angles * bootstrap_choose), float64)
           no_boot_weights = True

    code = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int bin1;
     int bin2;
     int mybootstrap, permut;
     long anglenum, mynumangles, counts1, counts2, counts12 ; 
     double mysign1, mysign2, mysign12, counts1d, counts2d, counts12d, dig1, dig2, dig12;
     
     #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,angle1_bin,angle2_bin,weight,bin1, bin2, mysign1, mysign2, mysign12, counts1, counts1d, counts2, counts2d, counts12, counts12d, dig1, dig2, dig12 )
     for( mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);
      for (permut=0; permut < permutations + 1; permut++) {          
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          //if(mybootstrap == bootstrap_sets - 1) {
          //  //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
          //  }
           if(anglenum == mynumangles - 1) {
             printf(""); //just to make sure count matrix values are written to memory before the next loop
           }
           //if(anglenum % markov_interval[mybootstrap] == 0) { 
              angle1_bin = *(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum);
              angle2_bin = *(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight;
           //} 
          }
           // do singlet entropies here
           for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1 = long(*(chi_counts1_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1);
                
                
                counts1d = 1.0 * counts1;
                *(ent_1_boots + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts1 + 1.0)))); 
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2 = long(*(chi_counts2_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin1);
              }
             }
 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts2d / mynumangles)*(log(mynumangles) - dig2 - ((double)mysign2 / ((double)(counts2 + 1.0)))); 
              }



            // do doublet entropy here
            for(bin2=0; bin2 < nbins; bin2++)
             {
         
             // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
             counts12 = long(*(count_matrix  + (long)( mybootstrap*(permutations + 1)*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 )));
 
              if(counts12 > 0)
              {
                mysign12 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                dig12 = xDiGamma_Function(counts12);
                
                
                counts12d = 1.0 * counts12;
                *(ent_1_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts12d / mynumangles)*(log(mynumangles) - dig12 - ((double)mysign1 / ((double)(counts12 + 1.0)))); 
              }
            }
         }
       
    
        }
       }
      """

    code_no_grassberger = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int bin1;
     int bin2;
     int mybootstrap, permut;
     long anglenum, mynumangles, counts1, counts2, counts12 ; 
     double mysign1, mysign2, mysign12, counts1d, counts2d, counts12d, dig1, dig2, dig12;
     
     #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,angle1_bin,angle2_bin,weight,bin1, bin2, mysign1, mysign2, mysign12, counts1, counts1d, counts2, counts2d, counts12, counts12d, dig1, dig2, dig12 )
     for( mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);
      for (permut=0; permut < permutations + 1; permut++) {          
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          //if(mybootstrap == bootstrap_sets - 1) {
          //  //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
          //  }
           if(anglenum == mynumangles - 1) {
             printf(""); //just to make sure count matrix values are written to disk before the next loop
           }
           //if(anglenum % markov_interval[mybootstrap] == 0) { 
              angle1_bin = *(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum);
              angle2_bin = *(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight;
           //} 
          }
           // do singlet entropies here
           for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1 = long(*(chi_counts1_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1);
                
                
                counts1d = 1.0 * counts1;
                *(ent_1_boots + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALLER));
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2 = long(*(chi_counts2_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin1);
              }
             }
 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts2d / mynumangles)*(log((double)counts2d / mynumangles + SMALLER)); 
              }



            // do doublet entropy here
            for(bin2=0; bin2 < nbins; bin2++)
             {
         
             // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
             counts12 = long(*(count_matrix  + (long)( mybootstrap*(permutations + 1)*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 )));
 
              if(counts12 > 0)
              {
                mysign12 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                dig12 = xDiGamma_Function(counts12);
                
                
                counts12d = 1.0 * counts12;
                *(ent_1_2_boots + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts12d / mynumangles)*(log((double)counts12d / mynumangles + SMALLER)); 
              }
            }
         }
       
    
        }
       }
      """

 

    if(VERBOSE >= 2): 
           print "about to populate count_matrix"
           print "chi counts 1"
           print chi_counts1[bootstrap]
           print "chi counts 2"
           print chi_counts2[bootstrap]

    if(NO_GRASSBERGER == False):
           weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots','markov_interval','SMALL'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    else:
           weave.inline(code_no_grassberger, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots','markov_interval','SMALLER'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    if (no_boot_weights != False ):
           count_matrix /= (bootstrap_choose*max_num_angles * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be bootstrap_choose*min(numangles)

    if(VERBOSE >=2):
           print "count matrix first pass:"
           print count_matrix
           print "chi counts 1"
           print chi_counts1[bootstrap]
           print "chi counts 2"
           print chi_counts2[bootstrap]
    

    ### Redundant Sanity checks
    ninj_flat = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
        my_flat = outer(chi_counts1[bootstrap] + 0.0 ,chi_counts2[bootstrap] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
        if(VERBOSE >=1):
               assert(all(my_flat >= 0))
        my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0]))
        ninj_flat[bootstrap,:,:] = my_flat[:,:]
        #now without the Bayes prior added into the marginal distribution
        my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
        my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
        ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
        nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix[:,:,:] == 0)) and (sum(chi_counts1) > 0) and (sum(chi_counts2) > 0):
           count_matrix[:,:] =  (outer(chi_counts1[bootstrap] ,chi_counts2[bootstrap] ).flatten()  ) / (numangles_bootstrap[0] * 1.0)
    
    if(VERBOSE >=1):
           assert(all(ninj_flat >= 0))
    Pij, PiPj = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
    permutation = 0
    Pij[1:,1:]  = (count_matrix[0,permutation,:]).reshape((nbins,nbins)) 
    PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    
    if(VERBOSE >= 1 and permutation == 0):
        print "First Pass:"
        print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
        print "Marginal Pij, summed over j:\n"
        print sum(Pij[1:,1:],axis=1)
        print "Marginal PiPj, summed over j:\n"
        print sum(PiPj[1:,1:],axis=1)   
        print "Marginal Pij, summed over i:\n"
        print sum(Pij[1:,1:],axis=0)
        print "Marginal PiPj, summed over i:\n"
        print sum(PiPj[1:,1:],axis=0)
    ### end redundant sanity checks
        
        
    #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
    if(VERBOSE >=1):
           assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
           assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
           assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    if(permutations > 0):
           permutation = 1
           Pij[1:,1:]  = (count_matrix[0,permutation,:]).reshape((nbins,nbins)) 
           PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    

           if(VERBOSE >=1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))









    #######################################################################################################
    ## Now for the star terms, if they are present, otherwise set the same as the normal ones     
    ########################################################################################################


    if count_matrix_star is None:    
       # max_num_angles_star = ???    
       count_matrix_star = zeros((bootstrap_sets, permutations + 1 , nbins*nbins), float64)
       count_matrix_sequential_star = zeros((bootstrap_sets, num_pair_runs,permutations_sequential + 1 , nbins_cor*nbins_cor), float64)
       min_angles_boot_pair_runs_matrix_star = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor*nbins_cor),int32)
       min_angles_boot_pair_runs_vector_star = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor),int32)
       for bootstrap in range(bootstrap_sets):
            for which_pair in range(num_pair_runs):
                if VERBOSE >= 2:
                       print "run1 "+str(pair_runs[bootstrap,which_pair,0])+" run2 "+str(pair_runs[bootstrap,which_pair,1])
                       print "numangles shape:" +str(numangles.shape)
                #my_flat = outer(chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]],chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]]).flatten()
                #my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0])) #replicate original data over n permutations
                min_angles_boot_pair_runs_matrix_star[bootstrap,which_pair,:,:] = resize(array(min(numangles_star[pair_runs[bootstrap,which_pair,0]],numangles_star[pair_runs[bootstrap,which_pair,1]]),int32),(permutations_sequential + 1, nbins_cor*nbins_cor)) #replicate original data over permutations and nbins_cor*nbins_cor
                min_angles_boot_pair_runs_vector_star[bootstrap,which_pair,:,:] = resize(array(min(numangles_star[pair_runs[bootstrap,which_pair,0]],numangles_star[pair_runs[bootstrap,which_pair,1]]),int32),(nbins_cor)) #replicate original data over nbins_cor, no permutations for marginal dist.            
                #ninj_flat_Bayes_sequential[bootstrap,which_pair,:,:] = my_flat[:,:]

       
           
       
    if(numangles_bootstrap_matrix_star is None):
        if(VERBOSE >= 2):
               print "numangles bootstrap: "+str(numangles_bootstrap_star)
        numangles_bootstrap_matrix_star = zeros((bootstrap_sets,permutations+1,nbins*nbins),float64)
        for bootstrap in range(bootstrap_sets):
            for permut in range(permutations+1):
                numangles_bootstrap_matrix_star[bootstrap,permut,:]=numangles_bootstrap_star[bootstrap]
        numangles_bootstrap_vector_star = numangles_bootstrap_matrix_star[:,:,:nbins]
        #print "Numangles Bootstrap Vector:\n"+str(numangles_bootstrap_vector)
    
    #if(chi_counts2_matrix is None):
    chi_counts1_matrix_star = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions 
    chi_counts2_matrix_star = zeros((bootstrap_sets,0 + 1,nbins*nbins),float64)        #no permutations for marginal distributions
    

    
    count_matrix_star[:,:,:] = 0
    count_matrix_sequential_star[:,:,:,:] =0
    # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
    chi_counts1_vector_star = reshape(chi_counts1_star.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    chi_counts2_vector_star = reshape(chi_counts2_star.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    chi_counts1_vector_star = repeat(chi_counts1_vector_star, permutations + 1, axis=1)     #but we need to repeat along permutations axis
    chi_counts2_vector_star = repeat(chi_counts2_vector_star, permutations + 1, axis=1)     #but we need to repeat along permutations axis

    ent_1_boots_star = zeros((bootstrap_sets,permutations + 1),float64)
    ent_2_boots_star = zeros((bootstrap_sets,permutations + 1),float64)
    ent_1_2_boots_star = zeros((bootstrap_sets,permutations + 1),float64)
    
    
    for bootstrap in range(bootstrap_sets): #copy here is critical to not change original arrays
        chi_counts2_matrix_star[bootstrap,0,:] = repeat(chi_counts2_star[bootstrap].copy(),nbins,axis=-1) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
        #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
        #handling the slower-varying index will be a little bit more tricky
        chi_counts1_matrix_star[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1_star[bootstrap].copy(), nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
    
    no_boot_weights = False
    if (boot_weights_star is None):
           boot_weights_star = ones((bootstrap_sets, max_num_angles * bootstrap_choose), float64)
           no_boot_weights = True

    code_star = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int bin1;
     int bin2;
     int mybootstrap, permut;
     long anglenum, mynumangles, counts1, counts2, counts12, counts1_star, counts2_star, counts12_star ; 
     double mysign1, mysign2, mysign12, counts1d, counts2d, counts12d, dig1, dig2, dig12;
     
     #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,angle1_bin,angle2_bin,weight,bin1, bin2, mysign1, mysign2, mysign12, counts1, counts1d, counts2, counts2d, counts12, counts12d, dig1, dig2, dig12 )
     for( mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);
      mynumangles_star = *(numangles_bootstrap_star + mybootstrap);
      for (permut=0; permut < permutations + 1; permut++) {          
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          //if(mybootstrap == bootstrap_sets - 1) {
          //  //printf("bin12 %i \\n",(*(bins1_star  +  mybootstrap*bootstrap_choose*max_num_angles_star  +  anglenum))*nbins +   (*(bins2_star + permut*bootstrap_sets*bootstrap_choose*max_num_angles_star + mybootstrap*bootstrap_choose*max_num_angles_star  +  anglenum)));
          //  }
           if(anglenum == mynumangles - 1) {
             printf(""); //just to make sure count matrix values are written to memory before the next loop
           }
           //if(anglenum % markov_interval[mybootstrap] == 0) { 
              angle1_bin = *(bins1_star  +  mybootstrap*bootstrap_choose*max_num_angles_star  +  anglenum);
              angle2_bin = *(bins2_star + permut*bootstrap_sets*bootstrap_choose*max_num_angles_star + mybootstrap*bootstrap_choose*max_num_angles_star  +  anglenum - offset);
              weight = *(boot_weights_star + mybootstrap*bootstrap_choose*max_num_angles_star + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight;
           //} 
          }
           // do singlet entropies here
           for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1      = long(*(chi_counts1_vector  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
              counts1_star = long(*(chi_counts1_vector_star  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              counts1_star = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1      += *(count_matrix  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
               counts1_star += *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1_star % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1_star);
                
                
                counts1d = 1.0 * counts1;
                *(ent_1_boots_star + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts1d / mynumangles)*(log(mynumangles_star) - dig1 - ((double)mysign1 / ((double)(counts1 + 1.0)))); 
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2      = long(*(chi_counts2_vector       + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
               counts2_star = long(*(chi_counts2_vector_star  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin1);
              }
             }
 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent_2_boots_star + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts2d / mynumangles)*(log(mynumangles_star) - dig2 - ((double)mysign2 / ((double)(counts2 + 1.0)))); 
              }



            // do doublet entropy here
            for(bin2=0; bin2 < nbins; bin2++)
             {
         
             // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
             counts12 = long(*(count_matrix_star  + (long)( mybootstrap*(permutations + 1)*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 )));
 
              if(counts12 > 0)
              {
                mysign12 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                dig12 = xDiGamma_Function(counts12);
                
                
                counts12d = 1.0 * counts12;
                *(ent_1_2_boots_star + (long)(mybootstrap*(permutations + 1) + permut)) += ((double)counts12d / mynumangles)*(log(mynumangles_star) - dig12 - ((double)mysign1 / ((double)(counts12 + 1.0)))); 
              }
            }
         }
       
    
        }
       }
      """

    code_no_grassberger_star = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight;
     int angle1_bin;
     int angle2_bin;
     int bin1;
     int bin2;
     int mybootstrap, permut;
     long anglenum, mynumangles, counts1, counts2, counts12 ; 
     double mysign1, mysign2, mysign12, counts1d, counts2d, counts12d, dig1, dig2, dig12;
     
     #pragma omp parallel for private(mybootstrap,mynumangles,permut,anglenum,angle1_bin,angle2_bin,weight,bin1, bin2, mysign1, mysign2, mysign12, counts1, counts1d, counts2, counts2d, counts12, counts12d, dig1, dig2, dig12 )
     for( mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);
      for (permut=0; permut < permutations + 1; permut++) {          
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          //if(mybootstrap == bootstrap_sets - 1) {
          //  //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
          //  }
           if(anglenum == mynumangles - 1) {
             printf(""); //just to make sure count matrix values are written to disk before the next loop
           }
           //if(anglenum % markov_interval[mybootstrap] == 0) { 
              angle1_bin = *(bins1_star +  mybootstrap*bootstrap_choose*max_num_angles_star  +  anglenum);
              angle2_bin = *(bins2_star + permut*bootstrap_sets*bootstrap_choose*max_num_angles_star + mybootstrap*bootstrap_choose*max_num_angles_star  +  anglenum - offset);
              weight = *(boot_weights_star + mybootstrap*bootstrap_choose*max_num_angles_star + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight;
           //} 
          }
           // do singlet entropies here
           for(bin1=0; bin1 < nbins; bin1++) 
           { 
             if(markov_interval[mybootstrap] < 2) {
              counts1 = long(*(chi_counts1_vector_star  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
              counts1 = 0;
              for(bin2=0; bin2 < nbins; bin2++) {
               counts1 += *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2);
              }
             }
   
              if(counts1 > 0)
              {
                mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                dig1 = xDiGamma_Function(counts1);
                
                
                counts1d = 1.0 * counts1;
                *(ent_1_boots_star + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALLER));
              }

             if(markov_interval[mybootstrap] < 2) {
               counts2 = long(*(chi_counts2_vector_star  + (long)( mybootstrap*(permutations + 1)*nbins   +  permut*nbins  +  bin1  )));
             }
             else {
               counts2 = 0;
               for(bin2=0; bin2 < nbins; bin2++) {
                counts2 += *(count_matrix_star  +  mybootstrap*(permutations + 1)*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin1);
              }
             }
 
              if(counts2 > 0)
              {
                mysign2 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                dig2 = xDiGamma_Function(counts2);
                
                
                counts2d = 1.0 * counts2;
                *(ent_2_boots_star + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts2d / mynumangles)*(log((double)counts2d / mynumangles + SMALLER)); 
              }



            // do doublet entropy here
            for(bin2=0; bin2 < nbins; bin2++)
             {
         
             // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
             counts12 = long(*(count_matrix_star  + (long)( mybootstrap*(permutations + 1)*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 )));
 
              if(counts12 > 0)
              {
                mysign12 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                dig12 = xDiGamma_Function(counts12);
                
                
                counts12d = 1.0 * counts12;
                *(ent_1_2_boots_star + (long)(mybootstrap*(permutations + 1) + permut)) += -1.0 * ((double)counts12d / mynumangles)*(log((double)counts12d / mynumangles + SMALLER)); 
              }
            }
         }
       
    
        }
       }
      """

 

    if(VERBOSE >= 2): 
           print "about to populate count_matrix"
           print "chi counts 1"
           print chi_counts1[bootstrap]
           print "chi counts 2"
           print chi_counts2[bootstrap]

    if(NO_GRASSBERGER == False):
           weave.inline(code, ['num_sims', 'numangles_bootstrap_star', 'nbins', 'bins1_star', 'bins2_star', 'count_matrix_star','bootstrap_sets','permutations','max_num_angles_star','bootstrap_choose','boot_weights_star','offset', 
                 'chi_counts1_vector_star','chi_counts2_vector_star','ent_1_boots_star','ent_2_boots_star','ent_1_2_boots_star','markov_interval','SMALL'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    else:
           weave.inline(code, ['num_sims', 'numangles_bootstrap_star', 'nbins', 'bins1_star', 'bins2_star', 'count_matrix_star','bootstrap_sets','permutations','max_num_angles_star','bootstrap_choose','boot_weights_star','offset', 
                 'chi_counts1_vector_star','chi_counts2_vector_star','ent_1_boots_star','ent_2_boots_star','ent_1_2_boots_star','markov_interval','SMALL','SMALLER'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    if (no_boot_weights != False ):  ## May need to fix this below if it is not correct
           count_matrix /= (bootstrap_choose*max_num_angles_star * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be bootstrap_choose*min(numangles)

    if(VERBOSE >=2):
           print "count matrix first pass:"
           print count_matrix
           print "chi counts 1"
           print chi_counts1[bootstrap]
           print "chi counts 2"
           print chi_counts2[bootstrap]
    

    ### Redundant Sanity checks
    ninj_flat = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
        my_flat = outer(chi_counts1_star[bootstrap] + 0.0 ,chi_counts2_star[bootstrap] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
        if(VERBOSE >=1):
               assert(all(my_flat >= 0))
        my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0]))
        ninj_flat[bootstrap,:,:] = my_flat[:,:]
        #now without the Bayes prior added into the marginal distribution
        #my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
        #my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
        #ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
        nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix_star[:,:,:] == 0)) and (sum(chi_counts1_star) > 0) and (sum(chi_counts2_star) > 0):
           count_matrix[:,:] =  (outer(chi_counts1_star[bootstrap] ,chi_counts2_star[bootstrap] ).flatten()  ) / (numangles_bootstrap_star[0] * 1.0)
    
    if(VERBOSE >=1):
           assert(all(ninj_flat >= 0))
    Pij, PiPj = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
    permutation = 0
    Pij[1:,1:]  = (count_matrix_star[0,permutation,:]).reshape((nbins,nbins)) 
    PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    
    if(VERBOSE >= 1 and permutation == 0):
        print "First Pass p-Star:"
        print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
        print "Marginal Pij, summed over j:\n"
        print sum(Pij[1:,1:],axis=1)
        print "Marginal PiPj, summed over j:\n"
        print sum(PiPj[1:,1:],axis=1)   
        print "Marginal Pij, summed over i:\n"
        print sum(Pij[1:,1:],axis=0)
        print "Marginal PiPj, summed over i:\n"
        print sum(PiPj[1:,1:],axis=0)
    ### end redundant sanity checks
        
        
    #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
    if(VERBOSE >=1):
           assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
           assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
           assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    if(permutations > 0):
           permutation = 1
           Pij[1:,1:]  = (count_matrix[0,permutation,:]).reshape((nbins,nbins)) 
           PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    

           if(VERBOSE >=1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    if(chi_counts1_star==None):
            count_matrix_star = count_matrix
            chi_counts1_vector_star = chi_counts1_vector
            chi_counts1_matrix_star = chi_counts1_matrix
            chi_counts2_vector_star = chi_counts2_vector
            chi_counts2_matrix_star = chi_counts2_matrix
            numangles_bootstrap_vector_star = numangles_bootstrap_vector
            numangles_bootstrap_matrix_star = numangles_bootstrap_matrix
        
    
    #######################################################################################################
    
    dKLtot_dKL1_dKL2 = zeros((bootstrap_sets), float64)
    
    #if(VERBOSE >= 2):
    #    print "nonzero bins"
    #    print nonzero_bins * 1.0
    #    print (count_matrix[bootstrap,0])[nonzero_bins]
    #    print (chi_counts1_matrix[bootstrap,0])[nonzero_bins]
    #    print (chi_counts2_matrix[bootstrap,0])[nonzero_bins]
        

    #######################################################################################################
    
    if(VERBOSE >=1):
           assert(all(chi_counts1 >= 0))
           assert(all(chi_counts2 >= 0))
    
    ##### to work around bug in _star code, in the meantime fill count_matrix again
    if(chi_counts1_star!=None):
           count_matrix[:,:,:] = 0
           #weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose'],
           #      #type_converters = converters.blitz,
           #      compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
           if(NO_GRASSBERGER == False):
              weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
           else:
              weave.inline(code_no_grassberger, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'count_matrix','bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset', 
                 'chi_counts1_vector','chi_counts2_vector','ent_1_boots','ent_2_boots','ent_1_2_boots'],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 support_code=my_support_code)
    #print "count matrix:"
    #print count_matrix
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix[:,:,:] == 0)) and (sum(chi_counts1) > 0) and (sum(chi_counts2) > 0):
           count_matrix[:,:] =  (outer(chi_counts1[bootstrap] ,chi_counts2[bootstrap] ).flatten()  ) / (numangles_bootstrap[0] * 1.0)

    ##################################################################

    ninj_flat = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
        my_flat = outer(chi_counts1[bootstrap] + 0.0 ,chi_counts2[bootstrap] + 0.0).flatten() # have to add 0.0 for outer() to work reliably
        if(VERBOSE >=1):
               assert(all(my_flat >= 0))
        my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0]))
        ninj_flat[bootstrap,:,:] = my_flat[:,:]
        #now without the Bayes prior added into the marginal distribution
        my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
        my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
        ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
        nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    if(VERBOSE >=1):
           assert(all(ninj_flat >= 0))
    
    #print "chi counts 1 after bin filling:"
    #print chi_counts1
    
    # print out the 2D population histograms to disk
    # erase data matrix files after creation to save space; put files in sub-directories so we don't end up with too many files in a directory for linux to handle
    Pij, PiPj = zeros((nbins+1, nbins+1), float64) - 1, zeros((nbins+1, nbins+1), float64) - 1
    mypermutation = 0
    Pij[1:,1:]  = (count_matrix[0,mypermutation,:]).reshape((nbins,nbins))
    PiPj[1:,1:] = (ninj_flat[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
    #sanity checks
    if(VERBOSE >= 2):
        print "Third Pass:"
        print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
        print "Marginal Pij, summed over j:\n"
        print sum(Pij[1:,1:],axis=1)
        print "Marginal PiPj, summed over j:\n"
        print sum(PiPj[1:,1:],axis=1)   
        print "Marginal Pij, summed over i:\n"
        print sum(Pij[1:,1:],axis=0)
        print "Marginal PiPj, summed over i:\n"
        print sum(PiPj[1:,1:],axis=0)
    #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
    if(VERBOSE >= 1):
           assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
           assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
           assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    if permutations > 0:
           mypermutation = 1
           Pij[1:,1:]  = (count_matrix[0,mypermutation,:]).reshape((nbins,nbins))
           PiPj[1:,1:] = (ninj_flat[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
           #sanity checks
           if(VERBOSE >= 1):
                  print "Third Pass: permutation 1"
                  print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
                  print "Marginal Pij, summed over j:\n"
                  print sum(Pij[1:,1:],axis=1)
                  print "Marginal PiPj, summed over j:\n"
                  print sum(PiPj[1:,1:],axis=1)   
                  print "Marginal Pij, summed over i:\n"
                  print sum(Pij[1:,1:],axis=0)
                  print "Marginal PiPj, summed over i:\n"
                  print sum(PiPj[1:,1:],axis=0)
                  #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
           if(VERBOSE >= 1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

           mypermutation = -1
           Pij[1:,1:]  = (count_matrix[0,mypermutation,:]).reshape((nbins,nbins))
           PiPj[1:,1:] = (ninj_flat[0,mypermutation,:]).reshape((nbins,nbins)) / (numangles_bootstrap[0] * 1.0)
           #sanity checks
           if(VERBOSE >= 1):
                  print "Third Pass: permutation -1"
                  print "Sum Pij: "+str(sum(Pij[1:,1:]))+" Sum PiPj: "+str(sum(PiPj[1:,1:]))
                  print "Marginal Pij, summed over j:\n"
                  print sum(Pij[1:,1:],axis=1)
                  print "Marginal PiPj, summed over j:\n"
                  print sum(PiPj[1:,1:],axis=1)   
                  print "Marginal Pij, summed over i:\n"
                  print sum(Pij[1:,1:],axis=0)
                  print "Marginal PiPj, summed over i:\n"
                  print sum(PiPj[1:,1:],axis=0)
                  #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))
           if(VERBOSE >= 1):
                  assert(abs(sum(Pij[1:,1:]) - sum(PiPj[1:,1:])) < nbins)
                  assert(all(abs(sum(Pij[1:,1:],axis=1) - sum(PiPj[1:,1:],axis=1)) < nbins))
                  assert(all(abs(sum(Pij[1:,1:],axis=0) - sum(PiPj[1:,1:],axis=0)) < nbins))

    
    #now sum over bootstraps for 
    Counts_ij = zeros((bootstrap_sets, nbins, nbins),float64)
    for mybootstrap in range(bootstrap_sets):
           mypermutation = 0
           Counts_ij[mybootstrap,:,:] = (count_matrix[mybootstrap,mypermutation,:]).reshape(nbins,nbins)
           #mypermutation = 1 #first independent dataset
           #Counts_ij_ind[mybootstrap,:,:] = (count_matrix[mybootstrap,mypermutation,:]).reshape(nbins,nbins)

    ### Old plot 2d histograms -- now a file with the twoD histogram counts is made if plot_2d_histograms == True  for use by another script
    #if plot_2d_histograms and file_prefix != None:
    #    file_prefix = file_prefix.replace(":", "_").replace(" ", "_")
    #    print file_prefix
    # 
    #    Pij[1:, 0] = PiPj[1:,0] = bins #i.e. bin cut points 0 to 360, nbins in length
    #    Pij[0, 1:] = PiPj[0,1:] = bins #i.e. bin cut points 0 to 360, nbins in length
    #
    #    #Pij[1:,1:]  = average(count_matrix[:,permutation,:], axis=0).reshape((nbins,nbins))
    #    #PiPj[1:,1:] = average(ninj_flat_Bayes[:,permutation,:], axis=0).reshape((nbins,nbins))
    #    Pij[1:,1:] /= sum(Pij[1:,1:])
    #    PiPj[1:,1:] /= sum(PiPj[1:,1:])
    #
    #    res1_str = "_".join(file_prefix.split("_")[:2])
    #    dirname = "plots_of_Pij_PiPj_nsims%d_nstructs%d_p%d_a%s/%s" % (num_sims, numangles_bootstrap[0]/len(which_runs[0]), 0, adaptive_partitioning, res1_str)
    #    utils.mkdir_cd(dirname)
    #    
    #    open("%s_Pij.dat"%file_prefix, "w").write(utils.arr2str2(Pij, precision=8))
    #    open("%s_PiPj.dat"%file_prefix, "w").write(utils.arr2str2(PiPj, precision=8))
    #    #open("%s_Pij_div_PiPj.dat"%file_prefix, "w").write(utils.arr2str2(Pij_div_PiPj, precision=8))
    #    utils.run("R --no-restore --no-save --no-readline %s_Pij.dat < ~/bin/dihedral_2dhist_plots.R" % (file_prefix))
    #    utils.run("R --no-restore --no-save --no-readline %s_PiPj.dat < ~/bin/dihedral_2dhist_plots.R" % (file_prefix))
    #    #utils.run("rm -f %s_*.dat" % file_prefix)
    #
    #    utils.cd("../..")
    


    ####
    #### ***  Important Note: don't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
    #### as we're using the same number of bins in each calculation *****
    ####

    #if(numangles_bootstrap[0] > 0 or nbins >= 6):
    if(True): #small sample stuff turned off for now because it's broken
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):  
        #ent1_boots = sum((chi_counts1_vector_star * 1.0 / numangles_bootstrap_vector_star) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_vector + SMALL) - (1 - 2*(chi_counts1_vector % 2)) / (chi_counts1_vector + 1.0)),axis=2) 
        
        #ent2_boots = sum((chi_counts2_vector_star * 1.0 / numangles_bootstrap_vector_star) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_vector + SMALL) - (1 - 2*(chi_counts2_vector % 2)) / (chi_counts2_vector + 1.0)),axis=2)
        
        #ent12_boots = sum((count_matrix_star * 1.0 /numangles_bootstrap_matrix_star)  \
        #                   * ( log(numangles_bootstrap_matrix) - \
        #                       special.psi(count_matrix + SMALL)  \
        #                       - (1 - 2*(count_matrix % 2)) / (count_matrix + 1.0) \
        #                       ),axis=2)
        # MI = H(1)+H(2)-H(1,2)
        # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
        #print "Numangles Bootstrap Matrix:\n"+str(numangles_bootstrap_matrix)
        
        mutinf_thisdof = ent_1_boots + ent_2_boots  \
                     - ent_1_2_boots

        ent12_boots_for_MI_norm = ent_1_2_boots[:,0] #value to return of MI_norm
    else:
        ent1_boots = sum((chi_counts1_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts1_vector,numangles_bootstrap_vector,permutations),axis=2)

        ent2_boots = sum((chi_counts2_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts2_vector,numangles_bootstrap_vector,permutations),axis=2)

        if(VERBOSE >= 2):
            print "shapes:"+str(ent1_boots)+" , "+str(ent2_boots)+" , "+str(sum((count_matrix + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix,numangles_bootstrap_matrix,permutations_multinomial),axis=2))
        
        mutinf_thisdof = ent1_boots + ent2_boots \
                         -sum((count_matrix + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix,numangles_bootstrap_matrix,permutations),axis=2)

    
    if (VERBOSE >=2): 
           print "Avg Descriptive Mutinf:    "+str(average(mutinf_thisdof[:,0]))
    if(permutations == 0):
           if(VERBOSE >= 2):
                  print "Avg Descriptive MI ind:    "+str(average(mutinf_multinomial,axis=1))
    else:
           if(VERBOSE >=2):
                  print "Avg Descriptive MI ind:    "+str(average(mutinf_thisdof[:,1:]))
                  print "Number of permutations:    "+str(mutinf_thisdof.shape[1] -1)
                  print "distribution of MI ind:    "+str((mutinf_thisdof[:,1:]))
    #Now, if permutations==0, filter according to Bayesian estimate of distribution
    # of mutual information, M. Hutter and M. Zaffalon 2004 (or 2005).
    # Here, we will discard those MI values with p(I | data < I*) > 0.05.
    # Alternatively, we could use the permutation method or a more advanced monte carlo simulation
    # over a Dirichlet distribution to empirically determine the distribution of mutual information of the uniform
    # distribution.  The greater variances of the MI in nonuniform distributions suggest this approach
    # rather than a statistical test against the null hypothesis that the MI is the same as that of the uniform distribution.
    # The uniform distribution or sampling from a Dirichlet would be appropriate since we're using adaptive partitioning.

    #First, compute  ln(nij*n/(ni*nj) = logU, as we will need it and its powers shortly.
    #Here, use Perks' prior nij'' = 1/(nbins*nbins)
    
    if (markov_samples > 0): #if using markov model to get distribution under null hypothesis of independence           
           for bootstrap in range(bootstrap_sets):
                mutinf_multinomial_this_bootstrap = mutinf_multinomial[bootstrap]  # since our markov samples are in axis 1 , and we aren't averaging over bootstraps since their transition matrices are different
                num_greater_than_obs_MI = sum(1.0 * (mutinf_multinomial_this_bootstrap > mutinf_thisdof[bootstrap,0]))

                pvalue_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_multinomial_this_bootstrap.shape[0])
                
                pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_multinomial)
                if(VERBOSE >=1):
                       print "Descriptive P(I=I_markov):"+str(pvalue_multinomial)
                       print "number of markov samples with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later
    elif (permutations == 0):  #then use Bayesian approach to approximate distribution of mutual information given data,prior
        count_matrix_wprior = count_matrix + ninj_prior
        count_matrix_wprior_star = count_matrix_star + ninj_prior #for alternative ensemble, weighting, as in cross terms like p* ln p.
        numangles_bootstrap_matrix_wprior = numangles_bootstrap_matrix + ninj_prior*nbins*nbins
        numangles_bootstrap_vector_wprior = numangles_bootstrap_vector + ninj_prior*nbins*nbins
        numangles_bootstrap_matrix_wprior_star = numangles_bootstrap_matrix_star + ninj_prior*nbins*nbins
        numangles_bootstrap_vector_wprior_star = numangles_bootstrap_vector_star + ninj_prior*nbins*nbins
        Uij = (numangles_bootstrap_matrix_wprior) * (count_matrix_wprior) / (ninj_flat_Bayes)
        logUij = log(Uij) # guaranteed to not have a zero denominator for non-zero prior (non-Haldane prior)

        Jij=zeros((bootstrap_sets, permutations + 1, nbins*nbins),float64)
        Jij = (count_matrix_wprior_star / (numangles_bootstrap_matrix_wprior)) * logUij

        #you will see alot of "[:,0]" following. This means to take the 0th permutation, in case we're permuting data
    
        J = (sum(Jij, axis=-1))[:,0] #sum over bins ij
        K = (sum((count_matrix_wprior_star / (numangles_bootstrap_matrix_wprior_star)) * logUij * logUij, axis=-1))[:,0] #sum over bins ij
        L = (sum((count_matrix_wprior_star / (numangles_bootstrap_matrix_wprior_star)) * logUij * logUij * logUij, axis=-1))[:,0] #sum over bins ij
    
        #we will need to allocate Ji and Jj for row and column sums over matrix elemenst Jij:

        Ji=zeros((bootstrap_sets, permutations + 1, nbins),float64)
        Jj=zeros((bootstrap_sets, permutations + 1, nbins),float64)
        chi_counts_bayes_flat1 = zeros((bootstrap_sets,permutations + 1, nbins*nbins),float64)
        chi_counts_bayes_flat2 = zeros((bootstrap_sets,permutations + 1, nbins*nbins),float64)
    
    
        
        #repeat(chi_counts2[bootstrap] + ni_prior,permutations +1,axis=0)
        for bootstrap in range(bootstrap_sets):
            chi_counts_matrix1 = reshape(resize(chi_counts1[bootstrap] + ni_prior, bootstrap_sets*(permutations+1)*nbins),(bootstrap_sets,permutations+1,nbins))
            chi_counts_matrix2 = reshape(resize(chi_counts2[bootstrap] + ni_prior, bootstrap_sets*(permutations+1)*nbins),(bootstrap_sets,permutations+1,nbins))
        
            if(VERBOSE >= 2):
                print "chi counts 1:" + str(chi_counts1[bootstrap])
                print "chi counts 2:" + str(chi_counts2[bootstrap])

            mycounts_mat = reshape(count_matrix[bootstrap],(nbins,nbins))
            if(VERBOSE >= 2):
                print "counts:\n" + str(mycounts_mat)
            if(VERBOSE >=1):
                   if(VERBOSE >= 2):
                          print sum(mycounts_mat,axis=1)
                          print sum(mycounts_mat,axis=0)
                          print chi_counts1[bootstrap]
                          print chi_counts2[bootstrap]
                   assert(all(abs(chi_counts1[bootstrap] - sum(mycounts_mat,axis=1)) < nbins))
                   assert(all(abs(chi_counts2[bootstrap] - sum(mycounts_mat,axis=0)) < nbins))

            #now we need to reshape the marginal counts into a flat "matrix" compatible with count_matrix
            # including counts from the prior
            if(VERBOSE >= 2):
                print "chi_counts2 shape:"+str(shape(chi_counts2))
            #print "chi_counts2[bootstrap]+ni_prior shape:"+str((chi_counts2[bootstrap] + ni_prior).shape)
            
            chi_counts_bayes_flat2[bootstrap,0,:] = repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
            #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
            #handling the slower-varying index will be a little bit more tricky
            chi_counts_bayes_flat1[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1[bootstrap] + ni_prior, nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
            #we will also need to calculate row and column sums Ji and Jj:
            Jij_2D_boot = reshape(Jij[bootstrap,0,:],(nbins,nbins))
            #Ji is the sum over j for row i, Jj is the sum over i for column j, j is fastest varying index
            #do Ji first
            Ji[bootstrap,0,:] = sum(Jij_2D_boot, axis=1)
            Jj[bootstrap,0,:] = sum(Jij_2D_boot, axis=0)


        #ok, now we calculated the desired quantities using the matrices we just set up
    
    
        numangles_bootstrap_wprior = numangles_bootstrap + ninj_prior*nbins*nbins
        
        M = (sum((1.0/(count_matrix_wprior + SMALL) - 1.0/chi_counts_bayes_flat1 -1.0/chi_counts_bayes_flat2 \
                  + 1.0/numangles_bootstrap_matrix_wprior) \
                 * count_matrix_wprior * logUij, axis=2))[:,0]

        Q = (1 - sum(count_matrix_wprior * count_matrix_wprior / ninj_flat_Bayes, axis=2))[:,0]

        #####DEBUGGING STATEMENTS
        #print "shapes"
        #print "Ji:   "+str(shape(Ji))
        #print "Jj:   "+str(shape(Jj))
        #print "chi counts matrix 1:   "+str(chi_counts_matrix1)
        #print "chi counts matrix 2:   "+str(chi_counts_matrix2)
        #print "numangles bootstrap wprior:   "+str(shape(numangles_bootstrap_wprior))

        P = (sum((numangles_bootstrap_vector_wprior) * Ji * Ji / chi_counts_matrix1, axis=2) \
             + sum(numangles_bootstrap_vector_wprior * Jj * Jj / chi_counts_matrix2, axis=2))[:,0]

        #####DEBUGGING STATEMENTS
        #print "ni prior:\n"+str(ni_prior)
        #print "numangles bootstrap wprior:\n"+str(numangles_bootstrap_wprior)
        #print "chi_counts_bayes_flat1:\n"+str(chi_counts_bayes_flat1)
        #print "chi_counts_bayes_flat2:\n"+str(chi_counts_bayes_flat2)

        #print intermediate values

        #print "J:"+str(J)
        #print "K:"+str(K)
        #print "L:"+str(L)
        #print "M:"+str(M)
        #print "P:"+str(P)
        #print "Q:"+str(Q)

        #Finally, we are ready to calculate moment approximations for p(I | Data)
        E_I_mat = ((count_matrix_wprior_star) / (numangles_bootstrap_matrix_wprior_star * 1.0)) \
                  * (  special.psi(count_matrix_wprior + 1.0) \
                     - special.psi(chi_counts_bayes_flat1 + 1.0) \
                     - special.psi(chi_counts_bayes_flat2 + 1.0) \
                     + special.psi(numangles_bootstrap_matrix_wprior + 1.0))
        #print "\n"
        if(VERBOSE >= 2):
            print "E_I_mat:\n"
            print E_I_mat
        E_I = (sum(E_I_mat,axis = 2))[:,0] # to get rid of permutations dimension 
        #print "Stdev of E_I over bootstraps:\n"
        #print stats.std(average(sum(E_I_mat,axis = 2), axis = 1))
        #print "estimated pop std of E_I over bootstraps, dividing by sqrt n"
        #print stats.std(average(sum(E_I_mat,axis = 2), axis = 1)) / sqrt(num_sims)
        #print "Stdev_mat:\n"
        #print sqrt(((K - J*J))/(numangles_bootstrap_wprior + 1) + (M + (nbins-1)*(nbins-1)*(0.5 - J) - Q)/((numangles_bootstrap_wprior + 1)*(numangles_bootstrap_wprior + 2)))

        Var_I = abs( \
             ((K - J*J))/(numangles_bootstrap_wprior + 1) + (M + (nbins-1)*(nbins-1)*(0.5 - J) - Q)/((numangles_bootstrap_wprior + 1)*(numangles_bootstrap_wprior + 2))) # a different variance for each bootstrap sample

        #now for higher moments, leading order terms

        #E_I3 = (1.0 / (numangles_bootstrap_wprior * numangles_bootstrap_wprior) ) \
        #       * (2.0 * (2 * J**3 -3*K*J + L) + 3.0 * (K + J*J - P))

        #E_I4 = (3.0 / (numangles_bootstrap_wprior * numangles_bootstrap_wprior)) * ((K - J*J) ** 2)

        #convert to skewness and kurtosis (not excess kurtosis)

        
        if(VERBOSE >= 2):
               print "Moments for Bayesian p(I|Data):"
               print "E_I:                   "+str(E_I)
               print "Var_I:                 "+str(Var_I)
        #print "Stdev_I:               "+str(sqrt(Var_I))
        #print "skewness:              "+str(E_I3/ (Var_I ** (3/2)) )
        #print "kurtosis:              "+str(E_I4/(Var_I ** (4/2)) )

        
        
        def Edgeworth_pdf(u1,u2,u3,u4):
            #convert moments to cumulants for u0=1, u1=0, u2=1, central moments normalized to zero mean and unit variance
            skewness = u3 / (u2 ** (3/2))
            excess_kurtosis = u4 / (u2 ** (4/2)) - 3
            s = sqrt(u2)
            k3 = skewness
            k4 = excess_kurtosis
            
            return lambda x: stats.norm.pdf((x-u1)/s) * (1.0 + (1.0/6)*k3*(special.hermitenorm(3)((x-u1)/s)) \
                              + (1.0/24)*k4*(special.hermitenorm(4)((x-u1)/s)))

        def Edgeworth_quantile(crit_value,u1,u2):
            #only Gaussian for now for speed #,u3,u4):
            #convert critical value to z-score
            #func = Edgeworth_pdf(u1,u2,u3,u4)
            #print func
            #normalization = integrate.quad( func, -100*sqrt(u2), 100*sqrt(u2) )[0]
            #integral = integrate.quad( func, -integrate.inf, crit_value)[0] #output just the definite integral
            #print "integral:              "+str(integral)
            #print "normalization:         "+str(normalization)
            #pval = abs(integral)
            #print "plusminus_sigma_check:"+str(integrate.quad( func, u1-sqrt(u2), u1+sqrt(u2))[0]/normalization)
            pval_gauss = stats.norm.cdf((crit_value - u1)/sqrt(u2))
            #print "p-value Bayes Edgeworth: "+str(pval)
            if(VERBOSE>=2):
                   print "p-value Bayes Gaussian:  "+str(pval_gauss)
            return pval_gauss

        #if(bootstrap_sets > 1):
        #    numangles_bootstrap_avg_wprior = average(numangles_bootstrap_wprior)
        #else:
        #    numangles_bootstrap_avg_wprior = numangles_bootstrap_wprior[0]

        if E_I_uniform is None:
            E_I_uniform = average(special.psi(numangles_bootstrap_wprior / (nbins * nbins) + 1) \
                                  - special.psi(numangles_bootstrap_wprior / nbins + 1) \
                                  - special.psi(numangles_bootstrap_wprior / nbins + 1) \
                                  + special.psi(numangles_bootstrap_wprior + 1))

        #####DEBUGGING STATEMENTS
        #print "Edgeworth pdf for E_I and E_I_uniform:"


        
        #print Edgeworth_pdf( E_I, Var_I, E_I3, E_I4)(E_I)
        #print Edgeworth_pdf( E_I, Var_I, E_I3, E_I4)(E_I_uniform)

        
        
        #print "E_I_uniform            :"+str(E_I_uniform)
        #print "E_I_multinomial_constr :"+str(E_I_multinomial)
        #now, determine the probability given the data that the true mutual information is
        #greater than that expected for the uniform distribution plus three sigma

        #pvalue for false positive
        #lower pvalue is better
        ##############################
        
        
        for bootstrap in range(bootstrap_sets):
            #for speed, use shortcuts for obviously significant or obviously insignificant mutual information
            if (E_I[bootstrap] < E_I_multinomial[bootstrap]):
                pvalue[bootstrap] = 1.0
            else:
                if(E_I[bootstrap] > E_I_multinomial[bootstrap] + 10 * sqrt(Var_I[bootstrap])):
                    pvalue[bootstrap] = 0.0
                else:
                    pvalue[bootstrap] = Edgeworth_quantile(E_I_multinomial[bootstrap], E_I[bootstrap], Var_I[bootstrap]) # , E_I3[bootstrap], E_I4[bootstrap])
            if(VERBOSE >=2 ):
                   print "Bayesian P(I<E[I]mult) bootstrap sample:"+str(bootstrap)+" = "+str(pvalue[bootstrap])
            num_greater_than_obs_MI = sum(1.0 * (mutinf_multinomial[bootstrap] > mutinf_thisdof[bootstrap,0]))
            if num_greater_than_obs_MI < 1:
                num_greater_than_obs_MI = 0
            pvalue_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_multinomial[bootstrap].shape[0])
            if (VERBOSE >= 2):
                   #print "Mutinf Multinomial Shape:"+str(mutinf_multinomial.shape)
                   print "Num Ind Greater than Obs:"+str(num_greater_than_obs_MI)
                   print "bootstrap: "+str(bootstrap)+" Descriptive P(I=I_mult):"+str(pvalue_multinomial)
            pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_multinomial)
            
        if(VERBOSE >= 2):
               print "Max pvalue             :"+str(pvalue[bootstrap])
               #print "integrate check: "+str(Edgeworth_quantile(integrate.inf,  E_I, Var_I)) #, E_I3, E_I4))
        #could use a monte carlo simulation to generate distribution of MI of uniform distribution over adaptive partitioning
        #this would be MI of independent variables
        #for non-Bayesian significance test against null hypothesis of independence
        #but not at the present time
           
    else:  #use permutation test to filter out true negatives, possibly in addition to the Bayesian filter above
        
        #pvalue is the fraction of mutinf values from samples of permuted data that are greater than the observed MI
        #pvalue for false negative
        #lower pvalue is better
        if(permutations > 0): #otherwise, keep pvalue as 0 for now, use wilcoxon signed ranks test at the end
            for bootstrap in range(bootstrap_sets):
                num_greater_than_obs_MI = sum(mutinf_thisdof[bootstrap,1:] > mutinf_thisdof[bootstrap,0])
                pvalue[bootstrap] += num_greater_than_obs_MI * 1.0 / permutations
                if(VERBOSE >=2):
                       print "number of permutations with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later
        else:
            for bootstrap in range(bootstrap_sets):
                num_greater_than_obs_MI = sum(1.0 * (mutinf_multinomial[bootstrap] > mutinf_thisdof[bootstrap,0]))

                pvalue_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_multinomial[bootstrap].shape[0])
                
                pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_multinomial)
                if(VERBOSE >=2):
                       print "Descriptive P(I=I_mult):"+str(pvalue_multinomial)
                       print "number of permutations with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later

    
    #print mutinf_thisdof
    if(bootstrap_sets > 1 and calc_variance == True):
       if(permutations > 0):
           var_mi_thisdof = (vfast_cov_for1D_boot(reshape(mutinf_thisdof[:,0],(mutinf_thisdof.shape[0],1))) - average(mutinf_thisdof[:,1:],axis=1))[0,0]
       else:
           var_mi_thisdof = (vfast_cov_for1D_boot(reshape(mutinf_thisdof[:,0],(mutinf_thisdof.shape[0],1))))[0,0]
    else:
       var_mi_thisdof = sum(Var_I)
    if(VERBOSE >=2):
           print "var_mi_thisdof: "+str(var_mi_thisdof)+"\n"
    if(calc_mutinf_between_sims == "no" or num_pair_runs <= 1):
        mutinf_thisdof_different_sims= zeros((bootstrap_sets,permutations_sequential + 1),float64)
        return mutinf_thisdof, var_mi_thisdof , mutinf_thisdof_different_sims, 0, average(mutinf_multinomial, axis=1), zeros((bootstrap_sets),float64), pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots_for_MI_norm
    #########################################################################################################
    ##
    ##  Now we will calculate mutinf between torsions between different sims for the undersampling correction 
    ##
    ##
    ##  For now, we will use the same undersampling correction for both regular and "_star" terms
    ##  
    ## 
    #########################################################################################################

    count_matrix_sequential[:,:,:,:] = 0
    
    #max_num_angles = int(max(numangles))
    max_num_angles = int(min(numangles))

    if(permutations == 0):
        if(mutinf_multinomial_sequential is None or adaptive_partitioning == 0):
            E_I_multinomial_sequential, Var_I_multinomial_sequential, E_I3_multinomial_sequential, E_I4_multinomial_sequential, Var_I_runs_multinomial_sequential, mutinf_multinomial_sequential, var_mutinf_multinomial_sequential = \
                                        calc_mutinf_multinomial_constrained(nbins_cor,chi_counts_sequential1.copy(),chi_counts_sequential2.copy(),adaptive_partitioning, bootstraps = bootstrap_sets )
            if VERBOSE >= 2:
                   print "shape of mutinf_multinomial_sequential: "+str(shape(mutinf_multinomial_sequential))
    else:
        mutinf_multinomial_sequential = var_mutinf_multinomial_sequential = zeros((bootstrap_sets,1),float64)
    
    #no_weights = False
    if (weights is None):
           weights = ones((num_sims, max_num_angles), float64)
    #       no_weights = True
    if VERBOSE >= 3:       
           print "weights"
           print weights
           print "bins1_sequential"
           print bins1_sequential
           print "bins2_sequential"
           print bins2_sequential
    code = """
     // weave7
     #include <math.h>
     
     int mynumangles, mynumangles1, mynumangles2, run1, run2, fetch1, fetch2, bin1, bin2, permut, anglenum = 0;
     double weight, weights_sum = 0 ;
     for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
       weights_sum = 0; // accumulator for normalization
       for(int which_run_pair=0; which_run_pair < num_pair_runs; which_run_pair++) {
         mynumangles1 = *(numangles + (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2 + 0)));
         mynumangles2 = *(numangles + (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2 + 1)));
         if(mynumangles1 <= mynumangles2) mynumangles = mynumangles1; else mynumangles = mynumangles2;
         run1 = (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2  + 0));
         run2 = (*(pair_runs + mybootstrap*num_pair_runs + which_run_pair*2 + 1));
         for (permut=0; permut < permutations_sequential + 1; permut++) {
           for (anglenum=0; anglenum< mynumangles; anglenum++) {
             bin1 = *(bins1_sequential  +  run1*max_num_angles  +  anglenum);
             bin2 = *(bins2_sequential  + permut*num_sims*max_num_angles + run2*max_num_angles  +  anglenum);
             //printf("bin1: %i bin2: %i \\n",bin1,bin2); 
             // take effective weight as product of the individual runs weights, will later renormalize
             weight = *(weights + run1*max_num_angles + anglenum) * (*(weights + run2*max_num_angles + anglenum));  
             weights_sum += weight;
             *(count_matrix_sequential  +  mybootstrap*num_pair_runs*(permutations_sequential + 1)*nbins_cor*nbins_cor + which_run_pair*(permutations_sequential + 1)*nbins_cor*nbins_cor  +  permut*nbins_cor*nbins_cor  +  bin1*nbins_cor +   bin2) += 1.0 * weight ;
           }
           // now normalize so that sum of snapshots weights equals numangles
           for(bin1 = 0; bin1 < nbins_cor; bin1++) {
             for(bin2 = 0; bin2 < nbins_cor; bin2++) {
                *(count_matrix_sequential  +  mybootstrap*num_pair_runs*(permutations_sequential + 1)*nbins_cor*nbins_cor + which_run_pair*(permutations_sequential + 1)*nbins_cor*nbins_cor  +  permut*nbins_cor*nbins_cor  +  bin1*nbins_cor +   bin2) *= (mynumangles / weights_sum);
                }
           }
           weights_sum = 0; //reset 
         }
        }
       }
    """                                           
                                          
                                           
    weave.inline(code, ['num_sims', 'numangles','max_num_angles', 'nbins_cor', 'bins1_sequential', 'bins2_sequential', 'count_matrix_sequential','pair_runs','num_pair_runs','bootstrap_sets','permutations_sequential','weights'],
                 #type_converters = converters.blitz,
                  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
    #if (no_weights != False):
    #count_matrix_sequential /= (min(numangles) * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be min(numangles)
    #for bootstrap in range(bootstrap_sets):
    #    for which_run_pair
    #    count_matrix_sequential[bootstrap,:,:] /= min((numangles[pair_runs[bootstrap,0]], numangles[pair_runs[bootstrap,1]]))
    

                                                                              
                                     
     
    
    #logU_sequential[:,:,:] = 0
    #U_sequential = count_matrix_sequential / (ninj_flat_Bayes_sequential + SMALL)
    if(VERBOSE >= 2):
        print "count_matrix_sequential\n"
        print count_matrix_sequential[0,0,0:]
        print "chi counts sequential1"
        print chi_counts_sequential1[0]
        print "chi counts sequential2"
        print chi_counts_sequential2[0]
    ##### DEBUGGING STUFF ##############################
    
    #print "shape of count_matrix_sequential\n"
    #print shape(count_matrix_sequential)
    #print "chi pop hist sequential1\n"
    #print chi_pop_hist_sequential1[pair_runs[0,0,0]]*numangles[pair_runs[0,0,0]]
    #print "chi pop hist sequential1\n"
    #print chi_pop_hist_sequential1[0]*numangles[0]
    #print "sum"
    #print sum(chi_pop_hist_sequential1[pair_runs[0,0,0]]*numangles[pair_runs[0,0,0]])
    #print "chi pop hist sequential2\n"
    #print chi_pop_hist_sequential2[pair_runs[0,0,1]]*numangles[pair_runs[0,0,1]]
    #print "chi pop hist sequential2\n"
    #print chi_pop_hist_sequential2[1]*numangles[1]
    #print "sum"
    #print sum(chi_pop_hist_sequential2[1]*numangles[1])
    #print "before log\n"
    #logU_sequential = log(U_sequential + SMALL)
                                                                              
    #print "min angles boot pair runs vector shape:"
    #print  shape(min_angles_boot_pair_runs_vector)
    ######################################################


    #take entropy for sims in chi counts sequential, summing over sims in each bootstrap
    ent1_boots_sims = zeros((bootstrap_sets, num_pair_runs, permutations_sequential + 1),float64)
    ent2_boots_sims = zeros((bootstrap_sets, num_pair_runs, permutations_sequential + 1),float64)

    if(NO_GRASSBERGER == False):
      for bootstrap in range(bootstrap_sets):
        for which_pair in range(num_pair_runs):
            #pick index 0 for axis=2 at the end because the arrays are over 
            
            myent1                                \
                                                  =\
                                                  sum((chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) \
                                                      * (log(min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) \
                                                         - special.psi( \
                chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] + SMALL) \
                                                         - (1 - 2*( \
                int32(chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]]) % 2)) / (chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] + 1.0)),axis=0) #sum over bins
            
            #print "ent1 boots sims thispair shape: "
            #print myent1.shape
            #print "ent1 boots: "+str(myent1)

            ent1_boots_sims[bootstrap,which_pair,:] = myent1
            
            myent2                                \
                                                  =\
                                                  sum((chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) \
                                                      * (log(min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) \
                                                         - special.psi( \
                chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] + SMALL) \
                                                         - (1 - 2*( \
                int32(chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]]) % 2)) / (chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] + 1.0)),axis=0) #sum over bins


            #print "ent1 boots sims thispair shape: "
            #print myent2.shape
            #print "ent1 boots: "+str(myent2)

            ent2_boots_sims[bootstrap,which_pair,:] = myent2
            #ent1_boots = average(ent1_boots_sims,axis=1) # avg over sims in each bootstrap
            #ent2_boots = average(ent2_boots_sims,axis=1) # avg over sims in each bootstrap


    else:  #do not use Grassberger corrected entropies
      for bootstrap in range(bootstrap_sets):
        for which_pair in range(num_pair_runs):
            #pick index 0 for axis=2 at the end because the arrays are over 
            
            myent1                                \
                                                  =\
                                                  sum((chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) \
                                                      * (log((chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,0],0,:]) + SMALLER)),axis=0) #sum over bins
            
            #print "ent1 boots sims thispair shape: "
            #print myent1.shape
            #print "ent1 boots: "+str(myent1)

            ent1_boots_sims[bootstrap,which_pair,:] = myent1
            
            myent2                                \
                                                  =\
                                                  sum((chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) \
                                                      * (log((chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]] \
                                                       * 1.0 / min_angles_boot_pair_runs_vector[bootstrap,pair_runs[bootstrap,which_pair,1],0,:]) + SMALLER )),axis=0) #sum over bins


            #print "ent1 boots sims thispair shape: "
            #print myent2.shape
            #print "ent1 boots: "+str(myent2)

            ent2_boots_sims[bootstrap,which_pair,:] = myent2
            #ent1_boots = average(ent1_boots_sims,axis=1) # avg over sims in each bootstrap
            #ent2_boots = average(ent2_boots_sims,axis=1) # avg over sims in each bootstrap



    #print "ent1 boots sims shape:"
    #print shape(ent1_boots_sims)
    #print "ent1 boots sims :"
    #print ent1_boots_sims
    #print "ent2 boots sims :"
    #print ent2_boots_sims
    ent1_boots = ent1_boots_sims
    ent2_boots = ent2_boots_sims
    
    
    
    # MI = H(1)+H(2)-H(1,2)
    # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI

    ent12_boots = sum((count_matrix_sequential * 1.0 /min_angles_boot_pair_runs_matrix)  \
                           * ( log(min_angles_boot_pair_runs_matrix) - \
                               special.psi(count_matrix_sequential + SMALL)  \
                               - (1 - 2*(int16(count_matrix_sequential) % 2)) / (count_matrix_sequential + 1.0) \
                               ),axis=3)

    #print "ent12_boots shape:"
    #print ent12_boots.shape
    #print "ent12_boots:      "
    #print ent12_boots
    
    mutinf_thisdof_different_sims_bootstrap_pairs = ent1_boots + ent2_boots - ent12_boots
                                                                                          
    #print "bootstrap sets:"+str(bootstrap_sets)
    #print "permutations:  "+str(permutations)
    #print "num pair runs: "+str(num_pair_runs)
    #print "mutinf sim1 sim2 shape:"

    #print mutinf_thisdof_different_sims_bootstrap_pairs.shape
    if(permutations == 0):
           if(VERBOSE >=2):
                  print "mutinf_multinomial_difsm:"+str(average(mutinf_multinomial_sequential))
                  print "stdev mutinf multi difsm:"+str(sqrt(average(var_mutinf_multinomial_sequential)))
    if(VERBOSE >=2):
           print "avg mutinf sim1 sim2        :"+str(average(mutinf_thisdof_different_sims_bootstrap_pairs[0,:,0]))
    mutinf_thisdof_different_sims = average(mutinf_thisdof_different_sims_bootstrap_pairs, axis=1) #average over pairs of runs in each bootstrap sample
    #now the nbins_cor*nbins_cor dimensions and num_pair_runs dimensions have been removed, leaving only bootstraps and permutations
    #print "mutinf values between sims for over original data for bootstrap=0 and permuted data\n"
    #print mutinf_thisdof_different_sims[0,0]
    #print mutinf_thisdof_different_sims[0,1:]
    if(VERBOSE >= 2):
        if bootstrap_sets > 1:
            print "mutinf values between sims for over original data for bootstrap=1 and permuted data\n"
            print mutinf_thisdof_different_sims[1,0]
            print mutinf_thisdof_different_sims[1,1:]
    #just first pair for debugging 
    #mutinf_thisdof_different_sims = mutinf_thisdof_different_sims_bootstrap_pairs[:,0]
    if(permutations > 0):
        var_ind_different_sims_pairs=zeros((bootstrap_sets,num_pair_runs),float64)
        #for bootstrap in range(bootstrap_sets):
        #   var_ind_different_sims_pairs[bootstrap,:] = vfast_var_for1Dpairs_ind(mutinf_thisdof_different_sims_bootstrap_pairs[bootstrap,:,1:])
        var_ind_different_sims_pairs = vfast_var_for1Dpairs_ind(mutinf_thisdof_different_sims_bootstrap_pairs)
        if(VERBOSE >= 2):
               print "variance of mutinf between sims for orig data and permuts"
               print var_ind_different_sims_pairs
        var_ind_different_sims = average(var_ind_different_sims_pairs, axis=1) #average variance over pairs of runs
        #NOTE: var_ind_different_sims is of shape = (bootstrap_sets), while var_mutinf_multinomial_sequential is only 1 number
        # But this will be used in array context -- for permutations == 0, it will be implicity repeated in calc_excess_mutinf
        # for permutations > 0, it is already of shape = (bootstrap_sets)
        var_mutinf_multinomial_sequential = var_ind_different_sims 
        #print "variance of mutinf between sims for orig data and permuts"     
        
    if calc_variance == False:
       #if VERBOSE: print "   mutinf = %.5f," % (mutinf_thisdof),
       return mutinf_thisdof, Var_I , mutinf_thisdof_different_sims, var_mutinf_multinomial_sequential, average(mutinf_multinomial,axis=1),average(mutinf_multinomial_sequential,axis=1), pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots_for_MI_norm
    # print out the number of nonzero bins
    if VERBOSE >=2:
     #  num_nonzero_jointpop, num_nonzero_pxipyj = len(nonzero(av_joint_pop)[0]), len(nonzero(pxipyj_flat)[0]), 
     #  print "   nonzero bins (tot=%d): jointpop=%d, pxipyj=%d, combined=%d" % (nbins*nbins, num_nonzero_jointpop, num_nonzero_pxipyj, len(rnonzero_bins[nonzero_bins==True]))
       print "   mutinf this degree of freedom "+str(mutinf_thisdof)
    ##### No longer used #######################################################################
    # calculate the variance of the mutual information using its derivative and error-propagation
    #pxi_plus_pyj_flat = add.outer(chi_pop_hist1, chi_pop_hist2).flatten()
    #deriv_vector = 1 + logU[0,:] - (pxi_plus_pyj_flat[0,:]) * U
    ##############################################################################################
    
    return mutinf_thisdof, var_mi_thisdof, mutinf_thisdof_different_sims, var_mutinf_multinomial_sequential, average(mutinf_multinomial,axis=1), average(mutinf_multinomial_sequential,axis=1), pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots_for_MI_norm



######################################################################


### calculation of independent information using permutations
independent_mutinf_thisdof = None

def calc_excess_mutinf(chi_counts1, chi_counts2, bins1, bins2, chi_counts_sequential1, chi_counts_sequential2, bins1_sequential, bins2_sequential, num_sims, nbins, numangles_bootstrap,numangles, sigalpha, permutations, bootstrap_choose, calc_variance=False, which_runs=None, pair_runs=None, calc_mutinf_between_sims = "yes", markov_samples=0, chi_counts1_markov=None, chi_counts2_markov=None, ent1_markov_boots=None, ent2_markov_boots=None, bins1_markov=None, bins2_markov=None, file_prefix=None, plot_2d_histograms=False, adaptive_partitioning = 0, bins1_slowest_timescale = None, bins2_slowest_timescale = None, bins1_slowest_lagtime = None, bins2_slowest_lagtime= None, lagtime_interval=None, boot_weights = None, weights = None, num_convergence_points = None, cyclic_permut = False):

    mutinf_tot_thisdof, var_mi_thisdof, mutinf_tot_thisdof_different_sims, var_ind_different_sims, mutinf_multinomial, mutinf_multinomial_sequential, pvalue, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, ent12_boots \
        = calc_mutinf_corrected(chi_counts1,chi_counts2, bins1, bins2, chi_counts_sequential1, chi_counts_sequential2, bins1_sequential, bins2_sequential, num_sims, nbins, numangles_bootstrap, numangles, calc_variance=calc_variance, bootstrap_choose=bootstrap_choose, permutations=permutations,which_runs=which_runs,pair_runs=pair_runs, calc_mutinf_between_sims=calc_mutinf_between_sims, markov_samples=markov_samples, chi_counts1_markov=chi_counts1_markov, chi_counts2_markov=chi_counts2_markov,ent1_markov_boots=ent1_markov_boots, ent2_markov_boots=ent2_markov_boots, bins1_markov=bins1_markov, bins2_markov=bins2_markov, file_prefix=file_prefix, plot_2d_histograms=plot_2d_histograms, adaptive_partitioning = adaptive_partitioning, bins1_slowest_timescale=bins1_slowest_timescale, bins2_slowest_timescale=bins2_slowest_timescale, bins1_slowest_lagtime = bins1_slowest_timescale, bins2_slowest_lagtime = bins2_slowest_timescale, lagtime_interval = lagtime_interval, boot_weights = boot_weights, weights = weights)
    
    
    
    #need to filter using p-value: for , use p-value (descriptive) to pick which terms to discard from average.
    #for bootstrap_sets=1, use p-value (Bayes) to filter mutinf values
    
    
    bootstrap_sets = mutinf_tot_thisdof.shape[0]
    sd_ind_different_sims = sqrt(var_ind_different_sims)
    if(VERBOSE >= 2):
           print "mutinf_tot_thisdof shape:"+str(mutinf_tot_thisdof.shape)
           print "mutinf_tot_thisdof different sims shape:"+str(mutinf_tot_thisdof_different_sims.shape)
           print "tot_mutinfs from bootstrap samples\n:"+str(mutinf_tot_thisdof[:,0])
    if(permutations > 0 ):
           if(VERBOSE >= 2):   
                  print "independent mutinf averaged over permutations\n:"+str(average(mutinf_tot_thisdof[:,1:], axis=1))
                  print "independent mutinf different sims averaged over permutations\n:"+str(average(mutinf_tot_thisdof_different_sims[:,1:], axis=1))
    num_pair_runs = pair_runs.shape[0]
    independent_mutinf_thisdof                = zeros((bootstrap_sets),float64)
    corrections_mutinf_thisdof                = zeros((bootstrap_sets),float64)
    independent_mutinf_thisdof_different_sims = zeros((bootstrap_sets),float64)
    if(permutations == 0):
        independent_mutinf_thisdof[:] = mutinf_multinomial #average over samples not bootstraps already performed before it is returned
        independent_mutinf_thisdof_different_sims[:] = average(mutinf_multinomial_sequential)  #replicate up to bootstraps, average over samples not set of pairs of sims already performed before it is returned ### average(mutinf_multinomial_sequential,axis=1) #average over samples not bootstraps here
    else:
        if(cyclic_permut == True):
               independent_mutinf_thisdof = zeros(bootstrap_sets)
               for mybootstrap in range(mutinf_tot_thisdof.shape[0]):
                      ind_mutinf_thisdof_myboot = mutinf_tot_thisdof[mybootstrap,1:]
                      try:
                             independent_mutinf_thisdof[mybootstrap] = average(ind_mutinf_thisdof_myboot)
                      except:
                             independent_mutinf_thisdof[mybootstrap] = 0
                      if(independent_mutinf_thisdof[mybootstrap] < 0):
                                independent_mutinf_thisdof[mybootstrap] = 0
        else:
               independent_mutinf_thisdof = average(mutinf_tot_thisdof[:,1:], axis=1) # average over permutations
        independent_mutinf_thisdof_different_sims = average(mutinf_tot_thisdof_different_sims[:,1:], axis=1) #avg over permutations
    
    sd_ind_different_sims = sqrt(var_ind_different_sims)
            
    #print independent_mutinf_thisdof
    #print average(independent_mutinf_thisdof)
    #if(permutations > 0):
    #    independent_mutinf_thisdof_different_sims = average(mutinf_tot_thisdof_different_sims[:,1:],axis=1)
                
    #print "ind_mutinfs:"+str(independent_mutinf_thisdof)
    #print "tot_mutinfs_diff_sims:"+str(mutinf_tot_thisdof_different_sims)
    uncorrected_mutinf_thisdof = array(mutinf_tot_thisdof[:,0],float64)
    print "uncorrected mutinfs thisdof:"+str(uncorrected_mutinf_thisdof)
    
    if(markov_samples > 0  ):
        excess_mutinf_thisdof = mutinf_tot_thisdof[:,0] #no correction to mutinf value for markov model
        corrections_mutinf_thisdof  += independent_mutinf_thisdof
    elif(sigalpha < 1.0): #if we're doing statistics at all...
        excess_mutinf_thisdof = mutinf_tot_thisdof[:,0] - independent_mutinf_thisdof
        corrections_mutinf_thisdof  += independent_mutinf_thisdof
    else:
        excess_mutinf_thisdof = mutinf_tot_thisdof[:,0]
    excess_mutinf_thisdof_different_sims = mutinf_tot_thisdof_different_sims[:,0] - independent_mutinf_thisdof_different_sims
    corrections_mutinf_thisdof += (mutinf_tot_thisdof_different_sims[:,0] - independent_mutinf_thisdof_different_sims )
    excess_mutinf_thisdof_different_sims -= sd_ind_different_sims
    #last term is for high pass filter, will be added back later
    #could consider having a different excess_mutinf_thisdof_different_sims for each bootstrap sample depending on the correlations between the runs it has in it
    #print "excess_mutinf_thisdof_different_sims:"+str(excess_mutinf_thisdof_different_sims)
    nonneg_excess_thisdof = logical_and((excess_mutinf_thisdof_different_sims > 0), (excess_mutinf_thisdof > 0))
    #print "nonneg excess thisdof: "+str(nonneg_excess_thisdof)
    old_excess_mutinf_thisdof = excess_mutinf_thisdof.copy()
    excess_mutinf_thisdof_different_sims += sd_ind_different_sims  #adding back the cutoff value
    
    #only do this next step if markov model is not used
    if(markov_samples == 0):
           excess_mutinf_thisdof[nonneg_excess_thisdof] -= excess_mutinf_thisdof_different_sims[nonneg_excess_thisdof] # subtract out high-pass-filtered mutinf for torsions in different sims
           
    #remember to zero out the excess mutinf thisdof from different sims that were below the cutoff value
    excess_mutinf_thisdof_different_sims[excess_mutinf_thisdof_different_sims <= sd_ind_different_sims ] = 0
    #print "corrected excess_mutinfs:"+str(excess_mutinf_thisdof)
    test_stat = 0
    mycutoff = 0
    sigtext = " "

    #now filter out those with too high of a probability for being incorrectly kept
    #print "pvalues (Bayes)"
    #print pvalue
    pvalue_toolow = pvalue > sigalpha
    if(sum(pvalue_toolow) > 0):
           if(VERBOSE >= 2):   
                  print "one or more values were not significant!"
    #remove values less than zero, as pairwise mutual information should be zero or greater
    excess_mutinf_thisdof[excess_mutinf_thisdof < 0] = 0
    mutinf_tot_gt_eq_0 = array(mutinf_tot_thisdof[:,0],float64)
    mutinf_tot_gt_eq_0[ mutinf_tot_gt_eq_0 < 0 ] = 0 #only values greater than or equal to zero
    #remove values that are not significant
    excess_mutinf_thisdof *= (1.0 - pvalue_toolow * 1.0) #zeros elements with pvalues that are below threshold
    
    mutinf_tot_thisdof_for_MI_norm = mutinf_tot_gt_eq_0 * (1.0 - pvalue_toolow * 1.0)

    MI_norm = mutinf_tot_thisdof_for_MI_norm / (ent12_boots + SMALL*SMALL)  #from Relly Brandman
    MI_norm[MI_norm < 0 ] = 0 #zero values that are less than zero if for some reason ent12_boots was less than zero

    #dKLtot_dKL1_dKL2 *= (1.0 - pvalue_toolow * 1.0)      #zeros KLdiv Hessian matrix elements that aren't significant
    if(VERBOSE >= 1):   
           print "var_mi_thisdof: "+str(var_mi_thisdof)+"\n"
           print "mutinf tot thisdof for MI norm:"
           print mutinf_tot_thisdof_for_MI_norm
           print "two-dimensional entropy S(1,2): "
           print ent12_boots
           print "MI_norm for bootsraps: "+str(MI_norm)
    if(VERBOSE >= 0): #usually want to print this anyways
           if(num_convergence_points < 2):
                  print "   mutinf/ind_mutinf = cor:%.3f ex_btw:%.3f exc:%.3f ind:%.4f tot:%.3f ind_btw:%.3f tot_btw:%.3f MI_norm:%.3f (sd= %.3f)   %s" % (average(excess_mutinf_thisdof),  average(excess_mutinf_thisdof_different_sims), average(old_excess_mutinf_thisdof),  average(independent_mutinf_thisdof), average(mutinf_tot_thisdof[:,0]), average(independent_mutinf_thisdof_different_sims), average(mutinf_tot_thisdof_different_sims[:,0]), average(MI_norm), sqrt(average(var_mi_thisdof)),sigtext)
           #debug mutinf between sims
           else:    #use last convergence point only
                   print "   mutinf/ind_mutinf = cor:%.3f ex_btw:%.3f exc:%.3f ind:%.4f tot:%.3f ind_btw:%.3f tot_btw:%.3f MI_norm:%.3f (sd= %.3f)   %s" % ((excess_mutinf_thisdof)[-1],  (excess_mutinf_thisdof_different_sims)[-1], old_excess_mutinf_thisdof[-1],  (independent_mutinf_thisdof[-1]), (mutinf_tot_thisdof[:,0])[-1], (independent_mutinf_thisdof_different_sims)[-1], (mutinf_tot_thisdof_different_sims[:,0])[-1], (MI_norm)[-1], (sqrt((var_mi_thisdof))),sigtext)
           
           
    assert(all(MI_norm[:] <=1.001 ))
    assert(all(MI_norm[:] >=0.0 ))
    MI_norm[MI_norm > 1.0] = 1.0
    print "shape of MI_norm: "+str(shape(MI_norm))
    #print "   mutinf/ind_mutinf = cor:%.3f ex_btw:%.3f exc:%.3f ind:%.3f tot:%.3f ind_btw:%.3f tot_btw:%.3f (sd= %.3f)   %s" % (average(excess_mutinf_thisdof),  average(excess_mutinf_thisdof_different_sims), average(old_excess_mutinf_thisdof),  average(independent_mutinf_thisdof), average(mutinf_tot_thisdof[:,0]), average(independent_mutinf_thisdof_different_sims), average(mutinf_tot_thisdof_different_sims[0,0]), var_mi_thisdof,sigtext)


    

    return excess_mutinf_thisdof, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, var_mi_thisdof, excess_mutinf_thisdof_different_sims, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, MI_norm





xtc_and_pdb_data = []
tot_residues = 0

class XTC_wrapper:
    coords =  zeros((10, 50, 3, 200000), float32) #allocate plenty of space so it doesn't get stomped on later
    #coords_data = []
    numangles = 0

xtc_coords = XTC_wrapper()

def output_distance_matrix_variances(bootstrap_sets,bootstrap_choose,which_runs,numangles,numangles_bootstrap, name_num_list):
       global xtc_coords
       (num_sims, num_res, mythree, max_num_angles) = shape(xtc_coords.coords) #
       xtc_dist_squared_matrix = zeros((bootstrap_sets, num_res, num_res), float64)
       print "num_res: "+str(num_res)
       print "name num list len: "+str(len(name_num_list))
       print "numangles: "+str(numangles)
       xtc_dist_matrix = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_accumulator = zeros((bootstrap_sets, num_res, num_res), int64)
       xtc_dist_squared_matrix_accumulator = zeros((bootstrap_sets, num_res, num_res), int64)
       xtc_dist_squared_matrix_dev_sum = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_squared_matrix_compensation = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_variance_matrix = 99 * ones((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_snapshots = zeros((bootstrap_sets, numangles_bootstrap[0] ), float64)
       xtc_dist_matrix_cutoff_filter = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter2 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter3 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter4 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter5 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter6 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter7 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter8 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter9 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter10 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter11 = zeros((bootstrap_sets, num_res, num_res), float64)
       xtc_dist_matrix_cutoff_filter12 = zeros((bootstrap_sets, num_res, num_res), float64)

       mynumangles = int(min(numangles))
       print "mynumangles: "+str(mynumangles)
       
       print "allocated memory successfully"
       which_runs = array(which_runs, int16)
       coords = xtc_coords.coords
       
       code = """
       // weave_distance_matrix_variances
       // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
       #include <math.h>
       #include <stdio.h> 
       int mybootstrap = 0;
       int mysim, simnum, res1, res2, cartnum = 0;
       int mynumangles, mynumangles_sum = 0;
       float cart1x, cart1y, cart1z, cart2x, cart2y, cart2z = 0;
       double dist2 = 0;
       double mean=0;

       for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
         
                mynumangles_sum = *(numangles_bootstrap + mybootstrap);
                for (mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                    
                    //for(res1=0;res1 < num_res; res1++) {                    
                    //   for(res2=res1+1; res2 < num_res; res2++) {
                         for (cartnum=0; cartnum< max_num_angles; cartnum++) {
                           cart1x = *(coords + simnum*num_res*3*max_num_angles + res1*3*max_num_angles + 0*max_num_angles + cartnum);
                           cart1y = *(coords + simnum*num_res*3*max_num_angles + res1*3*max_num_angles + 1*max_num_angles + cartnum);
                           cart1z = *(coords + simnum*num_res*3*max_num_angles + res1*3*max_num_angles + 2*max_num_angles + cartnum);
                           cart2x = *(coords + simnum*num_res*3*max_num_angles + res2*3*max_num_angles + 0*max_num_angles + cartnum);
                           cart2y = *(coords + simnum*num_res*3*max_num_angles + res2*3*max_num_angles + 1*max_num_angles + cartnum);
                           cart2z = *(coords + simnum*num_res*3*max_num_angles + res2*3*max_num_angles + 2*max_num_angles + cartnum);
                           dist2 = (cart2x - cart1x) * (cart2x - cart1x) + (cart2y - cart1y)*(cart2y - cart1y) + (cart2z - cart1z)*(cart2z - cart1z);
                           //*(xtc_dist_matrix_snapshots + mybootstrap*num_res*num_res*bootstrap_choose*mynumangles_sum + res2*num_res*bootstrap_choose*mynumangles_sum + res1*bootstrap_choose*mynumangles_sum + mysim*mynumangles_sum + cartnum) = dist2;
                           if(dist2 < 0) {
                           printf("ERROR: distance less than zero: %f \n",dist2);
                           }
                           //if(cartnum < 10) {
                           //  printf("res1: %i  res2: %i  cartnum: %i  distance: %f \\n", res1,res2,cartnum,dist2);
                           //}

                           // multiply by 1e5 to bring up to six decimal places into integer range for integer arithmetic
                           //*(xtc_dist_squared_matrix_accumulator + mybootstrap*num_res*num_res + res1*num_res + res2) += long((dist2 + SMALL) * 100000);
                           //*(xtc_dist_matrix_accumulator +         mybootstrap*num_res*num_res + res1*num_res + res2) += long(sqrt(dist2 - SMALL) * 100000);  
                           
                         }
                   //    } 
                   // }
                }

       }
       """


       code = """
       // weave_distance_matrix_variances
       // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
       #include <math.h>
       #include <stdio.h> 
       int mybootstrap = 0;
       int mysim, simnum, cartnum = 0;
       int mynumangles_sum = 0;
       float cart1x, cart1y, cart1z, cart2x, cart2y, cart2z = 0;
       double dist2 = 0;
       double mean=0;
       
       for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
         
                mynumangles_sum = 0; //*(numangles_bootstrap + mybootstrap);
                for (mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                         //mynumangles is the minimum -- and maximum -- number of snapshots to look at from each sim. 
                         // We use mynumangles as the max index here to essentially "slice" the array, so that we don't put extra junk into xtc_dist_matrix_snapshots
                         // mysim is which of the bootstrap_choose runs we're taking from the bootstrap sample, while simnum is the original sim number 
                         for (cartnum=0; cartnum< mynumangles; cartnum++) {
                           cart1x = *(coords + simnum*num_res*3*max_num_angles + res1*3*max_num_angles + 0*max_num_angles + cartnum);
                           cart1y = *(coords + simnum*num_res*3*max_num_angles + res1*3*max_num_angles + 1*max_num_angles + cartnum);
                           cart1z = *(coords + simnum*num_res*3*max_num_angles + res1*3*max_num_angles + 2*max_num_angles + cartnum);
                           cart2x = *(coords + simnum*num_res*3*max_num_angles + res2*3*max_num_angles + 0*max_num_angles + cartnum);
                           cart2y = *(coords + simnum*num_res*3*max_num_angles + res2*3*max_num_angles + 1*max_num_angles + cartnum);
                           cart2z = *(coords + simnum*num_res*3*max_num_angles + res2*3*max_num_angles + 2*max_num_angles + cartnum);
                           dist2 = (cart2x - cart1x) * (cart2x - cart1x) + (cart2y - cart1y)*(cart2y - cart1y) + (cart2z - cart1z)*(cart2z - cart1z);
                           *(xtc_dist_matrix_snapshots + mybootstrap*bootstrap_choose*mynumangles + mynumangles_sum + cartnum) = sqrt(dist2);

                           if (dist2 < 7.5*7.5) {
                                    *(xtc_dist_matrix_cutoff_filter + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 8.0*8.0) {
                                    *(xtc_dist_matrix_cutoff_filter2 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 8.5*8.5) {
                                    *(xtc_dist_matrix_cutoff_filter3 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 9.0*9.0) {
                                    *(xtc_dist_matrix_cutoff_filter4 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 9.5*9.5) {
                                    *(xtc_dist_matrix_cutoff_filter5 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 10.0*10.0) {
                                    *(xtc_dist_matrix_cutoff_filter6 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 11.0*11.0) {
                                    *(xtc_dist_matrix_cutoff_filter7 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 12.0*12.0) {
                                    *(xtc_dist_matrix_cutoff_filter8 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 14.0*14.0) {
                                    *(xtc_dist_matrix_cutoff_filter9 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 16.0*16.0) {
                                    *(xtc_dist_matrix_cutoff_filter10 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 18.0*18.0) {
                                    *(xtc_dist_matrix_cutoff_filter11 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           if (dist2 < 20.0*20.0) {
                                    *(xtc_dist_matrix_cutoff_filter12 + mybootstrap*num_res*num_res + res1*num_res + res2) += 1.0;
                           }
                           
                           //if(dist2 < 0) {
                           //printf("ERROR: distance less than zero: %f \\n",dist2);
                           //}
                           //if(cartnum < 1000000) {
                           //  printf("res1: %i  res2: %i  cartnum: %i  distance squared: %f \\n", res1,res2,cartnum,dist2);
                           //}

                           
                         }
                         mynumangles_sum += mynumangles;
                }

       }
       """

       mynumangles_sum = numangles_bootstrap[0] # assumes same number of datapoints per bootstrap
       for res1 in range(num_res):
              for res2 in range(res1, num_res):
                     #print "res1: "+str(num_res)+" res2: "+str(num_res)
                     weave.inline(code, ['num_sims', 'max_num_angles','numangles_bootstrap', 'num_res', 'which_runs', 'coords', 'xtc_dist_squared_matrix','xtc_dist_matrix','xtc_dist_variance_matrix','bootstrap_choose','bootstrap_sets','xtc_dist_squared_matrix_dev_sum','xtc_dist_squared_matrix_compensation','xtc_dist_matrix_accumulator','xtc_dist_squared_matrix_accumulator','xtc_dist_matrix_snapshots','xtc_dist_matrix_cutoff_filter','res1','res2','mynumangles', 'xtc_dist_matrix_cutoff_filter2', 'xtc_dist_matrix_cutoff_filter3', 'xtc_dist_matrix_cutoff_filter4', 'xtc_dist_matrix_cutoff_filter5' , 'xtc_dist_matrix_cutoff_filter6', 'xtc_dist_matrix_cutoff_filter7', 'xtc_dist_matrix_cutoff_filter8', 'xtc_dist_matrix_cutoff_filter9', 'xtc_dist_matrix_cutoff_filter10', 'xtc_dist_matrix_cutoff_filter11', 'xtc_dist_matrix_cutoff_filter12'  ],    extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                     compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                     #,type_converters = converters.blitz)
                     #print "residue1: "+str(name_num_list[res1])+ "residue2: "+str(name_num_list[res2])
                     #for mysim in range(bootstrap_choose):
                     #       simnum = which_runs[mybootstrap,mysim]
                     #       xtc_dist_matrix_snapshots[mybootstrap,mysim*numangles:] =
                     xtc_dist_variance_matrix[:,res1,res2] = var(xtc_dist_matrix_snapshots[:, :mynumangles_sum], axis=-1)
                     xtc_dist_matrix[:,res1,res2] = mean(xtc_dist_matrix_snapshots[:, :mynumangles_sum], axis=-1 )
                     xtc_dist_variance_matrix[:,res2,res1] = xtc_dist_variance_matrix[:,res1,res2]
                     xtc_dist_matrix[:,res2,res1] = xtc_dist_matrix[:,res1,res2]
                     
                     ## Create a matrix of ones/zeros to indicate elements within 7.5A
                     xtc_dist_matrix_cutoff_filter[:,res2,res1] = xtc_dist_matrix_cutoff_filter[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter2[:,res2,res1] = xtc_dist_matrix_cutoff_filter2[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter3[:,res2,res1] = xtc_dist_matrix_cutoff_filter3[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter4[:,res2,res1] = xtc_dist_matrix_cutoff_filter4[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter5[:,res2,res1] = xtc_dist_matrix_cutoff_filter5[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter6[:,res2,res1] = xtc_dist_matrix_cutoff_filter6[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter7[:,res2,res1] = xtc_dist_matrix_cutoff_filter7[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter8[:,res2,res1] = xtc_dist_matrix_cutoff_filter8[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter9[:,res2,res1] = xtc_dist_matrix_cutoff_filter9[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter10[:,res2,res1] = xtc_dist_matrix_cutoff_filter10[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter11[:,res2,res1] = xtc_dist_matrix_cutoff_filter11[:,res1,res2]
                     xtc_dist_matrix_cutoff_filter12[:,res2,res1] = xtc_dist_matrix_cutoff_filter12[:,res1,res2]

       xtc_dist_matrix_cutoff_filter_avg = average( xtc_dist_matrix_cutoff_filter, axis=0)
       xtc_dist_matrix_cutoff_filter_avg2 = average( xtc_dist_matrix_cutoff_filter2, axis=0)
       xtc_dist_matrix_cutoff_filter_avg3 = average( xtc_dist_matrix_cutoff_filter3, axis=0)
       xtc_dist_matrix_cutoff_filter_avg4 = average( xtc_dist_matrix_cutoff_filter4, axis=0)
       xtc_dist_matrix_cutoff_filter_avg5 = average( xtc_dist_matrix_cutoff_filter5, axis=0)
       xtc_dist_matrix_cutoff_filter_avg6 = average( xtc_dist_matrix_cutoff_filter6, axis=0)
       xtc_dist_matrix_cutoff_filter_avg7 = average( xtc_dist_matrix_cutoff_filter7, axis=0)
       xtc_dist_matrix_cutoff_filter_avg8 = average( xtc_dist_matrix_cutoff_filter8, axis=0)
       xtc_dist_matrix_cutoff_filter_avg9 = average( xtc_dist_matrix_cutoff_filter9, axis=0)
       xtc_dist_matrix_cutoff_filter_avg10 = average( xtc_dist_matrix_cutoff_filter10, axis=0)
       xtc_dist_matrix_cutoff_filter_avg11 = average( xtc_dist_matrix_cutoff_filter11, axis=0)
       xtc_dist_matrix_cutoff_filter_avg12 = average( xtc_dist_matrix_cutoff_filter12, axis=0)

       for res1 in range(num_res):
              for res2 in range(res1, num_res):
                     if(xtc_dist_matrix_cutoff_filter_avg[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg2[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg2[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg2[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg2[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg2[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg3[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg3[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg3[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg3[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg3[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg4[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg4[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg4[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg4[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg4[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg5[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg5[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg5[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg5[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg5[res2,res1] = 1.0
                     
                     if(xtc_dist_matrix_cutoff_filter_avg6[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg6[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg6[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg6[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg6[res2,res1] = 1.0


                     if(xtc_dist_matrix_cutoff_filter_avg7[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg7[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg7[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg7[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg7[res2,res1] = 1.0
                     

                     if(xtc_dist_matrix_cutoff_filter_avg8[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg8[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg8[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg8[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg8[res2,res1] = 1.0

                            
                     if(xtc_dist_matrix_cutoff_filter_avg9[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg9[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg9[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg9[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg9[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg9[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg9[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg9[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg9[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg9[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg10[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg10[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg10[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg10[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg10[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg11[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg11[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg11[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg11[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg11[res2,res1] = 1.0

                     if(xtc_dist_matrix_cutoff_filter_avg12[res1,res2] < 0.75 * mynumangles_sum):
                            xtc_dist_matrix_cutoff_filter_avg12[res1,res2] = 0
                            xtc_dist_matrix_cutoff_filter_avg12[res2,res1] = 0
                     else:
                            xtc_dist_matrix_cutoff_filter_avg12[res1,res2] = 1.0
                            xtc_dist_matrix_cutoff_filter_avg12[res2,res1] = 1.0
                     
                     

       code = """
       // weave_distance_matrix_variances
       // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
       #include <math.h>
       #include <stdio.h> 
       int mybootstrap = 0;
       int mysim, simnum, res1, res2, cartnum = 0;
       int mynumangles, mynumangles_sum = 0;
       double cart1x, cart1y, cart1z, cart2x, cart2y, cart2z = 0;
       double dist2 = 0;
       double mean=0;
       for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                for(res1=0;res1 < num_res; res1++) {                    
                   for(res2=res1+1; res2 < num_res; res2++) {
                       *(xtc_dist_matrix         + mybootstrap*num_res*num_res + res1*num_res + res2) = *(xtc_dist_matrix_accumulator +         mybootstrap*num_res*num_res + res1*num_res + res2)   / (100000.0 * mynumangles_sum) ; //normalize and remove extra digits and convert to double
                       *(xtc_dist_squared_matrix+ mybootstrap*num_res*num_res + res1*num_res + res2) = *(xtc_dist_squared_matrix_accumulator +         mybootstrap*num_res*num_res + res1*num_res + res2)   / (100000.0 * mynumangles_sum);  //normalize and remove extra digits and convert to double
                   }
                }

                for(res1=0;res1 < num_res; res1++) {                    
                       for(res2=res1+1; res2 < num_res; res2++) {  
                          
                          //compensated sum variance
                          //*(xtc_dist_variance_matrix  + mybootstrap*num_res*num_res + res1*num_res + res2) = ( *(xtc_dist_squared_matrix_dev_sum + mybootstrap*num_res*num_res + res1*num_res + res2) - (pow(*(xtc_dist_squared_matrix_compensation +         mybootstrap*num_res*num_res + res1*num_res + res2), 2))/mynumangles_sum ) / (mynumangles_sum - 1);  //unbiased variance

                          ////variance = <r^2> - <r>^2
                           //printf("res1: %i  res2: %i  cartnum: %i  <r^2>, <r>^2: %f \\n", res1,res2,cartnum,  *(xtc_dist_squared_matrix + mybootstrap*num_res*num_res + res1*num_res + res2) , (( *(xtc_dist_matrix  + mybootstrap*num_res*num_res + res1*num_res + res2))*(*(xtc_dist_matrix+ mybootstrap*num_res*num_res + res1*num_res + res2) ))     );
                          *(xtc_dist_variance_matrix  + mybootstrap*num_res*num_res + res1*num_res + res2) = *(xtc_dist_squared_matrix + mybootstrap*num_res*num_res + res1*num_res + res2) 
                     - (( *(xtc_dist_matrix  + mybootstrap*num_res*num_res + res1*num_res + res2))*(*(xtc_dist_matrix+ mybootstrap*num_res*num_res + res1*num_res + res2) ));
                          
                          //symmetrize
                          *(xtc_dist_squared_matrix + mybootstrap*num_res*num_res + res2*num_res + res1 ) = 
                                    *(xtc_dist_squared_matrix + mybootstrap*num_res*num_res + res1*num_res + res2) ;
                          *(xtc_dist_matrix + mybootstrap*num_res*num_res + res2*num_res + res1) = 
                                     *(xtc_dist_matrix + mybootstrap*num_res*num_res + res1*num_res + res2);
                    
                          *(xtc_dist_variance_matrix  + mybootstrap*num_res*num_res + res2*num_res + res1) = 
                                     *(xtc_dist_variance_matrix  + mybootstrap*num_res*num_res + res1*num_res + res2);
                       }
                }
       }
      """
       if(VERBOSE >= 2): print "about to populate xtc_dist_variance_matrix"
       
       #weave.inline(code, ['num_sims', 'max_num_angles','numangles_bootstrap', 'num_res', 'which_runs', 'coords', 'xtc_dist_squared_matrix','xtc_dist_matrix','xtc_dist_variance_matrix','bootstrap_choose','bootstrap_sets','xtc_dist_squared_matrix_dev_sum','xtc_dist_squared_matrix_compensation','xtc_dist_matrix_accumulator','xtc_dist_squared_matrix_accumulator' ],    extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
       #          extra_link_args=['-lgomp'],
       #          compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
       #            #,type_converters = converters.blitz)

       average_dist_variance_matrix = average(xtc_dist_variance_matrix,axis=0)
       #average_dist_variance_matrix[average_dist_variance_matrix < SMALL ] = 0.0
       output_matrix(prefix+"_bootstrap_avg_dist_matrix.txt",            average(xtc_dist_matrix,axis=0) ,name_num_list,name_num_list)
       #output_matrix(prefix+"_bootstrap_avg_dist_squared_matrix.txt",            average(xtc_dist_squared_matrix,axis=0) ,name_num_list,name_num_list,zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_variance_matrix.txt",            average_dist_variance_matrix ,name_num_list,name_num_list)
       output_matrix(prefix+"_bootstrap_avg_dist_variance_matrix_0diag.txt",            average_dist_variance_matrix ,name_num_list,name_num_list,zero_diag=True)
       #output_matrix(prefix+"_bootstrap_avg_dist_stdev_matrix.txt",            sqrt(average_dist_variance_matrix) ,name_num_list,name_num_list)
       #output_matrix(prefix+"_bootstrap_avg_dist_stdev_matrix_0diag.txt",            sqrt(average_dist_variance_matrix) ,name_num_list,name_num_list)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_7.5.txt", xtc_dist_matrix_cutoff_filter_avg, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_8.0.txt", xtc_dist_matrix_cutoff_filter_avg2, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_8.5.txt", xtc_dist_matrix_cutoff_filter_avg3, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_9.0.txt", xtc_dist_matrix_cutoff_filter_avg4, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_9.5.txt", xtc_dist_matrix_cutoff_filter_avg5, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_10.0.txt", xtc_dist_matrix_cutoff_filter_avg6, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_11.0.txt", xtc_dist_matrix_cutoff_filter_avg7, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_12.0.txt", xtc_dist_matrix_cutoff_filter_avg8, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_14.0.txt", xtc_dist_matrix_cutoff_filter_avg9, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_16.0.txt", xtc_dist_matrix_cutoff_filter_avg10, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_18.0.txt", xtc_dist_matrix_cutoff_filter_avg11, name_num_list, name_num_list, zero_diag=True)
       output_matrix(prefix+"_bootstrap_avg_dist_contacts_cutoff_filter_20.0.txt", xtc_dist_matrix_cutoff_filter_avg12, name_num_list, name_num_list, zero_diag=True)

       return


class ResidueChis:
   name = 'XXX'
   num = 0
   nchi = 0
   angles = []
   rank_order_angles = []
   angles_complex = [] #for spectral density calculations
   angles_input = []
   chi_pop_hist = []
   chi_var_pop = []
   bins = []
   entropy = 0
   var_ent = 0
   inv_numangles = []
   counts = []
   dKLtot_dchis2 = []
   weights = []
   max_num_chis = 99
   chain = ''
   sequential_res_num = 0
   expansion_factors = None
   # load angle info from xvg files
   def __str__(self): return "%s %s%s" % (self.name,self.num,self.chain)
   
   def get_num_chis(self,name):
    if NumChis.has_key(name): 
        if name == "SER" or name == "THR" or name == "NSER" or name == "NTHR" or name == "CSER" or name == "CTHR":
            if(not (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(2)+str(name)+".xvg") or os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(2)+str(name)+".xvg.gz"))): 
                return min(self.max_num_chis, 1)
            else: 
                return min(self.max_num_chis, NumChis[name])
        else:
            return min(self.max_num_chis, NumChis[name])
    else:
        mychi = 1
        numchis = 0
        #while (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+name+self.num+".xvg")):
        if(not (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+".xvg") or os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+".xvg.gz"))):
            while (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+str(self.num)+".xvg") or os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+str(self.num)+".xvg.gz")  ):
                numchis += 1
            if numchis == 0:
                print "cannot find file"+self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+".xvg"
        else:
            while (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+".xvg") or os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"chi"+str(numchis+1)+str(name)+".xvg.gz")  ):
                #quick fix until ligand mc writes residue number as well after their name
                numchis += 1
        if(numchis == 0): print "cannot find any xvg files for residue " + str(name) + str(self.num) + "\n"
        return numchis  

   def has_phipsi(self,name):
     has_phipsi = False
     if NumChis.has_key(name): 
         has_phipsi = True
     else:
         if (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"phi"+str(name)+".xvg")):
            has_phipsi = True
            if (os.path.exists(self.xvg_basedir+"run1"+self.xvg_chidir+"psi"+str(name)+".xvg")):
                has_phipsi = True
            else:
                has_phipsi = False
         else:
            has_phipsi = False
     return has_phipsi

   def _load_xvg_data(self, basedir, num_sims, max_angles, chi_dir = "/dihedrals/g_chi/", skip=1, skip_over_steps=0, last_step=None, coarse_discretize=None, split_main_side=None, backbone_only=0):
      myname = self.name
      mynumchis = self.get_num_chis(myname)
      shifted_angles = zeros((self.nchi,num_sims,max_angles),float64)
      nchi = self.nchi 
      backbone_only = self.backbone_only
      if(split_main_side == True):
             print "in loading data, treating main chain and side chain separately"
             if self.chain == "S":
                    print "using chi"
                    nchi = self.nchi  #just chi
             else:
                    print "using phi/psi"
                    mynumchis = 0
                    nchi = 2   #just phi/psi
      
      #self.numangles[:] = run_params.num_structs
          
      #shifted_angles[:,:,:] = -999 #a null value other than zero
      #weird fix for residue type "CYS2" in Gromacs
      #if myname == "CYS": myname += "2"
      #assert num_sims == len(self.which_runs[0])
          
      res_num = str(self.xvg_resnum)

      if(coarse_discretize is None ):
       for chi_num in range(nchi):
         #print "Chi:"+str(chi_num+1)+"\n"
         for sequential_sim_num in range(num_sims):
            #sim_index_str = str(self.which_runs[sequential_sim_num])
            if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                if(chi_num == 0):
                    xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname+res_num+".xvg"
                if(chi_num == 1):
                    xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname+res_num+".xvg"
                if(chi_num > 1 and chi_num <= mynumchis + 1):
                    xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname+res_num+".xvg"
            else:
                if(chi_num < mynumchis):
                    xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname+res_num+".xvg"
                if(chi_num == mynumchis):
                    xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname+res_num+".xvg"
                if(chi_num == mynumchis + 1):
                    xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname+res_num+".xvg"
            # gf: gromacs is changing some of my residue names when I use it to read in a PDB trajectory
            if not (os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz")):
               if myname == "LYS":
                  for myname_alt in ("LYSH", "LYP"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break
               if myname == "ASP" or myname == "ASH" or myname == "AS4":
                  for myname_alt in ("ASH","AS4", "ASP"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break   
               if myname == "GLU" or myname == "GLH" or myname == "GL4":
                  for myname_alt in ("GLH","GL4","GLU"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break   
               if myname == "HIS" or myname == "HID" or myname == "HIE" or myname == "HID" or myname == "HIP":
                  for myname_alt in ("HISA", "HISB", "HIE", "HID", "HIS", "HIP"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break
	       if myname == "CYS" or myname == "CYM" or myname == "CYT":
		  for myname_alt in ("CYS2", "CYX", "CYN", "F3G", "CYS", "CYM", "CYT"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break
               if myname == "ALA":
		  for myname_alt in ("ALA", "NALA", "NAL"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break
               if myname == "T2P" or myname == "TPO" or myname == "THR":
		  for myname_alt in ("THR", "T2P", "TPO"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break
               if myname == "S2P" or myname == "SEP" or myname == "SER":
		  for myname_alt in ("SER", "SEP", "S2P"):
                      if(nchi == mynumchis + 2 or nchi == mynumchis + 1 or backbone_only == 1): #phi/psi
                          if(chi_num == 0):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                          if(chi_num > 1 and chi_num <= mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num-1)+myname_alt+res_num+".xvg"
                      else:
                          if(chi_num < mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"chi"+str(chi_num+1)+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg"
                          if(chi_num == mynumchis + 1):
                              xvg_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"
                      if os.path.exists(xvg_fn) or os.path.exists(xvg_fn+".gz"): break

            if os.path.exists(xvg_fn):
                (data,titlestr)=readxvg(xvg_fn,skip,skip_over_steps,last_step)
            elif os.path.exists(xvg_fn+".gz"):
                (data,titlestr)=readxvg(xvg_fn+".gz",skip,skip_over_steps,last_step)
            else:
                print "ERROR: unable to find file '%s[.gz]'" % xvg_fn
                sys.exit(1)
                
            self.numangles[sequential_sim_num] = len(data[:,1])
            assert(self.numangles[sequential_sim_num] > 1)
            targ_shape = shape(self.angles_input)
            data_shape = len(data[:,1])
            #print "sim_num "+str(sequential_sim_num)+" numangles: "+str(self.numangles[sequential_sim_num])+" shape of target array: "+str(targ_shape)+" shape of data array: "+str(data_shape)
            data[:,1] = (data[:,1] + 180)%360 - 180 #Check and make sure within -180 to 180; I think this should do it 
            self.angles_input[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]] = data[:,1]
      
      else: #coarse_discretize 
             for sequential_sim_num in range(num_sims):
                    # gf: gromacs is changing some of my residue names when I use it to read in a PDB trajectory
                    xvg_fn_phi = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname+res_num+".xvg"
                    xvg_fn_psi = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname+res_num+".xvg"
                    def xvg_filenames_phipsi(thisname):
                           return [basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname_alt+res_num+".xvg", \
                                   basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname_alt+res_num+".xvg"]
                                         
                    if not (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                           (os.path.exists(xvg_fn_psi) or os.path.exists(xvg_fn_psi+".gz")) :
                           if myname == "LYS":
                                  for myname_alt in ("LYSH", "LYP"):
                                         (xvg_fn_phi, xvg_fn_psi) = xvg_filenames_phipsi(myname_alt) 
                                         if (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                                            (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")): break
                           if myname == "ASP" or myname == "ASH" or myname == "AS4":
                                  for myname_alt in ("ASH","AS4", "ASP"):
                                         (xvg_fn_phi, xvg_fn_psi) = xvg_filenames_phipsi(myname_alt)
                                         if (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                                            (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")): break
                           if myname == "GLU" or myname == "GLH" or myname == "GL4":
                                  for myname_alt in ("GLH","GL4","GLU"):
                                         (xvg_fn_phi, xvg_fn_psi) = xvg_filenames_phipsi(myname_alt)
                                         if (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                                            (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")): break
                           if myname == "HIS" or myname == "HID" or myname == "HIE" or myname == "HID" or myname == "HIP":
                                  for myname_alt in ("HISA", "HISB", "HIE", "HID", "HIS", "HIP"):
                                         (xvg_fn_phi, xvg_fn_psi) = xvg_filenames_phipsi(myname_alt)
                                         if (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                                            (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")): break
                           if myname == "CYS":
                                  for myname_alt in ("CYS2", "CYX", "CYN", "F3G"):
                                         (xvg_fn_phi, xvg_fn_psi) = xvg_filenames_phipsi(myname_alt)
                                         if (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                                            (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")): break
                           if myname == "ALA":
                                  for myname_alt in ("NALA", "NAL"):
                                         (xvg_fn_phi, xvg_fn_psi) = xvg_filenames_phipsi(myname_alt)
                                         if (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")) and \
                                            (os.path.exists(xvg_fn_phi) or os.path.exists(xvg_fn_phi+".gz")): break
                    else:
                           xvg_fn_phi = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"phi"+myname+res_num+".xvg"
                           xvg_fn_psi = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"psi"+myname+res_num+".xvg"
                    print "pathname phi: "
                    print xvg_fn_phi
                    print "pathname psi: "
                    print xvg_fn_psi
                    if os.path.exists(xvg_fn_phi):
                           (data_phi,titlestr)=readxvg(xvg_fn_phi,skip,skip_over_steps,last_step)
                    elif os.path.exists(xvg_fn_phi+".gz"):
                           (data_phi,titlestr)=readxvg(xvg_fn_phi+".gz",skip,skip_over_steps,last_step)
                    else:
                           print "ERROR: unable to find file '%s[.gz]'" % xvg_fn_phi
                           sys.exit(1)
                    if os.path.exists(xvg_fn_psi):
                           (data_psi,titlestr)=readxvg(xvg_fn_psi,skip,skip_over_steps,last_step)
                    elif os.path.exists(xvg_fn_phi+".gz"):
                           (data_psi,titlestr)=readxvg(xvg_fn_psi+".gz",skip,skip_over_steps,last_step)
                    else:
                           print "ERROR: unable to find file '%s[.gz]'" % xvg_fn_psi
                           sys.exit(1)
                    self.numangles[sequential_sim_num] = len(data_phi[:,1])   
                    assert(self.numangles[sequential_sim_num] > 1)
                    data_phi[:,1] = (data_phi[:,1] + 180)%360 - 180 #Check and make sure within -180 to 180; I think this should do it
                    data_psi[:,1] = (data_phi[:,1] + 180)%360 - 180 #Check and make sure within -180 to 180; I think this should do it
                    #print data_phi
                    for i in range(self.numangles[sequential_sim_num]):
                          phi = data_phi[i,1]
                          psi = data_psi[i,1]
                          ## Alpha   ... these angle numbers are just fixed values...
                          if ( -180 < phi < 0 and -100 < psi < 45):
                                 self.angles_input[0,sequential_sim_num,i] = -179.9
                                 #print "alpha"
                          ## Beta
                          elif ( -180 < phi < -45 and (45 < psi or psi > -135) ):
                                 self.angles_input[0,sequential_sim_num,i] = -89.9
                                 #print "beta"
                          ## Turn
                          elif ( 0 < phi < 180 and -90 < psi < 90):
                                 self.angles_input[0,sequential_sim_num,i] = 0.1
                                 #print "turn"
                          ## Other
                          else:  
                                 self.angles_input[0,sequential_sim_num,i] = 90.1
                                 #print "other"
                    #print "self.angles_input:"+str(self.angles_input[0,sequential_sim_num])
                    #print self.angles_input[0,sequential_sim_num]


          
      for sequential_sim_num in range(self.num_sims): ## PATCH to fix problem with different # of angles in different sims
              self.numangles[sequential_sim_num] = min(self.numangles[:])     ## PATCH to fix problem with different # of angles in different sims
      
      self.weights = zeros((self.num_sims,min(self.numangles)), float64)
      for sequential_sim_num in range(self.num_sims): ## PATCH to fix problem with different # of angles in different sims
             self.numangles[sequential_sim_num] = min(self.numangles[:])     ## PATCH to fix problem with different # of angles in different sims
      for sequential_sim_num in range(self.num_sims): 
          #weights for different snapshots in different sim nums for MBAR
          weights_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"weights.xvg"
          if os.path.exists(weights_fn):
              print "using per-snapshot weights"    
              (data,titlestr)=readxvg(weights_fn,skip,skip_over_steps)
              self.weights[sequential_sim_num,:min(self.numangles)] = data[:,1]
          else:
              print "assuming all snapshots are created equal\n"   
              self.weights[sequential_sim_num,:min(self.numangles)] = 1.0
          # Now, normalize weights so that their sum is the number of angles; this is so we can use a kind of continuous analog of "counts" with the Grassberger entropy terms
          self.weights[sequential_sim_num,:min(self.numangles)] = self.weights[sequential_sim_num,:min(self.numangles)] * (min(self.numangles) / sum(self.weights[sequential_sim_num,:min(self.numangles)])) 
      
          print "sum of weights: "+str(sum(self.weights[sequential_sim_num,:self.numangles[sequential_sim_num]]))
# for phi/psi discretization, use made-up numbers like 45, 135, 215, etc. and one torsion per residue, even making temporary file if needed
##



   
  
   def load_xtc_data(self, basedir, num_sims, max_angles, chi_dir = "/dihedrals/g_chi/", skip=1, skip_over_steps=0, pdbfile=None, xtcfile=None):
          
          #dataarray=zeros((int((inlength-skip_over_steps)/skip) + extra_record,numfields),float64) #could start with zero, so add + extra_record
          
      skiplines=0
   #Read data into array
      #for i in range(int((inlength-skip_over_steps)/skip) + extra_record): #could start with zero, so add + extra_record ...
      #   if(i*skip + skip_over_steps < inlength): # ... but make sure we don't overshoot
      #       entries=inlines[i*skip+skip_over_steps].split()
      global xtc_and_pdb_data
      global xtc_coords
      global tot_residues
      myname = self.name
      self.nchi = 3
      mynumchis = 3 #self.get_num_chis(myname)
      #tot_residues = 0
      #shifted_angles = zeros((self.nchi,num_sims,max_angles),float64)
      #self.numangles[:] = run_params.num_structs
      #shifted_angles[:,:,:] = -999 #a null value other than zero
      #weird fix for residue type "CYS2" in Gromacs
      #if myname == "CYS": myname += "2"
      #assert num_sims == len(self.which_runs[0])
      #res_num = str(self.sequential_res_num + 1) #since MDAnalysis starts with atom 1 
      if(skip == 1):
             extra_record = 0
      else:
             extra_record = 1
      print "sequential residue number: "+str(self.sequential_res_num)
      for sequential_sim_num in range(num_sims):
             xtc_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+xtcfile+str(sequential_sim_num+1)+".xtc"
             pdb_fn = pdbfile
             if os.path.exists(xtc_fn) and os.path.exists(pdb_fn):
                 if (xtc_and_pdb_data == []):
                     i = 0
                     tempdata = MDAnalysis.Universe(pdb_fn, xtc_fn)
                     tot_residues = shape(tempdata.atoms.coordinates())[0]
                     print "total residues:"+str(tot_residues)
                     count = 0
                     for ts in tempdata.trajectory:
                            if(count > skip_over_steps): 
                                   if( count % skip == 0):
                                          i += 1
                            count += 1
                     print str(i) + " snapshots read from xtc"
                     xtc_coords.numangles = i
                     self.numangles[sequential_sim_num] = i
                     xtc_coords.coords = resize(xtc_coords.coords,(num_sims, tot_residues, 3, self.numangles[0] + 1000)) #shrink down to proper limits with buffer
                     tempdata.trajectory.close_trajectory()
                 if len(xtc_and_pdb_data) < num_sims: #if we haven't opened all the trajectory files yet
                     xtc_and_pdb_data.append(MDAnalysis.Universe(pdb_fn, xtc_fn))
                     if(len(xtc_and_pdb_data) >= 1):
                         i = 0
                         count = 0
                         print "loading cartesian data, run: "+str(sequential_sim_num)
                         for ts in xtc_and_pdb_data[sequential_sim_num].trajectory:
                                #print shape((xtc_and_pdb_data[sequential_sim_num].atoms.coordinates())[:,:])
                                #print shape(xtc_coords.coords[sequential_sim_num, :tot_residues, :, i])
                                if(count > skip_over_steps): 
                                   if( count % skip == 0):
                                          xtc_coords.coords[sequential_sim_num, :tot_residues, :3, i] = (xtc_and_pdb_data[sequential_sim_num].atoms.coordinates())[:tot_residues,:3]
                                          i += 1
                                count += 1
                         self.numangles[sequential_sim_num] = i
                         assert(self.numangles[sequential_sim_num] > 1)
                         if sequential_sim_num == 0: #test mutual information between first and second Cartesian for debugging
                                print "debug: naive histogram mutual information of cart 1 x vs cart 2 x: "+str(python_mi(xtc_coords.coords[0,0,0,:], xtc_coords.coords[0,1,0,:]))
                                #print "debug: kernel density  mutual information of cart 1 x vs cart 2 x: "+str(kde_python_mi(xtc_coords.coords[0,0,0,:], xtc_coords.coords[0,1,0,:]))
                         xtc_and_pdb_data[sequential_sim_num].trajectory.rewind() #reset for next use
                 else: #get numangles from xtc_coords
                     self.numangles[sequential_sim_num] = xtc_coords.numangles
                     assert(self.numangles[sequential_sim_num] > 1)
                 #print shape(xtc_coords.coords)
                 #print xtc_coords.coords[sequential_sim_num, self.sequential_res_num, :, :self.numangles[sequential_sim_num]]
                 #print xtc_coords.coords[sequential_sim_num, :, 0, :self.numangles[sequential_sim_num]]
                 #print xtc_coords.coords[0, :, 0, :]
                 
                 
                 
                 
                 self.angles_input[:3,sequential_sim_num,:self.numangles[sequential_sim_num]] =  xtc_coords.coords[sequential_sim_num, self.sequential_res_num, :, :self.numangles[sequential_sim_num]]
                 #if( i % 100 == 0): 
                 #print "timestep "+str(i)
                 
                 
             else:
                 if os.path.exists(xtc_fn):
                     print "ERROR: unable to find file '%s[.gz]'" % pdb_fn
                 else:
                     print "ERROR: unable to find file '%s[.gz]'" % xtc_fn
                 sys.exit(1)
      self.weights = zeros((self.num_sims,min(self.numangles)), float64)
      for sequential_sim_num in range(self.num_sims): ## PATCH to fix problem with different # of angles in different sims
             self.numangles[sequential_sim_num] = min(self.numangles[:])     ## PATCH to fix problem with different # of angles in different sims
             xtc_coords.numangles = min(self.numangles[:])                   ## PATCH to fix problem with different # of angles in different sims
      for sequential_sim_num in range(self.num_sims): 
          #weights for different snapshots in different sim nums for MBAR
          weights_fn = basedir+"run"+str(sequential_sim_num+1)+chi_dir+"weights.xvg"
          if os.path.exists(weights_fn):
              print "using per-snapshot weights"    
              (data,titlestr)=readxvg(weights_fn,skip,skip_over_steps)
              self.weights[sequential_sim_num,:min(self.numangles)] = data[:,1]
          else:
              print "assuming all snapshots are created equal\n"   
              self.weights[sequential_sim_num,:min(self.numangles)] = 1.0
          # Now, normalize weights so that their sum is the number of angles; this is so we can use a kind of continuous analog of "counts" with the Grassberger entropy terms
          self.weights[sequential_sim_num,:min(self.numangles)] = self.weights[sequential_sim_num,:min(self.numangles)] * (min(self.numangles) / sum(self.weights[sequential_sim_num,:min(self.numangles)])) 
      xtc_coords.coords = xtc_coords.coords[:, :, :, :min(self.numangles)] #shrink down to proper limits with buffer

      #need to figure out how to add this back
      #if len(xtc_and_pdb_data) == num_sims and self.sequential_res_num >= tot_residues : #if we have opened all the trajectory files and processed all the data, resize
      #  xtc_coords.coords = xtc_coords.coords[:, :, :, :self.numangles[0]] #shrink down to proper limits with buffer
      #  xtc_and_pdb_data = []  #reset global variable
      
      ## RESCALE CARTESIANS 
      
      ## first, zero centroid of all particles
      ## ideally filter out the first eigenvector, assuming rot+trans alignment already performed, so this is in effect filters out #7
      
      ##Get coordinate ranges for all 3 cartesians over all residues
      #Perhaps should have box sizes grabbed somehow from xtc file? 
      #coordmin = amin(amin(self.angles_input[:3,:,:min(self.numangles)],axis=2),axis=1)
      #coordrange = amax(amax(self.angles_input[:3,:,:min(self.numangles)],axis=2),axis=1) - coordmin + SMALL #+ SMALL to avoid div by zero
      #coordavg = average(average(self.angles_input[:3,:,:min(self.numangles)],axis=2),axis=1)
   
      # perform rescaling so that cartesians are in range (-120, 120), this will give wiggle room for substantial shape changes
      #for sequential_sim_num in range(self.num_sims):
      #    offset = zeros((3,self.numangles[sequential_sim_num]), float64)
      #    scaling_factor = ones((3,self.numangles[sequential_sim_num]), float64)
      #    for mycoord in range(3):
      #        offset[mycoord,:] = coordavg[mycoord]
      #        scaling_factor = (240 - SMALL) / (coordrange[mycoord])
      #    self.angles_input[:3,sequential_sim_num,:self.numangles[sequential_sim_num]] = \
      #        (self.angles_input[:3,sequential_sim_num,:self.numangles[sequential_sim_num]]  -  offset) * scaling_factor - 120 
      
             

                 
          



       


      ### Rank-order data for adaptive partitioning -- note that I give the rank over the pooled data for each angle in each sim
##      shifted_angles = self.angles_input.copy()
##     shifted_angles[shifted_angles < 0] += 360 # wrap these around so that -180 and 180 are part of the same bin
##      for chi_num in range(self.nchi):
##         #print "Chi:"+str(mychi+1)+"\n"
##         shifted_flat = resize(((swapaxes((shifted_angles[chi_num,:,:]).copy(),0,1))),(sum(self.numangles)))
##         sorted_angles = sort(shifted_flat)
##         #print "numangles: "+str(self.numangles)+" sum_numangles: "+str(sum(self.numangles))+" num of sorted angles: "+str(len(sorted_angles))+" num of datapoints: "+str(len(shifted_flat))
##         #print (searchsorted(sorted_angles,shifted_angles[chi_num,0,:]))
##         for sequential_sim_num in range(num_sims):
##             #print "Shifted Angles: Length: "+str(shape(shifted_angles[chi_num,0,:]))+"\n"
##             #print shifted_angles[chi_num,sim_num,:]
##             #print "Sorted Angles: Length\n"+str(shape(sorted_angles))+"\n"
##             #print sorted_angles[:]
##             #print "Rank ordered Angles \n"
##            self.rank_order_angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]] = \
##                                      searchsorted(sorted_angles,shifted_angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]]) # rank-ordered dihedral angles
             #print self.rank_order_angles[chi_num,sim_num,:]
	     #print "\n"
   
   # load angle info from an all_angle_info object
   def _load_pdb_data(self, all_angle_info, max_angles):
      #shifted_angles = zeros((self.nchi,all_angle_info.num_sims,max_angles),float64)
      #shifted_angles[:,:,:] = -999 #a null value other than zero      
      for sequential_sim_num in range(self.num_sims): #range(all_angle_info.num_sims):
          curr_angles, numangles = all_angle_info.get_angles(sequential_sim_num, int(self.xvg_resnum), self.name, self.backbone_only, self.phipsi, self.max_num_chis)
          self.angles_input[:, sequential_sim_num, 0:numangles] = (curr_angles + 180)%360 - 180
          self.numangles[sequential_sim_num]= numangles
          assert(self.numangles[sequential_sim_num] > 1)
      self.weights = zeros((self.num_sims,min(self.numangles)), float64)
      for sequential_sim_num in range(self.num_sims): ## PATCH to fix problem with different # of angles in different sims
          self.numangles[sequential_sim_num] = min(self.numangles[:])     ## PATCH to fix problem with different # of angles in different sims
          self.weights[sequential_sim_num,:self.numangles[sequential_sim_num]] = 1.0
      ### Rank-order data for adaptive partitioning -- note that I give the rank over the pooled data for each angle in each sim



##      shifted_angles = self.angles_input.copy()
##      shifted_angles[shifted_angles < -90] += 360 # wrap these around so that -180 and 180 are part of the same bin
##      for chi_num in range(self.nchi):
##         #print "Chi:"+str(mychi+1)+"\n"
##         shifted_flat = resize(((swapaxes((shifted_angles[chi_num,:,:]).copy(),0,1))),(sum(self.numangles)))
##         sorted_angles = sort(shifted_flat)
##         #print "numangles: "+str(self.numangles)+" sum_numangles: "+str(sum(self.numangles))+" num of sorted angles: "+str(len(sorted_angles))+" num of datapoints: "+str(len(shifted_flat))
##         #print shifted_angles[chi_num,0,:200]
##         #print sorted_angles[:200]
##         #print (searchsorted(sorted_angles,shifted_angles[chi_num,0,:]))
##         for sequential_sim_num in range(self.num_sims):
##             self.rank_order_angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]] = \
##                                   searchsorted(sorted_angles,shifted_angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]]) # rank-ordered dihedral angles
             #print self.rank_order_angles[chi_num,sim_num,:]
            
   # load the angles and calculate the total entropy of this residue and its variance.
   # The entropy of a residue with two chi angles is calculated as
   #     H(X1,X2) = H(X1) + H(X2) - I(X1,X2)
   # where I() is the mutual information between X1 and X2.
   # For residues with more than 3 chi angles, we assume the 2nd order approximation is sufficient.
   # The variance residue's entropy is the sum of the variances of all the terms.

   def correct_and_shift_carts(self,num_sims,bootstrap_sets,bootstrap_choose, num_convergence_points = 1):
       shifted_carts = zeros((shape(self.angles)[0], shape(self.angles)[1], min(self.numangles)),float64)
       shifted_carts[:,:,:min(self.numangles)] = self.angles[:,:,:min(self.numangles)]
       myname = self.name
       mynumchis = self.get_num_chis(myname)
       print "correcting and shifting cartesians"
       # shift cartesians to [0, 360) so binning will be compatible for angular data
       # we don't know if all the sims are aligned togther.. if so, then shifting shoud be done for whole dataset not each sim separately 
       for i in range((shape(shifted_carts))[0]):
              for j in range ((shape(shifted_carts))[1]):                     
                     shifted_carts[i,j,:] = shifted_carts[i,j,:] - min(shifted_carts[i,j,:]) #shift so min at zero
                     if( max(shifted_carts[i,j,:] - min(shifted_carts[i,j,:] ) > 0 ) ):
                         shifted_carts[i,j,:] = (shifted_carts[i,j,:] * (360.0 - SMALL) / \
                                            (max(shifted_carts[i,j,:]) - min(shifted_carts[i,j,:]))) 
                         #shifted_carts[i,j,:] = shifted_carts[i,j,:] + min(shifted_carts[i,j,:])
       
       self.angles = shifted_carts[:,:,:]
       if VERBOSE > 1:
              print shifted_carts
       if num_convergence_points < 2:
              self.numangles_bootstrap[:] = min(self.numangles) * bootstrap_choose
       else:
              for convergence_point in range(num_convergence_points):
                      self.numangles_bootstrap[convergence_point] = int(min(self.numangles) * bootstrap_choose * convergence_point * 1.0 / num_convergence_points)
       print "done with correcting and shifting cartesians"
       del shifted_carts

   def expand_contract_data(self,num_sims,bootstrap_sets,bootstrap_choose):
       shifted_data = self.angles.copy()
       myname = self.name
       mynumchis = self.get_num_chis(myname)
       mymin = self.minmax[0,:]
       mymax = self.minmax[1,:]
       mymin_array1 = zeros((num_sims), float64)
       mymin_array2 = zeros((num_sims, shape(self.angles)[-1]), float64)
       mymax_array1 = zeros((num_sims), float64)
       mymax_array2 = zeros((num_sims, shape(self.angles)[-1]), float64)
       
       # shift data to [0, 360] so binning will be compatible for max and min of data
       # have center of data be max - min / 2 rather than average
       for mychi in range(self.nchi):
              mymin_array1[:] = mymin[mychi]
              mymin_array2[:, :] = resize(mymin_array1,shape(mymin_array2))
              mymax_array1[:] = mymax[mychi]
              mymax_array2[:, :] = resize(mymax_array1,shape(mymax_array2))

              #midpoint = ( mymax_array2[:,:] - mymin_array2[:,:] ) / 2.0 
              zeropoint =  mymin_array2[:,:] 
              # no recentering, as I think it might introduce bias
              # rather, we just stretch the range by setting the min to zero,
              # then grabbing on the max and pulling it to 360
              shifted_data[mychi,:,:] = shifted_data[mychi,:,:] - zeropoint
              if( all(mymax_array2[:,:] - mymin_array2[:,:] > 0) ):
                         shifted_data[mychi,:,:] = shifted_data[mychi,:,:] * 360.0 / \
                              ( mymax_array2[:,:] - mymin_array2[:,:] ) #+ midpoint ##if using midpoint recentering
                         shifted_data[mychi,:,:] = (shifted_data[mychi,:,:])%360  #Check and make sure within 0 to 360
            
       for mysim in range(num_sims):
              shifted_data[:,mysim,self.numangles[mysim]:] = 0
       print "orig data:"
       print self.angles[mychi,0,:min(self.numangles)]
       print "data range:"
       print mymin_array1
       print mymax_array1
       print "reshaped data:"
       print shifted_data[mychi,0,:min(self.numangles)]
       self.angles[:,:,:] = shifted_data[:,:,:]
       del shifted_data
   
   def correct_and_shift_angles(self,num_sims,bootstrap_sets,bootstrap_choose, coarse_discretize = None):
       ### Rank-order data for adaptive partitioning --
       ### note that I give the rank over the pooled data for angles from all sims
   
       myname = self.name
       try:
              mynumchis = self.get_num_chis(myname)
       except:
              mynumchis = NumChis[myname]
       print "residue name: "+str(self.name)+" num chis: "+str(mynumchis)+"\n"
       shifted_angles = self.angles.copy()
       shifted_angles[shifted_angles < -180] = -180 # bring in to limit in case it is out of range
       shifted_angles[shifted_angles > 360] = 360 # bring in to limit in case it is out of range
       shifted_angles[shifted_angles < 0] += 360 # wrap these around so that -180 and 180 are part of the same bin
       assert(self.nchi > 0) # if we didn't have any dihedrals, we should have dropped out by now
       
       #wrap torsions of 2-fold symmetric and 3-fold symmetric terminal chi angles
       # ARG and LYS's "chi5" is symmetric, but these isn't one of the standard chi angles
       # and typically we only have up to chi4 for these anyways.

       #However, we do not correct protonated ASP or GLU residues, as binding of a proton to these breaks symmetry.

       if(CORRECT_FOR_SYMMETRY == 1):
        if(self.nchi == mynumchis + 2): #phi/psi            
               if (myname == "ASP" or myname == "GLU" or myname == "PHE" or myname == "TYR" \
                   or myname == "NASP" or myname == "NGLU" or myname == "NPHE" or myname == "NTYR" \
                   or myname == "CASP" or myname == "CGLU" or myname == "CPHE" or myname == "CTYR") and coarse_discretize is None:
                      #last chi angle is 2-fold symmetric
                      myangles = shifted_angles[mynumchis + 1,:,:]
                      myangles[myangles > 180] = myangles[myangles > 180] - 180
                      shifted_angles[mynumchis + 1,:,:] = myangles
                      self.symmetry[mynumchis + 1] = 2
              
        else:
            if(self.nchi == mynumchis):
                   if (myname == "ASP" or myname == "GLU" or myname == "PHE" or myname == "TYR" \
                      or myname == "NASP" or myname == "NGLU" or myname == "NPHE" or myname == "NTYR" \
                      or myname == "CASP" or myname == "CGLU" or myname == "CPHE" or myname == "CTYR") and coarse_discretize is None :
                          #last chi angle is 2-fold symmetric
                          myangles = shifted_angles[mynumchis - 1,:,:]
                          myangles[myangles > 180] = myangles[myangles > 180] - 180
                          shifted_angles[mynumchis - 1,:,:] = myangles
                          self.symmetry[mynumchis - 1] = 2
            
              
       self.angles[:,:,:] = shifted_angles[:,:,:]  #now actually shift the angles, important for entropy and non-adaptive partitioning correlations
       
       ### Circular PCA
       ## uses "resultant" instead of a mean, following "Topics on circular statistics by S. Rao Jammalamadaka, Ambar Sengupta" 
       self.expansion_factors = None
       if (Circular_PCA == True):
              self.expansion_factors = zeros((self.nchi),float64)
              min_num_angles = min(self.numangles)
              cosangles = cos(self.angles)
              sinangles = sin(self.angles)

              sumcos = sum(sum(cosangles, axis=-1),axis=-1)
              sumsin = sum(sum(sinangles, axis=-1),axis=-1)
              mag_resultants = sqrt(sumcos * sumcos + sumsin * sumsin)
              resultants = arctan2(sumsin , mag_resultants)
              resultants_array = zeros((self.nchi,num_sims,min_num_angles),float64)
              for chi_num in range(self.nchi):
                     for sim_num in range(num_sims):
                            resultants_array[chi_num, sim_num, :] = resultants[chi_num]
              dists = minimum(abs(self.angles - resultants_array), 360 - abs(self.angles - resultants_array)) #since data now ranges 0 to 360 
              counter_clockwise_from_resultant = (dists == abs(self.angles - resultants_array)) * 2 - 1  #gives 1 if clockwise, -1 if counterclockwise
              displacements = dists * counter_clockwise_from_resultant
              print "shape of displacements:"
              print displacements.shape
              if VERBOSE > 1:
                     print "displacements:"
                     print displacements
              #perform PCA on displacements from resultant
              displacements_matrix = zeros((self.nchi, sum(self.numangles)), float64) 
              for chi_num in range(self.nchi):
                     displacements_matrix[chi_num, :] = resize((displacements[chi_num,:,:]).copy(),sum(self.numangles))
                     #use Singular Value Decomposition for PCA: X = U s VT
              print "displacements matrix shape: "
              print displacements_matrix.shape
              U, s, VT = linalg.svd(displacements_matrix, full_matrices=False)
              projected_displacements = dot(diag(s), VT)
              #try:
              blah = 0
              if (blah == 1):
                     #projected_displacements_pca = VT
                     myICA = mdp.nodes.CuBICANode()
                     #myICA = mdp.nodes.XSFANode(whitened=False,verbose=True)
                     myICA.train(transpose(displacements_matrix))
                     myICA = myICA.execute(transpose(displacements_matrix))
                     #myICA.train(transpose(VT))
                     #myICA = myICA.execute(transpose(VT))
                     #myICA = mdp.fastica(transpose(displacements_matrix))
                     projected_displacements = transpose(myICA)
                     if (isnan(projected_displacements)).any == True: #if any NaN's
                            print "NaN's present\n"
                            raise ICA_nanerror
              #except:
              #       print "warning: ICA did not converge, just using vanilla circular PCA instead"
              #       #U, s, VT = linalg.svd(displacements_matrix, full_matrices=False)
              #       print "eigenvalues:"
              #       print s
              #       print U.shape, VT.shape, s.shape
              #       #now need to change coordinates
              #       #note that since U is orthonormal, U^T X = s VT , and there is no Jacobian for the transformation
              #       
              #       projected_displacements = dot(diag(s), VT)
              print "projected displacements shape:"
              print projected_displacements.shape
              #print projected_displacements
              #scale projected displacements to range [0, 360]
              for i in range((shape(projected_displacements))[0]):
                     projected_displacements[i,:] = projected_displacements[i,:] - min(projected_displacements[i,:])
                     projected_displacements[i,:] = projected_displacements[i,:] *360.0 / \
                                (max(projected_displacements[i,:]) - min(projected_displacements[i,:]))
                     self.expansion_factors[i] = TWOPI / \
                                (max(projected_displacements[i,:]) - min(projected_displacements[i,:]))
                     print str(min(projected_displacements[i,:])) + " " + str(max(projected_displacements[i,:]))
              self.angles = zeros((self.nchi,num_sims,min_num_angles),float64)
              for chi_num in range(self.nchi):
                     for sequential_sim_num in range(num_sims):
                            for anglenum in range(min_num_angles):
                                   self.angles[chi_num, sequential_sim_num, anglenum] = projected_displacements[chi_num, sequential_sim_num * min_num_angles + anglenum]
              #self.angles = resize(projected_displacements,(sum(self.nchi), sum(self.num_sims), min(self.numangles)))
              #print new angles after PCA:
              print "new angles after Circular PCA"
              if VERBOSE > 1:
                     print self.angles
              #for mychi in range(self.nchi):
                     #print "chi: "+str(mychi)
                     #print self.angles[mychi]
                     #self.angles[self.angles < 0] += 360 #to move back to [0, 360]
                     #self.angles += 180 #to move back to [0, 360]
              # shift cartesians to [0, 360] so binning will be compatible for angular data
       #unshift 
       #self.angles[self.angles > 180] -= 360 # back to [-180, 180]
       
   def sort_angles(self,num_sims,bootstrap_sets,bootstrap_choose, num_convergence_points = 1, coarse_discretize = None):
       ## Note: the weights are the same across all torsions for this residue, as they only depend upon sim number and timepoint
       #self.rank_order_angles_weights = zeros((num_sims,min(self.numangles)),float64)  
       self.rank_order_angles_sequential_weights =zeros((num_sims,min(self.numangles)),float64)
       self.boot_ranked_angles_weights = zeros((bootstrap_sets,min(self.numangles)*bootstrap_choose),float64)
       self.boot_sorted_angles_weights = zeros((bootstrap_sets,min(self.numangles)*bootstrap_choose),float64)
       self.sorted_angles_weights = zeros(sum(self.numangles),float64)
       
       min_num_angles = min(self.numangles)
       ### NOTE: here that ranks are determined by taking all of the data, aggregating it together, then sorting,  for self.rank_order_angles (data are over nsims), and self.boot_ranked_angles, which are over bootstrap.
       ### NOTE: Approximation: or the mutinf-between-torisons-in-different-sims "correction term", there isn't a separate mapping from angles to ranks for each pair of sims -- even though there is a separate mapping for each bootstrap set. Perhaps this approximation could be removed, but it might come at a greater storage cost. 
       for chi_num in range(self.nchi):
         #print "Chi:"+str(mychi+1)+"\n"
         shifted_flat = resize(((swapaxes((self.angles[chi_num,:,:]).copy(),0,1))),(sum(self.numangles)))
         self.sorted_angles[chi_num,:] = sort(shifted_flat)
         weights_flat = resize(((swapaxes((self.weights[:,:]).copy(),0,1))),(sum(self.numangles)))
         self.sorted_angles_weights[:] = [ weights_flat[i] for i in argsort(shifted_flat)] #need this for later 
         print "sorted angles weights:"
         print self.sorted_angles_weights
         #print "numangles: "+str(self.numangles)+" sum_numangles: "+str(sum(self.numangles))+" num of sorted angles: "+str(len(sorted_angles))+" num of datapoints: "+str(len(shifted_flat))
         #print (searchsorted(sorted_angles,shifted_angles[chi_num,0,:]))
         print "Sorted Angles: Length\n"+str(shape(self.sorted_angles))+"\n"
         print self.sorted_angles[chi_num,:]
         for sequential_sim_num in range(num_sims):
             #print "Shifted Angles: Length: "+str(shape(shifted_angles[chi_num,0,:]))+"\n"
             #print shifted_angles[chi_num,sim_num,:]
             #print "Rank ordered Angles \n"
             #self.rank_order_angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]] = \
             # searchsorted(self.sorted_angles[chi_num,:], self.angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]]) # rank-ordered dihedral angles for all sims together
             self.rank_order_angles[chi_num,sequential_sim_num,:min_num_angles] = \
              searchsorted(self.sorted_angles[chi_num,:], self.angles[chi_num,sequential_sim_num,:min_num_angles]) # rank-ordered dihedral angles for all sims together
             #sorted_angles_sequential = sort((self.angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]]).copy())
             sorted_angles_sequential = sort((self.angles[chi_num,sequential_sim_num,:min_num_angles]).copy())
             #self.rank_order_angles_sequential[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]] = \
             #                         searchsorted(sorted_angles_sequential, self.angles[chi_num,sequential_sim_num,:self.numangles[sequential_sim_num]]) # rank-ordered dihedral angles, each sim ranked separately ... for corr between sims                                                              
             self.rank_order_angles_sequential[chi_num,sequential_sim_num,:min_num_angles] = \
                                      searchsorted(sorted_angles_sequential, self.angles[chi_num,sequential_sim_num,:min_num_angles]) # rank-ordered dihedral angles, each sim ranked separately ... for corr between sims                                                              
             #print self.rank_order_angles[chi_num,sim_num,:]
	     #print "\n"
             #now for the weights
             for i in range(self.numangles[sequential_sim_num]):
                    #self.rank_order_angles_weights[sequential_sum_num,i] = \
                    #    self.weights[sequential_sim_num, self.rank_order_angles[chi_num,sequential_sim_num,i]] 
                    self.rank_order_angles_sequential_weights[sequential_sim_num,i]  = \
                        self.weights[sequential_sim_num, self.rank_order_angles_sequential[chi_num,sequential_sim_num,i]] 
                       
         
         for bootstrap in range(bootstrap_sets):
             if num_convergence_points > 1:
                convergence_interval = int(min_num_angles * 1.0 / num_convergence_points) * (bootstrap + 1) # if using bootstraps for convergence rather than subsampling, use numangles proportional to bootstrap number
             else:
                convergence_interval = min_num_angles                                                 # otherwise use minimum number of angles per sim
             boot_numangles  = 0
             for boot_sim_num in range(bootstrap_choose):
                 sequential_sim_num = self.which_runs[bootstrap,boot_sim_num]
                 #boot_numangles += self.numangles[sequential_sim_num]
                 boot_numangles += convergence_interval
             boot_angles = zeros((bootstrap_choose,boot_numangles),float64)
             boot_weights = zeros((bootstrap_choose,boot_numangles),float64)
             for boot_sim_num in range(bootstrap_choose):
                 #sequential_sim_num = self.which_runs[bootstrap,boot_sim_num]
                 #copy angles
                 #self.boot_sorted_angles[chi_num,bootstrap,:]
                 #boot_angles[boot_sim_num,:self.numangles[sequential_sim_num]]= self.angles[chi_num, self.which_runs[bootstrap,boot_sim_num], :self.numangles[sequential_sim_num]]
                 #boot_weights[boot_sim_num,:self.numangles[sequential_sim_num]]= self.weights[self.which_runs[bootstrap,boot_sim_num], :self.numangles[sequential_sim_num]]
                 boot_angles[boot_sim_num,:convergence_interval]= self.angles[chi_num, self.which_runs[bootstrap,boot_sim_num], :convergence_interval]
                 boot_weights[boot_sim_num,:convergence_interval]= self.weights[self.which_runs[bootstrap,boot_sim_num], :convergence_interval]   
             # resize for sorting so the extra angle slots go into axis 0 and so will appear after all the data
             boot_flat = resize(swapaxes(boot_angles,0,1),(boot_numangles))
             boot_weights_flat = resize(swapaxes(boot_weights,0,1),(boot_numangles))
             print "shape of boot_flat: "+str(shape(boot_flat))
             print "boot_numangles: "+str(boot_numangles)
             self.boot_sorted_angles_weights[bootstrap,:boot_numangles] = [ boot_weights_flat[i] for i in argsort(boot_flat)] #weights doesn't depend on which chi this is
             self.boot_sorted_angles[chi_num,bootstrap,:boot_numangles] = (sort(boot_flat))[:boot_numangles]
             self.boot_ranked_angles[chi_num,bootstrap,:boot_numangles] = searchsorted( \
                 self.boot_sorted_angles[chi_num,bootstrap,:boot_numangles], \
                 boot_flat)
             self.boot_weights[bootstrap,:boot_numangles] = resize(boot_weights_flat,(boot_numangles))
             self.numangles_bootstrap[bootstrap] = boot_numangles
             print "boot ranked angles, chi_num: "+str(chi_num)+" bootstrap: "+str(bootstrap)
             print self.boot_ranked_angles[chi_num,bootstrap,:boot_numangles]
             if (chi_num == 0):
                    print "boot weights, bootstrap:"+str(bootstrap)
                    print self.boot_weights[bootstrap,:boot_numangles]
                    print "boot sorted angles weights, bootstrap:"+str(bootstrap)
                    print self.boot_sorted_angles_weights[bootstrap]
             
       return
   
   def __init__(self,myname,mynum,mychain,xvg_resnum,basedir,num_sims,max_angles,xvgorpdb,binwidth,sigalpha=1,
                permutations=0,phipsi=0,backbone_only=0,adaptive_partitioning=0,which_runs=None,pair_runs=None,bootstrap_choose=3,
                calc_variance=False, all_angle_info=None, xvg_chidir = "/dihedrals/g_chi/", skip=1, skip_over_steps=0, last_step=None, calc_mutinf_between_sims="yes", max_num_chis=99,
                sequential_res_num = 0, pdbfile = None, xtcfile = None, output_timeseries = "no", minmax=None, bailout_early = False, lagtime_interval = None, markov_samples = 250, num_convergence_points=1, cyclic_permut=False):
      global xtc_coords 
      global last_good_numangles # last good value for number of dihedrals
      global NumChis, NumChis_Safe
      self.name = myname
      self.num = mynum
      self.chain = mychain
      self.xvg_basedir = basedir
      self.xvg_chidir = xvg_chidir
      self.xvg_resnum = xvg_resnum
      self.sequential_res_num = sequential_res_num
      self.backbone_only = backbone_only
      self.phipsi = phipsi
      self.max_num_chis = max_num_chis
      self.markov_samples = markov_samples
      coarse_discretize = None
      split_main_side = None

      # we will look at mutual information convergence by taking linear subsets of the data instead of bootstraps, but use the bootstraps data structures and machinery. The averages over bootstraps then won't be meaningful
      # however the highest number bootstrap will contain the desired data -- this could be fixed later at the bottom of the code if desired
      # I also had to change some things in routines above that this code references in order to change numangles_bootstrap. We will essentially look at convergence by only looking at subsets of the data
      # in the weaves below, numangles will vary with 


      if(phipsi >= 0): 
             try:
                    self.nchi = self.get_num_chis(myname) * (1 - backbone_only) + phipsi * self.has_phipsi(myname)
             except:
                    NumChis = NumChis_Safe #don't use Ser/Thr hydroxyls for pdb trajectories
                    self.nchi = NumChis[myname] * (1 - backbone_only) + phipsi * self.has_phipsi(myname)
      elif(phipsi == -2):
             split_main_side = True
             if(self.chain == "S"):
                    self.nchi =  self.get_num_chis(myname)
             else:
                    self.nchi = 2 * self.has_phipsi(myname)
      elif(phipsi == -3):
             self.nchi = 3 #C-alpha x, y, z
      elif(phipsi == -4):
             print "doing analysis of stress data"
             self.nchi = 1 # just phi as a placeholder for a single variable
      else:             #coarse discretize phi/psi into 4 bins: alpha, beta, turn, other
             self.nchi = self.get_num_chis(myname) * (1 - backbone_only) + 1 * self.has_phipsi(myname)
             coarse_discretize = 1
             phipsi = 1
      if(xtcfile != None):
          self.nchi = 3 # x, y, z
      self.symmetry = ones((self.nchi),int16)
      self.numangles = zeros((num_sims),int32)
      self.num_sims = num_sims
      self.which_runs = array(which_runs)
      which_runs = self.which_runs
      #which_runs=array(self.which_runs)
      self.pair_runs = pair_runs
      self.permutations= permutations
      self.calc_mutinf_between_sims = calc_mutinf_between_sims
      if(bootstrap_choose == 0):
        bootstrap_choose = num_sims
      #print "bootstrap set size: "+str(bootstrap_choose)+"\n"
      #print "num_sims: "+str(num_sims)+"\n"
      #print self.which_runs
      #print "\n number of bootstrap sets: "+str(len(self.which_runs))+"\n"

      #check for free memory at least 15%
      #check_for_free_mem()
      
      #allocate stuff
      bootstrap_sets = self.which_runs.shape[0]

      #check num convergence points
      if num_convergence_points > 1:
             assert(num_convergence_points == bootstrap_sets)

      self.entropy =  zeros((bootstrap_sets,self.nchi), float64)
      self.entropy2 =  zeros((bootstrap_sets,self.nchi), float64) #entropy w/fewer bins
      self.entropy3 =  zeros((bootstrap_sets,self.nchi), float64) #entropy w/fewer bins
      self.entropy4 =  zeros((bootstrap_sets,self.nchi), float64) #entropy adaptive
      self.var_ent =  zeros((bootstrap_sets,self.nchi), float64)
      self.numangles_bootstrap = zeros((bootstrap_sets),int32)
      print "\n#### Residue: "+self.name+" "+self.num+" "+self.chain+" torsions: "+str(self.nchi), utils.flush()
      binwidth = float(binwidth)
      bins = arange(0,360, binwidth) #  bin edges global variable
      nbins=len(bins) # number of bins
      nbins_cor = int(nbins * FEWER_COR_BTW_BINS);
      self.nbins=nbins
      self.nbins_cor=nbins_cor
      sqrt_num_sims=sqrt(num_sims)
      self.chi_pop_hist=zeros((bootstrap_sets, self.nchi,nbins),float64)
      self.chi_counts=zeros((bootstrap_sets, self.nchi, nbins), float64) #since these can be weighted in advanced sampling
      #self.chi_var_pop=zeros((bootstrap_sets, self.nchi,nbins),float64)
      self.chi_pop_hist_sequential=zeros((num_sims, self.nchi, nbins_cor), float64)
      num_histogram_sizes_to_try = 2  # we could try more and pick the optimal size
      self.chi_counts_sequential=zeros((num_sims, self.nchi, nbins_cor), float64) #half bin size
      self.chi_counts_sequential_varying_bin_size=zeros((num_histogram_sizes_to_try, num_sims, self.nchi, int(nbins*(num_histogram_sizes_to_try/2)) ), float64) #varying bin size
      self.angles_input = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles, with a bigger array than will be needed later

      #self.sorted_angles = zeros((self.nchi,num_sims,max_angles),float64) # the dihedral angles sorted
      self.ent_hist_left_breaks = zeros((self.nchi, nbins * MULT_1D_BINS + 1),float64)
      self.adaptive_hist_left_breaks = zeros((bootstrap_sets, nbins + 1),float64) #nbins plus one to define the right side of the last bin
      self.adaptive_hist_left_breaks_sequential = zeros(( num_sims, nbins_cor + 1 ),float64)  #nbins_cor plus one to define the right side of the last bin
      self.adaptive_hist_binwidths = zeros((bootstrap_sets, nbins ),float64) 
      self.adaptive_hist_binwidths_sequential = zeros(( num_sims, nbins_cor ),float64)
      self.ent_hist_binwidths = zeros((bootstrap_sets, self.nchi, nbins * MULT_1D_BINS),float64)
      self.ent_from_sum_log_nn_dists = zeros((bootstrap_sets, self.nchi, MAX_NEAREST_NEIGHBORS),float64)
      self.minmax = zeros((2,self.nchi))
      self.minmax[1,:] += 1 #to avoid zero in divide in expand_contract angles

      #transition matrix
      #self.bins_transition_matrix = zeros((self.nchi, bootstrap_sets, nbins, nbins),float64)
      self.slowest_implied_timescale = ones((self.nchi, bootstrap_sets),float64) * int(min(self.numangles))
      self.slowest_lagtime = zeros((self.nchi, bootstrap_sets),float64) 
      self.mutinf_autocorr_time = zeros((self.nchi, bootstrap_sets),float64)  #easy to do while making transition matrix
      self.angles_autocorr_time = zeros((self.nchi, bootstrap_sets),float64)  #easy to do while making transition matrix
      self.bins_autocorr_time = zeros((self.nchi, bootstrap_sets),float64)  #easy to do while making transition matrix
      #count_matrix_multiple_lagtimes = zeros((self.nchi, bootstrap_sets, NUM_LAGTIMES, nbins,nbins ), float64)
      transition_matrix_multiple_lagtimes = zeros((self.nchi, bootstrap_sets, NUM_LAGTIMES, nbins,nbins ), float64)
      #mutinf_autocorrelation_vs_lagtime =  zeros((self.nchi, bootstrap_sets, NUM_LAGTIMES, nbins,nbins ), float64)
      

      self.transition_matrix = zeros((self.nchi, bootstrap_sets, nbins,nbins ), float64)
      #
      tau_lagtimes = zeros((NUM_LAGTIMES+1),float64)
      ### load DATA
      if(xvgorpdb == "xvg"):
         self._load_xvg_data(basedir, num_sims, max_angles, xvg_chidir, skip,skip_over_steps,last_step, coarse_discretize, split_main_side)
      if(xvgorpdb == "pdb"):
         self._load_pdb_data(all_angle_info, max_angles)
      if(xvgorpdb == "xtc"):
         self.load_xtc_data(basedir, num_sims, max_angles, xvg_chidir, skip, skip_over_steps, pdbfile, xtcfile)

      #resize angles array to get rid of trailing zeros, use minimum number
      print "weights"
      print self.weights

      #print "resizing angles array, and creating arrays for adaptive partitioning" 
      min_num_angles = int(min(self.numangles))
      max_angles = int(min_num_angles)
      if(min_num_angles > 0):
             last_good_numangles = min_num_angles
      self.angles = zeros((self.nchi, num_sims, min_num_angles))
      angles_autocorrelation = zeros((self.nchi, bootstrap_sets, min_num_angles), float64)
      #bins_autocorrelation =   zeros((self.nchi, bootstrap_sets, min_num_angles), float64)
      self.boot_sorted_angles = zeros((self.nchi,bootstrap_sets,bootstrap_choose*max_angles),float64)
      self.boot_ranked_angles = zeros((self.nchi,bootstrap_sets,bootstrap_choose*max_angles),int32) 
      self.boot_weights = zeros((bootstrap_sets,bootstrap_choose*max_angles),float64) 
      self.rank_order_angles = zeros((self.nchi,num_sims,max_angles),int32) # the dihedral angles
                                                                        # rank-ordered with respect to all sims together
      self.rank_order_angles_sequential = zeros((self.nchi,num_sims,max_angles),int32) # the dihedral angles
                                                                        # rank-ordered for each sim separately
                                                                        # for mutinf between sims
      #max_num_angles = int(max(self.numangles))
      max_num_angles = int(min(self.numangles)) #to avoid bugs
      self.bins = zeros((self.nchi, self.permutations + 1, bootstrap_sets, bootstrap_choose * max_num_angles ), int8) # the bin for each dihedral
      self.simbins = zeros((self.nchi, self.permutations + 1, num_sims, max_num_angles), int8)
      self.counts=zeros((bootstrap_sets,self.nchi,MULT_1D_BINS * nbins),float64) # number of counts per bin
      self.counts2=zeros((bootstrap_sets,self.nchi, nbins),float64) # number of counts per bin w/fewer bins
      self.counts3=zeros((bootstrap_sets,self.nchi, SAFE_1D_BINS),float64) # number of counts per bin w/ even fewer bins
      self.ent_markov_boots=zeros((self.nchi, bootstrap_sets, markov_samples), float64)      

      #counts_marginal=zeros((bootstrap_sets,self.nchi,nbins),float32) # normalized number of counts per bin, 
      counts_adaptive=zeros((bootstrap_sets,self.nchi,MULT_1D_BINS * nbins),float64)

      #print "initialized angles_new array"
      self.numangles[:] = min(self.numangles)
      #print "new numangles"
      #print self.numangles
      for mychi in range(self.nchi):
             for num_sim in range(num_sims):
                    self.angles[mychi,num_sim,:min_num_angles] = self.angles_input[mychi,num_sim,:min_num_angles]
      #print "done copying angles over"
      del self.angles_input #clear up memory space
      #print self.angles
      
      
      self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8) # the bin for each dihedral
      self.chi_counts_markov=zeros((self.nchi, bootstrap_sets, self.markov_samples, nbins), int64) #since these can be weighted in advanced sampling

      if ((self.name in ("GLY", "ALA")) and (phipsi == 0 or phipsi == -2)): 
           # First prepare chi pop hist and chi pop hist sequential needed for mutual information -- just dump everything into one bin, 
           # giving entropy zero, which should also give MI zero, but serve as a placeholder in the mutual information matrix
           if(last_good_numangles > 0):  
                  self.numangles[:] = last_good_numangles # dummy value  
           self.numangles_bootstrap[:] = bootstrap_choose *  int(min(self.numangles))
           for mybootstrap in range(bootstrap_sets):
                  self.chi_counts[mybootstrap,:,0] = bootstrap_choose *  int(min(self.numangles)) # should be numangles_bootstrap but this isn't initialized yet
                  self.chi_pop_hist[mybootstrap,:,0] = 1.0
                  
           for mysim in range(num_sims):
                  self.chi_counts_sequential[mysim,:,0] = int(min(self.numangles)) 
                  self.chi_pop_hist_sequential[mysim,:,0] = 1.0
           return   #if the side chains don't have torsion angles, drop out

      self.sorted_angles = zeros((self.nchi, sum(self.numangles)),float64)
      
      if(xvgorpdb == "xvg" or (xvgorpdb == "pdb" and phipsi != -3)): #if not using C-alphas from pdb
             self.correct_and_shift_angles(num_sims,bootstrap_sets,bootstrap_choose, coarse_discretize)
      elif(xvgorpdb == "xtc" or (xvgorpdb == "pdb" and phipsi == -3)) : #if using xtc cartesians or pdb C-alphas 
             self.correct_and_shift_carts(num_sims,bootstrap_sets,bootstrap_choose, num_convergence_points)
             
      if(minmax is None):
             print "getting min/max values"
             mymin = zeros(self.nchi)
             mymax = zeros(self.nchi)
             for mychi in range(self.nchi):
                    mymin[mychi] =  min((self.angles[mychi,:,:min(self.numangles)]).flatten())
                    mymax[mychi] =  max((self.angles[mychi,:,:min(self.numangles)]).flatten())
             print "__init__ mymin: "
             print mymin
             print "__init__ mymax: "
             print mymax
             self.minmax[0, :] = mymin
             self.minmax[1, :] = mymax
             for mychi in range(self.nchi):
                    if(self.minmax[1,mychi] - self.minmax[0,mychi] <= 0):
                           self.minmax[1,mychi] = self.minmax[0,mychi] + 1
             print self.minmax
      else:
             self.minmax = minmax
             self.expand_contract_data(num_sims,bootstrap_sets,bootstrap_choose)
      

      #now rank-order the data in case adaptive partitioning is needed
      self.sort_angles(num_sims,bootstrap_sets,bootstrap_choose, num_convergence_points, coarse_discretize)

      
      if(bailout_early == True): #if we just want the angles and especially the min and max values without the binning, etc....
             print "bailing out early with min/max values"
             return
      

      inv_binwidth = 1.0 / binwidth
      inv_binwidth_cor = nbins_cor / 360.0;
      #print "binwidth:"+str(binwidth)
      #if(adaptive_partitioning == 1):
      if(VERBOSE >= 2): 
             print "numangles:"+str((self.numangles))+"sum numangles: "+str(sum(self.numangles))+" binwidth = "+str(sum(self.numangles)/nbins)+"\n"
      inv_binwidth_adaptive_bootstraps = ((nbins * 1.0) /self.numangles_bootstrap)  #returns a vector
      if(VERBOSE >= 2): 
             print "inv_binwidth_adaptive_bootstraps: "+str(inv_binwidth_adaptive_bootstraps)
      inv_binwidth_adaptive_sequential = ((nbins_cor * 1.0) /(self.numangles * 1.0))  #returns a vector
      if(VERBOSE >= 2):
             print "inv binwidth adaptive sequential:"
             print inv_binwidth_adaptive_sequential
             print "binwidth adaptive sequential:"
             print 1.0 / inv_binwidth_adaptive_sequential
      
      number_per_ent_bin = sum(self.numangles) / (nbins * MULT_1D_BINS)
      
      if(VERBOSE >=2):
             print shape(self.ent_hist_left_breaks)
             print shape(self.ent_hist_binwidths)
      self.ent_pdf_adaptive=zeros((bootstrap_sets,self.nchi,MULT_1D_BINS * nbins),float64) # normalized number of counts per bin,
      for bootstrap in range(bootstrap_sets):
             for mychi in range(self.nchi):
                    for i in range(nbins*MULT_1D_BINS):   #use sorted angles. Note, we've only sorted angles once, could be done for each bootstrap sample just for the 1D entropies -- sorting each bootstrap sample separately would confuse correlations in rank order space
                           self.ent_hist_left_breaks[mychi,i] = self.sorted_angles[mychi, number_per_ent_bin * i]
                    self.ent_hist_left_breaks[mychi, -1] = self.sorted_angles[mychi,sum(self.numangles) - 1] + 0.0001
                    for i in range(nbins*MULT_1D_BINS):   #use sorted angles. Note, we've only sorted angles once, could be done for each bootstrap sample just for the 1D entropies -- sorting each bootstrap sample separately would confuse correlations in rank order space
                           self.ent_hist_binwidths[:,mychi, i]  = resize((TWOPI / 360.0) * (self.ent_hist_left_breaks[mychi, i + 1] - self.ent_hist_left_breaks[mychi, i]) / self.symmetry[mychi], self.ent_hist_binwidths.shape[0])
                    self.ent_hist_binwidths[:,mychi, -1]  = resize((TWOPI / 360.0) * (self.ent_hist_left_breaks[mychi, -1] - self.ent_hist_left_breaks[mychi, i]) / self.symmetry[mychi], self.ent_hist_binwidths.shape[0])


      ## NOTE: though we cannot guarantee complete uniform density in each bin, we approximate it as such
      ## find left breaks using cumulative density from sorted angles in each bootstrap and their weights (sorted according to angle)
      

      code_cumulative_boot_weights = """
              // weave_cumulative_boot_weights
              int mynumangles, angle;
              double weight, cumul_weight;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = *(numangles_bootstrap + mybootstrap);
                *(cumulative_boot_weights + mybootstrap*bootstrap_choose*max_angles + 0) = *(boot_sorted_angles_weights + mybootstrap*bootstrap_choose*max_angles + 0);
                for(int angle=1; angle < mynumangles; angle++)
                {
                  weight = *(boot_sorted_angles_weights + mybootstrap*bootstrap_choose*max_angles + angle) ;
                  cumul_weight = *(cumulative_boot_weights + mybootstrap*bootstrap_choose*max_angles + angle - 1) + weight;
                  //printf("angle: %i weight: %f cumul_weight: %f\\n ",angle,weight, cumul_weight);
                  *(cumulative_boot_weights + mybootstrap*bootstrap_choose*max_angles + angle) = cumul_weight ;
                }
              }
      """

      numangles_bootstrap = self.numangles_bootstrap # this will be overwritten in the later weaves, but overwritten correctly before each weave is done
      boot_sorted_angles_weights = self.boot_sorted_angles_weights

      print "shape of adaptive hist left breaks:"
      print shape(self.adaptive_hist_left_breaks)
      print "shape of adaptive_hist_binwidths:"
      print shape(self.adaptive_hist_binwidths)

      cumulative_boot_weights = zeros((bootstrap_sets, bootstrap_choose*max_angles), float64)

      weave.inline(code_cumulative_boot_weights, ['numangles_bootstrap', 'nbins', 'bootstrap_sets','bootstrap_choose','boot_sorted_angles_weights','cumulative_boot_weights','max_angles'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])

      # We will use the cumulative distribution of the uniform distribution to set the amount of weight (density) in each bin constant, and find appropriate bin breaks based on the c.d.f. of the bootstrap-sorted angles
      # NEED TO MAKE THIS IS CONSISTENT FOR BOOTSTRAPS THAT ARE SUBSETS OF THE DATA -- i.e. DIFFERENT SIZE
      for bootstrap in range(bootstrap_sets):
          cumulative_distribution_numangles_bootstrap = self.numangles_bootstrap[bootstrap] #* (self.numangles_bootstrap[bootstrap] + 1) / 2  # Sum_1^n of i = n(n+1)/2   
          if(VERBOSE >=2 ):
                 print "adaptive histogram bin left-side breaks: bootstrap "+str(bootstrap)             
                 print "cumulative distribution numangles bootstrap"
                 print cumulative_distribution_numangles_bootstrap
                 print "cumulative weights bootstrap"
                 print cumulative_boot_weights[bootstrap,self.numangles_bootstrap[bootstrap] - 1 ]
          #assert(cumulative_boot_weights[bootstrap,-1] <= cumulative_distribution_numangles_bootstrap + 1) #some tolerance
          assert(cumulative_boot_weights[bootstrap,self.numangles_bootstrap[bootstrap] - 1] <= cumulative_distribution_numangles_bootstrap) #no tolerance
          number_per_ent_bin = ((cumulative_distribution_numangles_bootstrap * 1.0) / (nbins * 1.0)) # to make sure we do float division instead of integer division
          if(VERBOSE >=2):
                 print "number per ent bin: "+str(number_per_ent_bin)
                 #rv_adaptive = stats.rv_discrete(name='adaptiveweights',values=(range(self.numangles_bootstrap[bootstrap]),self.boot_sorted_angles_weights[bootstrap,:self.numangles_bootstrap[bootstrap]]))   
                 print cumulative_boot_weights[bootstrap]
          self.adaptive_hist_left_breaks[bootstrap,0] = 0
          for i in range(1,self.nbins):   #use cdf of weights from bootstraps                 
              self.adaptive_hist_left_breaks[bootstrap,i] = searchsorted( cumulative_boot_weights[bootstrap, :self.numangles_bootstrap[bootstrap] - 1 ], number_per_ent_bin * i) + SMALL #since searchsorted returns an array, have to get element 0 from it, corresponding to the insertion point of "number_per_ent_bin * i"
              if(VERBOSE >=2):
                     print "adaptive histogram bin left-side breaks: bin "+str(i)+ " value: "+str(self.adaptive_hist_left_breaks[bootstrap,i])
          self.adaptive_hist_left_breaks[bootstrap,-1] = self.numangles_bootstrap[bootstrap] + SMALL
          if (VERBOSE >= 2): print "adaptive histogram bin left-side breaks: bin "+str(nbins)+ " value: "+str(self.adaptive_hist_left_breaks[bootstrap,nbins])

      #### adaptive_hist_left_breaks sequential will be different for each sim
      #### does this need to be modified for num_convergence_points?
      code_cumulative_weights = """
              // weave_cumulative_weights
              int angle;
              double weight;
              for(int simnum=0;simnum < num_sims;simnum++) {
                 *(cumulative_weights + simnum*mynumangles + 0) = *(rank_order_angles_sequential_weights + simnum*mynumangles + 0);
                 for(int angle=1; angle < mynumangles; angle++)
                 {
                     weight = *(rank_order_angles_sequential_weights + simnum*mynumangles  + angle) ;
                     *(cumulative_weights + simnum*mynumangles + angle) =  *(cumulative_weights + simnum*mynumangles +angle - 1) + weight;
                 }             
              }
          """

      if(VERBOSE >=2):  
             print shape(self.adaptive_hist_left_breaks_sequential)
      #print shape(self.adaptive_hist_binwidths)
      cumulative_weights = zeros((num_sims,min(self.numangles)), float64)
      mynumangles = int(min(self.numangles))
      rank_order_angles_sequential_weights = self.rank_order_angles_sequential_weights
      weave.inline(code_cumulative_weights, ['mynumangles','num_sims','rank_order_angles_sequential_weights','cumulative_weights'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
      for mysim in range(num_sims):
             numangles_sum = int(sum(self.numangles)) # this will be overwritten in the later weaves, but overwritten correctly before each weave is done
             numangles_bootstrap = self.numangles_bootstrap # this will be overwritten in the later weaves, but overwritten correctly before each weave is done
             sorted_angles_weights = self.sorted_angles_weights
             

             cumulative_distribution_numangles = mynumangles * 1.0 #* (self.numangles_bootstrap[bootstrap] + 1) / 2  # Sum_1^n of i = n(n+1)/2
             if(VERBOSE >= 2):
                    print "cumulative distribution numangles"
                    print cumulative_distribution_numangles
                    print "cumulative weights"
                    print cumulative_weights[-1]
                    #assert(cumulative_weights[mysim,-1] <= cumulative_distribution_numangles + 1 )            # to have some tolerance
             assert(cumulative_weights[mysim,-1] <= cumulative_distribution_numangles )            # no tolerance
             number_per_ent_bin_sequential = (cumulative_distribution_numangles * 1.0) / (nbins_cor * 1.0)
             self.adaptive_hist_left_breaks_sequential[mysim,0] = 0
             for i in range(1,nbins_cor):   #use cdf of weights from bootstraps
                           self.adaptive_hist_left_breaks_sequential[mysim,i] = searchsorted(cumulative_weights[mysim,:], number_per_ent_bin_sequential * i) + SMALL #since searchsorted returns an array, have to get element 0 from it, corresponding to the insertion point of "number_per_ent_bin_sequential * i"
             self.adaptive_hist_left_breaks_sequential[mysim,-1] = min(self.numangles) + SMALL


      if(VERBOSE >= 2):
         print "Average Number of Angles Per Entropy Bin:\n"
         print number_per_ent_bin
         print "Entropy Histogram Left Breaks:\n"
         print self.ent_hist_left_breaks[:,:]
         print "Entropy Histogram Binwidths:\n"
         print self.ent_hist_binwidths[:,:,:]
         print "Sum of Histogram Binwidhths:\n"
         print sum(self.ent_hist_binwidths,axis=-1)
         #use these left histogram breaks and binwidths below
         print "Adaptive hist left break :"
         print self.adaptive_hist_left_breaks[:,:]
         print "Adaptive hist left breaks sequential:"
         print self.adaptive_hist_left_breaks_sequential[:,:]
         print "Number per ent bin sequential:"
         print number_per_ent_bin_sequential
      

      
      if VERBOSE >= 5: 
             print "Angles: ", map(int, list(self.angles[0,0,0:self.numangles[0]])), "\n\n"
             print self.angles
      assert( all( self.angles >= 0))
      assert( all( self.angles <= 360))
      
      # find the bin for each angle and the number of counts per bin
      # need to weave this loop for speed
      
      
#
#
#work zone for weaved replacement of loops below
#
#
#def binsingle(angle,inv_binwidth):
#   if angle == -180: angle = 180
#   return int(floor((angle-0.00001 + 180)*inv_binwidth)) #so we don't get an overshoot if angle is exactly 180
#
#def binsingle_adaptive(angle,inv_binwidth):
#    #print "rank: "+str(angle)+" binwidth: "+str(1.0/inv_binwidth)+" bin: "+str(int(floor(angle*inv_binwidth)))
#    return int(floor(angle*inv_binwidth)) #here "angle" is a rank-order for the angle over sum(numangles)
#
#
#                    printf("simnum %d mynumangles %d\\n",simnum,mynumangles);
#                                          printf("bin %d \\n",bin1);

      
      code_nonadaptive_doublebins = """
              // weave8
              int mynumangles, mynumangles_sum, bin1, bin2, bin3, simnum = 0; //bin2 and bin3 are for lower-res histograms
              double angle = 0;
              int mybootstrap, mysim, anglenum = 0;
              for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;   
                // #pragma omp parallel for private(mysim, simnum, mynumangles, mynumangles_sum, anglenum, angle, bin1, bin2, bin3)
                for(mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);

                    for (anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      if(angle > 360) angle = angle - 360;
                      if(angle <= 0.000001) angle = 0.0000011;
                      bin1 = int((angle-0.000001)*inv_binwidth*MULT_1D_BINS);
                      bin2 = int((angle-0.000001)*inv_binwidth);
                      bin3 = int((angle-0.000001)*SAFE_1D_BINS/360);
                     
                      // #pragma omp atomic
                      *(counts  + mybootstrap*nchi*nbins*MULT_1D_BINS + mychi*nbins*MULT_1D_BINS + bin1) += 1;
                      // #pragma omp atomic
                      *(counts2 + mybootstrap*nchi*nbins              + mychi*nbins              + bin2) += 1;
                      // #pragma omp atomic
                      *(counts3 + mybootstrap*nchi*SAFE_1D_BINS       + mychi*SAFE_1D_BINS       + bin3) += 1;
                      }
                    // #pragma omp atomic
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                    }
                
              }
              """

           #for this next one, counts now will be a properly normalized pdf
           #bin1 = int((angle-0.0000001 + 180)*inv_binwidth*MULT_1D_BINS);
      code_adaptive_doublebins = """
              // weave8b
              int mynumangles, mynumangles_sum, bin1, simnum, found = 0;
              double angle, bin_bottom, bin_top, binwidth = 0;
              int mybootstrap, mysim, anglenum = 0;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;   
                // #pragma omp parallel for private(mynumangles, mysim, simnum, mynumangles_sum, anglenum,angle,found,bin_bottom,bin1,bin_top,binwidth)
                for(mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);   
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);
                    for (anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      //angle = 360.0 * (anglenum / mynumangles); 
                      //if(mybootstrap==0 && mysim == 0) printf("angle:%f\\n",angle);
                      if(angle <= 0) angle = 0;
                      if(angle > 360) angle = 360;
                      found = 0;
                      bin_bottom = *(ent_hist_left_breaks + 
                                     + mychi*(nbins*MULT_1D_BINS + 1) + 0);
                      for(bin1=0; bin1 < nbins*MULT_1D_BINS; bin1++) {
                          bin_top = *(ent_hist_left_breaks + 
                                     + mychi*(nbins*MULT_1D_BINS + 1) + (bin1 + 1));
                          if(found == 0 && angle >= bin_bottom && angle < bin_top) {
                              // #pragma omp atomic
                              *(counts_adaptive + mybootstrap*nchi*nbins*MULT_1D_BINS
                                     + mychi*nbins*MULT_1D_BINS + bin1) += 1;
                              found = 1;
                              //if(mybootstrap==0 && mysim == 0 ) printf("bin bot:%f bin top:%f\\n", bin_bottom, bin_top);
                          
                          }
                          bin_bottom = bin_top;
                          if(found == 1) break;
                       }
                              
                    }
                    // #pragma omp atomic
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                }
                for(bin1 = 0; bin1 < nbins*MULT_1D_BINS; bin1++) {
                    binwidth = *(ent_hist_binwidths +  mybootstrap*nchi*nbins*MULT_1D_BINS
                                  + mychi*nbins*MULT_1D_BINS + bin1);
                    // #pragma omp atomic
                    *(counts_adaptive + mybootstrap*nchi*nbins*MULT_1D_BINS + mychi*nbins*MULT_1D_BINS + bin1)
                                    /= (*(numangles_bootstrap + mybootstrap) * binwidth );
                    //printf("%f , %f \\n",*(counts_adaptive + mybootstrap*nchi*nbins*MULT_1D_BINS + mychi*nbins*MULT_1D_BINS + bin1),binwidth);
                }
              }
              """
      
      code_nonadaptive_doublebins_weights = """
              // weave8
              int mynumangles, mynumangles_sum, bin1, bin2, bin3, simnum; //bin2 and bin3 are for lower-res histograms
              double angle, weight;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;   
                for(int mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);

                    for (int anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      weight = *(weights + simnum*max_angles + anglenum); //assumes mynumangles same for all sims
                      if(angle > 360) angle = angle - 360;
                      if(angle <= 0.000001) angle = 0.0000011;
                      bin1 = int((angle-0.000001)*inv_binwidth*MULT_1D_BINS);
                      bin2 = int((angle-0.000001)*inv_binwidth);
                      bin3 = int((angle-0.000001)*SAFE_1D_BINS/360);
                     
                      *(counts  + mybootstrap*nchi*nbins*MULT_1D_BINS + mychi*nbins*MULT_1D_BINS + bin1) += 1.0 * weight;
                      *(counts2 + mybootstrap*nchi*nbins              + mychi*nbins              + bin2) += 1.0 * weight;
                      *(counts3 + mybootstrap*nchi*SAFE_1D_BINS       + mychi*SAFE_1D_BINS       + bin3) += 1.0 * weight;
                      }
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                    }
                
              }
              """


      code_adaptive_singlebins_weights = """
              // weave8b
              int mynumangles, mynumangles_sum, bin1, simnum, found, angle;
              double bin_bottom, bin_top, binwidth, weight;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;   
                for(int mysim=0; mysim < bootstrap_choose; mysim++) {
                    found = 0;
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);   
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);
   
                    for (int anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(boot_ranked_angles + mychi*bootstrap_sets*bootstrap_choose*max_angles + mybootstrap*bootstrap_choose*max_angles +mynumangles_sum + anglenum);
                      weight = *(boot_weights + mybootstrap*bootstrap_choose*max_angles + mynumangles_sum + anglenum); //assumes mynumangles same for all dihedrals
                      //if(mybootstrap==0 && mysim == 0) printf("angle:%i, weight:%f \\n",angle, weight);
                      found = 0;
                      bin_bottom = *(adaptive_hist_left_breaks + mybootstrap*(nbins+1) 
                                     + 0);
                      for(int bin1=0; bin1 < nbins; bin1++) {
                          //printf("bin: %i",bin1);
                          bin_top = *(adaptive_hist_left_breaks + mybootstrap*(nbins+1) 
                                     + bin1 +1 );
                          if(found == 0 && angle >= bin_bottom && angle < bin_top) {
                              *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1.0 * weight;
                              *(chi_counts  + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1.0 * weight;
                              //temp = mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles
                              //   + mybootstrap*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum;

                              *(bins + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles
                                 + mybootstrap*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;
                      
                              found = 1;
                              //if(mybootstrap==0 && mysim == 0 ) printf("bin bot:%f bin top:%f\\n", bin_bottom, bin_top);
                              
                          }
                          bin_bottom = bin_top;
                          if(found == 1) break;
                       }
                       
                              
                    }
                    //for(int bin1=0; bin1 < nbins; bin1++) {
                    //       printf("bootstrap: %d, chi counts: bin: %d == %f \\n",mybootstrap, bin1,*(chi_counts + mybootstrap*nchi*nbins + mychi*nbins + bin1));
                    //}
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                }
                for(bin1 = 0; bin1 < nbins; bin1++) {
                    //binwidth = *(adaptive_hist_binwidths +  mybootstrap*nchi*nbins
                    //             + mychi*nbins + bin1);
                    //*(chi_counts  + mybootstrap*nchi*nbins + mychi*nbins + bin1) = *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1);
                    *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1)
                                    /= (*(numangles_bootstrap + mybootstrap) );
                    //printf("chi counts %f , bin %i \\n",*(chi_counts + mybootstrap*nchi*nbins + mychi*nbins + bin1),bin1);
                }
              }
              """

      code_nearest_neighbor_1D = """
          // weave8c2 
          #include <math.h>
              int mynumangles, mynumangles_sum, mynumangles_thisboot, bin1, simnum, found = 0;
              double angle, neighbor_left,neighbor_right,nearest_neighbor;
              double dleft,dright,dmin, logd, dleft_safe, weight;
              double log_nn_dists_sum = 0.0;
              double Lk_minus1 = 0.0;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles_thisboot = *(numangles_bootstrap + mybootstrap);
                for(int k_nn = 1; k_nn <= MAX_NEAREST_NEIGHBORS; k_nn++)
                {
                 log_nn_dists_sum = 0;
                 
                 double neighborlist[2*k_nn + 1];
                 double temp_neighbor_dist;
                 double dist_to_neighbors[2*k_nn + 1];
                 
                 for (int anglenum=0; anglenum < mynumangles_thisboot; anglenum++) {

                      angle = *(boot_sorted_angles + mychi*bootstrap_sets*bootstrap_choose*max_angles + mybootstrap*bootstrap_choose*max_angles + anglenum);
                      
                      // make a list of candidate nearest neighbors
                      
                      for(int myk = -k_nn; myk <= k_nn; myk++)
                      {
                       if(myk < 0 && anglenum < abs(myk)) neighborlist[k_nn +myk] = -720;
                       else if(myk > 0 && anglenum > mynumangles_thisboot - myk - 1) neighborlist[k_nn +myk] = 720;
                       else if(myk == 0) neighborlist[k_nn + myk] = 9999;
                       else neighborlist[k_nn + myk] = *(boot_sorted_angles + mychi*bootstrap_sets*bootstrap_choose*max_angles + mybootstrap*bootstrap_choose*max_angles + anglenum + myk);
                       if (myk < 0 )      dist_to_neighbors[k_nn + myk] = angle - neighborlist[k_nn + myk];
                       else if (myk == 0) dist_to_neighbors[k_nn + myk] = 999; // don't want vanishing nn dists
                       else if (myk > 0 ) dist_to_neighbors[k_nn + myk] = neighborlist[k_nn + myk] - angle;
                      }

                      
                      
                      // find the distance to the "k_nn"th nearest neighbor

                      // first, sort using bubble sort
                      // ANN nearest neighbor package uses k-d tree
                      // sort is easier than a k-d tree because it's only 1D :)
                      // http://www.cs.princeton.edu/~ah/alg_anim/gawain-4.0/BubbleSort.html
                      
                      for (int i=0; i< (2*k_nn +1) -1; i++) {
                        for (int j=0; j<(2*k_nn +1) -1 -i; j++)
                           if (dist_to_neighbors[j+1] < dist_to_neighbors[j]) { 
                           temp_neighbor_dist = dist_to_neighbors[j];       
                           dist_to_neighbors[j] = dist_to_neighbors[j+1];
                           dist_to_neighbors[j+1] = temp_neighbor_dist;
                        }
                      }

                     // for(int myk = -k_nn; myk <= k_nn; myk++)
                     // {
                     //    if(anglenum < 10) printf("k:%i dist to neighbor k:%f \\n ",
                     //       k_nn + myk, dist_to_neighbors[k_nn + myk]);
                     // }      

                      dmin = dist_to_neighbors[k_nn - 1 ]; //find kth nearest neighbor
                      if(dmin >= 0.00099) logd = log(dmin);
                      else logd = log(0.00099);
                      //logd = 1000 * log(360.0 / (5005000.0 / ( int((k_nn - 1)/2) + 1)));  //for uniform dist overwrite test
                      log_nn_dists_sum += logd ;

                      //printf("angle:%f nearest:%f next-nearest:%f boot:%i logd:%f, log_nn_dists_sum:%f \\n ",
                      //     angle, dist_to_neighbors[k_nn - 1],dist_to_neighbors[k_nn],mybootstrap, logd, log_nn_dists_sum) ;
                      
                      
                    } // end for(angle ... )
                
                // Now Calculate Entropy From NN Distances For This Bootstrap Sample
                // S = s/n * sum_log_nn_dists + ln (n * vol(s-dim sphere)) -L(k-1)=0 + EULER_GAMMA
                 //L(k-1) = sum(1/i,i=1 .. k)
                 Lk_minus1 = 0.0;
                 if( k_nn > 1) {
                    for(int iter = 1; iter <= k_nn - 1; iter++) {
                           Lk_minus1 += 1.0 / iter; //formula for L(k - 1)
                    }
                 }
                 mynumangles_sum =mynumangles_thisboot;
                 //mynumangles_sum *= 1000; // for uniform dist. overwrite test
                 //printf("log_nn_dists:%f, mynumangles_sum:%i, mybootstrap:%i, mychi:%i, k_nn=%i, Lk_minus1=%f\\n",(float)log_nn_dists_sum, (int)mynumangles_sum, mybootstrap, mychi, k_nn,Lk_minus1);
                  log_nn_dists_sum /= mynumangles_sum;
                  log_nn_dists_sum += log(3.141592653589793238462643383279502884197 / 180.0);
                  log_nn_dists_sum += log( 2 * mynumangles_sum);
                  log_nn_dists_sum += -Lk_minus1 + 0.57721566490153286060651209; //asymptotic bias
                 *(ent_from_sum_log_nn_dists + mybootstrap*nchi*MAX_NEAREST_NEIGHBORS + mychi*MAX_NEAREST_NEIGHBORS + k_nn - 1) = log_nn_dists_sum;
                  log_nn_dists_sum  = 0;
                }
                //*(numangles_bootstrap + mybootstrap) = mynumangles_sum;
              }
              """


      code_nearest_neighbor_1D_weights = """
          // weave8c2 
          #include <math.h>
              int mynumangles, mynumangles_sum, mynumangles_thisboot, bin1, simnum, found;
              double angle, neighbor_left,neighbor_right,nearest_neighbor;
              double dleft,dright,dmin, logd, dleft_safe, weight;
              double log_nn_dists_sum = 0.0;
              double Lk_minus1 = 0.0;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles_thisboot = *(numangles_bootstrap + mybootstrap);
                for(int k_nn = 1; k_nn <= MAX_NEAREST_NEIGHBORS; k_nn++)
                {
                 log_nn_dists_sum = 0;
                 
                 double neighborlist[2*k_nn + 1];
                 double temp_neighbor_dist;
                 double dist_to_neighbors[2*k_nn + 1];
                 
                 for (int anglenum=0; anglenum < mynumangles_thisboot; anglenum++) {

                      angle = *(boot_sorted_angles + mychi*bootstrap_sets*bootstrap_choose*max_angles + mybootstrap*bootstrap_choose*max_angles + anglenum);
                      weight = *(boot_sorted_angles_weights + mybootstrap*bootstrap_choose*max_angles + anglenum); //assumes mynumangles same for all dihedrals
                      // make a list of candidate nearest neighbors
                      
                      for(int myk = -k_nn; myk <= k_nn; myk++)
                      {
                       if(myk < 0 && anglenum < abs(myk)) neighborlist[k_nn +myk] = -720;
                       else if(myk > 0 && anglenum > mynumangles_thisboot - myk - 1) neighborlist[k_nn +myk] = 720;
                       else if(myk == 0) neighborlist[k_nn + myk] = 9999;
                       else neighborlist[k_nn + myk] = *(boot_sorted_angles + mychi*bootstrap_sets*bootstrap_choose*max_angles + mybootstrap*bootstrap_choose*max_angles + anglenum + myk);
                       if (myk < 0 )      dist_to_neighbors[k_nn + myk] = angle - neighborlist[k_nn + myk];
                       else if (myk == 0) dist_to_neighbors[k_nn + myk] = 999; // don't want vanishing nn dists
                       else if (myk > 0 ) dist_to_neighbors[k_nn + myk] = neighborlist[k_nn + myk] - angle;
                      }

                      
                      
                      // find the distance to the "k_nn"th nearest neighbor

                      // first, sort using bubble sort
                      // ANN nearest neighbor package uses k-d tree
                      // sort is easier than a k-d tree because it's only 1D :)
                      // http://www.cs.princeton.edu/~ah/alg_anim/gawain-4.0/BubbleSort.html
                      
                      for (int i=0; i< (2*k_nn +1) -1; i++) {
                        for (int j=0; j<(2*k_nn +1) -1 -i; j++)
                           if (dist_to_neighbors[j+1] < dist_to_neighbors[j]) { 
                           temp_neighbor_dist = dist_to_neighbors[j];       
                           dist_to_neighbors[j] = dist_to_neighbors[j+1];
                           dist_to_neighbors[j+1] = temp_neighbor_dist;
                        }
                      }

                     // for(int myk = -k_nn; myk <= k_nn; myk++)
                     // {
                     //    if(anglenum < 10) printf("k:%i dist to neighbor k:%f \\n ",
                     //       k_nn + myk, dist_to_neighbors[k_nn + myk]);
                     // }      

                      dmin = dist_to_neighbors[k_nn - 1 ]; //find kth nearest neighbor
                      if(dmin >= 0.0000099) logd = log(dmin);
                      else logd = log(0.0000099);
                      //logd = 1000 * log(360.0 / (5005000.0 / ( int((k_nn - 1)/2) + 1)));  //for uniform dist overwrite test
                      log_nn_dists_sum += logd * weight ;

                      //printf("angle:%f nearest:%f next-nearest:%f boot:%i logd:%f, log_nn_dists_sum:%f \\n ",
                      //     angle, dist_to_neighbors[k_nn - 1],dist_to_neighbors[k_nn],mybootstrap, logd, log_nn_dists_sum) ;
                      
                      
                    } // end for(angle ... )
                
                // Now Calculate Entropy From NN Distances For This Bootstrap Sample
                // S = s/n * sum_log_nn_dists + ln (n * vol(s-dim sphere)) -L(k-1)=0 + EULER_GAMMA
                 //L(k-1) = sum(1/i,i=1 .. k)
                 Lk_minus1 = 0.0;
                 if( k_nn > 1) {
                    for(int iter = 1; iter <= k_nn - 1; iter++) {
                           Lk_minus1 += 1.0 / iter; //formula for L(k - 1)
                    }
                 }
                 mynumangles_sum =mynumangles_thisboot;
                 //mynumangles_sum *= 1000; // for uniform dist. overwrite test
                 //printf("log_nn_dists:%f, mynumangles_sum:%i, mybootstrap:%i, mychi:%i, k_nn=%i, Lk_minus1=%f\\n",(float)log_nn_dists_sum, (int)mynumangles_sum, mybootstrap, mychi, k_nn,Lk_minus1);
                  log_nn_dists_sum /= mynumangles_sum;
                  log_nn_dists_sum += log(3.141592653589793238462643383279502884197 / 180.0);
                  log_nn_dists_sum += log( 2 * mynumangles_sum);
                  log_nn_dists_sum += -Lk_minus1 + 0.57721566490153286060651209; //asymptotic bias
                 *(ent_from_sum_log_nn_dists + mybootstrap*nchi*MAX_NEAREST_NEIGHBORS + mychi*MAX_NEAREST_NEIGHBORS + k_nn - 1) = log_nn_dists_sum;
                  log_nn_dists_sum  = 0;
                }
                //*(numangles_bootstrap + mybootstrap) = mynumangles_sum;
              }
              """
      
      
      
      code_nonadaptive_singlebins = """
              // weave9
              int mynumangles, mynumangles_sum, bin1, simnum = 0;
              double angle;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;   
                for (int mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);
                    for (int anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      if(angle > 360) angle = angle - 360;
                      if(angle <= 0.000001) angle = 0.0000011;
                      bin1 = int((angle-0.000001)*inv_binwidth);
                      if(bin1 < 0)
                      {
                         printf("WARNING: bin less than zero");
                         bin1 = 0;
                      }
                      //printf("bootstrap: %4i boot_sim: %4i sim_num: %4i angle: %3.3f, bin: %4i \\n", mybootstrap, mysim, simnum, angle, bin1);  
                      *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1;
                      *(chi_counts + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1;
                      *(bins + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles
                                 + mybootstrap*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;
                      //if(mybootstrap==0 && mysim == 0) printf("angle:%3.3f %3i\\n",angle, bin1);
                      }
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                  }
                for(bin1 = 0; bin1 < nbins; bin1++) {
                    *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) /=  *(numangles_bootstrap + mybootstrap);
                }
              
              }
              """
      #printf("data slot %d bin  %d \\n",temp,bin1);           
      # printf("simnum %d mynumangles %d mynumangles_sum %d\\n",simnum,mynumangles,mynumangles_sum);
      code_adaptive_singlebins = """
              // weave10
              // 
              int mynumangles, mynumangles_sum, bin1, simnum, temp;
              int angle;
              double inv_binwidth_adaptive = 0;
              int mysim, anglenum = 0;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;
                inv_binwidth_adaptive = *(inv_binwidth_adaptive_bootstraps + mybootstrap);
                
                for (mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);
     
                    // Due to the nature of mynumangles_sum, OpenMP parallelization is at this depth
                    // r// #pragma omp parallel for private(anglenum,angle,temp,bin1,inv_binwidth_adaptive)
                    for (anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(boot_ranked_angles + mychi*bootstrap_sets*bootstrap_choose*max_angles + mybootstrap*bootstrap_choose*max_angles +mynumangles_sum + anglenum);
                      //printf("chi:%i sim:%i rank:%i\\n",mychi, simnum, angle); 
                      bin1 = int(double(angle)*inv_binwidth_adaptive);
                      if (bin1 < 0) bin1 = 0;
                      else if (bin1 >= nbins) bin1 = nbins - 1;
                      // #pragma omp atomic
                      *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1;
                      // #pragma omp atomic
                      *(chi_counts  + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1;
                      //temp = mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles
                      //           + mybootstrap*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum;

                      *(bins + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles
                                 + mybootstrap*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;
                      
                      }
                    
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                  }
                for(bin1 = 0; bin1 < nbins; bin1++) {
                    // #pragma omp atomic
                    *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) /=  *(numangles_bootstrap + mybootstrap);
                }
              
              }
              """

      code_nonadaptive_singlebins_sequential = """
              // weave11
              int mynumangles, mynumangles_sum, bin1;
              int nbins_max = nbins_cor*num_histogram_sizes_to_try; // maximum bin size for varying bin size chi counts
              double angle;
              int simnum, anglenum, binmult = 0;
                // #pragma omp parallel for private(simnum,mynumangles,anglenum,angle,bin1,binmult)
                for (simnum=0; simnum < num_sims; simnum++) {
                    mynumangles = *(numangles + simnum);
                    for (anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      if(angle > 360) angle = angle - 360;
                      if(angle <= 0.000001) angle = 0.0000011;
                      bin1 = int((angle-0.000001)*(inv_binwidth_cor)); //chi counts for half bin size
                      
                      // #pragma omp atomic
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1;
                      // #pragma omp atomic
                      *(chi_counts_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1;
                      // #pragma omp critical
                      *(simbins + mychi*(permutations + 1)*num_sims*max_num_angles
                                 + simnum*max_num_angles + anglenum) = bin1;
                      
                      for (binmult = 0; binmult < num_histogram_sizes_to_try; binmult++)
                          {
                            bin1 = int((angle-0.000001)*(inv_binwidth_cor*(binmult+1))); //chi counts for half bin size times a multiplication factor
                            // #pragma omp atomic
                            *(chi_counts_sequential_varying_bin_size + binmult*num_sims*nchi*nbins_max + simnum*nchi*nbins_max + mychi*nbins_max + bin1) += 1;
                          }
                             
                      
                      }
                    for(bin1 = 0; bin1 < nbins_cor; bin1++) {
                      // #pragma omp atomic
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) /=  mynumangles;
                }
              
              }
              """

      
      code_adaptive_singlebins_sequential = """
              // weave12b
              int mynumangles, mynumangles_sum, bin1;
              int angle;
              int simnum, anglenum = 0;
              float inv_binwidth_adaptive_this_sim = 0;
                // #pragma omp parallel for private(simnum,mynumangles,inv_binwidth_adaptive_this_sim,anglenum,angle,bin1)
                for (simnum=0; simnum < num_sims; simnum++) {
                    mynumangles = *(numangles + simnum);
                    inv_binwidth_adaptive_this_sim = *(inv_binwidth_adaptive_sequential + simnum);
                    for (anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(rank_order_angles_sequential + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      bin1 = int(double(angle)*inv_binwidth_adaptive_this_sim);
                      if (bin1 < 0) bin1 = 0;
                      else if (bin1 >= nbins_cor) bin1 = nbins_cor - 1;
                      // #pragma omp atomic
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1;
                      // #pragma omp atomic
                      *(chi_counts_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1;
                      // #pragma omp critical
                      *(simbins + mychi*(permutations + 1)*num_sims*max_num_angles
                                 + simnum*max_num_angles + anglenum) = bin1;
                      
                      }
                   for(bin1 = 0; bin1 < nbins_cor; bin1++) {
                      // #pragma omp atomic
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) /=  mynumangles;
                }
              
              }
              """


###############################################

### Transition matrix calculation

      code_transition_matrix_1D_threads = """
      #include <math.h>
      //weave_trans_matrix_1D
       // matrix is such that i is the start bin and j is the final bin
       int mychi = 0;
       int mynumangles, mybootstrap, anglenum, lagtime, i, j, lbin1, bin1, bin2, simnum, mysim = 0;
       long totsum = 0;
       long myoffset = 0;
       long myoffset2 = 0;  //precompute array offsets
        
       //printf("start of routine: %i  : mychi: %i \\n ", mybootstrap, mychi);
       double pij, pi, pj, sum_pij, sum_pi, sum_pj = 0;
       
       
       //printf("preparing for loops bootstrap: %i  : mychi: %i \\n ", mybootstrap, mychi);
       for(mychi = 0; mychi < nchi; mychi++)  {
        for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {                

            myoffset = (long)(mychi*(permutations+1)*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + mysim*max_num_angles);
            myoffset2 = (long)(mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins) ;

            #pragma omp parallel for default(shared) private(lagtime,mysim, simnum, mynumangles, anglenum, i,j,totsum,sum_pi,sum_pj,sum_pij,bin1, bin2) 
            // shared(tempsums,tempsums_i,tempsums_j,tempprobs_i,tempprobs_j,myoffset,myoffset2) 
            for (lagtime = 0; lagtime < NUM_LAGTIMES; lagtime++) {   //use lagtime 0 for information autocorrelation
              
              // do transitions within blocks, not between
              for (mysim=0; mysim < bootstrap_choose; mysim++) {
               simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
               mynumangles = *(numangles + mybootstrap*num_sims + simnum);
               //printf("transition matrix bootstrap: %i  : numangles: %i \\n ", mybootstrap, mynumangles);

               

               for (anglenum=lagtime_interval*lagtime; anglenum< mynumangles; anglenum++) {
                // compute from bin and to bin for T(bin1(t-dt),bin1(t))  --- here we will symmetrize using the fast, original approach of Bowman JCP 2009 of averaging the matrix and its transpose in Progress and challenges in the automated construction of Markov Models for full protein systems -- but here we will actually just add them so we can store counts as integers
                i = (*(bins  + myoffset + anglenum - lagtime_interval*lagtime));
                j = (*(bins  + myoffset + anglenum));
                //printf("markov transition: lagtime: %i, from bin: %i   to bin: %i \\n", lagtime, i, j);
                // *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + i*nbins + j ) += 1;  // for mutinf autocorrelation
                // // *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + j*nbins + i ) += 1; 
                *(transition_matrix_multiple_lagtimes + myoffset2 + lagtime*nbins*nbins  + i*nbins + j ) += 1; 
                *(transition_matrix_multiple_lagtimes + myoffset2 + lagtime*nbins*nbins  + j*nbins + i ) += 1; 
                //tempsums[lagtime*nbins + i]   += 1;
                //tempsums[lagtime*nbins + j]   += 1;
                //tempsums_i[lagtime*nbins + i] += 1;
                //tempsums_j[lagtime*nbins + j] += 1;
                *(tempsums + lagtime*nbins + i ) += 1;
                *(tempsums + lagtime*nbins + j ) += 1;
                *(tempsums_i + lagtime*nbins + i) += 1;
                *(tempsums_j + lagtime*nbins + j) += 1;
                totsum += 1;
                }
               } // end for mysim

              // normalize, for example as in Swope et al 2004 eq. 26., by populations in from bin
              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                *(tempprobs_i  + lagtime*nbins + bin1) = (*(tempsums_i  + lagtime*nbins + bin1) * 1.0) / totsum;
                *(tempprobs_j  + lagtime*nbins + bin1) = (*(tempsums_j  + lagtime*nbins + bin1) * 1.0) / totsum;
                //*(tempprobs_i  + bin1) = (*(tempsums_i  + bin1) * 1.0) / totsum;
                //*(tempprobs_j + bin1) = (*(tempsums_j +  bin1) * 1.0) / totsum;
                sum_pi +=  *(tempprobs_i  + lagtime*nbins + bin1) ;
                sum_pj +=  *(tempprobs_j  + lagtime*nbins + bin1) ;
                //printf("tempsum %i\\n", *(tempsums + lagtime*nbins + bin1));
                if(*(tempsums + lagtime*nbins + bin1) > 0) // to avoid dividing by zero
                {
                   for (bin2 = 0; bin2 < nbins; bin2++)
                   {
                     *(transition_matrix_multiple_lagtimes + myoffset2 + bin1*nbins + bin2 ) /=  (*(tempsums + lagtime*nbins + bin1)) ; //we already essentially divided by two. Wow we have transition probs not just counts
                   }
                }
                *(tempsums + lagtime*nbins + bin1) = 0;
                
              }

              sum_pij = 0;
              sum_pi = 0;
              sum_pj = 0;
              //printf("totsum %i\\n", totsum);
              

              //for (bin1 = 0; bin1 < nbins; bin1++)
              //{
              //    
              //     for (bin2 = 0; bin2 < nbins; bin2++)
              //     {
              //       pij =  *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) / totsum ; // normalize 2-D PDF
              //       pi  =  (*(tempsums_i + lagtime*nbins + bin1) * 1.0) / totsum;
              //       pj  =  (*(tempsums_j + lagtime*nbins + bin2) * 1.0) / totsum;
              //       sum_pij += pij;
              //       if( pij > 0 && pi > 0 && pj > 0)
              //       {
              //          *(mutinf_autocorrelation_vs_lagtime + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) =  pij * log ( (pij ) / ((pi ) * (pj ))) ;
              //       
              //       }
              //     }
              //}
              //printf("sum_pij %f\\n", sum_pij);
              //printf("sum_pi %f\\n", sum_pi);
              //printf("sum_pj %f\\n", sum_pj);
             
              //for (bin1 = 0; bin1 < nbins; bin1++)
              //{
              //    for (bin2 = 0; bin2 < nbins; bin2++)
              //    {
              //        //normalize mutinf autocorrelation by value at lagtime = 1
              //        
              //        *(mutinf_autocorrelation_vs_lagtime + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) /= sum_pij; 
              //        
              //    }
              //}

              for (bin1 = 0; bin1<nbins; bin1++) 
              {
                *(tempsums_i + lagtime*nbins + bin1) = 0;
                *(tempsums_j + lagtime*nbins + bin1) = 0;
              }
              totsum = 0;


            }
         } 
       }

      """

      code_transition_matrix_1D = """
      #include <math.h>
      //weave_trans_matrix_1D
       // matrix is such that i is the start bin and j is the final bin
       int mychi = 0;
       long mynumangles, mybootstrap, anglenum, lagtime, i, j, bin1, bin2, simnum, totsum, mysim = 0;
       double pij, pi, pj, sum_pij, sum_pi, sum_pj = 0;
       //double smaller = SMALL * SMALL;
       for(mychi = 0; mychi < nchi; mychi++)  {
        for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {        
            #pragma omp parallel for private(lagtime, mysim, simnum, mynumangles, anglenum, i,j,totsum,bin1,sum_pi,sum_pj,bin2)
            for (lagtime = 0; lagtime < NUM_LAGTIMES; lagtime++) {   //use lagtime 0 for information autocorrelation
              totsum = 0;
              sum_pi = 0;
              sum_pj = 0;
              // do transitions within blocks, not between
              for (mysim=0; mysim < bootstrap_choose; mysim++) {
              simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
              mynumangles = *(numangles + mybootstrap*num_sims + simnum);
              //printf("transition matrix bootstrap: %i  : numangles: %i \\n ", mybootstrap, mynumangles);
              for (anglenum=lagtime_interval*lagtime; anglenum< mynumangles; anglenum++) {
                // compute from bin and to bin for T(bin1(t-dt),bin1(t))  --- here we will symmetrize using the fast, original approach of Bowman JCP 2009 of averaging the matrix and its transpose in Progress and challenges in the automated construction of Markov Models for full protein systems -- but here we will actually just add them so we can store counts as integers
                i = (*(bins  + (long)(mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum - lagtime_interval*lagtime)));
                j = (*(bins  + (long)(mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum)));
                //printf("markov transition: lagtime: %i, from bin: %i   to bin: %i \\n", lagtime, i, j);
                // *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + i*nbins + j ) += 1;  // for mutinf autocorrelation
                // // *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + j*nbins + i ) += 1; 
                *(transition_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + i*nbins + j ) += 1; 
                *(transition_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + j*nbins + i ) += 1; 
                //tempsums(lagtime,i) += 1;
                //tempsums(lagtime,j) += 1;
                //tempsums_i(lagtime,i) += 1;
                //tempsums_j(lagtime,j) += 1;
                *(tempsums + lagtime*nbins*PAD + i ) += 1;
                *(tempsums + lagtime*nbins*PAD + j ) += 1;
                *(tempsums_i + lagtime*nbins*PAD +  i) += 1;
                *(tempsums_j + lagtime*nbins*PAD + j) += 1;
                totsum += 1;
                }
              }
              // normalize, for example as in Swope et al 2004 eq. 26., by populations in from bin
              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                *(tempprobs_i + lagtime*nbins*PAD + bin1) = (*(tempsums_i + lagtime*nbins*PAD + bin1) * 1.0) / totsum;
                *(tempprobs_j + lagtime*nbins*PAD + bin1) = (*(tempsums_j + lagtime*nbins*PAD + bin1) * 1.0) / totsum;
                sum_pi +=  *(tempprobs_i + lagtime*nbins*PAD + bin1) ;
                sum_pj +=  *(tempprobs_j + lagtime*nbins*PAD + bin1) ;
                //printf("tempsum %i\\n", *(tempsums + lagtime*nbins*PAD + bin1));
                if(*(tempsums + lagtime*nbins*PAD + bin1) > 0) // to avoid dividing by zero
                {
                   for (bin2 = 0; bin2 < nbins; bin2++)
                   {
                     *(transition_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) /=  (*(tempsums + lagtime*nbins*PAD + bin1)) ; //we already essentially divided by two. Wow we have transition probs not just counts
                   }
                }
                *(tempsums + lagtime*nbins*PAD + bin1) = 0;
                
              }

              sum_pij = 0;
              sum_pi = 0;
              sum_pj = 0;
              //printf("totsum %i\\n", totsum);
              

              //for (bin1 = 0; bin1 < nbins; bin1++)
              //{
              //    
              //     for (bin2 = 0; bin2 < nbins; bin2++)
              //     {
              //       pij =  *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) / totsum ; // normalize 2-D PDF
              //       pi  =  (*(tempsums_i + bin1) * 1.0) / totsum;
              //       pj  =  (*(tempsums_j + bin2) * 1.0) / totsum;
              //       sum_pij += pij;
              //       if( pij > 0 && pi > 0 && pj > 0)
              //       {
              //          *(mutinf_autocorrelation_vs_lagtime + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) =  pij * log ( (pij ) / ((pi ) * (pj ))) ;
              //       
              //       }
              //     }
              //}
              //printf("sum_pij %f\\n", sum_pij);
              //printf("sum_pi %f\\n", sum_pi);
              //printf("sum_pj %f\\n", sum_pj);
             
              //for (bin1 = 0; bin1 < nbins; bin1++)
              //{
              //    for (bin2 = 0; bin2 < nbins; bin2++)
              //    {
              //        //normalize mutinf autocorrelation by value at lagtime = 1
              //        
              //        *(mutinf_autocorrelation_vs_lagtime + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) /= sum_pij; 
              //        
              //    }
              //}

              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                     *(tempsums_i + lagtime*nbins*PAD + bin1) = 0;
                     *(tempsums_j + lagtime*nbins*PAD + bin1) = 0;
              }
              totsum = 0;
            }
         } 
       }
  
      """






### Transition matrix stochastic simulation  -- I tried to make this multithreaded but it didn't yield correct results

      code_sample_from_transition_matrix_1D_new = """
       //weave_trans_matrix_1D
       // matrix is such that i is the start bin and j is the final bin
  
       long mynumangles, mynumangles_sum, mybootstrap, markov_chain, anglenum, lagtime, i, j, bin1,  bin2, simnum, mysim = 0;
       long counts = 0; 
       long temp_counts[markov_samples][nbins]; 
       unsigned long long offset = 0;
       double mycdf = 0.0;
       double myrand = 0.0 ;
       double mysign1, dig1, counts1d = 0.0 ;
       
       
       
       # define __GOM_NOTHROW __attribute__((__nothrow__))
       

       extern void omp_set_num_threads (int); //__GOMP_NOTHROW;
       extern int omp_get_num_threads (void); //__GOMP_NOTHROW;
       extern int omp_get_max_threads (void); //__GOMP_NOTHROW;
       extern int omp_get_thread_num (void ); //__GOMP_NOTHROW;
       extern int omp_get_num_procs (void)  ; //__GOMP_NOTHROW;

       
       //for (i=0; i < 200; i++) {
       //    myrand = drand48();
       //}
       anglenum = 0 ; // start at beginning of sim
       for (mychi=0; mychi < nchi; mychi++)
       {
        for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) 
        {  
           
           #pragma omp parallel for private(markov_chain, mynumangles, mynumangles_sum, mysim, simnum, lagtime, anglenum, bin1, offset, myrand, mycdf, bin2)
           for (markov_chain = 0; markov_chain < markov_samples; markov_chain++) 
           {
              unsigned short xi[3]; 
              xi[0] = 123 + 10*markov_chain;
              xi[1] = 825;
              xi[2] = 5 + mybootstrap;
              myrand = erand48(xi);
              // do transitions within blocks, not between
              mynumangles_sum = 0;
              for (mysim=0; mysim < bootstrap_choose; mysim++) 
              {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                    lagtime = *(slowest_lagtime + mychi*bootstrap_sets + mybootstrap);
                    //first datapoint
                    anglenum = 0 ; // start at beginning of sim
                    bin1 =  (*(bins + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles +  mybootstrap*bootstrap_choose*max_num_angles + simnum*max_num_angles + anglenum)) ;
                    offset = (unsigned long long)(mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum);  
                    //if(markov_chain == 0)
                    //{
                    //   printf("first offset: %llu mychi: %i mysim: %i mybootstrap: %i bin1: %i  \\n", offset, mychi, mysim, mybootstrap, bin1);
                    //}
                    if(offset >= 2000000000 and markov_chain == 0) {
                       printf("too many snapshots and/or markov samples , will likely segfault \\n ");
                    }
                    //*(bins_markov  + mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;
                    *(bins_markov + mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum ) = bin1 ;
                    *(chi_counts_markov + (long)(mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1)) += 1 ;  
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);  // Dont jump between sims within a bootstrap
                    //printf("mychi: %i bootstrap: %i mysim: %i mynumangles_sum: %i mynumangles: %i max_num_angles: %i lagtime: % i\\n", mychi, mybootstrap, mysim, mynumangles_sum, mynumangles, max_num_angles, lagtime);

                    //now stochastically sample from Markov model for the rest
                    for (anglenum=1; anglenum< mynumangles; anglenum++) {
                        
                        //lagtime = 1;
                        if(anglenum % lagtime == 0) {   // then take a discrete markov step
                           myrand = erand48(xi);
                           mycdf = 0.0;
                            
                           for (bin2=0; bin2 < nbins; bin2++) {
                               if(mychi > 0) {
                                  //printf("angle: %i myrand %f mycdf %f  bin1: %i bin2: %i \\n", anglenum, myrand, mycdf, bin1, bin2);
                               }
                               mycdf += *(transition_matrix + mychi*bootstrap_sets*nbins*nbins + mybootstrap*nbins*nbins + bin1*nbins + bin2);
                               if(myrand < mycdf) {
                                    if(mychi > 0) {
                                          //printf("myrand %f mycdf %f  bin1: %i bin2: %i \\n", myrand, mycdf, bin1, bin2);
                                    }
                                    
                                    if(mychi > 0) {
                                         //printf("bin markov: %i \\n", bin2);
                                    }
                                    bin1 = bin2;
                                    bin2 = nbins; //to push loop to the end
                                }
                           }
                        
                        
                              
                           
                        }
                        // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8) # the bin for each dihedral

                        //if(anglenum % lagtime == 0) {   // then record the data  (markov_interval)
                          offset = (unsigned long long )(mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum) ;
                          //*(bins_markov +  mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;   // will be overwritten if anglenum is a multiple of the lagtime ... this is just to avoid another branch
                        
                          *(bins_markov + mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum ) = bin1;   // will be overwritten if anglenum is a multiple of the lagtime ... this is just to avoid another branch
                          // python: self.chi_counts_markov=zeros((bootstrap_sets, self.markov_samples, self.nchi, nbins), float64)  since these can be weighted in advanced sampling like replica exchange
                          *(chi_counts_markov + (long)(mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1)) += 1 ;  
                         //}
                    }
              mynumangles_sum += mynumangles; //increment for next iteration
              //printf("mysim: %i mynumangles_sum: %i mynumangles: %i max_num_angles: %i \\n", mysim, mynumangles_sum, mynumangles, max_num_angles);
              }
      
       
      
          }
       }

       for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) 
       {        
           //lagtime = *(slowest_lagtime + mychi*bootstrap_sets + mybootstrap);
           #pragma omp parallel for private(markov_chain, mynumangles_sum, mysim, simnum, mynumangles, anglenum, counts, lagtime, bin1,  myrand, mycdf, mysign1, counts1d, dig1)
           for (markov_chain = 0; markov_chain < markov_samples; markov_chain++) 
           {        
              // check bins_markov against chi counts markov
              
              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                  temp_counts[markov_chain][bin1] = 0;
              }
              mynumangles_sum = 0;
              for (mysim=0; mysim < bootstrap_choose; mysim++) 
              {
                simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                mynumangles = *(numangles + mybootstrap*num_sims + simnum); 
                mynumangles_sum += mynumangles;
              }
                for (anglenum=0; anglenum< mynumangles_sum; anglenum++) {
                 //if(anglenum % lagtime == 0) {   // then read data from the discrete markov step (markov interval)
                  // anglenum goes up to mynumangles_sum, so no extra index for anglenum here
                  bin1 = *(bins_markov + (long )(mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + anglenum));
                  //printf("anglenum: %i bin1: %i\\n", anglenum, bin1);
                  temp_counts[markov_chain][bin1] += 1;
                 //}
              }
              for (bin1 = 0; bin1 < nbins; bin1++)
              {  
                 counts = *(chi_counts_markov + (long)(mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1));
                 //printf(" bootstrap: %i chi: %i markov_sample: %i bin: %i temp_counts: %i counts: %i \\n", mybootstrap, mychi, markov_chain, bin1, temp_counts[markov_chain][bin1], counts );

                 // do 1-D entropy calc here for efficiency
                 if(counts > 0)
                 { 
                   mysign1 = 1.0L - 2*(counts % 2); // == -1 if it is odd, 1 if it is even
                 
                 
 
           
           
           
                    counts1d = 1.0 * counts;
                    dig1 = DiGamma_Function(counts1d );
                    *(ent_markov_boots + (unsigned long long)(mychi*bootstrap_sets*markov_samples + mybootstrap*markov_samples + markov_chain )) += (double)( (counts1d ) / mynumangles_sum)*(log(mynumangles_sum) - dig1 - (mysign1 / ((double)(counts1d + 1.0L)))); 
                 }
                //  end of entropy calc

                 if (temp_counts[markov_chain][bin1] !=  *(chi_counts_markov + mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1) )
                 {
                   printf("WARNING: chi_counts_markov and bins_markov are inconsistent! bootstrap: %i chi: %i markov_sample: %i bin: %i temp_counts: %i bin_counts: %i \\n", mybootstrap, mychi, markov_chain, bin1, temp_counts[markov_chain][bin1], int(*(chi_counts_markov + mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1)) );
                 }
              }
          }
      }
     }
      """


####

#################################################
#### OLD transition matrix routines
###################################################

### Transition matrix calculation

      code_transition_matrix_1D_old = """
      #include <math.h>
      //weave_trans_matrix_1D
       // matrix is such that i is the start bin and j is the final bin
       int mychi = 0;
       int mynumangles, mybootstrap, anglenum, lagtime, i, j, bin1, bin2, simnum, totsum, mysim = 0;
       double pij, pi, pj, sum_pij, sum_pi, sum_pj = 0;
       double smaller = SMALL * SMALL;
       for(mychi = 0; mychi < nchi; mychi++)  {
        for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {        
            for (lagtime = 0; lagtime < NUM_LAGTIMES; lagtime++) {   //use lagtime 0 for information autocorrelation
              // do transitions within blocks, not between
              for (mysim=0; mysim < bootstrap_choose; mysim++) {
              simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
              mynumangles = *(numangles + mybootstrap*num_sims + simnum);
              //printf("transition matrix bootstrap: %i  : numangles: %i \\n ", mybootstrap, mynumangles);
              for (anglenum=lagtime_interval*lagtime; anglenum< mynumangles; anglenum++) {
                // compute from bin and to bin for T(bin1(t-dt),bin1(t))  --- here we will symmetrize using the fast, original approach of Bowman JCP 2009 of averaging the matrix and its transpose in Progress and challenges in the automated construction of Markov Models for full protein systems -- but here we will actually just add them so we can store counts as integers
                i = (*(bins  + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum - lagtime_interval*lagtime));
                j = (*(bins  + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum));
                //printf("markov transition: lagtime: %i, from bin: %i   to bin: %i \\n", lagtime, i, j);
                //*(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + i*nbins + j ) += 1;  // for mutinf autocorrelation
                // // *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + j*nbins + i ) += 1; 
                *(transition_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + i*nbins + j ) += 1; 
                *(transition_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + j*nbins + i ) += 1; 
                *(tempsums +  i ) += 1;
                *(tempsums +  j ) += 1;
                *(tempsums_i + i) += 1;
                *(tempsums_j + j) += 1;
                totsum += 1;
                }
              }
              // normalize, for example as in Swope et al 2004 eq. 26., by populations in from bin
              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                *(tempprobs_i + bin1) = (*(tempsums_i + bin1) * 1.0) / totsum;
                *(tempprobs_j + bin1) = (*(tempsums_j + bin1) * 1.0) / totsum;
                sum_pi +=  *(tempprobs_i + bin1) ;
                sum_pj +=  *(tempprobs_j + bin1) ;
                //printf("tempsum %i\\n", *(tempsums + bin1));
                if(*(tempsums + bin1) > 0) // to avoid dividing by zero
                {
                   for (bin2 = 0; bin2 < nbins; bin2++)
                   {
                     *(transition_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) /=  (*(tempsums + bin1)) ; //we already essentially divided by two. Wow we have transition probs not just counts
                   }
                }
                *(tempsums + bin1) = 0;
                
              }

              sum_pij = 0;
              sum_pi = 0;
              sum_pj = 0;
              //printf("totsum %i\\n", totsum);
              

              //for (bin1 = 0; bin1 < nbins; bin1++)
              //{
              //    
              //     for (bin2 = 0; bin2 < nbins; bin2++)
              //     {
              //       pij =  *(count_matrix_multiple_lagtimes + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) / totsum ; // normalize 2-D PDF
              //       pi  =  (*(tempsums_i + bin1) * 1.0) / totsum;
              //       pj  =  (*(tempsums_j + bin2) * 1.0) / totsum;
              //       sum_pij += pij;
              //       if( pij > 0 && pi > 0 && pj > 0)
              //       {
              //          *(mutinf_autocorrelation_vs_lagtime + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) =  pij * log ( (pij ) / ((pi ) * (pj ))) ;
              //       
              //       }
              //     }
              //}
              //printf("sum_pij %f\\n", sum_pij);
              //printf("sum_pi %f\\n", sum_pi);
              //printf("sum_pj %f\\n", sum_pj);
             
              //for (bin1 = 0; bin1 < nbins; bin1++)
              //{
              //    for (bin2 = 0; bin2 < nbins; bin2++)
              //    {
              //        //normalize mutinf autocorrelation by value at lagtime = 1
              //        
              //        *(mutinf_autocorrelation_vs_lagtime + mychi*bootstrap_sets*NUM_LAGTIMES*nbins*nbins + mybootstrap*NUM_LAGTIMES*nbins*nbins + lagtime*nbins*nbins + bin1*nbins + bin2 ) /= sum_pij; 
              //        
              //    }
              //}

              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                     *(tempsums_i + bin1) = 0;
                     *(tempsums_j + bin1) = 0;
              }
              totsum = 0;
            }
         } 
       }
  
      """







### Transition matrix stochastic simulation  -- need to look over this again, and then add code to handle this in independent mutinf to get distribution and plot it, etc.

      code_sample_from_transition_matrix_1D = """
       //weave_trans_matrix_1D
       // matrix is such that i is the start bin and j is the final bin
       int mynumangles, mynumangles_sum, mybootstrap, markov_chain, anglenum, lagtime, i, j, bin1,  bin2, simnum, mysim, counts = 0; 
       
       unsigned long long offset = 0;
       double mycdf = 0.0;
       double myrand = 0.0 ;
       double mysign1, dig1, counts1d = 0.0 ;
       
       for (i=0; i < 200; i++) {
           myrand = drand48();
       }
       anglenum = 0 ; // start at beginning of sim
       for (mychi=0; mychi < nchi; mychi++)
       {
        for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) 
        {        
           for (markov_chain = 0; markov_chain < markov_samples; markov_chain++) 
           {
              // do transitions within blocks, not between
              mynumangles_sum = 0;
              for (mysim=0; mysim < bootstrap_choose; mysim++) 
              {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                    lagtime = *(slowest_lagtime + mychi*bootstrap_sets + mybootstrap);
                    //first datapoint
                    anglenum = 0 ; // start at beginning of sim
                    bin1 =  (*(bins  +  mychi*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles + simnum*max_num_angles + anglenum)) ;
                    offset = mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum, mychi, mysim, mybootstrap ;
                    if(markov_chain == 0)
                    {
                       printf("first offset: %llu mychi: %i mysim: %i mybootstrap: %i bin1: %i  \\n", offset, mychi, mysim, mybootstrap, bin1);
                    }
                    if(offset >= 2000000000 and markov_chain == 0) {
                       printf("too many snapshots and/or markov samples , will likely segfault \\n ");
                    }
                    *(bins_markov  + mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;
                    *(chi_counts_markov + (long)(mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1)) += 1 ;  
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);  // Dont jump between sims within a bootstrap
                    printf("mychi: %i bootstrap: %i mysim: %i mynumangles_sum: %i mynumangles: %i max_num_angles: %i lagtime: % i\\n", mychi, mybootstrap, mysim, mynumangles_sum, mynumangles, max_num_angles, lagtime);

                    //now stochastically sample from Markov model for the rest
                    for (anglenum=1; anglenum< mynumangles; anglenum++) {
                        
                        //lagtime = 1;
                        if(anglenum % lagtime == 0) {   // then take a discrete markov step
                           myrand = drand48();
                           mycdf = 0.0;
                            
                           for (bin2=0; bin2 < nbins; bin2++) {
                               if(mychi > 0) {
                                  //printf("angle: %i myrand %f mycdf %f  bin1: %i bin2: %i \\n", anglenum, myrand, mycdf, bin1, bin2);
                               }
                               mycdf += *(transition_matrix + mychi*bootstrap_sets*nbins*nbins + mybootstrap*nbins*nbins + bin1*nbins + bin2);
                               if(myrand < mycdf) {
                                    if(mychi > 0) {
                                          //printf("myrand %f mycdf %f  bin1: %i bin2: %i \\n", myrand, mycdf, bin1, bin2);
                                    }
                                    offset = mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum, mychi, mysim, mybootstrap ;
                                    *(bins_markov  + mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin2;
                                    if(mychi > 0) {
                                         //printf("bin markov: %i \\n", bin2);
                                    }
                                    bin1 = bin2;
                                    bin2 = nbins; //to push loop to the end
                                }
                           }
                        
                        
                              
                           
                        }
                        // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8) # the bin for each dihedral
                        offset = mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mysim*max_num_angles + anglenum, mychi, mysim, mybootstrap ;
                        *(bins_markov +  mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;   // will be overwritten if anglenum is a multiple of the lagtime ... this is just to avoid another branch
                        // python: self.chi_counts_markov=zeros((bootstrap_sets, self.markov_samples, self.nchi, nbins), float64)  since these can be weighted in advanced sampling like replica exchange
                        *(chi_counts_markov + (long)(mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1)) += 1 ;  
                    }
              mynumangles_sum += mynumangles; //increment for next iteration
              printf("mysim: %i mynumangles_sum: %i mynumangles: %i max_num_angles: %i \\n", mysim, mynumangles_sum, mynumangles, max_num_angles);
              }
      
       
      
          }
       }

       for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) 
       {        
           for (markov_chain = 0; markov_chain < markov_samples; markov_chain++) 
           {        
              long temp_counts[nbins]; 
              // check bins_markov against chi counts markov
              
              for (bin1 = 0; bin1 < nbins; bin1++)
              {
                  temp_counts[bin1] = 0;
              }
              mynumangles_sum = 0;
              for (mysim=0; mysim < bootstrap_choose; mysim++) 
              {
                simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);                
                mynumangles = *(numangles + mybootstrap*num_sims + simnum); 
                mynumangles_sum += mynumangles;
              }
                for (anglenum=0; anglenum< mynumangles_sum; anglenum++) {
                  // anglenum goes up to mynumangles_sum, so no extra index for anglenum here
                  bin1 = *(bins_markov + mychi*bootstrap_sets*markov_samples*bootstrap_choose*max_num_angles +  mybootstrap*markov_samples*bootstrap_choose*max_num_angles + markov_chain*bootstrap_choose*max_num_angles + anglenum);
                  //printf("anglenum: %i bin1: %i\\n", anglenum, bin1);
                  temp_counts[bin1] += 1;
              }
              for (bin1 = 0; bin1 < nbins; bin1++)
              {  
                 counts = *(chi_counts_markov + (long)(mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1));
                 // do 1-D entropy calc here for efficiency
                 if(counts > 0)
                 { 
                   mysign1 = 1.0L - 2*(counts % 2); // == -1 if it is odd, 1 if it is even
                 
                 
 
           
           
           
                    counts1d = 1.0 * counts;
                    dig1 = DiGamma_Function(counts1d );
                    *(ent_markov_boots + (unsigned long long)(mychi*bootstrap_sets*markov_samples + mybootstrap*markov_samples + markov_chain )) += (double)( (counts1d ) / mynumangles_sum)*(log(mynumangles_sum) - dig1 - (mysign1 / ((double)(counts1d + 1.0L)))); 
                 }
                 printf(" bootstrap: %i chi: %i markov_sample: %i bin: %i temp_counts: %i counts: %i \\n", mybootstrap, mychi, markov_chain, bin1, temp_counts[bin1], counts );
                 if (temp_counts[bin1] !=  *(chi_counts_markov + mychi*bootstrap_sets*markov_samples*nbins mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1) )
                 {
                   printf("WARNING: chi_counts_markov and bins_markov are inconsistent! bootstrap: %i chi: %i markov_sample: %i bin: %i temp_counts: %i bin_counts: %i \\n", mybootstrap, mychi, markov_chain, bin1, temp_counts[bin1], int(*(chi_counts_markov + mychi*bootstrap_sets*markov_samples*nbins + mybootstrap*markov_samples*nbins + markov_chain*nbins + bin1)) );
                 }
              }
          }
      }
     }
      """




#################################################################

### Nonunity Weights for each snapshot 
      
      code_nonadaptive_singlebins_weights = """
              // weave9
              int mynumangles, mynumangles_sum, bin1, simnum;
              double angle, weight;
              for(int mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
                mynumangles = 0;
                *(numangles_bootstrap + mybootstrap) = 0 ;   
                for (int mysim=0; mysim < bootstrap_choose; mysim++) {
                    simnum = *(which_runs + mybootstrap*bootstrap_choose + mysim);
                    mynumangles = *(numangles + mybootstrap*num_sims + simnum);
                    mynumangles_sum = *(numangles_bootstrap + mybootstrap);
                    for (int anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      weight = *(weights + simnum*max_angles + anglenum); //assumes mynumangles same for all dihedrals
                      
                      if(angle > 360) angle = angle - 360;
                      if(angle <= 0.000001) angle = 0.0000011;
                      bin1 = int((angle-0.000001)*inv_binwidth);
                      if(bin1 < 0)
                      {
                         printf("WARNING: bin less than zero");
                         bin1 = 0;
                      }
                      //printf("bootstrap: %4i boot_sim: %4i sim_num: %4i angle: %3.3f, bin: %4i , weight: %3.3f \\n", mybootstrap, mysim, simnum, angle, bin1, weight);  
                      *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1.0 * weight;
                      *(chi_counts + mybootstrap*nchi*nbins + mychi*nbins + bin1) += 1.0 * weight;
                      *(bins + mychi*(permutations + 1)*bootstrap_sets*bootstrap_choose*max_num_angles
                                 + mybootstrap*bootstrap_choose*max_num_angles + mynumangles_sum + anglenum) = bin1;
                      //if(mybootstrap==0 && mysim == 0) printf("angle:%3.3f %3i\\n",angle, bin1);
                      }
                    *(numangles_bootstrap + mybootstrap) += mynumangles;
                  }
                for(bin1 = 0; bin1 < nbins; bin1++) {
                    *(chi_pop_hist + mybootstrap*nchi*nbins + mychi*nbins + bin1) /=  *(numangles_bootstrap + mybootstrap);
                }
              
              }
              """
      #printf("data slot %d bin  %d \\n",temp,bin1);           
      # printf("simnum %d mynumangles %d mynumangles_sum %d\\n",simnum,mynumangles,mynumangles_sum);

      code_nonadaptive_singlebins_sequential_weights = """
              // weave11
              int mynumangles, mynumangles_sum, bin1, binmult, nbins_max;
              nbins_max=nbins_cor * num_histogram_sizes_to_try;
              double angle, weight;
                for (int simnum=0; simnum < num_sims; simnum++) {
                    mynumangles = *(numangles + simnum);
                    for (int anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(angles + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      weight = *(weights + simnum*max_angles + anglenum); //assumes mynumangles same for all sims
                      
                      if(angle > 360) angle = angle - 360;
                      if(angle <= 0.000001) angle = 0.0000011;
                      bin1 = int((angle-0.000001)*inv_binwidth_cor);
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1.0 * weight;
                      *(chi_counts_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1.0 * weight;
                      *(simbins + mychi*(permutations + 1)*num_sims*max_num_angles
                                 + simnum*max_num_angles + anglenum) = bin1;
                      
                      for (binmult = 0; binmult < num_histogram_sizes_to_try; binmult++)
                          {
                            bin1 = int((angle-0.000001)*(inv_binwidth_cor*(binmult+1))); //chi counts for half bin size times a multiplication factor
                            // #pragma omp atomic
                            *(chi_counts_sequential_varying_bin_size + binmult*num_sims*nchi*nbins_max + simnum*nchi*nbins_max + mychi*nbins_max + bin1) += 1.0 * weight;
                          }

                      }
                    for(bin1 = 0; bin1 < nbins_cor; bin1++) {
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) /=  mynumangles;
                }
              
              }
              """

      code_adaptive_singlebins_sequential_weights = """
              // weave12b
              int mynumangles, mynumangles_sum, bin1, found;
              int angle;

              double weight, bin_bottom, bin_top;
                for (int simnum=0; simnum < num_sims; simnum++) {
                    found = 0;
                    mynumangles = *(numangles + simnum);
                    double inv_binwidth_adaptive_this_sim = *(inv_binwidth_adaptive_sequential + simnum);
                    //printf("\\n simnum: %i inv_binwidth_adaptive_sequential:%f \\n",simnum, inv_binwidth_adaptive_this_sim);
                    for (int anglenum=0; anglenum< mynumangles; anglenum++) {
                      angle = *(rank_order_angles_sequential + mychi*num_sims*max_angles  + simnum*max_angles + anglenum);
                      weight = *(weights + simnum*max_angles + anglenum); //assumes mynumangles same for all sims
                      //printf("\\n simnum: %i angle:%i, weight:%f \\n",simnum, angle, weight);
                      found = 0;
                      bin_bottom = *(adaptive_hist_left_breaks_sequential + simnum*(nbins_cor + 1)
                                     + 0);
                      for(int bin1=0; bin1 < nbins_cor; bin1++) {
                          //printf("bin: %i  ",bin1);
                          bin_top = *(adaptive_hist_left_breaks_sequential + simnum*(nbins_cor + 1)
                                     + bin1 + 1);
                          if(found == 0 && angle >= bin_bottom && angle < bin_top) {
                      
                             *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1.0*weight;
                             *(chi_counts_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) += 1.0*weight;
                             *(simbins + mychi*(permutations + 1)*num_sims*max_num_angles
                                 + simnum*max_num_angles + anglenum) = bin1;
                             found = 1;
                          }
                          bin_bottom = bin_top;
                          if(found == 1) break;
                      }
                   }
                   for(bin1 = 0; bin1 < nbins_cor; bin1++) {
                      *(chi_pop_hist_sequential + simnum*nchi*nbins_cor + mychi*nbins_cor + bin1) /=  mynumangles;
                }
              
              }
              """
      
      ############################################################
     
      code_ent1 = """
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mybootstrap, mynumangles,markov_chain,anglenum;
     //long  offset1, offset2, offset3, offset4;
     long  counts1, counts2, counts12;
     double counts1d, counts2d, counts12d;
     double dig1, dig2;
     double mysign1, mysign2 = 0 ;
     for(mychi=0; mychi < nchi; mychi++)
     {
      for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
       mynumangles = 0;
       mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
       //printf("mynumangles: %i ", mynumangles);
       //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
       //offset2 = mybootstrap*(markov_samples)*nbins; 
       //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
 
       //#pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, mysign1, mysign2, bin1, bin2, dig1, dig2, counts1d, counts2d, counts12, counts12d) 
       for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
        
        for(bin1=0; bin1 < nbins; bin1++) 
          {
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           // python: self.chi_counts_markov=zeros((bootstrap_sets, self.markov_samples, self.nchi, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
           
           counts1 = *(chi_counts1_markov  +   mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1 );
           mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
           if(counts1 > 0)
           {
 
           
           
           
           counts1d = 1.0 * counts1;
           dig1 = DiGamma_Function(counts1d );
           *(ent_markov_boots + (long)(mychi*bootstrap_sets*markov_samples + mybootstrap*markov_samples + markov_chain)) += (double)( (counts1d ) / mynumangles_sum)*(log(mynumangles_sum) - dig1 - (mysign1 / ((double)(counts1d + 1.0L)))); 
           
           
           //printf("bin1: %i counts1 index: %i \\n",  bin1, mybootstrap*(markov_samples)*nbins  +  markov_chain*nbins  +  bin1);
           //printf("counts to numangles ratio %f \\n", (( (counts1d) / mynumangles)));
           //printf("log numangles             %f \\n", (( (log(mynumangles)))));
           //printf("mysign1                   %f \\n",  mysign1 );
           //printf("mysign1 / counts+1        %e \\n",  mysign1 / ((double)(counts1d + 1.0L)));
           //printf("log numangles minus dig1  %f \\n", (( (log(mynumangles) - dig1))));
           //printf("corr                      %e \\n",  (mysign1 / ((double)((counts1d + 1.0L)))));
           //printf("log numangles minus corr. %e \\n", (( (log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts1d + 1.0L)))))));
           
           //printf("ent1 boots term counts1d:%f, dig:%f, term:%f sum:%e \\n",counts1d, dig1,(double)( (counts1d ) / mynumangles)*(log(mynumangles) - dig1 - (mysign1 / (double)(counts1d + 1.0L))), (double)(*(ent_markov_boots + (long)(mychi*bootstrap_sets*markov_samples + mybootstrap*markov_samples + markov_chain)))); 
           }
          }
        }
      }

         

      """

                                                                              
#
#
#
#
#              weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'pop_matrix','permutations','bootstrap_sets'],
#                 #type_converters = converters.blitz,
#                 compiler = mycompiler,runtime_library_dirs="/usr/lib/x86_64-linux-gnu/", library_dirs="/usr/lib/x86_64-linux-gnu/", libraries="stdc++")
#
#
#
#
#              weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'pop_matrix','permutations','bootstrap_sets'],
#                 #type_converters = converters.blitz,
#                 compiler = mycompiler,runtime_library_dirs="/usr/lib/x86_64-linux-gnu/", library_dirs="/usr/lib/x86_64-linux-gnu/", libraries="stdc++")
#
#

      angles = self.angles
      rank_order_angles = self.rank_order_angles
      boot_ranked_angles = self.boot_ranked_angles
      rank_order_angles_sequential = self.rank_order_angles_sequential
      nchi = self.nchi
                                                                              
      counts = self.counts
      counts2 = self.counts2
      counts3 = self.counts3
      chi_counts = self.chi_counts
      numangles = self.numangles
      numangles_bootstrap = self.numangles_bootstrap # this will be overwritten in the weaves,
                                                     #but overwritten correctly before each weave is done
      chi_pop_hist = self.chi_pop_hist
                                                                              
                                                                            
      chi_pop_hist_sequential = self.chi_pop_hist_sequential
      chi_counts_sequential = self.chi_counts_sequential
      chi_counts_sequential_varying_bin_size = self.chi_counts_sequential_varying_bin_size
      bins = self.bins
      simbins = self.simbins
      weights = self.weights
      ent_hist_left_breaks = self.ent_hist_left_breaks
      ent_hist_binwidths   = self.ent_hist_binwidths
      adaptive_hist_left_breaks = self.adaptive_hist_left_breaks
      adaptive_hist_left_breaks_sequential = self.adaptive_hist_left_breaks_sequential
      boot_sorted_angles = self.boot_sorted_angles
      boot_sorted_angles = self.boot_sorted_angles
      boot_weights = self.boot_weights
      boot_sorted_angles_weights = self.boot_sorted_angles_weights

      ent_from_sum_log_nn_dists = self.ent_from_sum_log_nn_dists

      chi_counts_markov = self.chi_counts_markov
      bins_markov = self.bins_markov
      slowest_lagtime = self.slowest_lagtime
      transition_matrix = self.transition_matrix
      markov_samples = self.markov_samples
      

      #resize numangles in case we are doing convergence stuff, otherwise, it will work equivalently
      numangles = resize(self.numangles, (bootstrap_sets, num_sims)) #should copy data appropriately 
      for mybootstrap in range(1, bootstrap_sets):
             numangles[mybootstrap,:] = numangles[0,:] #just in case it doesn't copy over data correctly
      if num_convergence_points > 1:
             for convergence_point in range(num_convergence_points):
                    for mysim in range(num_sims):
                           if convergence_point + 1 <= num_convergence_points:
                                  numangles[convergence_point, :] = int(self.numangles[mysim] * 1.0 * (convergence_point + 1) / num_convergence_points)
                           else:
                                  numangles[convergence_point, :] = self.numangles[mysim]
             print "numangles for convergence points:"
             print numangles
             print "shape of self.numangles"
             print shape(self.numangles)
      
             
      for myscramble in range(self.permutations + 1):
          for mychi in range(self.nchi):
             if (myscramble == 1): #resize
                 for bootstrap in range(bootstrap_sets):
                     self.bins[mychi, :, bootstrap, :] = self.bins[mychi, 0, bootstrap, :] #replicate data in preparation for scrambling or cyclicly permuting
                 for sequential_sim_num in range(num_sims):
                      self.simbins[mychi, :, sequential_sim_num, :] = self.simbins[mychi, 0, sequential_sim_num, :]
             if(myscramble > 0):
                 for bootstrap in range(bootstrap_sets): 
                      if(cyclic_permut == False):  
                             print "shuffling for calculation of independent information"
                             random.shuffle(self.bins[mychi, myscramble, bootstrap, :numangles_bootstrap[bootstrap]])  #only shuffle up to number of angles in this bootstrap -- for num_convergence_points > 1
                      else:
                             print "doing cyclic permutations" #will offset by more than one block 
                             #if(num_convergence_points == 1):
                             if(True):
                                    numangles_per_boot = int(numangles_bootstrap[bootstrap] / bootstrap_choose)
                             
                                    print "numangles per sim in each bootstrap: "+str(numangles_per_boot)
                                    print "bootstrap choose:                    "+str(bootstrap_choose)
                                    print "shape of temp array:                 "+str(shape(self.bins[mychi,myscramble,bootstrap]))
                                    tempbins = reshape(self.bins[mychi,myscramble,bootstrap,:numangles_bootstrap[bootstrap]], (bootstrap_choose, numangles_per_boot)) #prepare for cyclic permutation
                                    tempbins = roll(tempbins,myscramble ,axis=0) # do cyclic permutation .. will be redundant if myscramble  > bootstrap_choose
                                    self.bins[mychi, myscramble, bootstrap, :numangles_bootstrap[bootstrap]] = reshape(tempbins, (bootstrap_choose * numangles_per_boot ))
                             else:
                                    numangles_per_sim = numangles[bootstrap, 0] #use sim zero since all will have the same value after the above code is run
                                    print "numangles per sim in each bootstrap: "+str(numangles_per_sim)
                                    print "num sims  choose:                    "+str(num_sims)
                                    print "shape of temp array:                 "+str(shape(self.bins[mychi,myscramble,bootstrap,:numangles_bootstrap[bootstrap]]))
                                    tempbins = reshape(self.bins[mychi,myscramble,bootstrap,:numangles_bootstrap[bootstrap]], (num_sims, numangles_per_sim)) #prepare for cyclic permutation
                                    tempbins = roll(tempbins,myscramble ,axis=0) # do cyclic permutation .. will be redundant if myscramble > bootstrap_choose
                                    self.bins[mychi, myscramble, bootstrap, :numangles_bootstrap[bootstrap]] = reshape(tempbins, (num_sims * numangles_per_sim ))
                 for sequential_sim_num in range(num_sims):
                     random.shuffle(self.simbins[mychi, myscramble, sequential_sim_num, :numangles_bootstrap[bootstrap]]) #only shuffle up to number of angles in this bootstrap -- for num_convergence_points > 1
             else:             
                 #for bootstrap in range(bootstrap_sets):
                 #    self.numangles_bootstrap[bootstrap] = 0
                 #    for sequential_sim_num in self.which_runs[bootstrap]:
                 #        for anglenum in range(self.numangles[sequential_sim_num]): # numangles per sim is the same regardless of chi or residue
                 #            bin_num = binsingle(self.angles[mychi,sequential_sim_num,anglenum],MULT_1D_BINS * inv_binwidth) #no adaptive paritioning here
                 #            self.counts[bootstrap, mychi ,bin_num] +=1 #counts are for 1-D histograms for which we use naive binning
                 #        self.numangles_bootstrap[bootstrap] += self.numangles[sequential_sim_num]
                 #    self.counts[bootstrap, mychi, :] /= self.numangles_bootstrap[bootstrap]  # normalize
                
                 weave.inline(code_nonadaptive_doublebins_weights, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins', 'permutations','bootstrap_sets','bootstrap_choose','angles','numangles','counts','counts2','counts3','which_runs','nchi','inv_binwidth','mychi',"max_angles",'max_num_angles','MULT_1D_BINS','FEWER_COR_BTW_BINS','SAFE_1D_BINS','TWOPI','weights'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                 weave.inline(code_adaptive_doublebins, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins', 'permutations','bootstrap_sets','bootstrap_choose','angles','numangles','counts_adaptive','which_runs','nchi','ent_hist_binwidths','ent_hist_left_breaks','mychi',"max_angles",'max_num_angles','MULT_1D_BINS','weights'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                 weave.inline(code_nearest_neighbor_1D_weights, ['num_sims', 'MAX_NEAREST_NEIGHBORS','numangles_bootstrap','permutations','bootstrap_sets','bootstrap_choose','numangles','which_runs','nchi','mychi',"max_angles",'ent_from_sum_log_nn_dists','boot_sorted_angles','boot_sorted_angles_weights'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])


                 
                 if(VERBOSE >= 2): 
                        print "ent hist stuff compiled successfully"
                 if(adaptive_partitioning == 0):
                    #for bootstrap in range(bootstrap_sets):
                    #       self.numangles_bootstrap[bootstrap] =0
                    #       for sequential_sim_num in self.which_runs[bootstrap]:
                    #           for anglenum in range(self.numangles[sequential_sim_num]): # numangles per sim is the same regardless of chi or residue
                    #               bin_num = binsingle(self.angles[mychi,sequential_sim_num,anglenum],inv_binwidth) #no adaptive paritioning here
                    #               self.chi_pop_hist[bootstrap, mychi , bin_num] +=1 #counts are for 1-D histograms for which we use naive binning
                    #               self.bins[mychi, 0, bootstrap, self.numangles_bootstrap[bootstrap]  + anglenum] = bin_num #use naive binning for 2-D histograms
                    #               self.chi_counts[bootstrap, mychi, bin_num] += 1 
                    #           self.numangles_bootstrap[bootstrap]  += self.numangles[sequential_sim_num]
                    #       self.chi_pop_hist[bootstrap, mychi, :] /=  self.numangles_bootstrap[bootstrap] # normalize
                    
                    weave.inline(code_nonadaptive_singlebins_weights, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins', 'permutations','bootstrap_sets','bootstrap_choose','angles','numangles','chi_pop_hist','chi_counts','which_runs','nchi','inv_binwidth','mychi',"max_angles",'max_num_angles','weights'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                    
                    # use bins twice as wide for mutinf between sims
                    #       for sequential_sim_num in range(num_sims):
                    #           for anglenum in range(self.numangles[sequential_sim_num]): # numangles per sim is the same regardless of chi or residue
                    #               bin_num = binsingle(self.angles[mychi,sequential_sim_num,anglenum],inv_binwidth/2.0) #no adaptive paritioning here
                    #               self.chi_pop_hist_sequential[sequential_sim_num, mychi , bin_num] +=1 #counts are for 1-D histograms for which we use naive binning
                    #               self.simbins[mychi, 0, sequential_sim_num, anglenum] = bin_num #use naive binning for 2-D histograms
                    #           self.chi_pop_hist_sequential[sequential_sim_num, mychi, :] /=  self.numangles[sequential_sim_num] # normalize
                    weave.inline(code_nonadaptive_singlebins_sequential_weights, ['num_sims', 'numangles_bootstrap', 'nbins_cor', 'simbins', 'permutations','bootstrap_sets','angles','numangles','chi_pop_hist_sequential','chi_counts_sequential','chi_counts_sequential_varying_bin_size','which_runs','nchi','inv_binwidth_cor','mychi',"max_angles",'max_num_angles','MULT_1D_BINS','FEWER_COR_BTW_BINS','num_histogram_sizes_to_try','weights'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                 else:
                    # use bins twice as wide for mutinf between sims
                    # for bootstrap in range(bootstrap_sets):
                    #     self.numangles_bootstrap[bootstrap] = 0
                    #     for sequential_sim_num in self.which_runs[bootstrap]:
                    #         for anglenum in range(self.numangles[sequential_sim_num]): # numangles per sim is the same regardless of chi or residue
                    #             bin_num_adaptive = binsingle_adaptive(self.rank_order_angles[mychi,sequential_sim_num,anglenum],inv_binwidth_adaptive)
                    #             if(bin_num_adaptive < 0):
                    #                 print "warning!!: negative bin number wrapped to bin zero"
                    #                 bin_num_adaptive = 0
                    #             if(bin_num_adaptive >= nbins):
                    #                 print "warning!!: bin number overshot nbins, wrapping to bin nbins-1"
                    #                 bin_num_adaptive = nbins - 1
                    #             self.bins[mychi, 0, bootstrap, self.numangles_bootstrap[bootstrap]  + anglenum] = bin_num_adaptive  # overwrite bin value, this is used for adaptive partitioning for 2-D histograms
                    #             self.chi_pop_hist[bootstrap, mychi , bin_num_adaptive] +=1 #counts for 1-D histograms for 2-D mutual information calculation
                    #         self.numangles_bootstrap[bootstrap]  += self.numangles[sequential_sim_num]
                    #     self.chi_pop_hist[bootstrap, mychi , :] /= self.numangles_bootstrap[bootstrap]  # normalize and overwrite pop_hist value, this is used for adaptive partitioning for 2-D histograms
                    weave.inline(code_adaptive_singlebins_weights, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins', 'permutations','bootstrap_sets','bootstrap_choose','boot_ranked_angles','numangles','chi_pop_hist','chi_counts','which_runs','nchi','inv_binwidth_adaptive_bootstraps','mychi',"max_angles",'max_num_angles','MULT_1D_BINS','FEWER_COR_BTW_BINS','boot_weights','adaptive_hist_left_breaks'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                    #print "finished adaptive singlebins weights"
                    #   for sequential_sim_num in range(num_sims):
                    #         for anglenum in range(self.numangles[sequential_sim_num]): # numangles per sim is the same regardless of chi or residue
                    #             bin_num_adaptive = binsingle_adaptive(self.rank_order_angles_sequential[mychi,sequential_sim_num,anglenum],inv_binwidth_adaptive_sequential[sequential_sim_num])
                    #             if(bin_num_adaptive < 0):
                    #                 print "warning!!: negative bin number wrapped to bin zero."
                    #                 bin_num_adaptive = 0
                    #             if(bin_num_adaptive >= nbins_cor):
                    #                 print "warning!!: bin number overshot nbins, wrapping to bin nbins - 1. Rank:"+str(bin_num_adaptive/inv_binwidth_adaptive_sequential[sequential_sim_num])
                    #                 bin_num_adaptive = nbins_cor - 1
                    #             self.chi_pop_hist_sequential[sequential_sim_num, mychi , bin_num_adaptive] +=1 #counts are for 1-D histograms using adapative binning
                    #             self.chi_counts_sequential[sequential_sim_num, mychi , bin_num_adaptive] +=1
                    #             self.simbins[mychi, 0, sequential_sim_num, anglenum] = bin_num_adaptive #use adaptive binning for 2-D histograms
                    #         self.chi_pop_hist_sequential[sequential_sim_num, mychi, :] /=  self.numangles[sequential_sim_num] # normalize
                 #   weave.inline(code_nonadaptive_singlebins_sequential, ['num_sims', 'numangles_bootstrap', 'nbins_cor', 'simbins', 'permutations','bootstrap_sets','angles','numangles','chi_pop_hist_sequential','chi_counts_sequential','which_runs','nchi','inv_binwidth_cor','mychi',"max_angles",'max_num_angles'], compiler = mycompiler,runtime_library_dirs="/usr/lib/x86_64-linux-gnu/", library_dirs="/usr/lib/x86_64-linux-gnu/", libraries="stdc++")
                 
                 

                    weave.inline(code_adaptive_singlebins_sequential_weights, ['num_sims', 'numangles_bootstrap', 'nbins_cor', 'simbins', 'permutations','bootstrap_sets','rank_order_angles_sequential','chi_counts_sequential','numangles','chi_pop_hist_sequential','which_runs','nchi','inv_binwidth_adaptive_sequential','mychi',"max_angles",'max_num_angles','weights','adaptive_hist_left_breaks_sequential'], compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                 print "finished weaves"

      
      #autocorrelation time calculations -- available if PyMBAR import was sucessful
      for mychi in range(self.nchi):
             for mybootstrap in range(bootstrap_sets):
                    numangles_to_examine = int(numangles_bootstrap[mybootstrap]/(bootstrap_choose * 1.0)  )  #will this work with mutinf convergence calcs?
                    print "numangles to examine for this bootstrap set: "+str(numangles_to_examine)
                    angles_sims_list = []
                    for mysim in range(bootstrap_choose):
                           if(xvgorpdb != "xtc"):
                                  angles_sims_list.append(cos(self.angles[mychi,which_runs[mybootstrap,mysim],:numangles_to_examine])) #take cosine of angles
                           else:
                                  angles_sims_list.append(self.angles[mychi,which_runs[mybootstrap,mysim],:numangles_to_examine]) #take cosine of angles
                           try:
                                  self.angles_autocorr_time[mychi,bootstrap] =  integratedAutocorrelationTimeMultiple(angles_sims_list)
                           except:
                                  #self.angles_autocorr_time[mychi,bootstrap] =  integratedAutocorrelationTimeMultiple(angles_sims_list)
                                  self.angles_autocorr_time[mychi,bootstrap] = 0  # PyMBAR not present
                                  
                    print "autocorrelation, chi = "+str(mychi)+" bootstrap = "+str(mybootstrap)+ ": "+str(self.angles_autocorr_time[mychi, mybootstrap])

      #### Transition matrix 1D for Markov Model
      #if(lagtime_interval != None and myscramble == 0):
      if(lagtime_interval != None):
               tempsums = zeros((NUM_LAGTIMES, nbins*PAD), int64)
               tempsums_i = zeros((NUM_LAGTIMES, nbins*PAD), int64)
               tempsums_j = zeros((NUM_LAGTIMES, nbins*PAD), int64)
               tempprobs_i = zeros((NUM_LAGTIMES, nbins*PAD), float64)
               tempprobs_j = zeros((NUM_LAGTIMES, nbins*PAD), float64)
               #print "transition matrix first lagtime before calculation"
               #print transition_matrix_multiple_lagtimes[:,0,1]
               #print "transition matrix last lagtime before calculation:"
               #print transition_matrix_multiple_lagtimes[:,0,-1]
               
               #compute angles autocorrelation using fft
               def rect(r, w, deg=0):		# radian if deg=0; degree if deg=1
                      from math import cos, sin, pi
                      if deg:
                             w = pi * w / 180.0
                             #return complex(r * cos(w) , r * sin(w) )
                             return [cos(x) + 1j * sin(x) for x in w]
               crect1 = lambda x: rect(1, x, deg=1) #convert degrees to complex for fft
 
               def autocorr(x): #requires 1-d array
                      result = correlate(x, x, mode='full') 
                      result = result[result.size/2:]
                      result /= result[0]
                      return result #normalizes to max of 1
               
               
                         #     blah = 1 #just a placeholder 
                         #     #nobs = min_num_angles
                         #     #Frf = fft.fft(crect1(self.angles[mychi,which_runs[mybootstrap,mysim],:numangles_to_examine]), n=2*numangles_to_examine) # zero-pad for separability 
                         #     #Frf = fft.fft(cos(self.angles[mychi,which_runs[mybootstrap,mysim],:numangles_to_examine]), n=2*numangles_to_examine) # zero-pad for separability 
                         #     #Sf = Frf * Frf.conjugate() 
                         #     #acf2 = fft.ifft(Sf) 
                         #     #acf2 = acf2[1:numangles_to_examine+1]/numangles_to_examine
                         #     #acf2 /= acf2[0] 
                         #     #acf2 = acf2.real 
                         #     #angles_autocorrelation[mychi,mybootstrap,:numangles_to_examine] +=  acf2 #= zeros((self.nchi, bootstrap_sets),float64)  #easy to do while making transition matrix
                         #     #angles_autocorrelation[mychi,mybootstrap,:numangles_to_examine] +=  autocorr(cos(self.angles[mychi,which_runs[mybootstrap,mysim],:numangles_to_examine])) #= zeros((self.nchi, bootstrap_sets),float64)  #easy to do while making transition matrix
                         #
               #         #angles_autocorrelation[mychi,mybootstrap,:numangles_to_examine] /= (bootstrap_choose * 1.0)
               #         #angles_to_examine = angles_autocorrelation[mychi,mybootstrap,0:numangles_to_examine]
               #         ##bins_to_examine = bins_autocorrelation[mychi,mybootstrap,0:numangles_to_examine]
               #         #my_x_values2 = my_x_values[angles_to_examine > 0]
               #         ##my_x_values2b = my_x_values[bins_to_examine > 0]
               #         #self.angles_autocorr_time[mychi, mybootstrap] = -1.0 / linalg.lstsq(my_x_values2[0:, None] , log( angles_to_examine[angles_to_examine > 0] ))[0][0]
               #         ##self.bins_autocorr_time[mychi, mybootstrap] = -1.0 / linalg.lstsq(my_x_values2b[0:, None] , log( bins_to_examine[bins_to_examine > 0] ))[0][0]
               #         ##self.angles_autocorr_time[mychi,mybootstrap] = -1.0 / ((exponential_fit(my_x_values, angles_autocorrelation[mychi,mybootstrap,:numangles_to_examine] ))[1])
               #         #if(mybootstrap == bootstrap_sets):
               #         #       print "angles_autocorrelation"
               #         #       print angles_autocorrelation[mychi,mybootstrap,:numangles_to_examine]
               #         #       print "angles autocorrelation time: "+str( self.angles_autocorr_time[mychi,mybootstrap] )
               #         #       #print "bins_autocorrelation"
               #         #       #print bins_autocorrelation[mychi,mybootstrap,:numangles_to_examine]
               #         #       #print "bins autocorrelation time: "+str( self.bins_autocorr_time[mychi,mybootstrap] )
               #         #       
               #         #if(mybootstrap == 0):
               #         #       myfile=open("angles_autocorr_%s%schi%s.dat"%(str(self.name),str(self.num),str(mychi)), "w")
               #         #       for i in range(numangles_to_examine):
               #         #              myfile.write( str( angles_autocorrelation[mychi,mybootstrap,i]) + "\n")
               #
               #         #myfile=open("bins_autocorr_%s%schi%s.dat"%(str(self.name),str(self.num),str(mychi)), "w")
               #         #if(mybootstrap == 0):
               #         #       for i in range(numangles_to_examine):
               #         #              myfile.write( str( bins_autocorrelation[mychi,mybootstrap,i]) + "\n")
               print "calculating transition matrix "
               #this next piece of code contains a loop over chi's for speed, so we have the loop over chis below this 
               weave.inline(code_transition_matrix_1D, ['num_sims', 'numangles','numangles_bootstrap', 'max_num_angles','max_angles','bootstrap_choose', 'nbins', 'bins', 'bootstrap_sets', 'nchi', 'transition_matrix_multiple_lagtimes', 'NUM_LAGTIMES','lagtime_interval','which_runs',  'SMALL', 'tempsums', 'tempsums_i', 'tempsums_j', 'tempprobs_i', 'tempprobs_j','PAD','permutations' ], 
                                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"],  extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler]
                                 ) #'mutinf_autocorrelation_vs_lagtime',
               
               print "done calculating transition matrix"

               
                    
               for mychi in range(self.nchi):
                      #print "transition matrix last lagtime: chi: "+str(mychi)
                      #print transition_matrix_multiple_lagtimes[mychi,0,-1]
                      
                      

                      #if(mychi + 1 < self.nchi):
                           #print "transition matrix first lagtime: next chi: "+str(mychi+1)
                           #print transition_matrix_multiple_lagtimes[mychi+1,0,-1]
                           #print "transition matrix last lagtime: next chi: "+str(mychi+1)
                           #print transition_matrix_multiple_lagtimes[mychi+1,0,-1]
                      tau_lagtimes[:] = 0
                      tau_lagtimes[0] = -1000
             
                      for mybootstrap in range(bootstrap_sets):
                             
                             
                         print "bootstrap of transition matrix: "+str(mybootstrap)       
                         # compute autocorrelation of bins 
                         # compute ACF using FFT 
                         

                         ## find mutinf autocorrelation time using exponential fit to data  
                         # sum pij log (pij / pipj) over bins for each lagtime
                         #print "mutinf autocorrelation bin elements"
                         #print "lagtime 1"
                         #print mutinf_autocorrelation_vs_lagtime[mychi,mybootstrap,1,:,:]
                         #print "lagtime 10"
                         #print mutinf_autocorrelation_vs_lagtime[mychi,mybootstrap,10,:,:]
                         #print "lagtime 100"
                         #print mutinf_autocorrelation_vs_lagtime[mychi,mybootstrap,100,:,:]
                         #print "lagtime 1000"
                         #print mutinf_autocorrelation_vs_lagtime[mychi,mybootstrap,1000,:,:]
                         #mutinf_autocorrelation = sum(sum(mutinf_autocorrelation_vs_lagtime[mychi,mybootstrap,:,:,:],axis=-1),axis=-1)
                         #mutinf_autocorrelation /= mutinf_autocorrelation[0]
                         #print str(mutinf_autocorrelation)
                         #if(VERBOSE > 4): #explicitly write out autocorrelation
                         #       myfile=open("autocorr_%s.dat"%str(mychi), "w")
                         #       for i in range(NUM_LAGTIMES):
                         #              myfile.write( str( mutinf_autocorrelation[i]) + "\n")
                         #       #for 
                         #       #.write(utils.arr2str2(mutinf_autocorrelation, precision=8))
                         #my_x_values = array(range(NUM_LAGTIMES), float64)
                         #print "mutinf autocorrelation: "
                         #print mutinf_autocorrelation
                         #    
                         ##normalize mutinf autocorrelation by first non-zero element
                         #mutinf_autocorrelation /= mutinf_autocorrelation[0]

                         ## see exponential_fit for a better way, but this has two parameters and I want to fix A=1. This should work for now
                         ##start at index 1 since mutinf autocorrelation at time zero is zero since pij = pi*pj
                         #self.mutinf_autocorr_time[mychi, mybootstrap] = -1.0 / linalg.lstsq(my_x_values[1:, None], log( mutinf_autocorrelation[1:]))[0][0]
                         #self.mutinf_autocorr_time[mychi,mybootstrap] = -1.0 / ((exponential_fit(my_x_values[1:], mutinf_autocorrelation[1:]))[1])
                         #print "mutinf autocorrelation time: "+str( self.mutinf_autocorr_time[mychi,mybootstrap] )
                         
                         
                         self.slowest_lagtime[mychi, mybootstrap] = 0 # an effectively null value
                         if num_convergence_points > 1:
                                this_num_lagtimes = int(NUM_LAGTIMES * (mybootstrap + 1 ) / num_convergence_points) #look at a number of lagtimes proportional to the percentage of data used in convergence calcs
                         else:
                                this_num_lagtimes = NUM_LAGTIMES
                         for mylagtime in range(1,this_num_lagtimes):
                            #print "count data at lagtime: "+str(mylagtime)
                            #print count_matrix_multiple_lagtimes[mychi,mybootstrap,mylagtime]
                            
                            #if(mylagtime == this_num_lagtimes - 2): 
                                   #print "transition data at lagtime: "+str(mylagtime)
                                   #print transition_matrix_multiple_lagtimes[mychi,mybootstrap,mylagtime]
                                   #print "chi pop hist: "
                                   #print self.chi_pop_hist[mybootstrap,mychi]
                            ##print "lagtime: "+str(mylagtime)   

                            try:    
                                   # U, s, VT = linalg.svd(transition_matrix_multiple_lagtimes[mychi,mybootstrap,mylagtime], full_matrices=True)
                                   s = sorted(linalg.eigvals(transition_matrix_multiple_lagtimes[mychi,mybootstrap,mylagtime]),reverse=True)
                                   #print "transition matrix eigenvalues: mychi:"+str(mychi)+" bootstrap:"+str(mybootstrap)+ " lagtime:"+str(mylagtime*lagtime_interval)
                                   #print s
                                   s_last = s #store this temporarily
                            except:
                                   s = s_last
                                   #s[0] = 1
                                   #s[1] = 0.9999999999
                                   
                                   #print "linalg.eigvals failed, trying linalg.svd"
                                   #try:
                                   #       U, s, VT = linalg.svd(transition_matrix_multiple_lagtimes[mychi,mybootstrap,mylagtime], full_matrices=True)
                                   #       print "transition matrix eigenvalues: mychi:"+str(mychi)+" bootstrap:"+str(mybootstrap)+ " lagtime:"+str(mylagtime*lagtime_interval)
                                   #       print s
                                   #except:
                                   #if mylagtime > 4: #at least try a few first
                                   #       mylagtime = mylagtime - 1
                                   #       print "eigval calc failed, starting to be non-Markovian, using last good tau value though implied timescale is unconverged " 
                                   #       
                                   #       #break


                            #print VT
                            s1 = s[1:]
                            #s_gt_0 = array(s1[s1 > 0])
                            try:
                                   tau_lagtimes[mylagtime] = -mylagtime*lagtime_interval / log(s[1] + SMALL)
                            except:
                                   tau_lagtimes[mylagtime] = this_num_lagtimes - 1
                            #print "tau lagtime :"+str(tau_lagtimes[mylagtime])
                            #first eigenvalue should be unity, within some discretization error perhaps
                            if s[0] > 1.05 or s[0] < 0.95: #now starting to be non-Markovian, time to drop out
                                   #self.slowest_implied_timescale[mychi, mybootstrap] = 0
                                   #self.slowest_lagtime[mychi, mybootstrap] = 0
                                   if mylagtime > 1:
                                          mylagtime = mylagtime - 1
                                          s = s_last #use last good value
                                          try:
                                                 tau_lagtimes[mylagtime] = -mylagtime*lagtime_interval / log(s[1] + SMALL)
                                          except:
                                                 tau_lagtimes[mylagtime] = this_num_lagtimes - 1
                                          print "first eigval not unity: starting to be non-Markovian, using last good tau value though implied timescale is unconverged " 
                                          print "eigvals: "
                                          print s
                                          break
                            #print "tau: "+str(tau_lagtimes[mylagtime])
                            # Choose a lagtime for the dihedral markov model --- I'm being very stringent here. Not only looking for convergence for implied tiemscale but requiring the lagtime > mutinf "autocorrelation"  time. 
                            #look for four values in a row within 10% of each other
                            test_percent = 0.01 
                            if tau_lagtimes[mylagtime] == 0:
                                   tau_lagtimes[mylagtime] = this_num_lagtimes - 1
                            if mylagtime >= 4: # and mylagtime >= self.mutinf_autocorr_time[mychi,mybootstrap] :
                                   if((abs(tau_lagtimes[mylagtime] - tau_lagtimes[mylagtime-1]) / tau_lagtimes[mylagtime]) < test_percent and (abs(tau_lagtimes[mylagtime-1] - tau_lagtimes[mylagtime-2]) / tau_lagtimes[mylagtime-1]) < test_percent and (abs(tau_lagtimes[mylagtime] - tau_lagtimes[mylagtime-2]) / tau_lagtimes[mylagtime]) < test_percent ) and (abs(tau_lagtimes[mylagtime] - tau_lagtimes[mylagtime-3]) / tau_lagtimes[mylagtime]) < test_percent :
                            #if mylagtime >= 3:
                            #       if((abs(tau_lagtimes[mylagtime] - tau_lagtimes[mylagtime-1]) / tau_lagtimes[mylagtime]) < test_percent and (abs(tau_lagtimes[mylagtime-1] - tau_lagtimes[mylagtime-2]) / tau_lagtimes[mylagtime-1]) < test_percent and (abs(tau_lagtimes[mylagtime] - tau_lagtimes[mylagtime-2]) / tau_lagtimes[mylagtime]) < test_percent ) :
                                          self.slowest_implied_timescale[mychi, mybootstrap] = tau_lagtimes[mylagtime]
                                          self.slowest_lagtime[mychi, mybootstrap] = mylagtime
                                          break 
                         if (self.slowest_lagtime[mychi, mybootstrap] == 0): 
                                self.slowest_implied_timescale[mychi, mybootstrap] = tau_lagtimes[mylagtime] #use the last one tried
                                self.slowest_lagtime[mychi, mybootstrap] = mylagtime  # use maximum value if it did not converge, though should probably generate some sort of issue
                                print "did not converge, so using maximum lagtime"
                                print "slowest lagtime: "+ str(self.slowest_lagtime[mychi,mybootstrap] * lagtime_interval)
                                print "slowest lagtime implied timescale: "+str(self.slowest_implied_timescale[mychi, mybootstrap])
                         self.transition_matrix[mychi,mybootstrap] = transition_matrix_multiple_lagtimes[mychi,mybootstrap,self.slowest_lagtime[mychi,mybootstrap] ]
                         try:
                                print "eigenvalues, bootstrap: "+str(mybootstrap)
                                print s
                         except:
                                s = zeros(((shape(transition_matrix_multiple_lagtimes[mychi,mybootstrap,mylagtime]))[-1]), float64)
                                s[0] = 1.0
                                print "eigenvalue calc failed, so using dummy values instead "
                                self.slowest_implied_timescale[mychi, mybootstrap] = max_num_angles
                                self.slowest_lagtime[mychi, mybootstrap] = this_num_lagtimes*lagtime_interval
                         print "slowest lagtime*lagtime_interval: "+str(self.slowest_lagtime[mychi,mybootstrap]*lagtime_interval)
                         print "slowest implied timescale: "+str(self.slowest_implied_timescale[mychi, mybootstrap])
                         self.slowest_lagtime[mychi,mybootstrap] *= lagtime_interval #convert to trajectory lagtime in snapshots
                         #now set the slowest lagtime equal to the minimum of the slowest implied timescale for stochastic sampling
                         self.slowest_lagtime[mychi,mybootstrap] = int(min([self.slowest_lagtime[mychi,mybootstrap],self.slowest_implied_timescale[mychi,mybootstrap]]))
                         
               
               for mychi in range(self.nchi):
                      for mybootstrap in range(bootstrap_sets):
                       ### Now generate samples from this transition matrix
                              print "transition matrix: bootstrap: "+str(mybootstrap)+" shape:"
                              print shape(self.transition_matrix)
                              print "transition matrix: bootstrap: "+str(mybootstrap)+" chi: "+str(mychi)
                              print self.transition_matrix[mychi,mybootstrap]
                              
               print "sampling from transition matrix"
               
               


               ent_markov_boots = self.ent_markov_boots
               #this code already has a loop over chi's and bootstraps           
               weave.inline(code_sample_from_transition_matrix_1D_new, ['num_sims', 'numangles','numangles_bootstrap', 'max_num_angles','max_angles','bootstrap_choose', 'nbins', 'bins', 'bootstrap_sets', 'mychi', 'transition_matrix', 'lagtime_interval','which_runs','markov_samples', 'chi_counts_markov', 'nchi', 'bins_markov', 'slowest_lagtime', 'ent_markov_boots', 'permutations' ],
                            compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"] , extra_compile_args =my_extra_compile_args[mycompiler],extra_link_args=my_extra_link_args[mycompiler],
                            support_code=my_support_code )
                              # now, self.bins_markov :: (self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ) # the bin for each dihedral
               
               #print stuff and do assertions
               for mychi in range(self.nchi):
                      for mybootstrap in range(bootstrap_sets):
                         print "chi counts markov sample 0:"
                         print chi_counts_markov[mychi,0,0]
                         #print "chi counts markov sample 1:"
                         #print chi_counts_markov[0,1,mychi]
                         print "sum of chi counts markov, markov sample 0: "
                         print sum(chi_counts_markov[mychi,0,0])
                         #print "sum of chi counts markov, markov sample 1: "
                         #print sum(chi_counts_markov[0,1,mychi])

                         for mychain in range(markov_samples):
                                #print "my markov chain: "+str(mychain)
                                assert (sum(chi_counts_markov[mychi,mybootstrap,mychain]) == sum(chi_counts_markov[mychi,mybootstrap,0]))
                                assert (sum(chi_counts_markov[mychi, mybootstrap, mychain]) == numangles_bootstrap[mybootstrap] )
               
               #print "chi pop hist sequential1:"
               #print chi_pop_hist_sequential[0,0,:]*numangles[0]
               #print "chi pop hist sequential2:"
               #print chi_pop_hist_sequential[1,0,:]*numangles[1]
      #print "numangles_bootstrap:"+str(self.numangles_bootstrap)+"\n"
         # look out for bin values less than zero
      ## Done with permutations
      del transition_matrix_multiple_lagtimes #free up memory
      #del mutinf_autocorrelation_vs_lagtime #free up memory

      if len(self.bins[self.bins<0]) > 0:
          print "ERROR: bin values should not be less than zero: ",
          for i in range(num_sims):
              if(VERBOSE >=2):
                  print "Angles\n"
                  print utils.arr1str1( self.angles[mychi, i, :self.numangles[i]]),
                  print "Bins\n"
                  for duboot in range(bootstrap_sets):
                      print utils.arr1str1( self.bins[mychi, 0, duboot, :])
                  
          sys.exit(1)

      if(VERBOSE >= 2):
      #      print self.bins[0:2,:,0,:]
             print "counts adaptive:"
             print counts_adaptive[0,:,:]
             #print "Counts Nonadaptive:\n"
             #print counts
             print "chi_counts:"
             print self.chi_counts[0]
             print "chi pop hist:"
             print self.chi_pop_hist[0]

      #ADD kernel density estimation here to replace histogram entropy, also provide per-torsion entropy
      
      #calculate entropy for various binwidths, will take the max later
      calc_entropy(self.counts, self.nchi, numangles_bootstrap, calc_variance=calc_variance,entropy=self.entropy,var_ent=self.var_ent,symmetry=self.symmetry,expansion_factors=self.expansion_factors)
      calc_entropy(self.counts2, self.nchi, numangles_bootstrap, calc_variance=calc_variance,entropy=self.entropy2,var_ent=self.var_ent,symmetry=self.symmetry,expansion_factors=self.expansion_factors)
      calc_entropy(self.counts3, self.nchi, numangles_bootstrap, calc_variance=calc_variance,entropy=self.entropy3,var_ent=self.var_ent,symmetry=self.symmetry,expansion_factors=self.expansion_factors)
      #print (sum(counts_adaptive * self.ent_hist_binwidths, axis=2) * numangles_bootstrap)[0,:]
      calc_entropy_adaptive(counts_adaptive, num_sims, self.nchi, bootstrap_choose, calc_variance=calc_variance,entropy=self.entropy4,var_ent=self.var_ent,binwidths=self.ent_hist_binwidths)
      #print "Nearest Neighbor distances k=1 thru 10"
      #print self.ent_from_sum_log_nn_dists[:,:,:]
      #use average of NN estimates for k=7 thru k=10
      #lower values of k are not used as there might be NN dists below that which we can resolve
      #this approach gets around this problem


      #Alright, now compare entropy estimates achieved using different methods
      #For the KNN estimate, I found that the k=4 or k=5 case seemed to be close to the 
      #histogram entropy.
      #So pick the k-value that gives the lowest positive entropy, and then pick the max of the knn estimate
      # or the histogramming estimate.
      # Phi/Psi angles typically are poor for histogramming
      #This should use NN entropy estimate for highly spiked distributions
      #so that they do not have a corrected entropy lower than zero

      print "Entropies for various bin sizes:"
      print self.entropy
      print self.entropy2
      print self.entropy3
      print self.entropy4
      print "K Nearest neighbor entropies:"
      print (ent_from_sum_log_nn_dists)
      for bootstrap in range(bootstrap_sets):
          for mychi in range(nchi):
              thisent1 = self.entropy[bootstrap,mychi]
              thisent2 = self.entropy2[bootstrap,mychi]
              thisent3 = self.entropy3[bootstrap,mychi]
              thisent4 = self.entropy4[bootstrap,mychi]
              #knn_ent_estimates = self.ent_from_sum_log_nn_dists[bootstrap,mychi,:] 
              #my_knn_ent_estimate = min(knn_ent_estimates[knn_ent_estimates > 0]) 
              #my_knn_ent_estimate = knn_ent_estimates[0]
              ## Use Abel 2009 JCTC eq. 27 weighting for KNN estimates for k=1,2,3
              #my_knn_ent_estimate = sum(WEIGHTS_KNN * knn_ent_estimates)
              #my_knn_ent_estimate = knn_ent_estimates
              #if my_knn_ent_estimate > 0:
              #    self.entropy[bootstrap,mychi] = my_knn_ent_estimate
              #else: ## If the KNN algorithm doesn't numerically work well for our data
              if(thisent4 < 0):
                     thisent4 = 0
              thisent1b = max(thisent1,0)
              thisent2b = max(thisent2,0)
              thisent3b = max(thisent3,0)
              thisent4b = max(thisent4,0)
              thisent_final = 99
              if(thisent1b > 0 and thisent1b < thisent_final):
                     thisent_final = thisent1b
              if(thisent2b > 0 and thisent2b < thisent_final):
                     thisent_final = thisent2b
              if(thisent3b > 0 and thisent3b < thisent_final):
                     thisent_final = thisent3b
              if(thisent4b > 0 and thisent4b < thisent_final):
                     thisent_final = thisent4b
              if thisent_final == 99:
                     thisent_final = 0
              
              self.entropy[bootstrap,mychi] = thisent_final #min(max(thisent1,0),max(thisent2,0),max(thisent3,0),max(thisent4,0)) #min(thisent4, 0) # max((thisent1, thisent2, thisent3, thisent4))
              if self.entropy[bootstrap,mychi] < 0:
                  print "WARNING: NEGATIVE ENTROPY DETECTED! "


      #self.entropy = entropy
      #self.var_ent += var_ent
      print "Total entropy (before mutinf): %.5f (s2= %.5f )" % (sum(average(self.entropy,axis=0),axis=0), sum(self.var_ent))
      print self.entropy

      

      # derivative of Kullback-Leibler divergence for this residue's chi angles by themselves
      # this has no second-order terms
      # factor of nbins is because sum over index i is retained, while sum over index i just gives a factor of nbins
      numangles_bootstrap_vector = zeros((bootstrap_sets,nbins),float64)
      for bootstrap in range(bootstrap_sets):
          numangles_bootstrap_vector[bootstrap,:]=numangles_bootstrap[bootstrap]
      self.dKLtot_dchis2 = zeros((bootstrap_sets, self.nchi))

      for bootstrap in range(bootstrap_sets):
        for mychi in range(self.nchi):
            nonzero_bins = chi_counts[bootstrap,mychi,:] > 0    
            self.dKLtot_dchis2[bootstrap,mychi] = nbins * sum(((numangles_bootstrap_vector[bootstrap])[nonzero_bins]) * (- 1.0 / ((self.chi_counts[bootstrap,mychi])[nonzero_bins] * 1.0)), axis=0)
      
       
     #free up things we don't need any more, like angles, rank ordered angles, boot angles, etc.
      if(output_timeseries != "yes"):
             del self.angles
      del self.rank_order_angles
      del self.rank_order_angles_sequential
      del self.sorted_angles
      del self.boot_sorted_angles
      del self.boot_ranked_angles
      del self.ent_from_sum_log_nn_dists


#########################################################################################################################################
##### calc_pair_stats: Mutual Information For All Pairs of Torsions ##########################################################
#########################################################################################################################################


# Calculate the mutual information between all pairs of residues.
# The MI between a pair of residues is the sum of the MI between all combinations of res1-chi? and res2-chi?.
# The variance of the MI is the sum of the individual variances.
# Returns mut_info_res_matrix, mut_info_uncert_matrix
def calc_pair_stats(reslist, run_params, plot_2d_histograms):
    rp = run_params
    which_runs = rp.which_runs
    #print rp.which_runs
    bootstrap_sets = len(which_runs)
    #initialize the mut info matrix
    #check_for_free_mem()
    mut_info_res_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),6,6),float32)
    mut_info_norm_res_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),6,6),float32)
    mut_info_res_matrix_different_sims = zeros((bootstrap_sets, len(reslist),len(reslist),6,6),float32)
    mut_info_uncert_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),6,6),float32)
    dKLtot_dresi_dresj_matrix = zeros((bootstrap_sets, len(reslist),len(reslist)),float32)
    Counts_ij = zeros((bootstrap_sets,rp.nbins,rp.nbins),float64)
    if(run_params.plot_2d_histograms == True):
           twoD_hist_boot_avg = zeros((len(reslist),len(reslist),6,6,rp.nbins,rp.nbins),float32) #big matrix of 2D populations sum over bootstraps
           twoD_hist_boots = zeros((bootstrap_sets,len(reslist),len(reslist),6,6,rp.nbins,rp.nbins),float32) #big matrix of 2D populations sum over bootstraps
           twoD_hist_ind_boot_avg = zeros((len(reslist),len(reslist),6,6,rp.nbins,rp.nbins),float32) #big matrix of 2D populations sum over bootstraps
           twoD_hist_ind_boots = zeros((bootstrap_sets,len(reslist),len(reslist),6,6,rp.nbins,rp.nbins),float32) #big matrix of 2D populations sum over bootstraps
    else:
           twoD_hist_boot_avg = 0 #dummy value
           twoD_hist_boots = 0 #dummy value
           twoD_hist_ind_boot_avg = 0
           twoD_hist_ind_boots = 0
    numangles_bootstrap_nbins_nbins = zeros((bootstrap_sets,rp.nbins,rp.nbins))
    #Loop over the residue list
    print
    for res_ind1, myres1 in zip(range(len(reslist)), reslist):    
       print "##### Working on residue %s%s  (%s) and other residues" % (myres1.num, myres1.chain, myres1.name), utils.flush()
       if(res_ind1 == 0):
              for mybootstrap in range(bootstrap_sets):
                     numangles_bootstrap_nbins_nbins[mybootstrap,:,:] = myres1.numangles_bootstrap[mybootstrap]
       for res_ind2, myres2 in zip(range(res_ind1, len(reslist)), reslist[res_ind1:]):
        if (VERBOSE >= 0):
               print "#### Working on residues %s%s and %s%s (%s and %s):" % (myres1.num, myres1.chain, myres2.num, myres2.chain, myres1.name, myres2.name) , utils.flush()
        max_S = 0.
        mutinf_thisdof = uncorrected_mutinf_thisdof = corrections_mutinf_thisdof = var_mi_thisdof = mutinf_thisdof_different_sims = dKLtot_dKL1_dKL2 = MI_norm = 0 #initialize
    
        for mychi1 in range(myres1.nchi):
             for mychi2 in range(myres2.nchi):
                 if(VERBOSE >=1):
                        print 
                        print "%s %s , %s %s chi1/chi2: %d/%d" % (myres1.name,myres1.num, myres2.name,myres2.num, mychi1+1,mychi2+1)
                 #check_for_free_mem()
                 mutinf_thisdof = uncorrected_mutinf_thisdof = corrections_mutinf_thisdof = var_mi_thisdof = mutinf_thisdof_different_sims = dKLtot_dKL1_dKL2 = MI_norm = 0 #initialize
                 angle_str = ("%s_chi%d-%s_chi%d"%(myres1, mychi1+1, myres2, mychi2+1)).replace(" ","_")
                 if(VERBOSE >=2):
                        #print "twoD hist boot avg shape: " + str(twoD_hist_boot_avg.shape ) 
                        print "ent1_markov_boots:        " + str(myres1.ent_markov_boots[mychi1,:,:])
                        print "ent2_markov_boots:        " + str(myres2.ent_markov_boots[mychi2,:,:])
                 if((res_ind1 != res_ind2 and OFF_DIAG==1 ) or (res_ind1 == res_ind2 and mychi1 > mychi2)):
                     #print "slowest_lagtime: chi: "+str(mychi1)+" : "+str(max(myres1.slowest_lagtime[mychi1]))+"\n"
                     #print "slowest_lagtime: chi: "+str(mychi2)+" : "+str(max(myres2.slowest_lagtime[mychi2]))+"\n"
                     mutinf_thisdof, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, var_mi_thisdof, mutinf_thisdof_different_sims, dKLtot_dKL1_dKL2, Counts_ij, Counts_ij_ind, MI_norm = \
                                 calc_excess_mutinf(myres1.chi_counts[:,mychi1,:],myres2.chi_counts[:,mychi2,:],\
                                                    myres1.bins[mychi1,:,:,:], myres2.bins[mychi2,:,:,:], \
                                                    myres1.chi_counts_sequential[:,mychi1,:],\
                                                    myres2.chi_counts_sequential[:,mychi2,:],\
                                                    myres1.simbins[mychi1,:,:,:], myres2.simbins[mychi2,:,:,:],\
                                                    rp.num_sims, rp.nbins, myres1.numangles_bootstrap,\
                                                    myres1.numangles, rp.sigalpha, rp.permutations,\
                                                    rp.bootstrap_choose, calc_variance=rp.calc_variance,\
                                                    which_runs=rp.which_runs,pair_runs=rp.pair_runs,\
                                                    calc_mutinf_between_sims=rp.calc_mutinf_between_sims, \
                                                    markov_samples = rp.markov_samples, \
                                                    chi_counts1_markov = myres1.chi_counts_markov[mychi1,:,:,:], \
                                                    chi_counts2_markov = myres2.chi_counts_markov[mychi2,:,:,:], \
                                                    ent1_markov_boots =  myres1.ent_markov_boots[mychi1,:,:], \
                                                    ent2_markov_boots =  myres2.ent_markov_boots[mychi2,:,:], \
                                                    bins1_markov = myres1.bins_markov[mychi1,:,:,:], \
                                                    bins2_markov = myres2.bins_markov[mychi2,:,:,:], \
                                                    file_prefix=angle_str, plot_2d_histograms=rp.plot_2d_histograms, \
                                                    adaptive_partitioning = rp.adaptive_partitioning, \
                                                    bins1_slowest_timescale = myres1.slowest_implied_timescale[mychi1], \
                                                    bins2_slowest_timescale = myres2.slowest_implied_timescale[mychi2], \
                                                    bins1_slowest_lagtime = myres1.slowest_lagtime[mychi1], \
                                                    bins2_slowest_lagtime = myres2.slowest_lagtime[mychi2], \
                                                    lagtime_interval = rp.lagtime_interval, \
                                                    boot_weights = myres1.boot_weights , \
                                                    weights = myres1.weights, num_convergence_points=rp.num_convergence_points, cyclic_permut=rp.cyclic_permut )
                     print "mutinf this dof:"
                     print mutinf_thisdof
                     if rp.num_convergence_points > 1 and OFF_DIAG==1 and VERBOSE >=2 :
                            output_mutinf_convergence(str(myres1.name)+str(myres1.num)+"chi"+str(mychi1+1)+"_"+str(myres2.name)+str(myres2.num)+"chi"+str(mychi2+1)+"_mutinf_convergence.txt", mutinf_thisdof, bootstrap_sets)
                            output_mutinf_convergence(str(myres1.name)+str(myres1.num)+"chi"+str(mychi1+1)+"_"+str(myres2.name)+str(myres2.num)+"chi"+str(mychi2+1)+"_uncorrected_convergence.txt", uncorrected_mutinf_thisdof, bootstrap_sets)
                            output_mutinf_convergence(str(myres1.name)+str(myres1.num)+"chi"+str(mychi1+1)+"_"+str(myres2.name)+str(myres2.num)+"chi"+str(mychi2+1)+"_independent_convergence.txt", corrections_mutinf_thisdof, bootstrap_sets)
                     
                 if(res_ind1 == res_ind2 and mychi1 == mychi2):
                     mut_info_res_matrix[:,res_ind1, res_ind2, mychi1, mychi2] = myres1.entropy[:,mychi1]
                     mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] = myres1.var_ent[:,mychi1]
                     max_S = 0
                     ## still need to calc dKLdiv here
                     dKLtot_dresi_dresj_matrix[:,res_ind1, res_ind2] += 0 #myres1.dKLtot_dchis2[:,mychi1]
                 
                 elif(res_ind1 == res_ind2 and mychi1 > mychi2):
                     mut_info_res_matrix[:,res_ind1 , res_ind2, mychi1, mychi2] = mutinf_thisdof
                     mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] = var_mi_thisdof
                     mut_info_res_matrix_different_sims[:,res_ind1, res_ind2, mychi1, mychi2] = mutinf_thisdof_different_sims
                     mut_info_norm_res_matrix[:,res_ind1 , res_ind2, mychi1, mychi2] = MI_norm
                     if(plot_2d_histograms == True):
                            twoD_hist_boots[:,res_ind1, res_ind2, mychi1, mychi2, :, :] = Counts_ij / numangles_bootstrap_nbins_nbins
                            twoD_hist_boots[:,res_ind1, res_ind2, mychi2, mychi1, :, :] = swapaxes(Counts_ij / numangles_bootstrap_nbins_nbins, -2, -1) #symmetric matrix
                            twoD_hist_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :] = average(Counts_ij / numangles_bootstrap_nbins_nbins, axis=0)
                            twoD_hist_boot_avg[res_ind1, res_ind2, mychi2, mychi1, :, :] = swapaxes(average(Counts_ij / numangles_bootstrap_nbins_nbins, axis=0) , -2, -1) #symmetric matrix
                            twoD_hist_ind_boots[:,res_ind1, res_ind2, mychi1, mychi2, :, :] = Counts_ij_ind / numangles_bootstrap_nbins_nbins
                            twoD_hist_ind_boots[:,res_ind1, res_ind2, mychi2, mychi1, :, :] = swapaxes(Counts_ij_ind / numangles_bootstrap_nbins_nbins, -2, -1) #symmetric matrix
                            twoD_hist_ind_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :] = average(Counts_ij_ind / numangles_bootstrap_nbins_nbins, axis=0)
                            twoD_hist_ind_boot_avg[res_ind1, res_ind2, mychi2, mychi1, :, :] = swapaxes(average(Counts_ij_ind / numangles_bootstrap_nbins_nbins, axis=0) , -2, -1) #symmetric matrix
                     mut_info_res_matrix[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_res_matrix[:,res_ind1, res_ind2, mychi1, mychi2]
                     mut_info_uncert_matrix[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2]
                     max_S = 0
                     ## still need to calc dKLdiv here
                     dKLtot_dresi_dresj_matrix[:,res_ind1, res_ind2] += dKLtot_dKL1_dKL2
                     
                 else:
                     S = 0
                     mychi1 = int(mychi1)
                     mychi2 = int(mychi2)
                     blah = myres1.chi_pop_hist[:,mychi1,:]
                     blah = myres2.chi_pop_hist[:,mychi2,:]
                     blah = myres1.bins[mychi1,:,:,:]
                     blah =  myres2.bins[mychi2,:,:,:]
                     blah =  myres1.chi_pop_hist_sequential[:,mychi1,:]
                     dKLtot_dresi_dresj_matrix[:,res_ind1, res_ind2] += dKLtot_dKL1_dKL2
                     dKLtot_dresi_dresj_matrix[:,res_ind2, res_ind1] += dKLtot_dKL1_dKL2 #note res_ind1 neq res_ind2 here
                     mut_info_res_matrix[:,res_ind1 , res_ind2, mychi1, mychi2] = mutinf_thisdof
                     mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] = var_mi_thisdof
                     mut_info_res_matrix_different_sims[:,res_ind1, res_ind2, mychi1, mychi2] = mutinf_thisdof_different_sims
                     mut_info_norm_res_matrix[:,res_ind1 , res_ind2, mychi1, mychi2] = MI_norm
                     print "shape of Counts_ij:"+str(shape(Counts_ij))
                     if(plot_2d_histograms == True):
                            twoD_hist_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :] = average(Counts_ij / numangles_bootstrap_nbins_nbins , axis=0)
                            twoD_hist_boot_avg[res_ind2, res_ind1, mychi2, mychi1, :, :] =average(swapaxes(Counts_ij / numangles_bootstrap_nbins_nbins, -2, -1), axis=0)  #symmetric matrix
                            twoD_hist_boots[:,res_ind1, res_ind2, mychi1, mychi2, :, :] = Counts_ij  / numangles_bootstrap_nbins_nbins
                            twoD_hist_boots[:,res_ind2, res_ind1, mychi2, mychi1, :, :] = swapaxes(Counts_ij /numangles_bootstrap_nbins_nbins, -2, -1) #symmetric matrix
                            twoD_hist_ind_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :] = average(Counts_ij_ind / numangles_bootstrap_nbins_nbins , axis=0)
                            twoD_hist_ind_boot_avg[res_ind2, res_ind1, mychi2, mychi1, :, :] =average(swapaxes(Counts_ij_ind / numangles_bootstrap_nbins_nbins, -2, -1), axis=0)  #symmetric matrix
                            twoD_hist_ind_boots[:,res_ind1, res_ind2, mychi1, mychi2, :, :] = Counts_ij_ind  / numangles_bootstrap_nbins_nbins
                            twoD_hist_ind_boots[:,res_ind2, res_ind1, mychi2, mychi1, :, :] = swapaxes(Counts_ij_ind /numangles_bootstrap_nbins_nbins, -2, -1) #symmetric matrix
                     max_S = max([max_S,S])
                     mut_info_res_matrix[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_res_matrix[:,res_ind1, res_ind2, mychi1, mychi2] #symmetric matrix
                     #mut_info_uncert_matrix[res_ind1, res_ind2] = mut_info_uncert_matrix[res_ind1, res_ind2]
                     mut_info_uncert_matrix[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] #symmetric matrix
                     mut_info_res_matrix_different_sims[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_res_matrix_different_sims[:,res_ind1, res_ind2, mychi1, mychi2] #symmetric matrix
                     #twoD_hist_boot_avg[res_ind2, res_ind1, mychi2, mychi1, :, :] = swapaxes(twoD_hist_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :],0,1) #symmetric matrix
        #print "mutinf=%.3f (uncert=%.3f; max(S)=%.3f" % (average((mut_info_res_matrix[:,res_ind1, res_ind2, : ,:]).flatten()), sum((mut_info_uncert_matrix[0,res_ind1, res_ind2, :, :]).flatten()), max_S),
        #if max_S > 0.26: print "#####",
        
        print
        
    "mut info res matrix:"
    print mut_info_res_matrix
    return mut_info_res_matrix, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, mut_info_uncert_matrix, mut_info_res_matrix_different_sims, dKLtot_dresi_dresj_matrix, twoD_hist_boot_avg, twoD_hist_boots, twoD_hist_ind_boots, twoD_hist_ind_boot_avg, mut_info_norm_res_matrix








#########################################################################################################################################
##### Routines for Loading Data #########################################################################################################
#########################################################################################################################################

class ResListEntry:
    name = 'XXX'
    num = 0
    chain = ' '
    def __init__(self,myname,mynum):
        self.name = myname
        self.num = mynum
    def __init__(self,myname,mynum,mychain):
        self.name = myname
        self.num = mynum
        self.chain = mychain

def load_resfile(run_params, load_angles=True, all_angle_info=None):
    rp = run_params
    if rp.num_structs is None: rp.num_structs =  16777216 # # 1024 * 1024 * 16 #1500000
    sequential_num = 0
    resfile=open(rp.resfile_fn,'r')
    reslines=resfile.readlines()
    resfile.close()
    reslist = []
    for resline in reslines:
       if len(resline.strip()) == 0: continue
       xvg_resnum, res_name, res_numchain = resline.split()
       myexpr = re.compile(r"([0-9]+)([A-Z]*)")
       matches = myexpr.match(res_numchain)
       res_num = matches.group(1)
       if matches.group(2) != None:
              res_chain = matches.group(2)
       else:
              res_chain = ""
       if load_angles: 
              reslist.append(ResidueChis(res_name,res_num, res_chain, xvg_resnum, rp.xvg_basedir, rp.num_sims, rp.num_structs, rp.xvgorpdb, rp.binwidth, rp.sigalpha, rp.permutations, rp.phipsi, rp.backbone_only, rp.adaptive_partitioning, rp.which_runs, rp.pair_runs, bootstrap_choose = rp.bootstrap_choose, calc_variance=rp.calc_variance, all_angle_info=all_angle_info, xvg_chidir=rp.xvg_chidir, skip=rp.skip,skip_over_steps=rp.skip_over_steps,last_step=rp.last_step, calc_mutinf_between_sims=rp.calc_mutinf_between_sims,max_num_chis=rp.max_num_chis, sequential_res_num = sequential_num, pdbfile=rp.pdbfile, xtcfile=rp.xtcfile, output_timeseries=rp.output_timeseries, lagtime_interval=rp.lagtime_interval, markov_samples=rp.markov_samples, num_convergence_points=rp.num_convergence_points, cyclic_permut = rp.cyclic_permut  ))
       else:  reslist.append(ResListEntry(res_name,res_num,res_chain))
       sequential_num += 1 
    return reslist



# Load angle data and calculate intra-residue entropies
def load_data(run_params):
    load_resfile(run_params, load_angles=False) # make sure the resfile parses correctly (but don't load angle data yet)

    ### load trajectories from pdbs
    all_angle_info = None
    if run_params.xvgorpdb == "pdb":
       trajs = [PDBlite.PDBTrajectory(traj_fn) for traj_fn in run_params.traj_fns]
       traj_lens = array([traj.parse_len() for traj in trajs])
       run_params.num_structs = int(min(traj_lens)) # convert to python int, because otherwise it stays as a numpy int which weave has trouble interpreting
       run_params.num_res = trajs[0].parse_len()

       all_angle_info = AllAngleInfo(run_params)
       runs_to_load = set(array(run_params.which_runs).flatten())
       for sequential_sim_num, true_sim_num in zip(range(len(runs_to_load)), runs_to_load):
          all_angle_info.load_angles_from_traj(sequential_sim_num, trajs[true_sim_num-1], run_params, CACHE_TO_DISK)
       print "Shape of all angle matrix: ", all_angle_info.all_chis.shape

    print run_params
    print type(run_params.num_structs)

    ### load the residue list and angle info for those residues
    ### calculate intra-residue entropies and their variances
    reslist = load_resfile(run_params, load_angles=True, all_angle_info=all_angle_info)

    print "\n--Num residues: %d--" % (len(reslist))
    if (run_params.xvgorpdb): run_params.num_structs = int(max(reslist[0].numangles))

    return reslist

##########################################################################################################
### RUNTIME SELF-TESTING FOR MUTINF 
##########################################################################################################
def test_mutinf(test_options, xvg_basedir, resfile_fn, weights_fn):
    
    bins=arange(-180,180,test_options.binwidth) #Compute bin edges
    nbins = len(bins)

    chi_counts =   zeros((2, 2, nbins), int64)
    chi_pop_hist = zeros((2, 2, nbins),float64)
    angles = zeros((2,2,test_options.num_sims,10001),float64)
    numangles = 10001
    mutinf_matrix = zeros((2,2))
    tot_numangles = 10001 * test_options.num_sims 
    bins = zeros((2,2,tot_numangles),int64)
    bins = zeros((2,2,tot_numangles),int64)
    weights_all = zeros((tot_numangles),int64)
    thisnumangles = [numangles, numangles]
    theseresidues = ["PHE42", "PHE78"]
    #read angles from xvg file, map to bins
    kdir = xvg_basedir
    for dataset in range(2): 
      for jchi in range(2):
          for n in range(test_options.num_sims):
                (data, title) = readxvg(str(kdir)+"run"+str(n+1)+"/chi"+str(jchi+1)+str(theseresidues[dataset])+".xvg",1,0)
                thisnumangles[dataset] = min(thisnumangles[dataset],len(data[:,1]))
                angles[dataset,jchi,n,:thisnumangles[dataset]] = data[:thisnumangles[dataset],1]
                (data2, title) = readxvg(str(kdir)+"run"+str(n+1)+"/"+weights_fn,1,0)
                weights_this_sim = data2[:thisnumangles[dataset],1]
                anglestemp = angles[dataset,jchi,n,:thisnumangles[dataset]]
                anglestemp[anglestemp < 0] += 360 # wrap around
                angles[dataset,jchi,n,:thisnumangles[dataset]] = anglestemp
                if(jchi == 1): #correct PHE42 and PHE78 chi2 for symmetry
                    anglestemp = angles[dataset,jchi,n,:thisnumangles[dataset]]
                    anglestemp[anglestemp > 180] = anglestemp[anglestemp > 180] - 180
                    angles[dataset,jchi,n,:thisnumangles[dataset]] = anglestemp
                for anglenum in range(thisnumangles[dataset]): #map to bins
                    bin_num = binsingle(angles[dataset , jchi, n, anglenum], 1.0 / test_options.binwidth)
                    chi_counts[dataset, jchi, bin_num] += 1.0 * weights_this_sim[anglenum]
                    bins[dataset, jchi, n * tot_numangles + anglenum] = bin_num
                    weights_all[n*numangles + anglenum] = weights_this_sim[anglenum]
                print "chi_counts1, chi: "+str(jchi)+ " :" + str(chi_counts[0, jchi])
                print "chi_counts2, chi: "+str(jchi)+ " :" + str(chi_counts[0, jchi])
          
      ## Generate Pairwise Histogram
    for res_i in range(2): #index of theseresidues 
        for res_j in range(2):  #index of theseresidues
            for ichi in range(2):
                for jchi in range(2):
                    count_matrix = zeros((2,2,2,2,nbins,nbins),float64)
                    for anglenum in range(tot_numangles):
                      count_matrix[res_i,res_j,ichi,jchi,bins1[jchi,anglenum], bins2[jchi,anglenum]] += 1.0 * weights_all[anglenum]
      

    #now, calculate mutinf
    print "counts i1:"+str(chi_counts1[0,:])
    print "counts j1:"+str(chi_counts2[0,:])
    print "counts i2:"+str(chi_counts1[1,:])
    print "counts j2:"+str(chi_counts2[1,:])     
    for res_i in range(2): #index of theseresidues 
      for res_j in range(2):  #index of theseresidues
         for ichi in range(2):
          for jchi in range(2):
              ent1 = sum((chi_counts[res_i, ichi,:] * 1.0 / tot_numangles) * (log(tot_numangles) - special.psi(chi_counts1[ichi,:] + SMALL) - \
                                                                                  (1 - 2*(float64(chi_counts[res_i, ichi, :] % 2))) / (chi_counts[res_i, ichi, :] + 1.0)))
              ent2 = sum((chi_counts[res_j, jchi,:] * 1.0 / tot_numangles) * (log(tot_numangles) - special.psi(chi_counts1[jchi,:] + SMALL) - \
                                                                                  (1 - 2*(float64(chi_counts[res_j, jchi, :] % 2))) / (chi_counts[res_j, jchi, :] + 1.0)))     
              counts = count_matrix[res_i, res_j, ichi, jchi, :]
              mutinf_matrix[res_i,res_j,ichi,jchi] = ent1 + ent2 \
                  - sum((counts * 1.0 / tot_numangles) * (log(tot_numanles) - special.psi(counts + SMALL) - \
                                                           (1 - 2*(float64(counts % 2))) / (counts + 1.0)))
         print "residue i: "+theseresidues[res_i]
         print "residue j: "+theseresidues[res_j]
         print "mutual information between chis from 1 to 2:"
         print mutinf_matrix[res_i,res_j,:,:]
      
    for jchi in range(2):
        chi_pop_hist1[jchi,:] = chi_counts1[jchi,:] * 1.0 / (test_options.num_sims * thisnumangles[0])
        chi_pop_hist2[jchi,:] = chi_counts2[jchi,:] * 1.0 / (test_options.num_sims * thisnumangles[1])
    
    pi1 = chi_pop_hist1[0,:]
    pi2 = chi_pop_hist1[1,:]
    pj1 = chi_pop_hist2[0,:]
    pj2 = chi_pop_hist2[1,:]

    assert(sum(pi1) > 0.99999 and sum(pi1) < 1.00001)
    assert(sum(pi2) > 0.99999 and sum(pi2) < 1.00001)
    assert(sum(pj1) > 0.99999 and sum(pj1) < 1.00001)
    assert(sum(pj2) > 0.99999 and sum(pj2) < 1.00001)
    
    
    print "pi1:"+str(chi_pop_hist1[0,:])
    print "pj1:"+str(chi_pop_hist2[0,:])
    print "pi2:"+str(chi_pop_hist1[1,:])
    print "pj2:"+str(chi_pop_hist2[1,:])     
    
    chi_pop_hist1[chi_pop_hist1==0] = SMALL * 0.5 # to avoid NaNs
    chi_pop_hist2[chi_pop_hist2==0] = SMALL * 0.5 # to avoid NaNs

    
    for mychi in range(2):            
        pi = chi_pop_hist1[mychi,:]
        pj = chi_pop_hist2[mychi,:]
        
        #Kullback-Leibler Divergence
        #kldiv2[mychi] = sum(pj[pi > SMALL] * log (pj[pi > 0 + SMALL]/pi[pi > 0 + SMALL]), axis=-1)
        #Jensen-Shannon Divergence
        #jsdiv2[mychi] = 0.5 * sum(pj * log (pj/(0.5*pi+0.5*pj+SMALL)), axis=-1) + 0.5 * \
        #    sum(pi * log (pi/(0.5*pi + 0.5*pj+SMALL)), axis=-1)
        
    
    #values to compare: 
    #kldiv chi1: 0.602409  chi2: 0.109821
    #jsdiv chi1: 0.188742  chi2: 0.030788
    kldiv_targ = [0.602408, 0.109821]
    jsdiv_targ = [0.188742, 0.030788]
    print "kldiv     : "+str(kldiv2)+"\n"
    print "kldiv targ: "+str(kldiv_targ)+"\n"
    print "jsdiv     : "+str(jsdiv2)+"\n"
    print "jsdiv targ: "+str(jsdiv_targ)+"\n"


    #assert(kldiv2[0] > 0.602408 and kldiv2[0] < 0.602410)
    #assert(kldiv2[1] > 0.109820 and kldiv2[1] < 0.109822)
    #assert(jsdiv2[0] > 0.188741 and jsdiv2[0] < 0.188743)
    #assert(jsdiv2[1] > 0.030788 and jsdiv2[1] < 0.030788)


#####################################################

#should I create a special.psi lookup table ?
#myps

##########################################################################################################    
#===================================================
#READ INPUT ARGUMENTS
#===================================================
##########################################################################################################
def main():
    try:
       import run_profile
       run_profile.fix_args()
    except: pass
    global OFF_DIAG
    usage="%prog [-t traj1:traj2] [-x xvg_basedir] resfile [simulation numbers to use]  # where resfile is in the format <1-based-index> <aa type> <res num>"
    parser=OptionParser(usage)
    parser.add_option("-t", "--traj_fns", default=None, type="string", help="filenames to load PDB trajectories from; colon separated (e.g. fn1:fn2)")
    parser.add_option("-x", "--xvg_basedir", default=None, type="string", help="basedir to look for xvg files")
    parser.add_option("-s", "--sigalpha", default=0.01, type="float", help="p-value threshold for statistical filtering, lower is stricter")
    parser.add_option("-w", "--binwidth", default=15.0, type="float", help="width of the bins in degrees")
    parser.add_option("-n", "--num_sims", default=None, type="int", help="number of simulations")
    parser.add_option("-p", "--permutations", default=0, type="int", help="number of permutations for independent mutual information, for subtraction from total Mutual Information")
    parser.add_option("-d", "--xvg_chidir", default = "/dihedrals/g_chi/", type ="string", help="subdirectory under xvg_basedir/run# where chi angles are stored")
    parser.add_option("-a", "--adaptive", default = "yes", type ="string", help="adaptive partitioning (yes|no)")
    parser.add_option("-b", "--backbone", default = "phipsichi", type = "string", help="chi: just sc  phipsi: just bb  phipsichi: bb + sc")
    parser.add_option("-o", "--bootstrap_set_size", default = None, type = "int", help="perform bootstrapping within this script; value is the size of the subsets to use")
    parser.add_option("-i", "--skip", default = 1, type = "int", help="interval between snapshots to consider, in whatever units of time snapshots were output in") 
    parser.add_option("-c", "--correct_formutinf_between_sims", default = "no", type="string", help="correct for excess mutual information between sims")
    parser.add_option("--load_matrices_numstructs", default = 0, type = "int", help="if you want to load bootstrap matrices from a previous run, give # of structs per sim (yes|no)")
    parser.add_option("-l", "--last_step", default=None, type= "int", help="last step to read from input files, useful for convergence analysis")
    parser.add_option("--plot_2d_histograms", default = False, action = "store_true", help="makes 2d histograms for all pairs of dihedrals in the first bootstrap")
    parser.add_option("-z", "--zoom_to_step", default = 0, type = "int", help="skips the first n snapshots in xvg files")
    parser.add_option("-M","--markov_samples", default = 0, type = "int", help="markov state model samples to use for independent distribution")
    parser.add_option("-N","--max_num_lagtimes", default = 5000, type =	"int", help="maximum number of lagtimes for markov model")
    parser.add_option("-m","--max_num_chis", default = 99, type = "int", help="max number of sidechain chi angles per residue or ligand")
    parser.add_option("-f","--pdbfile", default = None, type = "string", help="pdb structure file for additional 3-coord cartesian per residue")
    parser.add_option("-q","--xtcfile", default = None, type = "string", help="gromacs xtc prefix in 'run' subdirectories for additional 3-coord cartesian per residue")
    #parser.add_option("-g","--gcc", default = 'intelem', type = "string", help="numpy distutils ccompiler to use. Recommended ones intelem or gcc")
    parser.add_option("-g","--gcc", default = 'gcc', type = "string", help="numpy distutils ccompiler to use. Recommended ones intelem or gcc")
    parser.add_option("-e","--output_timeseries", default = "no", type = "string", help="output corrected dihedral timeseries (requires more memory) yes|no ")
    parser.add_option("-y","--symmetry_number", default = 1, type = int, help="number of identical subunits in homo-oligomer for symmetrizing matrix")
    parser.add_option("-L","--lagtime_interval", default = None, type=int, help="base snapshot interval to use for lagtimes in Markov model of bin transitions")
    parser.add_option("-j","--offset", default = 0,type=int, help="offset for mutinf of (i,j) at (t, t - offset)")
    parser.add_option("--output_independent",default = 0, type=int, help="set equal to 1 to output independent mutinf values from markov model or multinomial distribution")
    parser.add_option("-C","--num_convergence_points", default = 0, type=int, help="for -n == -o , use this many subsets of the data to look at convergence statistics")   
    parser.add_option("-T","--triplet", default=None, type="string", help="wheter to perform triplet mutual information or not")
    parser.add_option("-P","--cyclic_permut",default=False, action = "store_true", help="for permutations, cyclicly permute blocks (run1 run2 run3 etc.) instead of scrambling or shuffling")
    parser.add_option("-D","--diag",default=False, action = "store_true", help="Examine only timescales and couplings within residues")
    (options,args)=parser.parse_args()
    mycompiler = options.gcc
    if len(filter(lambda x: x==None, (options.traj_fns, options.xvg_basedir))) != 1:
        parser.error("ERROR exactly one of --traj_fns or --xvg_basedir must be specified")

    print "COMMANDS: ", " ".join(sys.argv)
    
    
    # Initialize
    offset = options.offset
    resfile_fn = args[0]
    adaptive_partitioning = (options.adaptive == "yes")
    phipsi = 0
    backbone_only = 0
    if options.backbone == "calpha" or options.backbone == "calphas":
           if options.traj_fns != None:
                  phipsi = -3
                  backbone_only = 1
           else:
                  options.backbone = "phipsi"
    
    if options.backbone == "phipsichi":
        phipsi = 2;
        backbone_only =0
    if options.backbone == "phipsi":
        phipsi = 2;
        backbone_only = 1
    if options.backbone == "stress":
        phipsi = -4;
        NumChis["GLY"] = 1
        NumChis["ALA"] = 1
        backbone_only = 1
        print "performing stress analysis"
    if options.backbone == "coarse_phipsi":
        phipsi = -1
        backbone_only = 1
        print "overriding binwidth, using four bins for coarse discretized backbone"
        options.binwidth = 90.0
    if options.backbone == "split_main_side":
        print "treating backbone and sidechain separately according to residue list"
        phipsi = -2
        backbone_only = 0

    if options.diag == False: #default is to do all, not just diag
           OFF_DIAG = 1
    else:
           OFF_DIAG = 0

    if options.xvg_basedir.endswith("/"):
           pass
    else:
           options.xvg_basedir += "/" 

    if options.xvg_chidir.endswith("/"):
           pass
    else:
           options.xvg_chidir += "/" 


    #phipsi = options.backbone.find("phipsi")!=-1
    #backbone_only = options.backbone.find("chi")==-1
    bins=arange(-180,180,options.binwidth) #Compute bin edges
    nbins = len(bins)

    if options.traj_fns != None:
       xvgorpdb = "pdb"
       traj_fns = options.traj_fns.split(":")
       num_sims = len(traj_fns)
    else:
        if(options.xtcfile != None):
            xvgorpdb = "xtc"
            num_sims = options.num_sims
            traj_fns = None
        else:
            xvgorpdb = "xvg"
            traj_fns = None
            num_sims = options.num_sims
 
    print "num_sims:"+str(num_sims)+"\n"

    #if(len(args) > 1):
    #    which_runs = map(int, args[1:])
    #    num_sims = len(which_runs)
    #else:
    assert(num_sims != None)
    #which_runs = range(num_sims)
    #which_runs = None

    if options.bootstrap_set_size is None:
        options.bootstrap_set_size = num_sims

    which_runs = []
    pair_runs_list = []
    for myruns in xuniqueCombinations(range(num_sims), options.bootstrap_set_size):
        which_runs.append(myruns)
    print which_runs

    if (options.cyclic_permut == True):
           print "using cyclic permutations instead of scrambling as per Fenley, Muddana, Gilson et al PNAS 2012 "
    if options.num_convergence_points > 1: #create bootstrap samples for number of convergence points, this will also set variables like bootstrap_sets  
           print "looking at mutual information convergence using "+str(options.num_convergence_points)+" convergence points"
           #options.bootstrap_set_size = 1  #override options
           for convergence_point in range(options.num_convergence_points - 1):
                  which_runs.append(which_runs[0])


    NUM_LAGTIMES=options.max_num_lagtimes #overwrite global var
    OUTPUT_INDEPENDENT_MUTINF_VALUES = options.output_independent

    bootstrap_pair_runs_list = []
    for bootstrap in range((array(which_runs)).shape[0]):
        pair_runs_list = []
        for myruns2 in xcombinations(which_runs[bootstrap], 2):
            pair_runs_list.append(myruns2)
        bootstrap_pair_runs_list.append(pair_runs_list)

    pair_runs_array = array(bootstrap_pair_runs_list,int16)
    print pair_runs_array 


    

    #set num_structs = options.load_matrices in case we don't actually load any data, just want to get the filenames right for the bootstrap matrices

    if (options.load_matrices_numstructs > 0): num_structs = options.load_matrices_numstructs
    else: num_structs = None

    run_params = RunParameters(resfile_fn=resfile_fn, adaptive_partitioning=adaptive_partitioning, phipsi=phipsi, backbone_only=backbone_only, nbins = nbins,
      bootstrap_set_size=options.bootstrap_set_size, sigalpha=options.sigalpha, permutations=options.permutations, num_sims=num_sims, num_structs=num_structs,
      binwidth=options.binwidth, bins=bins, which_runs=which_runs, xvgorpdb=xvgorpdb, traj_fns=traj_fns, xvg_basedir=options.xvg_basedir, calc_variance=False, xvg_chidir=options.xvg_chidir,bootstrap_choose=options.bootstrap_set_size,pair_runs=pair_runs_array,skip=options.skip,skip_over_steps=options.zoom_to_step,last_step=options.last_step,calc_mutinf_between_sims=options.correct_formutinf_between_sims,load_matrices_numstructs=options.load_matrices_numstructs,plot_2d_histograms=options.plot_2d_histograms,max_num_chis=options.max_num_chis,pdbfile=options.pdbfile, xtcfile=options.xtcfile, output_timeseries=options.output_timeseries,lagtime_interval=options.lagtime_interval, markov_samples=options.markov_samples, num_convergence_points=options.num_convergence_points, cyclic_permut=options.cyclic_permut)


    print run_params


    #====================================================
    #DO ANALYSIS
    #===================================================

    print "Calculating Entropy and Mutual Information"

    
    independent_mutinf_thisdof = zeros((run_params.permutations,1),float32)
    timer = utils.Timer()


    ### load angle data, calculate entropies and mutual informations between residues (and error-propagated variances)
    if run_params.bootstrap_set_size is None:
        if run_params.load_matrices_numstructs == 0: 
               reslist = load_data(run_params)
               print "TIME to load trajectories & calculate intra-residue entropies: ", timer
               timer=utils.Timer()

               prefix = run_params.get_logfile_prefix()
               ##========================================================
               ## Output Timeseries Data in a big matrix
               ##========================================================
               name_num_list = make_name_num_list(reslist)
               if(xvgorpdb == "xtc"):
                      print "calculating distance matrices and variance:"
                      output_distance_matrix_variances(len(run_params.which_runs),run_params.bootstrap_set_size,run_params.which_runs, reslist[0].numangles, reslist[0].numangles_bootstrap, name_num_list) #uses global xtc_coords for data
               if run_params.output_timeseries == "yes":
                      timeseries_chis_matrix = output_timeseries_chis(prefix+"_timeseries",reslist,name_num_list,run_params.num_sims)
                      print "TIME to output timeseries data: ", timer
                      timer=utils.Timer()
               
               timescales_chis = output_timescales_chis(prefix+"_chis_implied_timescales_bootstrap_avg",reslist,name_num_list)
               timescales_chis_boots = output_timescales_chis(prefix+"_chis_implied_timescales_bootstrap_avg",reslist,name_num_list)
               timescales_chis_boots = output_timescales_chis_boots(prefix+"_chis_implied_timescales_bootstrap",reslist,len(run_params.which_runs),name_num_list)
               conv_bb_sc = output_conv_bb_sc_boots(prefix+"_chis_markov_conv_bb_sc_bootstrap",reslist,len(run_params.which_runs),name_num_list)
               lagtimes_chis = output_lagtimes_chis(prefix+"_chis_lagtimes_bootstrap_avg",reslist,name_num_list)
               lagtimes_chis = output_lagtimes_chis_last(prefix+"_chis_lagtimes_bootstrap_last",reslist,name_num_list)
               timescales_chis = output_timescales_chis_avg(prefix+"_avg_over_chis_implied_timescales_bootstrap_avg",reslist,name_num_list)
               timescales_chis = output_timescales_chis_max(prefix+"_avg_over_chis_implied_timescales_bootstrap_max",reslist,name_num_list)
               timescales_chis = output_timescales_chis_last(prefix+"_avg_over_chis_implied_timescales_bootstrap_last",reslist,name_num_list)
               
               #timescales_mutinf_autocorr = output_timescales_mutinf_autocorr_chis_max(prefix+"_chis_mutinf_autocorr_time_bootstrap_max",reslist,name_num_list)
               timescales_angles_autocorr = output_timescales_angles_autocorr_chis(prefix+"_chis_angles_autocorr_time_bootstrap_max",reslist,name_num_list)
               #this next one is useful for a convergence series
               timescales_angles_autocorr_boots = output_timescales_angles_autocorr_chis_boots(prefix+"_chis_angles_autocorr_time_bootstrap",reslist,len(run_params.which_runs),name_num_list) #bootstrap_sets are len(run_params.which_runs)

               
               mut_info_res_matrix, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, mut_info_uncert_matrix, mut_info_res_matrix_different_sims, dKLtot_dresi_dresj_matrix, twoD_hist_boot_avg, twoD_hist_boots, twoD_hist_ind_boots, twoD_hist_ind_boot_avg, mut_info_norm_res_matrix = calc_pair_stats(reslist, run_params)
               print "TIME to calculate pair stats: ", timer
               timer=utils.Timer()

        prefix = run_params.get_logfile_prefix() + "_sims" + ",".join(map(str, sorted(which_runs)))
    else:
        runs_superset, set_size = run_params.which_runs, run_params.bootstrap_set_size
        if set_size > len(runs_superset[0]) or len(runs_superset) < 1:
               print "FATAL ERROR: invalid values for bootstrap set size '%d' from runs '%s'" % (set_size, runs_superset)
               sys.exit(1)

        run_params.calc_variance = False
        print "\n----- STARTING BOOTSTRAP RUNS: %s -----" % run_params
        if run_params.load_matrices_numstructs == 0:
            reslist = load_data(run_params)
        print "TIME to load trajectories & calculate intra-residue entropies: ", timer
        timer=utils.Timer()

        

        ##========================================================
        ## Output Timeseries Data in a big matrix
        ##========================================================
        prefix = run_params.get_logfile_prefix()

        if run_params.load_matrices_numstructs == 0: 
           
           name_num_list = make_name_num_list(reslist)

           if(xvgorpdb == "xtc" and run_params.load_matrices_numstructs == 0 ):
                         name_num_list = make_name_num_list(reslist)
                         print "calculating distance matrices and variance:"
                         output_distance_matrix_variances(len(run_params.which_runs),run_params.bootstrap_set_size,run_params.which_runs, reslist[0].numangles, reslist[0].numangles_bootstrap, name_num_list) #uses global xtc_coords for data, len(which_runs) gives number of bootstrap_sets
           if run_params.output_timeseries == "yes":
                  timeseries_chis_matrix = output_timeseries_chis(prefix+"_timeseries",reslist,name_num_list,run_params.num_sims)
                  print "TIME to output timeseries data: ", timer
                  timer=utils.Timer()

           lagtimes_chis = output_lagtimes_chis(prefix+"_chis_lagtimes_bootstrap_avg",reslist,name_num_list)
           timescales_chis_boots = output_timescales_chis(prefix+"_chis_implied_timescales_bootstrap_avg",reslist,name_num_list)
           timescales_chis_boots = output_timescales_chis_boots(prefix+"_chis_implied_timescales_bootstrap",reslist,len(run_params.which_runs),name_num_list)
           conv_bb_sc = output_conv_bb_sc_boots(prefix+"_chis_markov_conv_bb_sc_bootstrap",reslist,len(run_params.which_runs),name_num_list)
           lagtimes_chis = output_lagtimes_chis(prefix+"_chis_lagtimes_bootstrap_avg",reslist,name_num_list)
           lagtimes_chis = output_lagtimes_chis_last(prefix+"_chis_lagtimes_bootstrap_last",reslist,name_num_list)
           timescales_chis = output_timescales_chis_avg(prefix+"_avg_over_chis_implied_timescales_bootstrap_avg",reslist,name_num_list)
           timescales_chis = output_timescales_chis_max(prefix+"_avg_over_chis_implied_timescales_bootstrap_max",reslist,name_num_list)
           timescales_chis = output_timescales_chis_last(prefix+"_avg_over_chis_implied_timescales_bootstrap_last",reslist,name_num_list)
           #timescales_mutinf_autocorr = output_timescales_mutinf_autocorr_chis_max(prefix+"_chis_mutinf_autocorr_time_bootstrap_max",reslist,name_num_list)
           timescales_angles_autocorr = output_timescales_angles_autocorr_chis(prefix+"_chis_angles_autocorr_time_bootstrap_max",reslist,name_num_list)
           #this next one is useful for a convergence series
           timescales_angles_autocorr_boots = output_timescales_angles_autocorr_chis_boots(prefix+"_chis_angles_autocorr_time_bootstrap",reslist,len(run_params.which_runs),name_num_list) #bootstrap_sets are len(run_params.which_runs)
           if run_params.load_matrices_numstructs == 0:
               mut_info_res_matrix, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, mut_info_uncert_matrix, mut_info_res_matrix_different_sims, dKLtot_dresi_dresj_matrix, twoD_hist_boot_avg, twoD_hist_boots, twoD_hist_ind_boots, twoD_hist_ind_boot_avg, mut_info_norm_res_matrix = calc_pair_stats(reslist, run_params, options.plot_2d_histograms)
               print mut_info_res_matrix
           print "TIME to calculate pair stats: ", timer
           timer=utils.Timer()



           # create a master matrix
           #matrix_list += [calc_pair_stats(reslist, run_params)[0]] # add the entropy/mut_inf matrix to the list of matrices
           #bootstraps_mut_inf_res_matrix = zeros(list(matrix_list[0].shape) + [len(matrix_list)], float32)
           #for i in range(len(matrix_list)): bootstraps_mut_inf_res_matrix[:,:,i] = matrix_list[i]

           #mut_info_res_matrix, mut_info_uncert_matrix = bootstraps_mut_inf_res_matrix.mean(axis=2), bootstraps_mut_inf_res_matrix.std(axis=2)
           #prefix = run_params.get_logfile_prefix() + "_sims%s_choose%d" % (",".join(map(str, sorted(runs_superset))), set_size)
        
    ### output results to disk
    
    ##==============================================================
    # setup output or read in previously-calculated mutual information matrices, if given
    ##==============================================================

    name_num_list=[]
    if run_params.load_matrices_numstructs == 0:
        name_num_list = make_name_num_list(reslist)
        #for res in reslist: name_num_list.append(res.name + str(res.num))
    else:
        rownames = []
        colnames = []
        (test_matrix, rownames, colnames) = read_matrix_chis(prefix+"_bootstrap_0_mutinf.txt")
        rownames = colnames
        #print test_matrix
        mut_info_res_matrix = zeros(((len(run_params.which_runs)),test_matrix.shape[0],test_matrix.shape[1],test_matrix.shape[2],test_matrix.shape[3]),float32)
        mut_info_res_matrix_different_sims = zeros(((len(run_params.which_runs)),test_matrix.shape[0],test_matrix.shape[1],test_matrix.shape[2],test_matrix.shape[3]),float32)
        mut_info_uncert_matrix = zeros(((len(run_params.which_runs)),test_matrix.shape[0],test_matrix.shape[1],test_matrix.shape[2],test_matrix.shape[3]),float32)
        mut_info_norm_res_matrix = zeros(((len(run_params.which_runs)),test_matrix.shape[0],test_matrix.shape[1],test_matrix.shape[2],test_matrix.shape[3]),float32)
        for bootstrap in range(len(run_params.which_runs)):
            (mut_info_res_matrix[bootstrap,:,:,:,:], rownames, colnames) = read_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf.txt")
            (mut_info_res_matrix_different_sims[bootstrap,:,:,:,:], rownames, colnames) = read_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_different_sims.txt")
            (mut_info_norm_res_matrix[bootstrap,:,:,:,:], rownames, colnames) = read_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_norm.txt")
        name_num_list = rownames
    
    
    
    ##########################################################################################################   
    ### FINAL STATISTICAL FILTERING USING WILCOXON TEST (NEW) AND OUTPUT MATRICES
    ##########################################################################################################
    
    
    #for bootstrap in range(len(run_params.which_runs)):
    if EACH_BOOTSTRAP_MATRIX == 1 and OFF_DIAG == 1 and run_params.load_matrices_numstructs == 0:
        for bootstrap in range(len(run_params.which_runs)):
            output_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf.txt",mut_info_res_matrix[bootstrap],name_num_list,name_num_list)
            output_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_0diag.txt",mut_info_res_matrix[bootstrap],name_num_list,name_num_list, zero_diag=True)
            output_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_different_sims.txt",mut_info_res_matrix_different_sims[bootstrap],name_num_list,name_num_list)
            
           #output_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_uncert.txt",mut_info_uncert_matrix[bootstrap],name_num_list,name_num_list)
            if(OUTPUT_DIAG == 1):
                   output_diag(prefix+"_bootstrap_"+str(bootstrap)+"_entropy_res.txt",mut_info_res_matrix[bootstrap],name_num_list)
                   output_diag(prefix+"_bootstrap_"+str(bootstrap)+"_entropy_res_uncert.txt",mut_info_uncert_matrix[bootstrap],name_num_list)
    
    tot_ent = zeros(mut_info_res_matrix.shape[0],float32)
    tot_ent_sig01 = zeros(mut_info_res_matrix.shape[0],float32)
    tot_ent_sig05 = zeros(mut_info_res_matrix.shape[0],float32)
    tot_ent_diag = zeros(mut_info_res_matrix.shape[0],float32)
    mut_info_res_matrix_different_sims_avg = zeros(mut_info_res_matrix.shape[1:],float32)
    mut_info_res_matrix_avg = zeros(mut_info_res_matrix.shape[1:],float32)
    mut_info_res_matrix_sig_01 = zeros(mut_info_res_matrix.shape,float32)
    mut_info_res_matrix_sig_05 = zeros(mut_info_res_matrix.shape,float32)
    mut_info_norm_res_matrix_avg = zeros(mut_info_norm_res_matrix.shape[1:],float32)
    mut_info_norm_res_matrix_sig_01 = zeros(mut_info_norm_res_matrix.shape,float32)
    mut_info_norm_res_matrix_sig_05 = zeros(mut_info_norm_res_matrix.shape,float32)
    mut_info_res_sumoverchis_matrix_sig_01 = zeros(mut_info_res_matrix.shape[0:3],float32)
    mut_info_res_sumoverchis_matrix_sig_05 = zeros(mut_info_res_matrix.shape[0:3],float32)
    mut_info_uncert_matrix_avg = zeros(mut_info_uncert_matrix.shape[1:],float32)
    mut_info_norm_res_sumoverchis_matrix_sig_01 = zeros(mut_info_res_matrix.shape[0:3],float32)
    mut_info_norm_res_sumoverchis_matrix_sig_05 = zeros(mut_info_res_matrix.shape[0:3],float32)
    mut_info_norm_uncert_matrix_avg = zeros(mut_info_uncert_matrix.shape[1:],float32)
    mut_info_pval_matrix = zeros(mut_info_uncert_matrix.shape[1:],float32)
    mut_info_res_sumoverchis_matrix_sig_avg_01 = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_sumoverchis_matrix_sig_avg_05 = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_sumoverchis_matrix_sig_avg_01_Snorm = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_sumoverchis_matrix_sig_avg_05_Snorm = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_sumoverchis_matrix_avg = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_norm_res_sumoverchis_matrix_avg = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_norm_res_sumoverchis_matrix_sig_avg_01 = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_norm_res_sumoverchis_matrix_sig_avg_05 = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_uncert_sumoverchis_matrix_avg = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_sumoverchis_matrix_different_sims_avg = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_maxoverchis_matrix_avg = zeros(mut_info_res_matrix.shape[1:3],float32)
    Kullback_Leibler_local_covar_boots = zeros(mut_info_res_matrix.shape[0:3], float32)
    Kullback_Leibler_local_covar_avg = zeros(mut_info_res_matrix.shape[1:3], float32)
    mut_info_res_sumoverchis_matrix_avg_symmetrized = zeros((int(mut_info_res_matrix.shape[1]/options.symmetry_number),int(mut_info_res_matrix.shape[2]/options.symmetry_number)),float32)

    #for bootstrap in range(len(run_params.which_runs)):
        ### invert dKLtot_dresi_dresj_matrix for each bootstrap then average over bootstraps
        ### Don't do inverse here in case matrix is singular -- do it post-processing
        #Kullback_Leibler_local_covar_boots[bootstrap,:,:] = dKLtot_dresi_dresj_matrix[bootstrap,:,:]
    #Kullback_Leibler_local_covar_avg = average(Kullback_Leibler_local_covar_boots,axis=0)
    
    #print mut_info_res_sumoverchis_matrix_avg.shape
    #print mut_info_res_matrix[:,0,0,0,1]
    testarray = zeros(mut_info_res_matrix.shape[0],float32)
    #print "Applying Wilcoxon test to filter matrix:"
    mutinf_vals = [] # store mutinf values for printing
    for i in range(mut_info_res_matrix.shape[1]):
        for j in range(i, mut_info_res_matrix.shape[2]):
            #mutinf_norm_these_res = zeros((mut_info_res_matrix.shape[0]), int16)
            #mutinf_norm_these_res = zeros((mut_info_res_matrix.shape[0]), int16)
            mutinf_norm_these_res =  0 #zeros((mut_info_res_matrix.shape[0]), int16)
            for k in range(mut_info_res_matrix.shape[3]):
                for m in range(mut_info_res_matrix.shape[4]):
                    if(k < reslist[i].nchi and m < reslist[j].nchi):   
                     if(any(mut_info_res_matrix[:,i,j,k,m]) > 0):
                        mutinf_boots = mut_info_res_matrix[:,i,j,k,m].copy()
                        mutinf_norm_boots = mut_info_norm_res_matrix[:,i,j,k,m].copy()
                        #print "mutinf boots avg: "+str(i)+" "+str(j)+" "+str(k)+" "+str(m)+":   "+str(mutinf_boots)
                        #print "mutinf boots avg: "+str(i)+" "+str(j)+" "+str(k)+" "+str(m)
                        #print average(mutinf_boots[mutinf_boots > 0], axis=0)
                        mutinf_boots[mutinf_boots < 0] = 0 #use negative and zero values for significance testing but not in the average
                        mutinf_norm_boots[mutinf_boots < 0] = 0 #use negative and zero values for significance testing but not in the average
                        #for myboot in range(mutinf_norm_boots.shape[0]):
                               #if(mutinf_norm_boots[myboot] > 0):
                        mutinf_norm_these_res += 1 
                                      
                        #zero values of mutinf_boots will include those zeroed out in the permutation test
                        if(options.num_convergence_points <= 1):
                               mutinf = mut_info_res_matrix_avg[i,j,k,m] = average(mutinf_boots[mutinf_boots > 0 ],axis=0) 
                               mutinf_norm = mut_info_norm_res_matrix_avg[i,j,k,m] = average(mutinf_norm_boots[mutinf_norm_boots > 0 ],axis=0) 
                        else:
                               mutinf = mut_info_res_matrix_avg[i,j,k,m] = mutinf_boots[-1] #take last bootstrap if doing convergence series
                               mutinf_norm = mut_info_norm_res_matrix_avg[i,j,k,m] = mutinf_norm_boots[-1] #take last bootstrap if doing convergence series
                        uncert = mut_info_uncert_matrix_avg[i,j,k,m] = sqrt(cov(mut_info_res_matrix[:,i,j,k,m]) / (run_params.num_sims))
                        uncert_norm = mut_info_uncert_matrix_avg[i,j,k,m] = sqrt(cov(mut_info_res_matrix[:,i,j,k,m]) / (run_params.num_sims))
                        #use vfast_cov_for1D_boot as a fast way to calculate a stdev instead of the function std()
                        uncert = pval = None
                        if(mut_info_res_matrix.shape[0] >= 10 and options.num_convergence_points == 1):
                            uncert = mut_info_uncert_matrix_avg[i,j,k,m] = sqrt((vfast_cov_for1D_boot(reshape(mut_info_res_matrix[:,i,j,k,m],(mut_info_res_matrix.shape[0],1)))[0,0]) / (run_params.num_sims))
                            pval = mut_info_pval_matrix[i,j,k,m] =(stats.wilcoxon(mut_info_res_matrix[:,i,j,k,m]+SMALL))[1] / 2.0 #offset by SMALL to ensure no values of exactly zero will be removed in the test
                            #for myboot in mutinf_norm_boots[:]:
                               #if(mutinf_norm_boots[myboot] > 0 and pval <=0.05):
                               #       mutinf_norm_these_res[myboot] += 1
                               #if(mutinf_norm_boots[myboot] > 0 and pval <=0.01):
                               #       mutinf_norm_these_res[myboot] += 1
                            if pval == 0.00 and (i != j): pval = 1.0
                            if pval <= 0.05:
                                mut_info_res_matrix_sig_05[:,i,j,k,m] = mutinf_boots.copy()
                                mut_info_norm_res_matrix_sig_05[:,i,j,k,m] = mutinf_norm_boots.copy()
                            if pval <= 0.01:   
                                mut_info_res_matrix_sig_01[:,i,j,k,m] = mutinf_boots.copy()
                                mut_info_norm_matrix_sig_01[:,i,j,k,m] = mutinf_norm_boots.copy()

                        else:
                            mut_info_res_matrix_sig_05[:,i,j,k,m] = mutinf_boots.copy()
                            mut_info_res_matrix_sig_01[:,i,j,k,m] = mutinf_boots.copy()
                            mut_info_norm_res_matrix_sig_05[:,i,j,k,m] = mutinf_norm_boots.copy()
                            mut_info_norm_res_matrix_sig_01[:,i,j,k,m] = mutinf_norm_boots.copy()
                            uncert = mut_info_uncert_matrix_avg[i,j,k,m] = sqrt((vfast_cov_for1D_boot(reshape(mut_info_res_matrix[:,i,j,k,m],(mut_info_res_matrix.shape[0],1)))[0,0]) / (run_params.num_sims))
                            pval = 0

                        if pval <= 0.05 and (i != j or k == m):
                            mut_info_res_sumoverchis_matrix_sig_avg_05[i,j] += mutinf
                            mut_info_res_sumoverchis_matrix_sig_05[:,i,j] += mutinf_boots.copy()
                            mut_info_norm_res_sumoverchis_matrix_sig_avg_05[i,j] += mutinf_norm
                            mut_info_norm_res_sumoverchis_matrix_sig_05[:,i,j] += mutinf_norm_boots.copy()

                            if(i == j):
                                tot_ent_sig05 += mutinf_boots.copy()
                            else:
                                tot_ent_sig05 -= mutinf_boots.copy()

                        if pval <= 0.05 and i == j and k != m:
                             mut_info_res_sumoverchis_matrix_sig_avg_05[i,j] += -0.5 * mutinf
                             mut_info_res_sumoverchis_matrix_sig_05[:,i,j] += -0.5 * mutinf_boots.copy()
                             mut_info_norm_res_sumoverchis_matrix_sig_avg_05[i,j] += -0.5 * mutinf_norm
                             mut_info_norm_res_sumoverchis_matrix_sig_05[:,i,j] += -0.5 * mutinf_norm_boots.copy()
                             tot_ent_sig05  -= 0.5 * mutinf_boots.copy() 


                        if pval <= 0.01 and (i != j or k == m):
                            mut_info_res_sumoverchis_matrix_sig_avg_01[i,j] += mutinf
                            mut_info_norm_res_sumoverchis_matrix_sig_avg_01[i,j] += mutinf_norm
                            #if(mutinf > 0):
                                #ent1 = 0
                                #ent2 = 0
                                #for count in range(6):
                                #    ent1 += mut_info_res_matrix_avg[i,i,count,count]
                                #    ent2 +=  mut_info_res_matrix_avg[j,j,count,count
                                #mut_info_res_sumoverchis_matrix_sig_avg_01_Snorm[i,j] +=  
                                #mutinf / min(mut_info_res_matrix_avg[i,i,k,k] + mut_info_res_matrix_avg[j,j,m,m])
                            mut_info_res_sumoverchis_matrix_sig_01[:,i,j] += mutinf_boots.copy()
                            mut_info_norm_res_sumoverchis_matrix_sig_01[:,i,j] += mutinf_norm_boots.copy()

                            if(i == j):
                                tot_ent_sig01 += mutinf_boots.copy()
                            else:
                                tot_ent_sig01 -= mutinf_boots.copy()

                        if pval <= 0.01 and i == j and k != m:
                             mut_info_res_sumoverchis_matrix_sig_avg_01[i,j] += -0.5 * mutinf
                             mut_info_res_sumoverchis_matrix_sig_01[:,i,j] += -0.5 * mutinf_boots.copy()
                             #mut_info_norm_res_sumoverchis_matrix_sig_avg_01[i,j] += -0.5 * mutinf_norm
                             #mut_info_norm_res_sumoverchis_matrix_sig_01[:,i,j] += -0.5 * mutinf_norm_boots.copy()
                             tot_ent_sig01  -= 0.5 * mutinf_boots.copy()

                        if mutinf > 0 and (i != j or k == m):
                            mut_info_res_sumoverchis_matrix_avg[i,j] += mutinf
                            mut_info_norm_res_sumoverchis_matrix_avg[i,j] += mutinf_norm
                            mut_info_res_sumoverchis_matrix_different_sims_avg[i,j] += average(mut_info_res_matrix_different_sims[:,i,j,k,m])
                            mut_info_res_uncert_sumoverchis_matrix_avg[i,j] += uncert ## error bars' sum over torsion pairs

                            if mut_info_res_maxoverchis_matrix_avg[i,j] < mutinf:  mut_info_res_maxoverchis_matrix_avg[i,j] = mutinf

                            if(i == j):
                                tot_ent += mutinf_boots.copy()
                                tot_ent_diag += mutinf_boots.copy()
                            else:
                                tot_ent -= mutinf_boots.copy()
                        if mutinf > 0 and i == j and k != m:
                             mut_info_res_sumoverchis_matrix_avg[i,j] += -0.5 * mutinf
                             #mut_info_norm_res_sumoverchis_matrix_avg[i,j] += -0.5 * mutinf
                        #     mut_info_res_sumoverchis_matrix_avg[i,j] += -0.5 * mutinf
                             mut_info_res_sumoverchis_matrix_different_sims_avg[i,j] += -0.5 * average(mut_info_res_matrix_different_sims[:,i,j,k,m])
                             mut_info_res_uncert_sumoverchis_matrix_avg[i,j] += 0.5 * uncert ## error bars' sum over torsion pairs, factor of 0.5 is because of double counting when looping over all pairs i,j in same residue
                             tot_ent -= 0.5 * mutinf_boots.copy()
                             tot_ent_diag -= 0.5 * mutinf_boots.copy()

                        for mat in (mut_info_pval_matrix, mut_info_res_matrix_avg, mut_info_uncert_matrix_avg):
                            if i != j: mat[j,i,m,k] = mat[i,j,k,m] #symmetric matrix
                        
                        if isnan(uncert):
                               uncert = 0.0
                        if (i==j and k==m): continue
                        elif (i == j and k < m): mutinf_vals.append([mutinf, uncert, pval, "%5s chi%d %5s chi%d (SAME RES)" % (str(name_num_list[i]), k+1, str(name_num_list[j]), m+1)])
                        elif (i != j): mutinf_vals.append([mutinf, uncert, pval, "%5s chi%d %5s chi%d (DIFF RES)" % (str(name_num_list[i]), k+1, str(name_num_list[j]), m+1)])
                    else:
                        mut_info_norm_res_matrix_avg[i,j,k,m] = 0.0   
                        mut_info_res_matrix_avg[i,j,k,m] = 0.0
                        mut_info_uncert_matrix_avg[i,j,k,m] = 0.0
                        mut_info_pval_matrix[i,j,k,m] = 0.0
                        #mutinf_vals.append([0, 0, 0, "%5s chi%d %5s chi%d --DEBUG--" % (str(name_num_list[i]), k+1, str(name_num_list[j]), m+1)]) # for debugging
            mut_info_res_sumoverchis_matrix_avg[j,i] = mut_info_res_sumoverchis_matrix_avg[i,j]
            mut_info_res_maxoverchis_matrix_avg[j,i] = mut_info_res_maxoverchis_matrix_avg[i,j]
            mut_info_res_sumoverchis_matrix_different_sims_avg[j,i] = mut_info_res_sumoverchis_matrix_different_sims_avg[i,j]
            mut_info_res_sumoverchis_matrix_sig_avg_01[j,i] = mut_info_res_sumoverchis_matrix_sig_avg_01[i,j]
            mut_info_res_sumoverchis_matrix_sig_avg_05[j,i] = mut_info_res_sumoverchis_matrix_sig_avg_05[i,j]
            mut_info_res_uncert_sumoverchis_matrix_avg[j,i] = mut_info_res_uncert_sumoverchis_matrix_avg[i,j]
            mut_info_res_sumoverchis_matrix_sig_01[:,j,i] = mut_info_res_sumoverchis_matrix_sig_01[:,i,j] 
            mut_info_res_sumoverchis_matrix_sig_05[:,j,i] = mut_info_res_sumoverchis_matrix_sig_05[:,i,j]

            

            mut_info_norm_res_sumoverchis_matrix_sig_01[:,i,j] /=  (mutinf_norm_these_res + SMALL)
            mut_info_norm_res_sumoverchis_matrix_sig_05[:,i,j] /=  (mutinf_norm_these_res + SMALL)
            mut_info_norm_res_sumoverchis_matrix_sig_01[:,j,i] = mut_info_norm_res_sumoverchis_matrix_sig_01[:,i,j] 
            mut_info_norm_res_sumoverchis_matrix_sig_05[:,j,i] = mut_info_norm_res_sumoverchis_matrix_sig_05[:,i,j] 
            
            mut_info_norm_res_sumoverchis_matrix_sig_avg_01[i,j] = average(mut_info_norm_res_sumoverchis_matrix_sig_01[:,i,j])
            mut_info_norm_res_sumoverchis_matrix_sig_avg_05[i,j] = average(mut_info_norm_res_sumoverchis_matrix_sig_05[:,i,j])
            mut_info_norm_res_sumoverchis_matrix_sig_avg_01[j,i] = mut_info_norm_res_sumoverchis_matrix_sig_avg_01[i,j]
            mut_info_norm_res_sumoverchis_matrix_sig_avg_05[j,i] = mut_info_norm_res_sumoverchis_matrix_sig_avg_05[i,j]
            mut_info_norm_res_sumoverchis_matrix_avg[i,j] /= (mutinf_norm_these_res + SMALL)
            mut_info_norm_res_sumoverchis_matrix_avg[j,i] = mut_info_norm_res_sumoverchis_matrix_avg[i,j] 
            #mut_info_res_sumoverchis_matrix_avg_symmetrized = average(average(mut_info_res_sumoverchis_matrix_chains, axis=0), axis=1) # average over symmetry-related molecules

    mut_info_res_sumoverchis_matrix_chains=reshape(mut_info_res_sumoverchis_matrix_avg,(options.symmetry_number,options.symmetry_number,int(mut_info_res_sumoverchis_matrix_avg.shape[0]/options.symmetry_number),int(mut_info_res_sumoverchis_matrix_avg.shape[1]/options.symmetry_number)))
    



    for i in range(options.symmetry_number):
           mut_info_res_sumoverchis_matrix_avg_symmetrized[:,:] += mut_info_res_sumoverchis_matrix_chains[i,i,:,:] # average over intra-chain couplings in symmetry-related molecules
           mut_info_res_sumoverchis_matrix_avg_symmetrized /= (1.0 * options.symmetry_number)
    
    short_name_num_list = name_num_list[0:int(len(name_num_list)/options.symmetry_number)]
    mut_info_res_sumoverchis_matrix_max_sig01 = zeros(mut_info_res_matrix.shape[1:3],float32)
    mut_info_res_matrix_avg_sig01 = zeros(mut_info_res_matrix_avg.shape,float64)
    for i in range(mut_info_res_matrix.shape[1]):
        for j in range(i, mut_info_res_matrix.shape[2]):
            for chi1 in range(mut_info_res_matrix.shape[3]):
                chis_mi = mut_info_res_matrix_avg[i,j,chi1,:]
                chis_p = mut_info_pval_matrix[i,j,chi1,:]
                chis_mi[chis_p >= 0.01] = 0
                mut_info_res_matrix_avg_sig01[i,j,chi1,:] = chis_mi[:] 
            mut_info_res_sumoverchis_matrix_max_sig01[i,j] = max( list(mut_info_res_matrix_avg_sig01[i,j,:,:].flatten()))

    #for i in range(mut_info_res_sumoverchis_matrix_sig_01.shape[0]): #normalizing mutinf btw res by min. ent of the 2 res.
    #    for j in range(mut_info_res_sumoverchis_matrix_sig_01.shape[0]):
    #        mut_info_res_sumoverchis_matrix_sig_avg_01_Snorm[i,j] = mut_info_res_sumoverchis_matrix_sig_avg_01[i,j] / min(mut_info_res_sumoverchis_matrix_sig_avg_01[i,i], mut_info_res_sumoverchis_matrix_sig_avg_01[j,j])
    #print out total entropies

    output_entropy(prefix+"_entropies_from_bootstraps.txt",tot_ent)
    output_entropy(prefix+"_entropies_from_bootstraps_sig01.txt",tot_ent_sig01)
    output_entropy(prefix+"_entropies_from_bootstraps_sig05.txt",tot_ent_sig05)
    output_entropy(prefix+"_entropies_from_bootstraps_norescorr.txt",tot_ent_diag)

    

    output_value(prefix+"_entropy.txt",average(tot_ent))
    output_value(prefix+"_entropy_sig01.txt",average(tot_ent_sig01))
    output_value(prefix+"_entropy_sig05.txt",average(tot_ent_sig05))
    output_value(prefix+"_entropy_norescorr.txt",average(tot_ent_diag))
    output_value(prefix+"_entropy_err.txt",sqrt((vfast_cov_for1D_boot(reshape(tot_ent,(tot_ent.shape[0],1)))[0,0]) / (run_params.num_sims)))
    output_value(prefix+"_entropy_sig01_err.txt",sqrt((vfast_cov_for1D_boot(reshape(tot_ent_sig01,(tot_ent_sig01.shape[0],1)))[0,0]) / (run_params.num_sims)))
    output_value(prefix+"_entropy_sig05_err.txt",sqrt((vfast_cov_for1D_boot(reshape(tot_ent_sig05,(tot_ent_sig05.shape[0],1)))[0,0]) / (run_params.num_sims)))
    output_value(prefix+"_entropy_norescorr_err.txt",sqrt((vfast_cov_for1D_boot(reshape(tot_ent_diag,(tot_ent_diag.shape[0],1)))[0,0]) / (run_params.num_sims)))




    # print out mutinf values

    if EACH_BOOTSTRAP_MATRIX == 1 and OFF_DIAG == 1:
        for bootstrap in range(len(run_params.which_runs)):
                    output_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_sig01.txt",mut_info_res_matrix_sig_01[bootstrap],name_num_list,name_num_list)
                    output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_sig05.txt",mut_info_res_matrix_sig_05[bootstrap],name_num_list,name_num_list)
                    output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_res_sig01.txt",mut_info_res_sumoverchis_matrix_sig_01[bootstrap],name_num_list,name_num_list)
                    output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_res_sig05.txt",mut_info_res_sumoverchis_matrix_sig_05[bootstrap],name_num_list,name_num_list)
                    output_matrix_chis(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_norm_sig01.txt",mut_info_norm_res_matrix_sig_01[bootstrap],name_num_list,name_num_list)
                    output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_norm_sig05.txt",mut_info_norm_res_matrix_sig_05[bootstrap],name_num_list,name_num_list)
                    output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_norm_res_sig01.txt",mut_info_norm_res_sumoverchis_matrix_sig_01[bootstrap],name_num_list,name_num_list)
                    output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_norm_res_sig05.txt",mut_info_norm_res_sumoverchis_matrix_sig_05[bootstrap],name_num_list,name_num_list)

                    





    for mutinf, uncert, pval, name in reversed(sorted(mutinf_vals)): print "BOOTSTRAP DIHEDRAL RESULTS: %s -> mi %.4f   sd/sqrt(n) %.1e   p %.1e" % (name, mutinf, uncert, pval)



    if(OFF_DIAG == 1):
         output_matrix(prefix+"_bootstrap_avg_mutinf_res_sum.txt",            mut_info_res_sumoverchis_matrix_avg ,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_avg_mutinf_res_sum_0diag.txt",      mut_info_res_sumoverchis_matrix_avg ,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_avg_mutinf_res_max_0diag.txt",      mut_info_res_maxoverchis_matrix_avg ,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_sigavg01_mutinf_res.txt",           mut_info_res_sumoverchis_matrix_sig_avg_01,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_sigavg01_mutinf_res_0diag.txt",     mut_info_res_sumoverchis_matrix_sig_avg_01,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_sigavg05_mutinf_res.txt",           mut_info_res_sumoverchis_matrix_sig_avg_05 ,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_sigavg05_mutinf_res_0diag.txt",     mut_info_res_sumoverchis_matrix_sig_avg_05 ,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_sigavg01_mutinf_norm_res.txt",           mut_info_norm_res_sumoverchis_matrix_sig_avg_01,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_sigavg01_mutinf_norm_res_0diag.txt",     mut_info_norm_res_sumoverchis_matrix_sig_avg_01,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_sigavg05_mutinf_norm_res.txt",           mut_info_norm_res_sumoverchis_matrix_sig_avg_05 ,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_sigavg05_mutinf_norm_res_0diag.txt",     mut_info_norm_res_sumoverchis_matrix_sig_avg_05 ,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_sigavg01_mutinf_norm_res_Snorm_0diag.txt", mut_info_res_sumoverchis_matrix_sig_avg_01_Snorm,name_num_list,name_num_list, zero_diag=True)
         output_matrix_chis(prefix+"_bootstrap_avg_mutinf.txt",                    mut_info_res_matrix_avg,name_num_list,name_num_list)
         output_matrix_chis(prefix+"_bootstrap_avg_mutinf_0diag.txt",              mut_info_res_matrix_avg,name_num_list,name_num_list, zero_diag=True)
         output_matrix_chis(prefix+"_bootstrap_avg_mutinf_norm.txt",                    mut_info_norm_res_matrix_avg,name_num_list,name_num_list)
         output_matrix_chis(prefix+"_bootstrap_avg_mutinf_norm_0diag.txt",              mut_info_norm_res_matrix_avg,name_num_list,name_num_list, zero_diag=True)
         output_matrix(prefix+"_bootstrap_sigavg_mutinf_res_uncert.txt",      mut_info_res_uncert_sumoverchis_matrix_avg,name_num_list,name_num_list)
         output_matrix_chis(prefix+"_bootstrap_sigavg_mutinf_pval.txt",       mut_info_pval_matrix,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_sigavg_mutinf_res_max_pval.txt",   amax(amax(mut_info_pval_matrix,axis=-1),axis=-1),name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_avg_mutinf_different_sims.txt",     mut_info_res_sumoverchis_matrix_different_sims_avg,name_num_list,name_num_list)
         output_matrix(prefix+"_bootstrap_sigmax01_mutinf_res_0diag.txt",     mut_info_res_sumoverchis_matrix_max_sig01,name_num_list,name_num_list, zero_diag=True)
         
         output_matrix(prefix+"_bootstrap_avg_mutinf_res_symmetrized_sum_0diag.txt",  mut_info_res_sumoverchis_matrix_avg_symmetrized ,short_name_num_list,short_name_num_list, zero_diag=True)
         
         #### Uncomment this for interactive visualization of 2D hists
         if(run_params.plot_2d_histograms == True):
                for row_num, row_name in zip(range(len(name_num_list)), name_num_list):
                       for col_num, col_name in zip(range(len(name_num_list)), name_num_list):
                              if col_num > row_num:
                                     for row_chi in range(6):
                                            for col_chi in range(6):
                                                   print "twoD hist boot avg shape: " + str(twoD_hist_boot_avg.shape )                                                             
                                                   #output_matrix_chis_2dhists(prefix+"_bootstrap_avg_2d_hists.txt",     twoD_hist_boot_avg, name_num_list, name_num_list, nchi=6, nbins = run_params.nbins, zero_diag=True)
                                                   output_2dhist(prefix+"_bootstrap_avg_"+str(row_name)+"_"+str(row_chi)+"_"+str(col_name)+"_"+str(col_chi)+"2d_hist.txt", twoD_hist_boot_avg[row_num,col_num,row_chi,col_chi], str(row_name)+"_"+str(row_chi), str(col_name)+"_"+str(col_chi))
                                                   for mybootstrap in range(len(run_params.which_runs)):
                                                   #output_matrix_chis_2dhists(prefix+"_bootstrap_"+str(mybootstrap)+"_2d_hists.txt",     twoD_hist_boots[mybootstrap], name_num_list, name_num_list, nchi=6, nbins = run_params.nbins, zero_diag=True)
                                                   #output_matrix_chis_2dhists(prefix+"_bootstrap_"+str(mybootstrap)+"_independent_markov_2d_hists.txt",     twoD_hist_ind_boots[mybootstrap], name_num_list, name_num_list, nchi=6, nbins = run_params.nbins, zero_diag=True)
                                                          output_2dhist(prefix+"_bootstrap_"+str(mybootstrap)+"_"+str(row_name)+"_"+str(row_chi)+"_"+str(col_name)+"_"+str(col_chi)+"_2d_hist.txt", twoD_hist_boots[mybootstrap,row_num,col_num,row_chi,col_chi], str(row_name)+"_"+str(row_chi), str(col_name)+"_"+str(col_chi))    
                                                          output_2dhist(prefix+"_bootstrap_"+str(mybootstrap)+"_"+str(row_name)+"_"+str(row_chi)+"_"+ str(col_name)+"_"+str(col_chi)+"_ind_2d_hist.txt", twoD_hist_ind_boots[mybootstrap,row_num,col_num,row_chi,col_chi], str(row_name)+"_"+str(row_chi), str(col_name)+"_"+str(col_chi))    
         ####
         #output_matrix(prefix+"_bootstrap_avg_KLdivpert_res_0diag.txt",     Kullback_Leibler_local_covar_avg,name_num_list,name_num_list, zero_diag=True)

        
    if(OUTPUT_DIAG == 1):
         output_diag(prefix+"_bootstrap_sigavg_entropy_res.txt",mut_info_res_matrix_avg,name_num_list)
         output_diag(prefix+"_bootstrap_sigavg_entropy_res_uncert.txt",mut_info_uncert_matrix_avg,name_num_list)


    #clean up stuff
    del twoD_hist_boot_avg
    del twoD_hist_boots
    del twoD_hist_ind_boot_avg
    del twoD_hist_ind_boots

    ########################################################
    #######  RUN TRIPLET STATS  ############################

    mutinf_triplet_vals = []
    if (options.triplet == "yes"):
       

       if run_params.load_matrices_numstructs == 0 and OFF_DIAG==1:
                  mut_info_triplet_res_matrix, mut_info_triplet_uncorrected_matrix, mut_info_triplet_corrections_matrix, mut_info_triplet_uncert_matrix, mut_info_triplet_res_matrix_different_sims, dKLtot_dresi_dresj_matrix, threeD_hist_boot_avg = calc_triplet_stats(reslist, run_params, mut_info_res_matrix)

       print "TIME to run triplet stats: ", timer
       timer=utils.Timer()
       mut_info_triplet_res_matrix_avg = zeros(mut_info_triplet_res_matrix.shape[1:7],float32)           
       mut_info_triplet_uncert_matrix_avg = zeros(mut_info_triplet_res_matrix.shape[1:7],float32)           
       mut_info_triplet_res_sumoverchis_matrix_sig_05 = zeros(mut_info_triplet_res_matrix.shape[0:4],float32)
       mut_info_triplet_res_sumoverchis_matrix_sig_01 = zeros(mut_info_triplet_res_matrix.shape[0:4],float32)
       mut_info_triplet_res_uncert_sumoverchis_matrix_avg = zeros(mut_info_triplet_res_matrix.shape[1:4],float32)
       mut_info_triplet_res_sumoverchis_matrix_sig_avg_05 = zeros(mut_info_triplet_res_matrix.shape[1:4],float32)
       mut_info_triplet_res_sumoverchis_matrix_sig_avg_01 = zeros(mut_info_triplet_res_matrix.shape[1:4],float32)
       mut_info_triplet_res_sumoverchis_matrix = zeros(mut_info_triplet_res_matrix.shape[0:4],float32)
       
       number_sig = 0
       for i in range(mut_info_triplet_res_matrix.shape[1]):
           for j in range(i, mut_info_triplet_res_matrix.shape[2]):
            for k in range(i, mut_info_triplet_res_matrix.shape[3]):
               number_sig = zeros(mut_info_triplet_res_matrix.shape[0],float64) #number of significant dihedral triplets for this combination of residues    
               for l in range(mut_info_triplet_res_matrix.shape[4]):
                   for m in range(mut_info_triplet_res_matrix.shape[5]):
                     for n in range(mut_info_triplet_res_matrix.shape[6]):
                        if(l <  reslist[i].nchi and m < reslist[j].nchi and n < reslist[k].nchi):
                           mutinf = 0 
                           
                           mutinf_boots = mut_info_triplet_res_matrix[:,i,j,k,l,m,n].copy()
                           #print "mutinf boots avg: "+str(average(mutinf_boots, axis=0))
                           #no separate treatment of positive and negative here, both may be important
                           #zero values of mutinf_boots will include those zeroed out in the permutation test
                           if(options.num_convergence_points <= 1):
                                  mutinf = mut_info_triplet_res_matrix_avg[i,j,k,l,m,n] = average(mutinf_boots,axis=0) 
                           else:
                                  mutinf = mut_info_triplet_res_matrix_avg[i,j,k,l,m,n] = mutinf_boots[-1] #take last bootstrap if doing convergence series
                                  mut_info_triplet_res_sumoverchis_matrix[:,i,j,k] += mutinf_boots  #### NEED TO FIX THIS FOR CONVERGENCE, AND ADD SUM OVER RES FOR PAIRS ABOVE 

                           if(mut_info_triplet_res_matrix.shape[0] > 5):
                                  uncert = mut_info_triplet_uncert_matrix_avg[i,j,k,m,n] = sqrt(cov(mut_info_triplet_res_matrix[:,i,j,k,l,m,n]) / (run_params.num_sims))
                           else:
                                  uncert = 0.0

                           #use vfast_cov_for1D_boot as a fast way to calculate a stdev instead of the function std()
                           pval = 1.0
                           if(mut_info_triplet_res_matrix.shape[0] >= 10 and options.num_convergence_points == 1):
                               uncert = mut_info_triplet_uncert_matrix_avg[i,j,k,l,m,n] = sqrt((vfast_cov_for1D_boot(reshape(mut_info_triplet_res_matrix[:,i,j,k,l,m,n],(mut_info_triplet_res_matrix.shape[0],1)))[0,0]) / (run_params.num_sims))
                               pval = mut_info_pval_matrix[i,j,k,l,m,n] =(stats.wilcoxon(mut_info_triplet_res_matrix[:,i,j,k,l,m,n]+SMALL))[1] / 2.0 #offset by SMALL to ensure no values of exactly zero will be removed in the test
                               if pval == 0.00 and (i != j): pval = 1.0
                               if pval <= 0.05:
                                   my_2nd_order_1_2 = mut_info_res_matrix[:,i,j,l,m]    
                                   my_2nd_order_1_3 = mut_info_res_matrix[:,i,k,l,n]    
                                   my_2nd_order_2_3 = mut_info_res_matrix[:,j,k,m,n]    
                                   mysig_2nd_order=zeros(my_2nd_order_1_2.shape, bool)
                                   for mybootstrap in range(my_2nd_order_1_2.shape[0]):
                                          mysig_2nd_order[mybootstrap] = (my_2nd_order_1_2[mybootstrap] > 0 ) or (my_2nd_order_1_3[mybootstrap] > 0 ) or (my_2nd_order_2_3[mybootstrap] > 0)     
                                   number_sig[mysig_2nd_order == True] += 1   
                                   mut_info_triplet_res_matrix_sig_05[:,i,j,k,l,m,n] = mutinf_boots.copy()
                                   mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,j,k] += mutinf_boots.copy()
                               if pval <= 0.01:
                                   #already accounted for number_sig here   
                                   mut_info_triplet_res_matrix_sig_01[:,i,j,k,l,m,n] = mutinf_boots.copy()
                           else:
                                  my_2nd_order_1_2 = mut_info_res_matrix[:,i,j,l,m]    
                                  my_2nd_order_1_3 = mut_info_res_matrix[:,i,k,l,n]    
                                  my_2nd_order_2_3 = mut_info_res_matrix[:,j,k,m,n]    
                                  mysig_2nd_order=zeros(my_2nd_order_1_2.shape, bool) 
                                  for mybootstrap in range(my_2nd_order_1_2.shape[0]):
                                          mysig_2nd_order[mybootstrap] = (my_2nd_order_1_2[mybootstrap] > 0 ) or (my_2nd_order_1_2[mybootstrap] > 0 ) or (my_2nd_order_1_2[mybootstrap] > 0)     

                                  number_sig[mysig_2nd_order == True] += 1   
                                  
                                  mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,j,k] += mutinf_boots.copy()
                           
                           #if (i != j or j != k or i != k ):
                           
                           #mut_info_triplet_res_sumoverchis_matrix_different_sims_avg[i,j] += average(mut_info_triplet_res_matrix_different_sims[:,i,j,k,m])
                           mut_info_triplet_res_uncert_sumoverchis_matrix_avg[i,j,k] += uncert ## error bars' sum over torsion pairs
                           
                               #if mut_info_triplet_res_maxoverchis_matrix_avg[i,j,k] < mutinf:  mut_info_triplet_res_maxoverchis_matrix_avg[i,j,k] = mutinf
                           
                           #if(i == j == k):
                           #        tot_ent += 0.333333 * mutinf_boots.copy()
                           #        tot_ent_diag += 0.333333 * mutinf_boots.copy()
                           #else:
                           #        tot_ent += mutinf_boots.copy()
                           #if i == j and j == k and (not (l == m == n)) :
                           #     mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,j,k] += 0.333333 * mutinf_boots.copy()
                                #mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,j,k] += -0.5 * mutinf_boots.copy()
                                #mut_info_triplet_res_sumoverchis_matrix_different_sims_avg[i,j,k] += -0.5 * average(mut_info_triplet_res_matrix_different_sims[:,i,j,k,m])
                                #mut_info_triplet_res_uncert_sumoverchis_matrix_sig_05[i,j] += 0.5 * uncert ## error bars' sum over torsion pairs, factor of 0.5 is because of double counting when looping over all pairs i,j in same residue
                           tot_ent += 0.333333 * mutinf_boots.copy()
                           tot_ent_diag += 0.333333 * mutinf_boots.copy()
                           #if (i==j and k==l): continue
                           #if (mutinf > 0.00009 or mutinf < -0.00009): print "about to append triplet mutinf: "+str(mutinf)
                           #else: print "mutinf too low: "+str(mutinf)
                           if ((i == j or j == k or i == k) and (mutinf > 0.00009 or mutinf < -0.00009 )): mutinf_triplet_vals.append([mutinf, uncert, pval, "%5s chi%d %5s chi%d %5s chi%d (SAME RES)" % (str(name_num_list[i]), l+1, str(name_num_list[j]), m+1,str(name_num_list[k]), n+1)])
                           if (i != j and j != k and i != k and (mutinf > 0.00009 or mutinf < -0.00009 )): mutinf_triplet_vals.append([mutinf, uncert, pval, "%5s chi%d %5s chi%d %5s chi%d (DIFF RES)" % (str(name_num_list[i]), l+1, str(name_num_list[j]), m+1, str(name_num_list[k]), n+1)])
               
               mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,j,k] /= (number_sig * 1.0 + SMALL) #average over number of significant ones in each bootstrap sample

       # HACK FOR NOW TO FIX TENSOR TRANSPOSE STUFF
       for i in range(len(reslist)):
              for j in range(i + 1,len(reslist)):
                     for k in range(j + 1,len(reslist)):
                            mutinf =  mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,j,k]
                            mut_info_triplet_res_sumoverchis_matrix_sig_05[:,i,k,j] = mutinf
                            mut_info_triplet_res_sumoverchis_matrix_sig_05[:,j,i,k] = mutinf
                            mut_info_triplet_res_sumoverchis_matrix_sig_05[:,j,k,i] = mutinf
                            mut_info_triplet_res_sumoverchis_matrix_sig_05[:,k,i,j] = mutinf
                            mut_info_triplet_res_sumoverchis_matrix_sig_05[:,k,j,i] = mutinf
                                   
       


       ########################################################

       if(OFF_DIAG == 1):
              if(options.num_convergence_points > 1):
                     output_tensor(prefix+"_bootstrap_avg_triplet_mutinf_res_sum.txt",            mut_info_triplet_res_sumoverchis_matrix_sig_05[-1,:,:,:] ,name_num_list,name_num_list, name_num_list)
              else:
                     output_tensor(prefix+"_bootstrap_avg_triplet_mutinf_res_sum.txt",            average(mut_info_triplet_res_sumoverchis_matrix_sig_05,axis=0) ,name_num_list,name_num_list, name_num_list)

       if EACH_BOOTSTRAP_MATRIX == 1 and OFF_DIAG == 1:
              for bootstrap in range(len(run_params.which_runs)):
                           output_tensor(prefix+"_bootstrap_"+str(bootstrap)+"_mutinf_triplet_res_sig_05.txt",mut_info_triplet_res_sumoverchis_matrix_sig_05[bootstrap],name_num_list,name_num_list,name_num_list)
                    #output_matrix(prefix+"_bootstrap_"+str(bootstrap)+"_KLdivpert_boots.txt",Kullback_Leibler_local_covar_boots[bootstrap],name_num_list,name_num_list)
                    



                           

       for mutinf, uncert, pval, name in reversed(sorted(mutinf_vals)): print "BOOTSTRAP DIHEDRAL RESULTS: %s -> mi %.4f   sd/sqrt(n) %.1e   p %.1e" % (name, mutinf, uncert, pval)
       
       if(options.triplet == "yes"):
           print 
           for mutinf, uncert, pval, name in reversed(sorted(mutinf_triplet_vals)): print "TRIPLET DIHEDRAL RESULTS: %s -> mi %.4f   sd/sqrt(n) %.1e   p %.1e" % (name, mutinf, uncert, pval)
           print 



    print "TIME at finish: ", timer

if __name__ == "__main__":
    main()