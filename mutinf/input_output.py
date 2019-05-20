### input_output.py #####################
### Input/Output Utilities ####
### Christopher McClendon ###########################
### acknowledgements to David Mobley for readxvg  ###
#####################################################
from numpy import *
import numpy as np
#import mdp
import re, os, sys, os.path, time, shelve
from optparse import OptionParser
import weave
from scipy import stats as stats
from scipy import special as special
from scipy import integrate as integrate
from scipy import misc as misc
from weave import converters
from constants import *
from input_output import *
import time
import PDBlite, utils

#########################################################################################################################################
##### Utility Functions: Various funtions for reading input, writing output, etc.  ##########################################################
#########################################################################################################################################

#Function definition to read an xvg file
def readxvg(filename,skip,skip_over_steps,last_step = None):
   """Read desired xvg file; strip headers and return data as array. First column of array is times of data points; remaining columns are the data. Should properly truncate the end of the data file if any of the lines are incomplete.
INPUT: Name or path of xvg file to read.
RETURN: As a tuple:
(1) An LxN data array containing the data from the file, less the header and any aberrant lines from the end (aberrant in the sense of truncated or not following the pattern of the rest of the lines). N is the number of columns in the xvg file, and L the number of lines. It is up to the user to interpret these.
Note that units are as in xvg file (normall kJ/mol for energies from GROMACS)
(2) The title as read from the xvg file
"""

   #Read input data
   print "filename: " + filename + "\n";
   if filename.endswith(".xvg.gz"):
       import gzip
       fil = gzip.GzipFile(filename, 'r')
   elif filename.endswith(".xvg"):
       fil = open(filename,'r');
   else:
       print "ERROR: Expected and .xvg or .xvg.gz file type: " + filename
       sys.exit(1)
   inlines=fil.readlines()
   fil.close()

   #Slice off headers
   #Find header lines beginning with @ or #.
   headerline=re.compile(r'[@#].*')
   match=True
   linenum=0
   title=''
   while (match):
     m=headerline.match(inlines[linenum])
     if not m:
        match=False
     else:
        #obtain title
        if inlines[linenum].find('title')>-1:
           tmp=inlines[linenum].split() 
           if(len(tmp) > 3):
                  title=tmp[2]+' '+tmp[3]
        #Go to next line
        linenum+=1
   #slice off headers
   inlines=inlines[linenum:]
   #print inlines[:10] #print first 10 lines to check
   #print "\n"
   #print inlines[9990:]
   #Detect how many fields on each line in body of xvg file. 
   numfields=len(inlines[0].split())

   #Length (including any aberrant lines at the end) 
   inlength=len(inlines)
   if last_step != None:
          if last_step > 0:
                 if inlength > last_step:
                        inlength = last_step
                 if inlength < last_step:
                        print "WARNING: last_step "+str(last_step)+" is beyond input length"
   print "inlength:"+str(inlength)+"\n"
   #Array to store data
   extra_record = 0
   if(skip == 1):
       extra_record = 0
   else:
       extra_record = 0
   dataarray=zeros((int((inlength-skip_over_steps)/skip) + extra_record,numfields),float64) #could start with zero, so add + extra_record
   
   skiplines=0
   #Read data into array
   for i in range(int((inlength-skip_over_steps)/skip) + extra_record): #could start with zero, so add + extra_record ...
      if(i*skip + skip_over_steps < inlength): # ... but make sure we don't overshoot
          entries=inlines[i*skip+skip_over_steps].split()
      #Make sure find expected number of entries on line
      tmpentries=len(entries)
      if tmpentries!=numfields:
        print "Found %(tmpentries)s on line %(i)s; expected %(numfields)s. Skipping line and continuing." % vars()
        skiplines+=1
      elif entries[1]=='nan':
        #Do a bit of checking also for corrupted data as in the case of corrupted trajectories
        #which sometimes give nan on this step.
        skiplines+=1
        print "Found some 'nan' entries on line %(i)s. Skipping." % vars()
      else:
        #Store data to data array, in packed format
        for j in range(numfields):
           dataarray[i-skiplines][j]=float(entries[j])

   #Last (skiplines) of dataarray will be empty, so pack data array
   dataarray=resize(dataarray,(int((inlength-skip_over_steps)/skip + extra_record)-skiplines,numfields))
   print "shape of dataarray:"+str(dataarray.shape)
   return (dataarray,title)


def bintouple(angle1,angle2,binwidth):
   bin1 = int(floor((angle1-0.00001 + 180) / binwidth))
   bin2 = int(floor((angle2-0.00001 + 180) / binwidth))
   return [bin1, bin2]

def binsingle(angle,inv_binwidth):
   if angle < 0: angle = 0.00000011
   if angle > 360: angle -= 360
   return int(floor((angle-0.0000001)*inv_binwidth)) #so we don't get an overshoot if angle is exactly 180

def binsingle_adaptive(angle,inv_binwidth):
    #print "rank: "+str(angle)+" binwidth: "+str(1.0/inv_binwidth)+" bin: "+str(int(floor(angle*inv_binwidth)))
    return int(floor(angle*inv_binwidth)) #here "angle" is a rank-order for the angle over sum(numangles)
   
# output the diagonal elements of a matrix
def output_diag(myfilename,mymatrix,rownames):
   #outputs only diagonal
   myfile = open(myfilename,'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
      myfile.write(row_name + " ")
      for col_num, col_name in zip(range(len(rownames)), rownames):
         if(row_num == col_num):
            myfile.write(str(mymatrix[row_num,col_num]))
            myfile.write("\n")
   myfile.close()

def output_entropy(myfilename,mylist):
   myfile = open(myfilename,'w')
   for i in range(mylist.shape[0]):
       myfile.write(str(mylist[i]))
       myfile.write("\n")
   myfile.close()

def output_value(myfilename,myvalue):
   myfile = open(myfilename,'w')
   myfile.write(str(myvalue))
   myfile.write("\n")
   myfile.close()

# output the elements of a matrix in string formatting, optionally zeroing the diagonal terms
def output_matrix(myfilename,mymatrix,rownames,colnames, zero_diag=False):
   myfile = open(myfilename,'w')

   for col_num, col_name in zip(range(len(colnames)), colnames):
      myfile.write(col_name + " ")
   myfile.write("\n")
   for row_num, row_name in zip(range(len(rownames)), rownames):
      myfile.write(row_name + " ")
      for col_num, col_name in zip(range(len(colnames)), colnames):
         if col_num == row_num and zero_diag:
            myfile.write(str(0))
         else:
            #print row_num, col_num, mymatrix[row_num,col_num]
            myfile.write(str(mymatrix[row_num,col_num]))
         myfile.write(" ")
      myfile.write("\n")
   myfile.close()

# output the elements of a rank 3 tensor in string formatting, with elem 1 and 2 as rows and elem 3 as columns, optionally zeroing the diagonal terms
def output_tensor(myfilename,mymatrix,rownames1,rownames2,colnames, zero_diag=False):
   myfile = open(myfilename,'w')

   for col_num, col_name in zip(range(len(colnames)), colnames):
      myfile.write(col_name + " ")
   myfile.write("\n")
   for row_num1, row_name1 in zip(range(len(rownames1)), rownames1): #concatenate names of elem1 and elem 2 together with _
      
      for row_num2, row_name2 in zip(range(len(rownames2)), rownames2):
        myfile.write(row_name1 + "_" + row_name2 + " ")
        for col_num, col_name in zip(range(len(colnames)), colnames):
          if (col_num == row_num1 or col_num == row_num2) and zero_diag:
             myfile.write(str(0))
          else:
            #print row_num, col_num, mymatrix[row_num,col_num]
            myfile.write(str(mymatrix[row_num1,row_num2,col_num]))
          myfile.write(" ")
        myfile.write("\n")
   myfile.close()

def output_matrix_chis(myfilename,mymatrix,rownames,colnames, nchi=6, zero_diag=False):
   myfile = open(myfilename,'w')
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   for col_num, col_name in zip(range(len(colnames)), colnames):
     for col_chi in range(nchi):
      myfile.write(col_name + "_" +str(col_chi) + " ")
   myfile.write("\n")
   for row_num, row_name in zip(range(len(rownames)), rownames):
     for row_chi in range(nchi):
      myfile.write(row_name + "_" + str(row_chi) + " ")
      for col_num, col_name in zip(range(len(colnames)), colnames):
        for col_chi in range(nchi):  
         if col_num == row_num and row_chi == col_chi and zero_diag:
            myfile.write(str(0))
         else:
            #print row_num, col_num, mymatrix[row_num,col_num,row_chi,col_chi]
            myfile.write(str(mymatrix[row_num,col_num,row_chi,col_chi]))
         myfile.write(" ")
      myfile.write("\n")
   myfile.close()


def output_timeseries_chis(myfilename_prefix,myreslist,colnames, nsims = 6, nchi=6, ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)

   timeseries_chis_matrix = zeros((nsims, len(myreslist) * nchi, min_num_angles), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timeseries_chis_matrix[:, res_ind1 * nchi + mychi1, :] = myres1.angles[mychi1, :, :min_num_angles]
   my_file_list = []                           
   for mysim in range(nsims):
          myfile = open(myfilename_prefix + "_" + str(mysim) + ".txt",'w')
          for col_num, col_name in zip(range(len(colnames)), colnames):
                 for col_chi in range(nchi):
                        myfile.write(col_name + "_" +str(col_chi) + " ")
                 myfile.write("\n")
   
          for myrow in range(min_num_angles):
                 for col_num, col_name in zip(range(len(colnames)), colnames):
                        for col_chi in range(nchi):  
                               myfile.write(str(timeseries_chis_matrix[mysim,col_num * nchi + col_chi, myrow]))
                               myfile.write(" ")
                 myfile.write("\n")
          myfile.close()

   return timeseries_chis_matrix

def output_timescales_chis(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   timescales_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timescales_chis[res_ind1,  mychi1 ] = average(myres1.slowest_implied_timescale[mychi1,:] )
                              
   myfile = open(myfilename_prefix + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(timescales_chis[row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return timescales_chis

def output_conv_bb_sc_boots(myfilename_prefix,myreslist,bootstrap_sets,rownames, nsims = 6, nchi=6 ):  ## NOTE: ASSUMES RESIDUE NUMBER STARTS AT 1 AND ALL RESIDUES SEQUENTIAL
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_array = zeros((len(myreslist), bootstrap_sets), int64)
   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
      for mybootstrap in range(bootstrap_sets):
         mynumangles_array[res_ind1, mybootstrap] = myres1.numangles_bootstrap[mybootstrap]

   timescales_chis = zeros((bootstrap_sets, len(myreslist), 2 ), float64) #initialize  mc, sc
   timescales_chis = zeros((bootstrap_sets, len(myreslist), 2 ), float64) #initialize  mc, sc
   timescales_stdevs_chis = zeros((bootstrap_sets, len(myreslist), 2 ), float64) #initialize  mc, sc
   RYG_timescales = zeros((bootstrap_sets, len(myreslist), 2 ), int8)  #green -- converged > 10 tau, yellow -- almost  1-10 tau,  red -- not converged
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for mybootstrap in range(bootstrap_sets):
      numangles_this_bootstrap = max(mynumangles_array[:,mybootstrap])
      nextbootstrap = min(mybootstrap+1,bootstrap_sets)
      minbootstrap = max(mybootstrap-3, 0)
      for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mc_sc in range(2):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              if(mc_sc == 0):
                                 try:
                                    timescales_chis[mybootstrap, res_ind1,  mc_sc ] = amax(amax(myres1.slowest_implied_timescale[0:1,minbootstrap:nextbootstrap], axis=1),axis=0) #max of the slowest implied timescales over this and previous three bootstraps
                                    timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] = amax(std(myres1.slowest_implied_timescale[0:1,minbootstrap:nextbootstrap],axis=1),axis=0)  #
                                 except:
                                    timescales_chis[mybootstrap, res_ind1,  mc_sc ] = 0
                                    timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] = 0
                              if(mc_sc == 1):
                                 try: #if it has a sidechain
                                    timescales_chis[mybootstrap, res_ind1,  mc_sc ] = amax(amax(myres1.slowest_implied_timescale[2:,minbootstrap:nextbootstrap],axis=1),axis=0) #max of the slowest implied timescales over this and previous three bootstraps
                                    timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] = amax(std(myres1.slowest_implied_timescale[2:,minbootstrap:nextbootstrap],axis=1),axis=0)  #
                                 except:
                                    timescales_chis[mybootstrap, res_ind1,  mc_sc ] = 0
                                    timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] = 0
                                    
                              #color by convergence based on slowest implied timescale
                              try:
                                 if(numangles_this_bootstrap >= 10 * timescales_chis[mybootstrap, res_ind1, mc_sc] and (timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] / (timescales_chis[mybootstrap, res_ind1, mc_sc] < 0.1 + SMALL) ) ) : #within 10%
                                    RYG_timescales[mybootstrap, res_ind1, mc_sc ] = 3 # "green"
                                 elif(numangles_this_bootstrap >= 10 * timescales_chis[mybootstrap, res_ind1, mc_sc] and (timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] / (timescales_chis[mybootstrap, res_ind1, mc_sc] < 0.2 + SMALL)) ): #within 20%
                                    RYG_timescales[mybootstrap, res_ind1, mc_sc ] = 5 # "cyan"
                                 elif(numangles_this_bootstrap >= 10 * timescales_chis[mybootstrap, res_ind1, mc_sc] and (timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] / (timescales_chis[mybootstrap, res_ind1, mc_sc] < 0.4  + SMALL)) ): #within 40%
                                    RYG_timescales[mybootstrap, res_ind1, mc_sc ] = 6 # "yellow"
                                 elif(numangles_this_bootstrap >= 10 * timescales_chis[mybootstrap, res_ind1, mc_sc] and (timescales_stdevs_chis[mybootstrap, res_ind1, mc_sc] / (timescales_chis[mybootstrap, res_ind1, mc_sc] < 0.8  + SMALL)) ): #within 80%
                                    RYG_timescales[mybootstrap, res_ind1, mc_sc ] = 13 # "orange"
                                 else:
                                    RYG_timescales[mybootstrap, res_ind1, mc_sc ] = 4 # "red"
                              except:
                                 RYG_timescales[mybootstrap, res_ind1, mc_sc ] = 4 # "red"
                                 

      myfile = open(myfilename_prefix + "_" + str(mybootstrap) + ".txt",'w')
      mypml = open(myfilename_prefix + "_" + str(mybootstrap) + ".pml",'w')
      mypse = open(myfilename_prefix + "_" + str(mybootstrap) + ".pse",'w')
      for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(2): #mainchain, sidechain 
                 myfile.write(str(RYG_timescales[mybootstrap, row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")

      for row_num, row_name in zip(range(len(rownames)), rownames):
          #myfile.write(str(row_name) + " ")
          #for col_chi in range(2): #mainchain, sidechain 
          mypml.write("color "+str(int(RYG_timescales[mybootstrap, row_num, 0 ]) )+",resi "+str(row_num)+" and n;ca,c,n,o,h" ) #mainchain
          mypml.write("\n")
          mypml.write("color "+str(int(RYG_timescales[mybootstrap, row_num, 1 ]) )+",resi "+str(row_num)+" and !(n;c,o,h|(n. n&!r. pro))" ) #sidechain
          mypml.write("\n")
   
      mypml.write("cmd.bg_color('white') \n")
      mypml.write("cmd.show('cartoon'   ,'all') \n")
      mypml.write("cmd.show('sticks','((byres (all))&(!(n;c,o,h|(n. n&!r. pro))))') \n")
      mypml.write("cmd.hide('((byres (all))&(n. c,o,h|(n. n&!r. pro)))') \n")
      mypml.write("cmd.hide('(hydro and (elem c extend 1))') \n")
      mypml.write("save "+str(mypse)+",format=pse \n")
      mypml.write("cmd.set('session_changed',0) \n")

      myfile.close()
      mypml.close()
   return timescales_chis

def output_timescales_chis_boots(myfilename_prefix,myreslist,bootstrap_sets,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   timescales_chis = zeros((bootstrap_sets, len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for mybootstrap in range(bootstrap_sets):
      for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timescales_chis[mybootstrap, res_ind1,  mychi1 ] = myres1.slowest_implied_timescale[mychi1,mybootstrap] 
                              
      myfile = open(myfilename_prefix + "_" + str(mybootstrap) + ".txt",'w')
      for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(timescales_chis[mybootstrap, row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
   
      myfile.close()

   return timescales_chis

def output_timescales_chis_avg(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   timescales_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timescales_chis[res_ind1,  mychi1 ] = average(myres1.slowest_implied_timescale[mychi1,:] )
                              
   myfile = open(myfilename_prefix + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")  
          myfile.write(str(average(timescales_chis[row_num, : ]) ))
          myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return timescales_chis

def output_timescales_chis_max(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   timescales_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timescales_chis[res_ind1,  mychi1 ] = average(myres1.slowest_implied_timescale[mychi1,:] )
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")  
          myfile.write(str(timescales_chis[row_num, : ]) )
          myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return timescales_chis

def output_timescales_chis_last_max(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   timescales_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timescales_chis[res_ind1,  mychi1 ] = myres1.slowest_implied_timescale[mychi1, -1 ]
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")  
          for col_chi in range(nchi):
             myfile.write(str(np.max(timescales_chis[row_num, col_chi ] )))
             myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return timescales_chis

def output_timescales_chis_last(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)

   timescales_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              timescales_chis[res_ind1,  mychi1 ] = myres1.slowest_implied_timescale[mychi1, -1 ]
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")  
          for col_chi in range(nchi):
             myfile.write(str(np.max(timescales_chis[row_num, col_chi ] )))
             myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return timescales_chis



def output_lagtimes_chis(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   lagtimes_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              lagtimes_chis[res_ind1,  mychi1 ] = average(myres1.slowest_lagtime[mychi1,:] )
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(lagtimes_chis[row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return lagtimes_chis

def output_lagtimes_chis_last(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   lagtimes_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              lagtimes_chis[res_ind1,  mychi1 ] = (myres1.slowest_lagtime[mychi1,-1 ] )
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(lagtimes_chis[row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return lagtimes_chis

def output_timescales_mutinf_autocorr_chis(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
#print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   autotimes_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              autotimes_chis[res_ind1,  mychi1 ] = average(myres1.mutinf_autocorr_time[mychi1,:] )
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(autotimes_chis[row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return lagtimes_chis

def output_timescales_angles_autocorr_chis(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
#print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   autotimes_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              autotimes_chis[res_ind1,  mychi1 ] = average(myres1.angles_autocorr_time[mychi1,:] )
                              
   myfile = open(myfilename_prefix + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(autotimes_chis[row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return autotimes_chis

def output_timescales_angles_autocorr_chis_boots(myfilename_prefix,myreslist,bootstrap_sets, rownames, nsims = 6, nchi=6 ):
#print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   autotimes_chis = zeros((bootstrap_sets, len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles
   
   for mybootstrap in range(bootstrap_sets):
      for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              autotimes_chis[mybootstrap, res_ind1,  mychi1 ] = myres1.angles_autocorr_time[mychi1,mybootstrap] 
                              
      myfile = open(myfilename_prefix + "_" + str(mybootstrap) + ".txt",'w')
      for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")
          for col_chi in range(nchi):  
                 myfile.write(str(autotimes_chis[mybootstrap, row_num, col_chi ] ))
                 myfile.write(" ")
          myfile.write("\n")
      
      myfile.close()

   return autotimes_chis

def output_timescales_mutinf_autocorr_chis_max(myfilename_prefix,myreslist,rownames, nsims = 6, nchi=6 ):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   mynumangles_list = []
   for myres1 in myreslist:
      mynumangles_list.append(min(myres1.numangles))
   min_num_angles = max(mynumangles_list)
   autotimescales_chis = zeros((len(myreslist), nchi ), float64) #initialize
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   for res_ind1, myres1 in zip(range(len(myreslist)), myreslist):
                       #print "\n#### Working on residue %s (%s):" % (myres1.num, myres1.name) , utils.flush()
                       for mychi1 in range(myres1.nchi):
                              #print "%s chi: %d/%d" % (myres1.name,int(myres1.num),mychi1+1)
                              #print "res_ind1: "+str(res_ind1)
                              #rint "mychi1: "+str(mychi1)
                              #print "nchi: " +str(nchi)
                              #print "min_num_angles: "+str(min_num_angles)
                              #print "res_ind1 * nchi + mychi1: "+str(res_ind1 * nchi + mychi1)
                              #print "myres1.angles: "
                              #print myres1.angles
                              #print "angle entries: "
                              #print myres1.angles[mychi1, :, :min_num_angles]
                              autotimescales_chis[res_ind1,  mychi1 ] = average(myres1.mutinf_autocorr_time[mychi1,:] )
                              
   myfile = open(myfilename_prefix  + ".txt",'w')
   for row_num, row_name in zip(range(len(rownames)), rownames):
          myfile.write(str(row_name) + " ")  
          myfile.write(str(np.max(autotimescales_chis[row_num, : ]) ))
          myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()

   return timescales_chis

def output_mutinf_convergence(myfilename,mutinf,bootstrap_sets):
   #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
   
   #self.angles = zeros((self.nchi,num_sims,max_angles),float64)         # the dihedral angles

   print "outputting mutinf "
   print mutinf
   myfile = open(myfilename,'w')
   for row_num, row_name in zip(range(bootstrap_sets),range(bootstrap_sets) ):
          myfile.write(str(row_name) + " ")  
          myfile.write(str(mutinf[row_num]))
          myfile.write(" ")
          myfile.write("\n")
   
   myfile.close()


def output_matrix_chis_2dhists(myfilename,mymatrix,rownames,colnames, nchi=6, nbins = 12, zero_diag=False):
    myfile = open(myfilename,'w')
    #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"
    for col_num, col_name in zip(range(len(colnames)), colnames):
        for col_chi in range(nchi):
                            for bin_j in range(nbins):
                                myfile.write(col_name + "_" +str(col_chi) +  "_" + str(bin_j) + " ")
    myfile.write("\n")
    for row_num, row_name in zip(range(len(rownames)), rownames):
        for row_chi in range(nchi):
           for bin_i in range(nbins):
              myfile.write(row_name + "_" + str(row_chi) + "_" + str(bin_i) + " ")
              for col_num, col_name in zip(range(len(colnames)), colnames):
                 for col_chi in range(nchi):  
                        for bin_j in range(nbins):
                            if col_num == row_num and row_chi == col_chi and zero_diag:
                                myfile.write(str(0))
                            else:
                                #print row_num, col_num, row_chi, col_chi, bin_i, bin_j 
                                #print mymatrix[row_num,col_num,row_chi,col_chi,bin_i,bin_j]
                                myfile.write(str(mymatrix[row_num,col_num,row_chi,col_chi,bin_i,bin_j]))
                            myfile.write(" ")
              myfile.write("\n") #newline before next row
    myfile.close()


def output_2dhist(myfilename,mymatrix,row_name, col_name, nbins = 12,  zero_diag=False):
    myfile = open(myfilename,'w')
    #print "shape of matrix to ouput:"+str(mymatrix.shape)+"\n"

    for bin_j in range(nbins):
       myfile.write(col_name +  "_" + str(bin_j) + " ")
    
    myfile.write("\n") 

    for bin_i in range(nbins):
              myfile.write(row_name + "_" + str(bin_i) + " ")
              for bin_j in range(nbins):
                 
                           
                 #print row_num, col_num, row_chi, col_chi, bin_i, bin_j 
                 #print mymatrix[row_num,col_num,row_chi,col_chi,bin_i,bin_j]
                 myfile.write(str(mymatrix[bin_i,bin_j]))
                 myfile.write(" ")
              myfile.write("\n") #newline before next row
    myfile.close()


def output_mutinfs_for_hists(myfilename, mutinfs_1_2, mutinfs_2_3, mutinfs_1_3,  uncorrected_mutinfs ,  independent_mutinfs, corrected_mutinfs ):
    myfile = open(myfilename,'w')
    for myindex in range(len(corrected_mutinfs)):
       myfile.write(str(mutinfs_1_2[myindex]) +" "+str(mutinfs_2_3[myindex]) + " " + str(mutinfs_1_3[myindex]) + " " + str(uncorrected_mutinfs[myindex]) + " " + str(independent_mutinfs[myindex]) + " " + str(corrected_mutinfs[myindex]) + "\n")
    myfile.close()


def read_matrix_chis(myfilename, nchi=6, zero_diag=False):
   rownames = []
   colnames = []
   myfile = open(myfilename,'r')
   inlines = myfile.readlines()
   #print inlines
   myfile.close()
   reschis = inlines[0].split()
   mymatrix = zeros((int(len(inlines[1:]) / nchi), int((len(reschis))/nchi),6,6),float64)
   #print mymatrix.shape
   for myname_num in reschis:
       (thisname, thisnum) = myname_num.split('_')
       if int(thisnum) == 0:
           colnames.append(thisname)
   #print colnames
   #print len(colnames)
   for row_num in range(int(len(inlines[1:]))):
       thisline = inlines[row_num + 1]
       thislinedata = thisline.split()
       (thisname, row_chi) = thislinedata[0].split('_')
       res_num = int(floor(row_num / nchi))
       row_chi = int(row_chi) #convert string value to integer
       thislinenums = map(float, thislinedata[1:]) #does this need to be float64 or another double precision thingy?
       #print thislinenums
       thislinearray = array(thislinenums,float64)
       #print thislinearray.shape
       if row_chi == 0:
           rownames.append(thisname)
       for col_num in range(len(colnames)):
           for col_chi in range(nchi):
               #print "name: "+str(thisname)+" chi: "+str(row_chi)+ " row_num: "+str(row_num)+" row_chi: "+str(row_chi)+ " col_num: "+str(col_num)+" col_chi: "+str(col_chi)+"\n"
               mymatrix[res_num,col_num,row_chi,col_chi] = float64(thislinearray[col_num*nchi + col_chi])
   #print rownames
   return mymatrix, rownames, colnames


def read_matrix_chis_2dhists(myfilename,mymatrix, nchi=6, nbins = 12, zero_diag=False):
    rownames = []
    colnames = []
    print "reading 2d histograms from: "+str(myfilename)
    myfile = open(myfilename,'r')
    inlines = myfile.readlines()
    #print inlines                                                                                                           
    print inlines[0]
    myfile.close()
    reschis = inlines[0].split()
    mymatrix = zeros((int(len(inlines[1:]) / (nchi*nbins)), int((len(reschis))/(nchi*nbins)),nchi,nchi,nbins,nbins),float64)
    for myname_num in reschis:
       (thisname, thischi, thisbin) = myname_num.split('_')
       if int(thischi) == 0 and int(thisbin) == 0:
           colnames.append(thisname)

    for row_num in range(int(len(inlines[1:]))):
       thisline = inlines[row_num + 1]
       thislinedata = thisline.split()
       (thisname, row_chi, row_bin) = thislinedata[0].split('_')
       res_num = int(floor(row_num / (nchi*nbins)))
       row_chi = int(row_chi) #convert string value to integer
       row_bin = int(row_bin) #convert string value to integer
       thislinenums = map(float64, thislinedata[1:]) #does this need to be float64 or another double precision thingy?
       #print thislinenums
       thislinearray = array(thislinenums,float64)
       #print thislinearray.shape
       if row_chi == 0:
           rownames.append(thisname)
       for col_num in range(len(colnames)):
           for col_chi in range(nchi):
              for col_bin in range(nbins):
               #print "name: "+str(thisname)+" chi: "+str(row_chi)+ " row_num: "+str(row_num)+" row_chi: "+str(row_chi)+ " col_num: "+str(col_num)+" col_chi: "+str(col_chi)+"\n"
               mymatrix[res_num,col_num,row_chi,col_chi,row_bin,col_bin] = float64(thislinearray[col_num*nchi*nbins + col_chi*nbins + col_bin])
    myfile.close()
    return mymatrix, rownames, colnames


def read_res_matrix(myfilename):
   rownames = []
   colnames = []
   myfile = open(myfilename,'r')
   inlines = myfile.readlines()
   myfile.close()
   res = inlines[0].split()
   mymatrix = zeros((int(len(inlines[1:])), int(len(res))),float64)
   #print mymatrix.shape
   for myname_num in res:
       colnames.append(myname_num)
   #print colnames
   #print len(colnames)
   for row_num in range(int(len(inlines[1:]))):
       thisline = inlines[row_num + 1]
       thislinedata = thisline.split()
       thisname = thislinedata[0]
       res_num = int(floor(row_num))
       thislinenums = map(float, thislinedata[1:]) #does this need to be float64 or another double precision thingy?
       #print thislinenums
       thislinearray = array(thislinenums,float64)
       #print thislinearray.shape
       rownames.append(thisname)
       for col_num in range(len(colnames)):
           #print "name: "+str(thisname)+" chi: "+str(row_chi)+ " row_num: "+str(row_num)+" row_chi: "+str(row_chi)+ " col_num: "+str(col_num)+" col_chi: "+str(col_chi)+"\n"
           mymatrix[res_num,col_num] = float64(thislinearray[col_num])
   #print rownames
   return mymatrix, rownames, colnames


