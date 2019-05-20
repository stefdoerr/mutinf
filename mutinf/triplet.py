
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
from constants import *
from input_output import *
import time
import PDBlite, utils
try:
       import MDAnalysis 
except: pass


### GLOBALS FOR TRIPLETS ###
E_I_triplet_multinomial = None
Var_I_triplet_multinomial = None
Var_I_runs_triplet_multinomial = None
mutinf_triplet_multinomial = None
var_mutinf_triplet_multinomial = None
mutinf_triplet_multinomial_sequential = None
var_mutinf_triplet_multinomial_sequential = None
E_I_triplet_uniform = None
numangles_bootstrap_tensor = None
numangles_bootstrap_matrix = None
count_matrix_triplet_sequential = None
count_matrix_triplet = None
count_matrix_1 = None
count_matrix_2 = None
count_matrix_3 = None
count_matrix_1_2 = None
count_matrix_1_3 = None
count_matrix_2_3 = None


#######   TRIPLET TERMS ##################

count_matrix_triplet_markov = None
count_matrix_triplet_markov_1 = None
count_matrix_triplet_markov_2 = None
count_matrix_triplet_markov_3 = None
count_matrix_triplet_markov_1_2 = None
count_matrix_triplet_markov_2_3 = None
count_matrix_triplet_markov_1_3 = None
#numangles_bootstrap_matrix_markov = None
def calc_triplet_mutinf_markov_independent( nbins, chi_counts1_markov, chi_counts2_markov, chi_counts3_markov, ent1_markov_boots, ent2_markov_boots, ent3_markov_boots, bins1, bins2, bins3, bootstrap_sets, bootstrap_choose, markov_samples, max_num_angles, numangles_bootstrap, bins1_slowest_lagtime, bins2_slowest_lagtime, bins3_slowest_lagtime):
    
    global count_matrix_triplet_markov, count_matrix_triplet_markov_1, count_matrix_triplet_markov_2, count_matrix_triplet_markov_3, count_matrix_triplet_markov_1_2, count_matrix_triplet_markov_2_3, count_matrix_triplet_markov_1_3
    #global numangles_bootstrap_matrix_markov
    global OUTPUT_INDEPENDENT_MUTINF_VALUES

    markov_interval = zeros((bootstrap_sets),int16)
    # allocate the matrices the first time only for speed
    #if(markov_samples > 0 and bins1_slowest_lagtime != None and bins2_slowest_lagtime != None and bins3_slowest_lagtime != None ):
    #       for bootstrap in range(bootstrap_sets):
    #              max_lagtime = max(bins1_slowest_lagtime[bootstrap], bins2_slowest_lagtime[bootstrap], bins3_slowest_lagtime[bootstrap])
    #              markov_interval[bootstrap] = int(max_num_angles / max_lagtime) #interval for final mutual information calc, based on max lagtime
    #else:
    #       markov_interval[:] = 1

    markov_interval[:] = 1
    #ninj_prior = 1.0 / float64(nbins*nbins) #Perks' Dirichlet prior
    #ni_prior = nbins * ninj_prior           #Perks' Dirichlet prior
    ninj_prior = 1.0                        #Uniform prior
    ni_prior = nbins * ninj_prior           #Uniform prior
    
    if count_matrix_triplet_markov is None:    

           count_matrix_triplet_markov = zeros((bootstrap_sets, markov_samples , nbins*nbins*nbins), float64)

           #1-D marginals
           count_matrix_triplet_markov_1 = zeros((bootstrap_sets, markov_samples, nbins), float64)
           count_matrix_triplet_markov_2 = zeros((bootstrap_sets, markov_samples, nbins), float64)
           count_matrix_triplet_markov_3 = zeros((bootstrap_sets, markov_samples, nbins), float64)

           #2-D marginals
           count_matrix_triplet_markov_1_2 = zeros((bootstrap_sets, markov_samples , nbins*nbins), float64)
           count_matrix_triplet_markov_2_3 = zeros((bootstrap_sets, markov_samples , nbins*nbins), float64)
           count_matrix_triplet_markov_1_3 = zeros((bootstrap_sets, markov_samples , nbins*nbins), float64)
       
    #print "shape of chi counts1_markov:"
    #print shape(chi_counts1_markov)
       
           
       
    #if(numangles_bootstrap_matrix_markov is None):
    #print "numangles bootstrap: "+str(numangles_bootstrap)
    numangles_bootstrap_tensor_markov = zeros((bootstrap_sets,markov_samples,nbins*nbins*nbins),float64)
    numangles_bootstrap_matrix_markov = zeros((bootstrap_sets,markov_samples,nbins*nbins),float64)
    for bootstrap in range(bootstrap_sets):
           for markov_chain in range(markov_samples):
                  numangles_bootstrap_tensor_markov[bootstrap,markov_chain,:]=numangles_bootstrap[bootstrap]
                  numangles_bootstrap_matrix_markov[bootstrap,markov_chain,:]=numangles_bootstrap[bootstrap]
    numangles_bootstrap_vector = numangles_bootstrap_matrix_markov[:,:,:nbins]
        #print "Numangles Bootstrap Vector:\n"+str(numangles_bootstrap_vector)
    
    #if(chi_counts2_markov_matrix is None):
    #chi_counts1_markov_matrix = zeros((bootstrap_sets, markov_samples, nbins*nbins),float64)        
    #chi_counts2_markov_matrix = zeros((bootstrap_sets, markov_samples, nbins*nbins),float64)       
    

    #initialize if we aren't allocating each time
    count_matrix_triplet_markov[:,:,:] = 0
    count_matrix_triplet_markov_1[:,:,:] = 0
    count_matrix_triplet_markov_2[:,:,:] = 0
    count_matrix_triplet_markov_3[:,:,:] = 0
    count_matrix_triplet_markov_1_2[:,:,:] = 0
    count_matrix_triplet_markov_2_3[:,:,:] = 0
    count_matrix_triplet_markov_1_3[:,:,:] = 0

    ent1_boots=zeros((bootstrap_sets, markov_samples), float64)
    ent2_boots=zeros((bootstrap_sets, markov_samples), float64)
    ent3_boots=zeros((bootstrap_sets, markov_samples), float64)
    
    ent_1_2_boots=zeros((bootstrap_sets, markov_samples), float64)
    ent_2_3_boots=zeros((bootstrap_sets, markov_samples), float64)
    ent_1_3_boots=zeros((bootstrap_sets, markov_samples), float64)

    ent_1_2_3_boots=zeros((bootstrap_sets, markov_samples), float64)
    ## 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
    

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

    code = """
    // weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     #include <math.h>

  
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mysign1, mysign2, mysign3 = 0 ;
     int mybootstrap, markov_chain;
     long mynumangles, anglenum;
     long  offset1, offset2, offset3, offset4, counts1, counts2, counts3, counts12, counts23, counts13;
     double dig1, dig2, dig3, counts1d, counts12d, counts23d, counts13d;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
      //offset4 = mybootstrap*(markov_samples)*nbins*nbins*nbins ;
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, counts3, counts12, counts23, counts13, counts1d, counts12d, counts13d, counts23d, mysign1, mysign2, mysign3, bin1, bin2, bin3, dig1) 
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
           if(anglenum == mynumangles - 1) {
            printf(""); 
            // the command above is just to stall for a little time and make sure that the code behaves when compiled and finishes writing to the arrays
            
            }
       
             // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8)  --- but with chi already dereferenced :  the bin for each dihedral
             // python: self.chi_counts_markov=zeros((bootstrap_sets, self.markov_samples, self.nchi, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
             //if(anglenum % markov_interval[mybootstrap] == 0) {

              angle1_bin = *(bins1  + mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
              angle2_bin = *(bins2  + mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
              angle3_bin = *(bins3  + mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
             
              *(count_matrix_triplet_markov_1  + (long)(mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  angle1_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_2  + (long)(mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  angle2_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_3  + (long)(mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  angle3_bin )) += 1.0 ;  //* weight * weight;

              *(count_matrix_triplet_markov_1_2  +   (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle2_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_2_3  +   (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle2_bin*nbins +   angle3_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_1_3  +   (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle3_bin )) += 1.0 ;  //* weight * weight;

              *(count_matrix_triplet_markov  +  (long)(mybootstrap*(markov_samples)*nbins*nbins*nbins  +  markov_chain*nbins*nbins*nbins  +  angle1_bin*nbins*nbins +   angle2_bin*nbins + angle3_bin )) += 1.0 ;  //* weight * weight;
             //}
             
          }

          
          

          
          // now actually compute the entropies for each order to be combined later
 
          
         for(bin1=0; bin1 < nbins; bin1++) 
          { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
         
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           counts12 = *(count_matrix_triplet_markov_1_2  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
           
           

          if(counts12 > 0)
           {
            mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts12);
            counts12d = 1.0 * counts12;
            *(ent_1_2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts12d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts12d + 1.0)))); 
           }
           
          counts23 = *(count_matrix_triplet_markov_2_3  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
          if(counts23 > 0)
           {
            mysign1 = 1.0L - 2*(counts23 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts23);
            counts23d = 1.0 * counts23;
            *(ent_2_3_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts23d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts23d + 1.0)))); 
           }
           
          counts13 = *(count_matrix_triplet_markov_1_3  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));

          if(counts13 > 0)
           {
            mysign1 = 1.0L - 2*(counts13 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts13);
            counts13d = 1.0 * counts13;
            *(ent_1_3_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts13d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts13d + 1.0)))); 
           }
           



           }
          }
         
           

        
         for(bin1=0; bin1 < nbins; bin1++) 
         { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
            for(bin3=0; bin3 < nbins; bin3++)
             { 
               counts1 = *(count_matrix_triplet_markov  +  (long)(mybootstrap*(markov_samples)*nbins*nbins*nbins   +  markov_chain*nbins*nbins*nbins  +  bin1*nbins*nbins + bin2*nbins + bin3 ));
               
 
                if(counts1 > 0)
                {
                  mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                  dig1 = xDiGamma_Function(counts1);
                  mysign1 = 1 - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                  counts1d = 1.0 * counts1 ;
                  *(ent_1_2_3_boots + (long)(mybootstrap*markov_samples + markov_chain)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)counts1d + 1.0))); 
                }
               
           }
          }
         }
    

        }
       }
      """

    code_no_grassberger = """
    // weave6_markov
    // bins dimensions: bootstrap_sets * markov_samples * bootstrap_choose * max_num_angles
     #include <math.h>

  
     double weight;
     int angle1_bin = 0;
     int angle2_bin = 0 ;
     int angle3_bin = 0;
     int bin1, bin2, bin3 = 0;
     int mysign1, mysign2, mysign3 = 0 ;
     int mybootstrap, markov_chain;
     long mynumangles, anglenum;
     long  offset1, offset2, offset3, offset4, counts1, counts2, counts3, counts12, counts23, counts13;
     double dig1, dig2, dig3, counts1d, counts12d, counts23d, counts13d;
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      mynumangles = *(numangles_bootstrap + mybootstrap);  // original data went in using each sim separately using which_sims and simnum, but is read on a per-bootstrap basis
      //printf("mynumangles: %i ", mynumangles);
      //offset1 = mybootstrap*markov_samples*bootstrap_choose*max_num_angles;
      //offset2 = mybootstrap*(markov_samples)*nbins; 
      //offset3 = mybootstrap*(markov_samples)*nbins*nbins ;
      //offset4 = mybootstrap*(markov_samples)*nbins*nbins*nbins ;
      #pragma omp parallel for private(markov_chain,anglenum, angle1_bin, angle2_bin, angle3_bin, counts1, counts2, counts3, counts12, counts23, counts13, counts1d, counts12d, counts13d, counts23d, mysign1, mysign2, mysign3, bin1, bin2, bin3, dig1) 
      for (markov_chain=0; markov_chain < markov_samples ; markov_chain++) {
          for (anglenum=0; anglenum< mynumangles; anglenum++) {
           if(anglenum == mynumangles - 1) {
            printf(""); 
            // the command above is just to stall for a little time and make sure that the code behaves when compiled and finishes writing to the arrays
            
            }
       
             // python: self.bins_markov = zeros((self.nchi, bootstrap_sets, self.markov_samples, bootstrap_choose * max_num_angles ), int8)  --- but with chi already dereferenced :  the bin for each dihedral
             // python: self.chi_counts_markov=zeros((bootstrap_sets, self.markov_samples, self.nchi, nbins), float64) --- but with chi already dereferenced   : since these can be weighted in advanced sampling like replica exchange
             //if(anglenum % markov_interval[mybootstrap] == 0) {

              angle1_bin = *(bins1  + mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
              angle2_bin = *(bins2  + mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
              angle3_bin = *(bins3  + mybootstrap*markov_samples*bootstrap_choose*max_num_angles  + markov_chain*bootstrap_choose*max_num_angles +  anglenum);
             
              *(count_matrix_triplet_markov_1  + (long)(mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  angle1_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_2  + (long)(mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  angle2_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_3  + (long)(mybootstrap*(markov_samples)*nbins   +  markov_chain*nbins  +  angle3_bin )) += 1.0 ;  //* weight * weight;

              *(count_matrix_triplet_markov_1_2  +   (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle2_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_2_3  +   (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle2_bin*nbins +   angle3_bin )) += 1.0 ;  //* weight * weight;
              *(count_matrix_triplet_markov_1_3  +   (long)(mybootstrap*(markov_samples)*nbins*nbins  +  markov_chain*nbins*nbins  +  angle1_bin*nbins +   angle3_bin )) += 1.0 ;  //* weight * weight;

              *(count_matrix_triplet_markov  +  (long)(mybootstrap*(markov_samples)*nbins*nbins*nbins  +  markov_chain*nbins*nbins*nbins  +  angle1_bin*nbins*nbins +   angle2_bin*nbins + angle3_bin )) += 1.0 ;  //* weight * weight;
             //}
             
          }

          
          

          
          // now actually compute the entropies for each order to be combined later
 
          
         for(bin1=0; bin1 < nbins; bin1++) 
          { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
         
          // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
           counts12 = *(count_matrix_triplet_markov_1_2  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
           
           

          if(counts12 > 0)
           {
            mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts12);
            counts12d = 1.0 * counts12;
            *(ent_1_2_boots + (long)(mybootstrap*markov_samples + markov_chain)) += -1.0 * ((double)counts12d / mynumangles)*(log((double)counts12d / mynumangles + SMALL)) ; 
           }
           
          counts23 = *(count_matrix_triplet_markov_2_3  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));
          if(counts23 > 0)
           {
            mysign1 = 1.0L - 2*(counts23 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts23);
            counts23d = 1.0 * counts23;
            *(ent_2_3_boots + (long)(mybootstrap*markov_samples + markov_chain)) += -1.0 * ((double)counts23d / mynumangles)*(log((double)counts23d / mynumangles + SMALL)) ; 
           }
           
          counts13 = *(count_matrix_triplet_markov_1_3  + (long)( mybootstrap*(markov_samples)*nbins*nbins   +  markov_chain*nbins*nbins  +  bin1*nbins + bin2 ));

          if(counts13 > 0)
           {
            mysign1 = 1.0L - 2*(counts13 % 2); // == -1 if it is odd, 1 if it is even
            dig1 = xDiGamma_Function(counts13);
            counts13d = 1.0 * counts13;
            *(ent_1_3_boots + (long)(mybootstrap*markov_samples + markov_chain)) += -1.0 * ((double)counts13d / mynumangles)*(log((double)counts13d / mynumangles + SMALL)) ; 
           }
           



           }
          }
         
           

        
         for(bin1=0; bin1 < nbins; bin1++) 
         { 
           for(bin2=0; bin2 < nbins; bin2++)
           {
            for(bin3=0; bin3 < nbins; bin3++)
             { 
               counts1 = *(count_matrix_triplet_markov  +  (long)(mybootstrap*(markov_samples)*nbins*nbins*nbins   +  markov_chain*nbins*nbins*nbins  +  bin1*nbins*nbins + bin2*nbins + bin3 ));
               
 
                if(counts1 > 0)
                {
                  mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                  dig1 = xDiGamma_Function(counts1);
                  mysign1 = 1 - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                  counts1d = 1.0 * counts1 ;
                  *(ent_1_2_3_boots + (long)(mybootstrap*markov_samples + markov_chain)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALL)); 
                }
               
           }
          }
         }
    

        }
       }
      """
     # //weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights


    if(VERBOSE >= 2): print "about to populate triplet  count_matrix_triplet_markov"
    if(NO_GRASSBERGER == False):
           weave.inline(code, ['numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'bins3', 'count_matrix_triplet_markov','count_matrix_triplet_markov_1_2','count_matrix_triplet_markov_2_3','count_matrix_triplet_markov_1_3','count_matrix_triplet_markov_1','count_matrix_triplet_markov_2','count_matrix_triplet_markov_3','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset','markov_interval','ent1_boots','ent2_boots','ent3_boots','ent_1_2_boots','ent_2_3_boots','ent_1_3_boots','ent_1_2_3_boots','SMALL'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"], extra_compile_args =['  -fopenmp -lgomp'], extra_link_args=['-lgomp'],
                 support_code=my_support_code )
    else:
           weave.inline(code_no_grassberger, ['numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'bins3', 'count_matrix_triplet_markov','count_matrix_triplet_markov_1_2','count_matrix_triplet_markov_2_3','count_matrix_triplet_markov_1_3','count_matrix_triplet_markov_1','count_matrix_triplet_markov_2','count_matrix_triplet_markov_3','bootstrap_sets','markov_samples','max_num_angles','bootstrap_choose','offset','markov_interval','ent1_boots','ent2_boots','ent3_boots','ent_1_2_boots','ent_2_3_boots','ent_1_3_boots','ent_1_2_3_boots','SMALL'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"], extra_compile_args =['  -fopenmp -lgomp'], extra_link_args=['-lgomp'],
                 support_code=my_support_code )
    #if (no_boot_weights != False ):
    #       count_matrix_triplet_markov /= (bootstrap_choose*max_num_angles * 1.0) #to correct for the fact that we used the product of the two weights -- total "weight" should be bootstrap_choose*min(numangles)



    chi_counts1_markov_vector = count_matrix_triplet_markov_1 #reshape(chi_counts1_markov.copy(),(bootstrap_sets, markov_samples ,nbins)) #no permutations for marginal distributions...
    chi_counts2_markov_vector = count_matrix_triplet_markov_2 #reshape(chi_counts2_markov.copy(),(bootstrap_sets, markov_samples ,nbins)) #no permutations for marginal distributions...
    chi_counts3_markov_vector = count_matrix_triplet_markov_3 #reshape(chi_counts2_markov.copy(),(bootstrap_sets, markov_samples ,nbins)) #no permutations for marginal distributions...

    if(VERBOSE >= 4):
           print "count matrix markov first pass:"
           print count_matrix_triplet_markov
           print "count matrix markov shape: "
           print shape(count_matrix_triplet_markov)
           print "chi count markov shape: "
           print shape(chi_counts1_markov)
           print "sum of count_matrix_triplet_markov for bootstrap zero and markov chain last one:"
           print sum(count_matrix_triplet_markov[0,-1,:])
    #just check markov chain 0
    ninj_flat = zeros((bootstrap_sets, markov_samples ,nbins*nbins*nbins),float64)
    

           #now without the Bayes prior added into the marginal distribution
           #my_flat_Bayes = outer(chi_counts1_markov[bootstrap] + ni_prior,chi_counts2_markov[bootstrap] + ni_prior).flatten() 
           #my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
           #ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
           #nbins_cor = int(nbins * FEWER_COR_BTW_BINS)

           ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just use outer product to give zero MI
    if(all(count_matrix_triplet_markov[:,:,:] == 0)) and (sum(chi_counts1_markov[bootstrap,markov_chain]) > 0) and (sum(chi_counts2_markov[bootstrap,markov_chain]) > 0):
           count_matrix_triplet_markov[bootstrap,markov_chain,:] =  (outer(chi_counts1_markov_vector[bootstrap,markov_chain] ,outer(chi_counts2_markov_vector[bootstrap,markov_chain], chi_counts3_markov_vector[bootstrap,markov_chain]) ).flatten()  ) / (numangles_bootstrap[0] * 1.0)
              

    if(VERBOSE >=1):
              assert(all(ninj_flat >= 0))
              
    if(VERBOSE >= 1):
           for markov_chain in range(markov_samples):
                  #print "chi counts1 markov bootstrap zero markov chain: "+str(markov_chain)+" : "
                  #print chi_counts1_markov[0,markov_chain]
                  for bootstrap in range(bootstrap_sets):
                         my_flat = outer(chi_counts1_markov_vector[bootstrap,markov_chain] + 0.0 ,outer(chi_counts2_markov_vector[bootstrap,markov_chain] + 0.0, chi_counts3_markov_vector[bootstrap,markov_chain])).flatten() # have to add 0.0 for outer() to work reliably
                         if(VERBOSE >=1):
                                assert(all(my_flat >= 0))
                         #my_flat = resize(my_flat,(1 ,(my_flat.shape)[0]))
                         ninj_flat[bootstrap,markov_chain,:] = my_flat #[,:]

           Pij, PiPj = zeros((nbins, nbins), float64)  , zeros((nbins, nbins), float64)  
           Pijk, PiPjPk = zeros((nbins, nbins, nbins), float64)  , zeros((nbins, nbins, nbins), float64)  
           for markov_sample  in range(markov_samples):
                  #PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins,nbins))
                  Pijk[:,:,:]  = (count_matrix_triplet_markov[0,markov_sample,:]).reshape((nbins,nbins,nbins))  / ((numangles_bootstrap[0] / (1.0 * markov_interval[0])) * 1.0)
                  PiPjPk[:,:,:] = (ninj_flat[0,markov_sample,:]).reshape((nbins,nbins,nbins)) / ((numangles_bootstrap[0] / (1.0 * markov_interval[0])) * 1.0) #convert sum of chi_counts^3 to sum of 1
                  PiPjPk[:,:,:] /= numangles_bootstrap[0] / (1.0 * markov_interval[0]) #to prevent integer overflow
                  PiPjPk[:,:,:] /= numangles_bootstrap[0] / (1.0 * markov_interval[0])#to prevent integer overflow
                  PiPj[:,:] = sum(PiPjPk[:,:,:], axis=2)
                  Pij[:,:]  = sum(Pijk[:,:,:], axis=2)
                  if(VERBOSE >= 1 and markov_sample==0):
                         print "First Pass:"
                         print "Sum Pij: "+str(sum(Pij[:,:]))+" Sum PiPj: "+str(sum(PiPj[:,:]))
                         print "Sum Pijk: "+str(sum(Pijk[:,:]))+" Sum PiPjPk: "+str(sum(PiPjPk[:,:]))
                         print "Marginal Pij, summed over j:\n"
                         print sum(Pij[:,:],axis=1)
                         print "Marginal PiPj, summed over j:\n"
                         print sum(PiPj[:,:],axis=1)   
                         print "Marginal Pij, summed over i:\n"
                         print sum(Pij[:,:],axis=0)
                         print "Marginal PiPj, summed over i:\n"
                         print sum(PiPj[:,:],axis=0)
                         ### end redundant sanity checks
                         if(VERBOSE >=1):
                             assert(abs(sum(Pij[:,:]) - sum(PiPj[:,:])) < SMALL)# nbins)
                             assert(all(abs(sum(Pij[:,:],axis=1) - sum(PiPj[:,:],axis=1)) < SMALL)), "Pij - PiPj summed over j: "+str(sum(Pij[:,:],axis=1) - sum(PiPj[:,:],axis=1)) # < nbins))
                             assert(all(abs(sum(Pij[:,:],axis=0) - sum(PiPj[:,:],axis=0)) < SMALL)), "Pij - PiPj summed over i: "+str(sum(Pij[:,:],axis=0) - sum(PiPj[:,:],axis=0)) #< nbins))
                             assert(all(abs(sum(sum(Pijk[:,:],axis=0),axis=0) - sum(sum(PiPjPk[:,:,:],axis=0),axis=0)) < SMALL)) # < nbins)) #second axis here is actual axis number minus 1
                             assert(all(abs(sum(sum(Pijk[:,:],axis=1),axis=1) - sum(sum(PiPjPk[:,:,:],axis=1),axis=1)) < SMALL ))# < nbins)) #second axis here is actual axis number minus 1
                             assert(all(abs(sum(sum(Pijk[:,:],axis=0),axis=1) - sum(sum(PiPjPk[:,:,:],axis=0),axis=1)) < SMALL ))# < nbins)) #second axis here is actual axis number minus 1
       


    #print floor(sum(Pij[1:,1:],axis=1)) == floor(sum(PiPj[1:,1:],axis=1))

    
    
    #print "done with count matrix setup for multinomial\n"
    for bootstrap in range(bootstrap_sets):
           my_flat = outer(chi_counts1_markov[bootstrap] + ni_prior,chi_counts2_markov[bootstrap] + ni_prior).flatten() 
           my_flat = resize(my_flat,(0 + 1,(my_flat.shape)[0]))
           #ninj_flat_Bayes[bootstrap,:,:] = my_flat[:,:]

    if(numangles_bootstrap[0] > 0): #small sample stuff turned off for now cause it's broken
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):
          #ent1_boots = sum((chi_counts1_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_markov_vector + SMALL) - ((-1) ** (chi_counts1_markov_vector % 2)) / (chi_counts1_markov_vector + 1.0)),axis=2)

          #ent2_boots = sum((chi_counts2_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_markov_vector + SMALL) - ((-1) ** (chi_counts2_markov_vector % 2)) / (chi_counts2_markov_vector + 1.0)),axis=2)

          #ent3_boots = sum((chi_counts3_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts3_markov_vector + SMALL) - ((-1) ** (chi_counts3_markov_vector % 2)) / (chi_counts3_markov_vector + 1.0)),axis=2)

          #ent_1_2_boots = sum((count_matrix_triplet_markov_1_2 * 1.0 / numangles_bootstrap_matrix_markov) * (log(numangles_bootstrap_matrix_markov) - special.psi(count_matrix_triplet_markov_1_2 + SMALL) - ((-1) ** (count_matrix_triplet_markov_1_2 % 2)) / (count_matrix_triplet_markov_1_2 + 1.0)),axis=2)

          #ent_2_3_boots = sum((count_matrix_triplet_markov_2_3 * 1.0 / numangles_bootstrap_matrix_markov) * (log(numangles_bootstrap_matrix_markov) - special.psi(count_matrix_triplet_markov_2_3 + SMALL) - ((-1) ** (count_matrix_triplet_markov_2_3 % 2)) / (count_matrix_triplet_markov_2_3 + 1.0)),axis=2)

          #ent_1_3_boots = sum((count_matrix_triplet_markov_1_3 * 1.0 / numangles_bootstrap_matrix_markov) * (log(numangles_bootstrap_matrix_markov) - special.psi(count_matrix_triplet_markov_1_3 + SMALL) - ((-1) ** (count_matrix_triplet_markov_1_3 % 2)) / (count_matrix_triplet_markov_1_3 + 1.0)),axis=2)
        
          #ent_1_2_3_boots = sum((count_matrix_triplet_markov * 1.0 / numangles_bootstrap_tensor_markov) * (log(numangles_bootstrap_tensor_markov) - special.psi(count_matrix_triplet_markov + SMALL) - ((-1) ** (count_matrix_triplet_markov % 2)) / (count_matrix_triplet_markov + 1.0)),axis=2)

          #assert(all(abs(ent1_markov_boots - sum((chi_counts1_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_markov_vector + SMALL) - (1 - 2*(chi_counts1_markov_vector % 2)) / (chi_counts1_markov_vector + 1.0)),axis=2)) < 0.0001))

          #assert(all(abs(ent2_markov_boots - sum((chi_counts2_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_markov_vector + SMALL) - (1 - 2*(chi_counts2_markov_vector % 2)) / (chi_counts2_markov_vector + 1.0)),axis=2)) < 0.0001))

          #assert(all(abs(ent3_markov_boots - sum((chi_counts3_markov_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts3_markov_vector + SMALL) - (1 - 2*(chi_counts3_markov_vector % 2)) / (chi_counts3_markov_vector + 1.0)),axis=2)) < 0.0001))

          #assert(all(abs(ent_1_2_boots - sum((count_matrix_triplet_markov_1_2 * 1.0 / numangles_bootstrap_matrix_markov) * (log(numangles_bootstrap_matrix_markov) - special.psi(count_matrix_triplet_markov_1_2 + SMALL) - (1 - 2*(count_matrix_triplet_markov_1_2 % 2)) / (count_matrix_triplet_markov_1_2 + 1.0)),axis=2)) < 0.0001))

          #assert(all(abs(ent_2_3_boots - sum((count_matrix_triplet_markov_2_3 * 1.0 / numangles_bootstrap_matrix_markov) * (log(numangles_bootstrap_matrix_markov) - special.psi(count_matrix_triplet_markov_2_3 + SMALL) - (1 - 2*(count_matrix_triplet_markov_2_3 % 2)) / (count_matrix_triplet_markov_2_3 + 1.0)),axis=2)) < 0.0001))

          #assert(all(abs(ent_1_3_boots - sum((count_matrix_triplet_markov_1_3 * 1.0 / numangles_bootstrap_matrix_markov) * (log(numangles_bootstrap_matrix_markov) - special.psi(count_matrix_triplet_markov_1_3 + SMALL) - (1 - 2*(count_matrix_triplet_markov_1_3 % 2)) / (count_matrix_triplet_markov_1_3 + 1.0)),axis=2)) < 0.0001))
        
          #assert(all(abs(ent_1_2_3_boots - sum((count_matrix_triplet_markov * 1.0 / numangles_bootstrap_tensor_markov) * (log(numangles_bootstrap_tensor_markov) - special.psi(count_matrix_triplet_markov + SMALL) - (1 - 2*(count_matrix_triplet_markov % 2)) / (count_matrix_triplet_markov + 1.0)),axis=2)) < 0.0001))


          mutinf_thisdof = ent_1_2_3_boots - ent_1_2_boots - ent_2_3_boots - ent_1_3_boots + ent1_markov_boots + ent2_markov_boots + ent3_markov_boots
        
          print "ind ent1_boots bootstrap 0: "+str(ent1_boots[0])
          print "ind ent2_boots bootstrap 0: "+str(ent2_boots[0])
          print "ind ent3_boots bootstrap 0: "+str(ent3_boots[0])
          print "ind ent_1_2_boots bootstrap 0: "+str(ent_1_2_boots[0])
          print "ind ent_2_3_boots bootstrap 0: "+str(ent_2_3_boots[0])
          print "ind ent_1_3_boots bootstrap 0: "+str(ent_1_3_boots[0])
          print "ind ent_1_2_3_boots,        0: "+str(ent_1_2_3_boots[0])
          # MI = H(1)+H(2)-H(1,2)
          # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
          #print "Numangles Bootstrap Matrix:\n"+str(numangles_bootstrap_matrix_markov)
        
        
    
    
    
    
            
    
    
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
           num_effective_snapshots = int((1.0 * numangles_bootstrap[0]) / (1.0 + 1.0 * max(bins1_slowest_lagtime[mybootstrap],bins2_slowest_lagtime[mybootstrap], bins3_slowest_lagtime[mybootstrap]))) + 1 # take the longer of the two lagtimes as the effective lagtime, add one to ensure no divide by zero failure
           #mutinf_per_order_expected_variance =  ((nbins*nbins - 1) ** (2)) * 1.0/(2.0 * num_effective_snapshots * num_effective_snapshots)
           #mutinf_per_order_expected          =  ((-1) ** 2) * ( (int(nbins*nbins - 1) ** 2) / (2.0*num_effective_snapshots))
           #print "num effective snapshots: "+str(num_effective_snapshots)
           #print "ind mutinf average: "+str(average(mutinf_thisdof[mybootstrap],axis=0))+" ind mutinf expected from num effective snapshots: "+str(mutinf_per_order_expected)
           #print "ind mutinf variance: "+str(var(mutinf_thisdof))+" ind mutinf variance expected from num effective snapshots: "+str(mutinf_per_order_expected_variance)
           if (mybootstrap == 0 and OUTPUT_INDEPENDENT_MUTINF_VALUES != 0): #global var set by options.output_independent
                  OUTPUT_INDEPENDENT_MUTINF_VALUES = 1 # in case it's something else
                  myfile = open("markov_independent_mutinf_"+str(OUTPUT_INDEPENDENT_MUTINF_VALUES)+"_bootstrap_"+str(mybootstrap)+".txt",'w')
                  for i in range((shape(mutinf_thisdof))[0]):
                         myfile.write(str(mutinf_thisdof[i])+"\n")
                  myfile.close()
    #print "shape of mutinf markov thisdof: "+str(shape(mutinf_thisdof))
    return E_I, Var_I , E_I3, E_I4, Var_I_runs, mutinf_thisdof, var_mi_thisdof


###################################################################################

def calc_triplet_mutinf_multinomial_constrained( nbins, counts1, counts2, counts3, adaptive_partitioning, bootstraps = None ):
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
    ref_counts3 = zeros((bootstrap_sets, nbins),int32)
    
    
    
    chi_counts1 = resize(ref_counts1,(bootstrap_sets,permutations_multinomial,nbins))
    chi_counts2 = resize(ref_counts2,(bootstrap_sets,permutations_multinomial,nbins))
    chi_counts3 = resize(ref_counts3,(bootstrap_sets,permutations_multinomial,nbins))
    chi_countdown1 = zeros((shape(chi_counts1)),float64)
    chi_countdown2 = zeros((shape(chi_counts2)),float64)
    chi_countdown3 = zeros((shape(chi_counts3)),float64)
    
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
          *(ref_counts3 + mybootstrap*nbins + bin) += 1;
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
           *(chi_counts3 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin) = *(ref_counts3 +  mybootstrap*nbins + bin) ; 
           *(chi_countdown3 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin) = *(ref_counts3 +  mybootstrap*nbins + bin) ; 
        }
      }
    }
    """
    if(adaptive_partitioning != 0):
        print "preparing multinomial distribution for two-D independent histograms, given marginal distributions"
        weave.inline(code_create_ref,['nbins','ref_counts1','ref_counts2','ref_counts3','numangles_bootstrap','bootstrap_sets','permutations_multinomial','chi_counts1','chi_counts2','chi_counts3','chi_countdown1','chi_countdown2','chi_countdown3'], compiler=mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
    else:
        ref_counts1 = counts1.copy()
        ref_counts2 = counts2.copy()
    
    
    
    #print "chi_counts1 trial 1 :"
    #print chi_counts1[0,:]
    #print "chi_counts2 trial 1:"
    #print chi_counts2[0,:]
    #print "chi_counts3 trial 1:"
    #print chi_counts3[0,:]
    #chi_countdown1 = zeros((bootstrap_sets,permutations_multinomial,nbins),int32)
    #chi_countdown2 = zeros((bootstrap_sets,permutations_multinomial,nbins),int32)
    chi_countdown1 = chi_counts1.copy()
    chi_countdown2 = chi_counts2.copy()
    chi_countdown3 = chi_counts3.copy()
    #chi_countdown1[:,:] = resize(chi_counts1[0,:].copy(),(bootstrap_sets,nbins)) #replicate up to bootstrap_sets
    #chi_countdown2[:,:] = resize(chi_counts2[0,:].copy(),(bootstrap_sets,nbins)) #replicate up to bootstrap_sets
    
    #print "counts1 to pick from without replacement, first sample"
    #print chi_countdown1[:,0]
    #print "counts1 to pick from without replacement, last sample"
    #print chi_countdown1[:,-1]
    #print "counts3 to pick from without replacement, first sample"
    #print chi_countdown3[:,0]
    #print "counts3 to pick from without replacement, last sample"
    #print chi_countdown3[:,-1]
    
    
    if(True):
           
       # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
       #U = zeros((bootstrap_sets,0 + 1, nbins*nbins), float64)
       #logU = zeros((bootstrap_sets,0 + 1, nbins*nbins),float64)

       count_matrix_multi = zeros((bootstrap_sets, permutations_multinomial , nbins*nbins*nbins), int32)
       count_matrix_multi_1_2   = zeros((bootstrap_sets, permutations_multinomial , nbins*nbins), int32)
       count_matrix_multi_1_3   = zeros((bootstrap_sets, permutations_multinomial , nbins*nbins), int32)
       count_matrix_multi_2_3   = zeros((bootstrap_sets, permutations_multinomial , nbins*nbins), int32)
       #ninj_flat_Bayes = zeros((bootstrap_sets, permutations_multinomial ,nbins*nbins),float64)
           
       
       
       numangles_bootstrap_tensor = zeros((bootstrap_sets,permutations_multinomial,nbins*nbins*nbins),float64)
       numangles_bootstrap_matrix = zeros((bootstrap_sets,permutations_multinomial,nbins*nbins),float64)
       for bootstrap in range(bootstrap_sets):
           for permut in range(permutations_multinomial):
               numangles_bootstrap_tensor[bootstrap,permut,:]=numangles_bootstrap[bootstrap]   
               numangles_bootstrap_matrix[bootstrap,permut,:]=numangles_bootstrap[bootstrap]
    
    chi_counts1_vector = chi_counts1.copy() #reshape(chi_counts1,(bootstrap_sets,permutations_multinomial,nbins)) 
    chi_counts2_vector = chi_counts2.copy() #reshape(chi_counts2,(bootstrap_sets,permutations_multinomial,nbins)) 
    chi_counts3_vector = chi_counts3.copy() #reshape(chi_counts2,(bootstrap_sets,permutations_multinomial,nbins)) 
    numangles_bootstrap_vector = numangles_bootstrap_matrix[:,:,:nbins]    
    count_matrix_multi[:,:,:] = 0
    
    
    
    code_multi = """
    // weave6a
     #include <math.h>
     int bin1, bin2, bin3, bin1_found, bin2_found, bin3_found = 0;
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
             bin3_found = 0;
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
             while(bin3_found == 0) {      //sampling without replacement
                bin3 = int(drand48() * int(nbins));
                if( *(chi_countdown3 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin3) > 0.999) { // the 0.999 is for fractional counts, which come into play with weights
                  bin3_found = 1;
                  // #pragma omp atomic
                  *(chi_countdown3 + mybootstrap*permutations_multinomial*nbins + permut*nbins + bin3) -= 1;
                }
             }
          //printf("bin1 %d bin2 %d bin3 %d\\n", bin1, bin2, bin3);
          // #pragma omp atomic
          *(count_matrix_multi  +  mybootstrap*permutations_multinomial*nbins*nbins*nbins  +  permut*nbins*nbins*nbins  +  bin1*nbins*nbins + bin2*nbins + bin3) += 1;
          *(count_matrix_multi_1_2  +  mybootstrap*permutations_multinomial*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin2) += 1;
          *(count_matrix_multi_2_3  +  mybootstrap*permutations_multinomial*nbins*nbins  +  permut*nbins*nbins  +  bin2*nbins + bin3) += 1;
          *(count_matrix_multi_1_3  +  mybootstrap*permutations_multinomial*nbins*nbins  +  permut*nbins*nbins  +  bin1*nbins + bin3) += 1;
             
          }
        }
       }
      """
    weave.inline(code_multi, ['numangles_bootstrap', 'nbins', 'count_matrix_multi','count_matrix_multi_1_2','count_matrix_multi_2_3','count_matrix_multi_1_3','chi_counts1','chi_counts2','chi_countdown1','chi_countdown2','chi_countdown3','bootstrap_sets','permutations_multinomial'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"])
                 #extra_compile_args =['  -fopenmp -lgomp'],
                 #extra_link_args=['-lgomp'])

    del chi_countdown1 #free up mem
    del chi_countdown2 #free up mem
    del chi_countdown3 #free up mem
    #print "done with count matrix setup for multinomial\n"
    for bootstrap in range(bootstrap_sets):
           for permut in range(permutations_multinomial):
              my_flat = outer(chi_counts1[bootstrap,permut], outer(chi_counts2[bootstrap,permut] + ni_prior,chi_counts3[bootstrap,permut] + ni_prior)).flatten() 
              #my_flat = resize(my_flat,(permutations_multinomial,(my_flat.shape)[0]))
              #ninj_flat_Bayes[bootstrap,permut,:] = my_flat[:] #[:,:]


    if(numangles_bootstrap[0] > 0): #small sample stuff turned off for now cause it's broken
        if(NO_GRASSBERGER == False):    
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):
           ent1_boots = sum((chi_counts1_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_vector + SMALL) - (1 - 2*(chi_counts1_vector % 2)) / (chi_counts1_vector + 1.0)),axis=2)

           ent2_boots = sum((chi_counts2_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_vector + SMALL) - (1 - 2*(chi_counts2_vector % 2)) / (chi_counts2_vector + 1.0)),axis=2)

           ent3_boots = sum((chi_counts3_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts3_vector + SMALL) - (1 - 2*(chi_counts3_vector % 2)) / (chi_counts3_vector + 1.0)),axis=2)

           ent_1_2_boots = sum((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix) * (log(numangles_bootstrap_matrix) - special.psi(count_matrix_multi_1_2 + SMALL) - (1 - 2*(count_matrix_multi_1_2 % 2)) / (count_matrix_multi_1_2 + 1.0)),axis=2) 
        
           ent_2_3_boots = sum((count_matrix_multi_2_3 * 1.0 / numangles_bootstrap_matrix) * (log(numangles_bootstrap_matrix) - special.psi(count_matrix_multi_2_3 + SMALL) - (1 - 2*(count_matrix_multi_2_3 % 2)) / (count_matrix_multi_2_3 + 1.0)),axis=2) 
        
           ent_1_3_boots = sum((count_matrix_multi_1_3 * 1.0 / numangles_bootstrap_matrix) * (log(numangles_bootstrap_matrix) - special.psi(count_matrix_multi_1_3 + SMALL) - (1 - 2*(count_matrix_multi_1_3 % 2)) / (count_matrix_multi_1_3 + 1.0)),axis=2) 

           ent_1_2_3_boots = sum((count_matrix_multi * 1.0 / numangles_bootstrap_tensor) * (log(numangles_bootstrap_tensor) - special.psi(count_matrix_multi + SMALL) - (1 - 2*(count_matrix_multi % 2)) / (count_matrix_multi + 1.0)),axis=2) 
        else:
           ent1_boots = sum((chi_counts1_vector * 1.0 / numangles_bootstrap_vector) * (log((chi_counts1_vector * 1.0 / numangles_bootstrap_vector) + SMALLER))  ,axis=2)

           ent2_boots = sum((chi_counts2_vector * 1.0 / numangles_bootstrap_vector) * (log((chi_counts2_vector * 1.0 / numangles_bootstrap_vector) + SMALLER))  ,axis=2) 

           ent3_boots = sum((chi_counts3_vector * 1.0 / numangles_bootstrap_vector) * (log((chi_counts3_vector * 1.0 / numangles_bootstrap_vector) + SMALLER))  ,axis=2) 

           ent_1_2_boots = sum((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix) * (log((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix))) ,axis=2)
        
           ent_2_3_boots = sum((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix) * (log((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix))) ,axis=2) 
        
           ent_1_3_boots = sum((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix) * (log((count_matrix_multi_1_2 * 1.0 / numangles_bootstrap_matrix))) ,axis=2) 

           ent_1_2_3_boots = sum((count_matrix_multi * 1.0 / numangles_bootstrap_tensor) * (log((count_matrix_multi * 1.0 / numangles_bootstrap_tensor))),axis=2)

           mutinf_thisdof = ent_1_2_3_boots - ent_1_2_boots - ent_2_3_boots - ent_1_3_boots + ent1_boots + ent2_boots + ent3_boots
        

        
        
    
        # MI = H(1)+H(2)-H(1,2)
        # where H(1,2) doesn't need absolute correction related to number of bins and symmetry factors because this cancels out in MI
        #print "Numangles Bootstrap Matrix:\n"+str(numangles_bootstrap_matrix)
        

    else:
        ent1_boots = sum((chi_counts1_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts1_vector,numangles_bootstrap_vector,permutations_multinomial),axis=2)

        ent2_boots = sum((chi_counts2_vector + 1) * (1.0 / numangles_bootstrap_vector) * sumstuff(chi_counts2_vector,numangles_bootstrap_vector,permutations_multinomial),axis=2)

        #print "shapes:"+str(ent1_boots)+" , "+str(ent2_boots)+" , "+str(sum((count_matrix_multi + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix_multi,numangles_bootstrap_matrix,permutations_multinomial),axis=2))
        
        mutinf_thisdof = ent1_boots + ent2_boots \
                         -sum((count_matrix_multi + 1)  * (1.0 / numangles_bootstrap_matrix) * sumstuff(count_matrix_multi,numangles_bootstrap_matrix,permutations_multinomial),axis=2)
        
        print "Descriptive Triplet Mutinf Multinomial:"
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
    


    var_mi_thisdof = zeros((bootstrap_sets), float64)
    #for bootstrap in range(bootstrap_sets):
    #       var_mi_thisdof[bootstrap] = vfast_cov_for1D_boot_multinomial(reshape(mutinf_thisdof[bootstrap,:].copy(),(mutinf_thisdof.shape[1],1)))[0,0]
    
    E_I = zeros((bootstrap_sets),float64)
    Var_I = zeros((bootstrap_sets),float64)
    Var_I_runs = zeros((bootstrap_sets),float64)
    E_I3 = E_I * 0.0 #zeroing now for speed since we're not using these
    E_I4 = E_I * 0.0 #zeroing now for speed since we're not using these
    #del count_matrix_multi_wprior, 1, chi_counts_matrix2, chi_counts_matrix3,
    count_matrix_multi, count_matrix_multi_1_2, count_matrix_multi_2_3, count_matrix_multi_1_3, chi_counts1_vector, chi_counts2_vector, chi_counts3_vector
    return E_I, Var_I , E_I3, E_I4, Var_I_runs, mutinf_thisdof, var_mi_thisdof









###################### TRIPLET MUTINF ################################

print "this is a test"



#min_angles_boot_pair_runs_matrix = None
#min_angles_boot_pair_runs_vector = None
def calc_triplet_mutinf_corrected(chi_counts1, chi_counts2, chi_counts3, bins1, bins2, bins3, chi_counts_sequential1, chi_counts_sequential2, chi_counts_sequential3, bins1_sequential, bins2_sequential, bins3_sequential, num_sims, nbins, numangles_bootstrap, numangles, calc_variance=False,bootstrap_choose=0,permutations=0,which_runs=None,pair_runs=None, calc_mutinf_between_sims="yes", markov_samples = 0, chi_counts1_markov=None, chi_counts2_markov=None, chi_counts3_markov=None,  ent1_markov_boots=None, ent2_markov_boots=None, ent3_markov_boots=None, bins1_markov=None, bins2_markov=None, bins3_markov=None, file_prefix=None, plot_2d_histograms=False, adaptive_partitioning = 0, lagtime_interval=None, bins1_slowest_timescale = None, bins2_slowest_timescale = None, bins3_slowest_timescale=None, bins1_slowest_lagtime = None, bins2_slowest_lagtime = None, bins3_slowest_lagtime=None, boot_weights = None, weights = None, num_convergence_points=None ):
    global count_matrix_triplet, count_matrix_1_2, count_matrix_1_3, count_matrix_2_3, count_matrix_1, count_matrix_2, count_matrix_3, count_matrix_triplet_sequential #, ninj_flat_Bayes, ninj_flat_Bayes_sequential # , ninj_flat
    global nbins_cor, min_angles_boot_pair_runs , numangles_bootstrap_tensor, numangles_bootstrap_matrix, numangles_boot_pair_runs_matrix, numangles_bootstrap_vector
    global min_angles_boot_pair_runs_matrix, min_angles_boot_pair_runs_vector
    global E_I_triplet_multinomial, Var_I_triplet_multinomial,  Var_I_runs_multinomial 
    global mutinf_triplet_multinomial, var_mutinf_triplet_multinomial, mutinf_triplet_multinomial_sequential, var_mutinf_triplet_multinomial_sequential
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

    markov_interval = zeros((bootstrap_sets), int16)
    if(markov_samples > 0 and bins1_slowest_lagtime != None and bins2_slowest_lagtime != None and bins3_slowest_lagtime != None ):
           for bootstrap in range(bootstrap_sets):
                  max_lagtime = max(bins1_slowest_lagtime[bootstrap], bins2_slowest_lagtime[bootstrap], bins3_slowest_lagtime[bootstrap])
                  markov_interval[bootstrap] = int(max_num_angles / max_lagtime) #interval for final mutual information calc, based on max lagtime
    else:
           markov_interval[:] = 1
    
    markov_interval[:] = 1
    num_pair_runs = pair_runs.shape[1]
    nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    if(VERBOSE >=2):
           print "nbins_cor: "+str(nbins_cor)
           print pair_runs
           print "bootstrap_sets: "+str(bootstrap_sets)+"num_pair_runs: "+str(num_pair_runs)+"\n"

    #print numangles_bootstrap
    #print bins1.shape
    
    #assert(bootstrap_sets == pair_runs.shape[0] == chi_counts1.shape[0] == chi_counts2.shape[0] == bins1.shape[0] == bins2.shape[0])
    
    pvalue = zeros((bootstrap_sets),float64)

    #must be careful to zero discrete histograms that we'll put data in from weaves
    if VERBOSE >= 2:
           print "chi counts 1 before triplet_multinomial:"
           print chi_counts1
           print "chi counts 2 before triplet_multinomial:"
           print chi_counts2
           print "chi counts 3 before triplet_multinomial:"
           print chi_counts2
    #print "mutinf triplet_multinomial: "+str(mutinf_triplet_multinomial)
    #initialize if permutations > 0
    if(permutations > 0):
           mutinf_triplet_multinomial = zeros((bootstrap_sets,1),float64)
           mutinf_triplet_multinomial_sequential = zeros((bootstrap_sets,1),float64)
           var_mutinf_triplet_multinomial_sequential = zeros((bootstrap_sets,1), float64)
    #only do multinomial if not doing permutations
    if(mutinf_triplet_multinomial is None and permutations == 0 and markov_samples == 0 ):
        E_I_triplet_multinomial, Var_I_triplet_multinomial, E_I3_triplet_multinomial, E_I4_triplet_multinomial, Var_I_runs_triplet_multinomial, \
                         mutinf_triplet_multinomial, var_mutinf_triplet_multinomial = \
                         calc_triplet_mutinf_multinomial_constrained(nbins,chi_counts1,chi_counts2,chi_counts3,adaptive_partitioning )
    #print "mutinf triplet_multinomial after run: "+str(mutinf_triplet_multinomial)
    if(permutations == 0 and markov_samples > 0 ): #run mutinf for independent markov samples for every dihedral
        E_I_triplet_multinomial, Var_I_triplet_multinomial, E_I3_triplet_multinomial, E_I4_triplet_multinomial, Var_I_runs_triplet_multinomial, \
                         mutinf_triplet_multinomial, var_mutinf_triplet_multinomial = \
                         calc_triplet_mutinf_markov_independent(nbins,chi_counts1_markov, chi_counts2_markov, chi_counts3_markov, ent1_markov_boots, ent2_markov_boots, ent3_markov_boots, bins1_markov,bins2_markov, bins3_markov, bootstrap_sets, bootstrap_choose, markov_samples, max_num_angles, numangles_bootstrap, bins1_slowest_lagtime, bins2_slowest_lagtime, bins3_slowest_lagtime)
    
    #NOTE: If markov_samples == 0, shape of mutinf_multinomial is (bootstrap_sets, 1). If markov_samples > 0, shape of mutinf_multinomial is (bootstrap_sets, markov_samples). 

    total_permutations = (permutations + 1) * (permutations + 1)  #includes no permutations case for indices 2 and 3
    
    if count_matrix_triplet is None:    

       count_matrix_triplet = zeros((bootstrap_sets, total_permutations , nbins*nbins*nbins), float64)

       #1-D marginals
       count_matrix_1 = zeros((bootstrap_sets, total_permutations, nbins), float64)
       count_matrix_2 = zeros((bootstrap_sets, total_permutations, nbins), float64)
       count_matrix_3 = zeros((bootstrap_sets, total_permutations, nbins), float64)
       
       #2-D marginals
       count_matrix_1_2 = zeros((bootstrap_sets, total_permutations , nbins*nbins), float64)
       count_matrix_2_3 = zeros((bootstrap_sets, total_permutations , nbins*nbins), float64)
       count_matrix_1_3 = zeros((bootstrap_sets, total_permutations , nbins*nbins), float64)

       #count_matrix_triplet_sequential = zeros((bootstrap_sets, num_pair_runs,permutations_sequential + 1 , nbins_cor*nbins_cor), float64)
       #min_angles_boot_pair_runs_matrix = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor*nbins_cor),int32)
       #min_angles_boot_pair_runs_vector = zeros((bootstrap_sets,num_pair_runs,permutations_sequential + 1,nbins_cor),int32)
       #for bootstrap in range(bootstrap_sets):
            #for which_pair in range(num_pair_runs):
            #    if VERBOSE >= 2:
            #           print "run1 "+str(pair_runs[bootstrap,which_pair,0])+" run2 "+str(pair_runs[bootstrap,which_pair,1])
            #           print "numangles shape:" +str(numangles.shape)
            #    #my_flat = outer(chi_counts_sequential1[pair_runs[bootstrap,which_pair,0]],chi_counts_sequential2[pair_runs[bootstrap,which_pair,1]]).flatten()
            #    #my_flat = resize(my_flat,(permutations + 1,(my_flat.shape)[0])) #replicate original data over n permutations
            #    min_angles_boot_pair_runs_matrix[bootstrap,which_pair,:,:] = resize(array(min(numangles[pair_runs[bootstrap,which_pair,0]],numangles[pair_runs[bootstrap,which_pair,1]]),int32),(permutations_sequential + 1, nbins_cor*nbins_cor)) #replicate original data over permutations and nbins_cor*nbins_cor
            #    min_angles_boot_pair_runs_vector[bootstrap,which_pair,:,:] = resize(array(min(numangles[pair_runs[bootstrap,which_pair,0]],numangles[pair_runs[bootstrap,which_pair,1]]),int32),(nbins_cor)) #replicate original data over nbins_cor, no permutations for marginal dist.            
            #    #ninj_flat_Bayes_sequential[bootstrap,which_pair,:,:] = my_flat[:,:]

           
           
       
    if(numangles_bootstrap_matrix is None):
        print "numangles bootstrap: "+str(numangles_bootstrap)
        numangles_bootstrap_tensor = zeros((bootstrap_sets,total_permutations,nbins*nbins*nbins),float64)
        numangles_bootstrap_matrix = zeros((bootstrap_sets,total_permutations,nbins*nbins),float64)
        for bootstrap in range(bootstrap_sets):
            for permut in range(total_permutations):
                numangles_bootstrap_matrix[bootstrap,permut,:]=numangles_bootstrap[bootstrap]
                numangles_bootstrap_tensor[bootstrap,permut,:]=numangles_bootstrap[bootstrap]
        numangles_bootstrap_vector = numangles_bootstrap_matrix[:,:,:nbins]
        #print "Numangles Bootstrap Vector:\n"+str(numangles_bootstrap_vector)
    
    #if(chi_counts2_matrix is None):


    
    
    
    count_matrix_triplet[:,:,:] = 0
    count_matrix_1[:,:,:] = 0
    count_matrix_2[:,:,:] = 0
    count_matrix_3[:,:,:] = 0
    count_matrix_1_2[:,:,:] = 0
    count_matrix_2_3[:,:,:] = 0
    count_matrix_1_3[:,:,:] = 0

    ent_1_boots=zeros((bootstrap_sets, (total_permutations)), float64)
    ent_2_boots=zeros((bootstrap_sets, (total_permutations)), float64)
    ent_3_boots=zeros((bootstrap_sets, (total_permutations)), float64)
    
    ent_1_2_boots=zeros((bootstrap_sets, (total_permutations)), float64)
    ent_2_3_boots=zeros((bootstrap_sets, (total_permutations)), float64)
    ent_1_3_boots=zeros((bootstrap_sets, (total_permutations)), float64)

    ent_1_2_3_boots=zeros((bootstrap_sets, (total_permutations)), float64)

    #count_matrix_triplet_sequential[:,:,:,:] =0
    # 0 is a placeholder for permuations, which are not performed here; instead, analytical corrections are used
    

    
    
   # for bootstrap in range(bootstrap_sets): #copy here is critical to not change original arrays
   #     chi_counts2_matrix[bootstrap,0,:] = repeat(chi_counts2[bootstrap].copy(),nbins,axis=-1) #just replicate along fastest-varying axis, this works because nbins is the same for both i and j
   #     #print repeat(chi_counts2[bootstrap] + ni_prior,nbins,axis=0)
   #     #handling the slower-varying index will be a little bit more tricky
   #     chi_counts1_matrix[bootstrap,0,:] = (transpose(reshape(resize(chi_counts1[bootstrap].copy(), nbins*nbins),(nbins,nbins)))).flatten() # replicate along fastest-varying axis, convert into a 2x2 matrix, take the transpose)
    
    no_boot_weights = False
    if (boot_weights is None):
           boot_weights = ones((bootstrap_sets, max_num_angles * bootstrap_choose), float64)
           no_boot_weights = True
    code = """
    // weave6
    // bins dimensions: (permutations + 1) * bootstrap_sets * bootstrap_choose * max_num_angles
     //#include <math.h>
     double weight, weight2, weight3;
     int angle1_bin = 0;
     int angle2_bin = 0;
     int angle3_bin = 0;
     int bin1, bin2, bin3 =0;
     int mybootstrap,permut;
     long anglenum;
     int permut1, permut2 = 0;
     long mynumangles;
     long  offset1, offset2, offset3, offset4, counts1, counts2, counts3, counts12, counts23, counts13, counts123 = 0;
     double mysign1, mysign2, mysign3, mysign12, mysign23, mysign13, mysign123, dig1, dig2, dig3, dig12, dig23, dig13, dig123, counts1d, counts12d, counts23d, counts13d, counts123d;

     #pragma omp parallel for private(mybootstrap,permut1, permut2, permut, mynumangles, anglenum, angle1_bin, angle2_bin, angle3_bin, weight, weight2, weight3,counts1, counts2, counts3, counts12, counts13, counts23, counts123, dig1, dig2, dig3, dig12, dig23, dig13, dig123, counts1d, counts12d, counts23d, counts13d, counts123d, bin1, bin2, bin3,mysign1, mysign2, mysign3, mysign12, mysign23, mysign13, mysign123) 
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      for (permut1=0; permut1 < permutations + 1; permut1++) {
       for(permut2=0; permut2 < permutations + 1; permut2++) { 
          permut = permut1*(permutations + 1 ) + permut2; //output permutation's index. index3 must be cyclicly permuted with respect to the others 
          mynumangles = *(numangles_bootstrap + mybootstrap);
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          if(anglenum == mynumangles - 1) {
            printf(""); 
            // the command above is just to stall for a little time and make sure that the code behaves when compiled and finishes writing to the arrays
            //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
            }
             //if(anglenum % markov_interval[mybootstrap] == 0) {
              angle1_bin = *(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum);
              angle2_bin = *(bins2 + permut1*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              angle3_bin = *(bins3 + permut2*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              weight2 = weight * weight;
              weight3 = weight2 * weight;

              *(count_matrix_1  + mybootstrap*(total_permutations)*nbins + permut*nbins +  angle1_bin  ) += 1.0 * weight ;
              *(count_matrix_2  + mybootstrap*(total_permutations)*nbins + permut*nbins +  angle2_bin  ) += 1.0 * weight ;
              *(count_matrix_3  + mybootstrap*(total_permutations)*nbins + permut*nbins +  angle3_bin  ) += 1.0 * weight ;

              *(count_matrix_1_2  + mybootstrap*(total_permutations)*nbins*nbins + permut*nbins*nbins +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight2 ;
              *(count_matrix_2_3  + mybootstrap*(total_permutations)*nbins*nbins + permut*nbins*nbins +  angle2_bin*nbins +   angle3_bin ) += 1.0 * weight2 ;
              *(count_matrix_1_3  + mybootstrap*(total_permutations)*nbins*nbins + permut*nbins*nbins +  angle1_bin*nbins +   angle3_bin ) += 1.0 * weight2 ;

              *(count_matrix_triplet  + (long)(mybootstrap*(total_permutations)*nbins*nbins*nbins +  permut*nbins*nbins*nbins  +  angle1_bin*nbins*nbins +   angle2_bin*nbins + angle3_bin )) += 1.0 * weight3;
             //}
            }


          for(bin1=0; bin1 < nbins; bin1++) 
            { 

               counts1 = *(count_matrix_1 + (long)(mybootstrap*(total_permutations)*nbins + permut*nbins + bin1)) ;


                     if(counts1 > 0)
                     {
                       mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                       dig1 = xDiGamma_Function(counts1);
                       mysign1 = 1 - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                       counts1d = 1.0 * counts1 ;
                       *(ent_1_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)counts1d + 1.0))); 
                     }

               counts2 = *(count_matrix_2 + (long)(mybootstrap*(total_permutations)*nbins + permut*nbins + bin1)) ;
               
               


                     if(counts2 > 0)
                     {
                       mysign1 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                       dig1 = xDiGamma_Function(counts2);
                       mysign1 = 1 - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                       counts1d = 1.0 * counts2 ;
                       *(ent_2_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)counts1d + 1.0))); 
                     }
               
               counts3 = *(count_matrix_3 + (long)(mybootstrap*(total_permutations)*nbins + permut*nbins + bin1)) ;
               
             

                     if(counts3 > 0)
                     {
                       mysign1 = 1.0L - 2*(counts3 % 2); // == -1 if it is odd, 1 if it is even
                       dig1 = xDiGamma_Function(counts3);
                       mysign1 = 1 - 2*(counts3 % 2); // == -1 if it is odd, 1 if it is even
                       counts1d = 1.0 * counts3 ;
                       *(ent_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts1d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)counts1d + 1.0))); 
                     }
  
             


             for(bin2=0; bin2 < nbins; bin2++)
              {
         
                // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
                counts12 = *(count_matrix_1_2  + (long)( mybootstrap*((total_permutations))*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 ));
                
                

               if(counts12 > 0)
                {
                 mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                 dig1 = xDiGamma_Function(counts12);
                 counts12d = 1.0 * counts12;
                 *(ent_1_2_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts12d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts12d + 1.0)))); 
                }

               counts23 = *(count_matrix_2_3  + (long)( mybootstrap*((total_permutations))*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 ));
               if(counts23 > 0)
                {
                 mysign1 = 1.0L - 2*(counts23 % 2); // == -1 if it is odd, 1 if it is even
                 dig1 = xDiGamma_Function(counts23);
                 counts23d = 1.0 * counts23;
                 *(ent_2_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts23d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts23d + 1.0)))); 
                }

               counts13 = *(count_matrix_1_3  + (long)( mybootstrap*((total_permutations))*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 ));
               if(counts13 > 0)
                {
                 mysign1 = 1.0L - 2*(counts13 % 2); // == -1 if it is odd, 1 if it is even
                 dig1 = xDiGamma_Function(counts13);
                 counts13d = 1.0 * counts13;
                 *(ent_1_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts13d / mynumangles)*(log(mynumangles) - dig1 - ((double)mysign1 / ((double)(counts13d + 1.0)))); 
                }




                }
               }




              for(bin1=0; bin1 < nbins; bin1++) 
              { 
                for(bin2=0; bin2 < nbins; bin2++)
                {
                 for(bin3=0; bin3 < nbins; bin3++)
                  { 
                    counts123 = int(*(count_matrix_triplet  +  (long)(mybootstrap*((total_permutations))*nbins*nbins*nbins   +  permut*nbins*nbins*nbins  +  bin1*nbins*nbins + bin2*nbins + bin3 )));


                     if(counts123 > 0)
                     {
                       mysign123 = 1.0L - 2*(counts123 % 2); // == -1 if it is odd, 1 if it is even
                       dig123 = xDiGamma_Function(counts123);
                       mysign123 = 1 - 2*(counts123 % 2); // == -1 if it is odd, 1 if it is even
                       counts123d = 1.0 * counts123 ;
                       *(ent_1_2_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += ((double)counts123d / mynumangles)*(log(mynumangles) - dig123 - ((double)mysign123 / ((double)counts123d + 1.0))); 
                     }

                }
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
     double weight, weight2, weight3;
     int angle1_bin = 0;
     int angle2_bin = 0;
     int angle3_bin = 0;
     int bin1, bin2, bin3 =0;
     int mybootstrap,permut;
     long anglenum;
     int permut1, permut2 = 0;
     long mynumangles;
     long  offset1, offset2, offset3, offset4, counts1, counts2, counts3, counts12, counts23, counts13, counts123 = 0;
     double mysign1, mysign2, mysign3, mysign12, mysign23, mysign13, mysign123, dig1, dig2, dig3, dig12, dig23, dig13, dig123, counts1d, counts12d, counts23d, counts13d, counts123d;

     #pragma omp parallel for private(mybootstrap,permut1, permut2, permut, mynumangles, anglenum, angle1_bin, angle2_bin, angle3_bin, weight, weight2, weight3,counts1, counts2, counts3, counts12, counts13, counts23, counts123, dig1, dig2, dig3, dig12, dig23, dig13, dig123, counts1d, counts12d, counts23d, counts13d, counts123d, bin1, bin2, bin3,mysign1, mysign2, mysign3, mysign12, mysign23, mysign13, mysign123) 
     for(mybootstrap=0; mybootstrap < bootstrap_sets; mybootstrap++) {
      mynumangles = 0;
      for (permut1=0; permut1 < permutations + 1; permut1++) {
       for(permut2=0; permut2 < permutations + 1; permut2++) { 
          permut = permut1*(permutations + 1 ) + permut2; //output permutation's index. index3 must be cyclicly permuted with respect to the others 
          mynumangles = *(numangles_bootstrap + mybootstrap);
          for (anglenum=offset; anglenum< mynumangles; anglenum++) {
          if(anglenum == mynumangles - 1) {
            printf(""); 
            // the command above is just to stall for a little time and make sure that the code behaves when compiled and finishes writing to the arrays
            //printf("bin12 %i \\n",(*(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum))*nbins +   (*(bins2 + permut*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum)));
            }
             //if(anglenum % markov_interval[mybootstrap] == 0) {
              angle1_bin = *(bins1  +  mybootstrap*bootstrap_choose*max_num_angles  +  anglenum);
              angle2_bin = *(bins2 + permut1*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              angle3_bin = *(bins3 + permut2*bootstrap_sets*bootstrap_choose*max_num_angles + mybootstrap*bootstrap_choose*max_num_angles  +  anglenum - offset);
              weight = *(boot_weights + mybootstrap*bootstrap_choose*max_num_angles + anglenum); //assumes mynumangles same for all dihedrals, for nonzero offsets assumes equal weights
              weight2 = weight * weight;
              weight3 = weight2 * weight;

              *(count_matrix_1  + mybootstrap*(total_permutations)*nbins + permut*nbins +  angle1_bin  ) += 1.0 * weight ;
              *(count_matrix_2  + mybootstrap*(total_permutations)*nbins + permut*nbins +  angle2_bin  ) += 1.0 * weight ;
              *(count_matrix_3  + mybootstrap*(total_permutations)*nbins + permut*nbins +  angle3_bin  ) += 1.0 * weight ;

              *(count_matrix_1_2  + mybootstrap*(total_permutations)*nbins*nbins + permut*nbins*nbins +  angle1_bin*nbins +   angle2_bin ) += 1.0 * weight2 ;
              *(count_matrix_2_3  + mybootstrap*(total_permutations)*nbins*nbins + permut*nbins*nbins +  angle2_bin*nbins +   angle3_bin ) += 1.0 * weight2 ;
              *(count_matrix_1_3  + mybootstrap*(total_permutations)*nbins*nbins + permut*nbins*nbins +  angle1_bin*nbins +   angle3_bin ) += 1.0 * weight2 ;

              *(count_matrix_triplet  + (long)(mybootstrap*(total_permutations)*nbins*nbins*nbins +  permut*nbins*nbins*nbins  +  angle1_bin*nbins*nbins +   angle2_bin*nbins + angle3_bin )) += 1.0 * weight3;
             //}
            }


          for(bin1=0; bin1 < nbins; bin1++) 
            { 

               counts1 = *(count_matrix_1 + (long)(mybootstrap*(total_permutations)*nbins + permut*nbins + bin1)) ;


                     if(counts1 > 0)
                     {
                       mysign1 = 1.0L - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                       dig1 = xDiGamma_Function(counts1);
                       mysign1 = 1 - 2*(counts1 % 2); // == -1 if it is odd, 1 if it is even
                       counts1d = 1.0 * counts1 ;
                       *(ent_1_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALL)); 
                     }

               counts2 = *(count_matrix_2 + (long)(mybootstrap*(total_permutations)*nbins + permut*nbins + bin1)) ;
               
               


                     if(counts2 > 0)
                     {
                       mysign1 = 1.0L - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                       dig1 = xDiGamma_Function(counts2);
                       mysign1 = 1 - 2*(counts2 % 2); // == -1 if it is odd, 1 if it is even
                       counts1d = 1.0 * counts2 ;
                       *(ent_2_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALL)); 
                     }
               
               counts3 = *(count_matrix_3 + (long)(mybootstrap*(total_permutations)*nbins + permut*nbins + bin1)) ;
               
             

                     if(counts3 > 0)
                     {
                       mysign1 = 1.0L - 2*(counts3 % 2); // == -1 if it is odd, 1 if it is even
                       dig1 = xDiGamma_Function(counts3);
                       mysign1 = 1 - 2*(counts3 % 2); // == -1 if it is odd, 1 if it is even
                       counts1d = 1.0 * counts3 ;
                       *(ent_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts1d / mynumangles)*(log((double)counts1d / mynumangles + SMALL)); 
                     }
  
             


             for(bin2=0; bin2 < nbins; bin2++)
              {
         
                // ent1_boots = sum((chi_counts1_markov * 1.0 / numangles_bootstrap) * (log(numangles_bootstrap) - special.psi(chi_counts1_markov + SMALL) - ((-1) ** (chi_counts1_markov % 2)) / (chi_counts1_markov + 1.0)),axis=2)
                counts12 = *(count_matrix_1_2  + (long)( mybootstrap*((total_permutations))*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 ));
                
                

               if(counts12 > 0)
                {
                 mysign1 = 1.0L - 2*(counts12 % 2); // == -1 if it is odd, 1 if it is even
                 dig1 = xDiGamma_Function(counts12);
                 counts12d = 1.0 * counts12;
                 *(ent_1_2_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts12d / mynumangles)*(log((double)counts12d / mynumangles + SMALL)); 
                }

               counts23 = *(count_matrix_2_3  + (long)( mybootstrap*((total_permutations))*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 ));
               if(counts23 > 0)
                {
                 mysign1 = 1.0L - 2*(counts23 % 2); // == -1 if it is odd, 1 if it is even
                 dig1 = xDiGamma_Function(counts23);
                 counts23d = 1.0 * counts23;
                 *(ent_2_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts23d / mynumangles)*(log((double)counts23d / mynumangles + SMALL)); 
                }

               counts13 = *(count_matrix_1_3  + (long)( mybootstrap*((total_permutations))*nbins*nbins   +  permut*nbins*nbins  +  bin1*nbins + bin2 ));
               if(counts13 > 0)
                {
                 mysign1 = 1.0L - 2*(counts13 % 2); // == -1 if it is odd, 1 if it is even
                 dig1 = xDiGamma_Function(counts13);
                 counts13d = 1.0 * counts13;
                 *(ent_1_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts13d / mynumangles)*(log((double)counts13d / mynumangles + SMALL)); 
                }




                }
               }




              for(bin1=0; bin1 < nbins; bin1++) 
              { 
                for(bin2=0; bin2 < nbins; bin2++)
                {
                 for(bin3=0; bin3 < nbins; bin3++)
                  { 
                    counts123 = int(*(count_matrix_triplet  +  (long)(mybootstrap*((total_permutations))*nbins*nbins*nbins   +  permut*nbins*nbins*nbins  +  bin1*nbins*nbins + bin2*nbins + bin3 )));


                     if(counts123 > 0)
                     {
                       mysign123 = 1.0L - 2*(counts123 % 2); // == -1 if it is odd, 1 if it is even
                       dig123 = xDiGamma_Function(counts123);
                       mysign123 = 1 - 2*(counts123 % 2); // == -1 if it is odd, 1 if it is even
                       counts123d = 1.0 * counts123 ;
                       *(ent_1_2_3_boots + (long)(mybootstrap*(total_permutations) + permut)) += -1.0 * ((double)counts123d / mynumangles)*(log((double)counts123d / mynumangles + SMALL)); 
                     }

                }
               }
              }
    
             
         } 
        }
       }
      """
    #
    #
    #
    #
    if(VERBOSE >= 2): print "about to populate count_matrix_triplet"
    if(NO_GRASSBERGER == False):
           weave.inline(code, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'bins3', 'count_matrix_triplet','count_matrix_1_2', 'count_matrix_2_3', 'count_matrix_1_3', 'count_matrix_1', 'count_matrix_2', 'count_matrix_3', 'bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset','markov_interval','chi_counts1','chi_counts2','chi_counts3','ent_1_boots','ent_2_boots','ent_3_boots','ent_1_2_boots','ent_2_3_boots','ent_1_3_boots','ent_1_2_3_boots','total_permutations'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"], extra_compile_args =['  -fopenmp -lgomp'], extra_link_args=['-lgomp'], support_code=my_support_code)
    else:
           weave.inline(code_no_grassberger, ['num_sims', 'numangles_bootstrap', 'nbins', 'bins1', 'bins2', 'bins3', 'count_matrix_triplet','count_matrix_1_2', 'count_matrix_2_3', 'count_matrix_1_3', 'count_matrix_1', 'count_matrix_2', 'count_matrix_3', 'bootstrap_sets','permutations','max_num_angles','bootstrap_choose','boot_weights','offset','markov_interval','chi_counts1','chi_counts2','chi_counts3','ent_1_boots','ent_2_boots','ent_3_boots','ent_1_2_boots','ent_2_3_boots','ent_1_3_boots','ent_1_2_3_boots','total_permutations','SMALL'],
                 #type_converters = converters.blitz,
                 compiler = mycompiler,runtime_library_dirs=["/usr/lib/x86_64-linux-gnu/"], library_dirs=["/usr/lib/x86_64-linux-gnu/"], libraries=["stdc++"], extra_compile_args =['  -fopenmp -lgomp'], extra_link_args=['-lgomp'], support_code=my_support_code)
    #if (no_boot_weights != False ):
    # scale histogram counts to sum to bootstrap_choose*max_angles , since we multiplied by the product of weights
    for mybootstrap in range(bootstrap_sets):
           print "count matrix triplet, bootstrap: "+str(mybootstrap)
           print count_matrix_triplet[mybootstrap,0,:]
           #print count_matrix_triplet[mybootstrap,1,:]
           print count_matrix_triplet[mybootstrap,-1,:]
           angles_this_boot = sum(count_matrix_triplet[mybootstrap,0,:])
           print "angles this bootstrap: "+str(angles_this_boot)
           count_matrix_1[mybootstrap,:,:]       *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
           count_matrix_2[mybootstrap,:,:]       *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
           count_matrix_3[mybootstrap,:,:]       *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
           count_matrix_1_2[mybootstrap,:,:]     *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
           count_matrix_1_3[mybootstrap,:,:]     *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
           count_matrix_2_3[mybootstrap,:,:]     *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
           count_matrix_triplet[mybootstrap,:,:] *= (numangles_bootstrap[mybootstrap] * 1.0) / angles_this_boot
                  

    #chi_counts1_vector = reshape(chi_counts1.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions...
    #chi_counts2_vector = reshape(chi_counts2.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions..
    #chi_counts3_vector = reshape(chi_counts3.copy(),(bootstrap_sets,0 + 1,nbins)) #no permutations for marginal distributions..
    #chi_counts1_vector = repeat(chi_counts1_vector, total_permutations, axis=1)     #but we need to repeat along permutations axis
    #chi_counts2_vector = repeat(chi_counts2_vector, total_permutations, axis=1)     #but we need to repeat along permutations axis
    #chi_counts3_vector = repeat(chi_counts3_vector, total_permutations, axis=1)     #but we need to repeat along permutations axis

    

    print "markov intervals :", str(markov_interval)

    if(VERBOSE >=2):
           print "markov intervals :", str(markov_interval)
           print "count matrix first pass:"
           print count_matrix_triplet[0,0,:]
    
    
    ### Redundant Sanity checks
    if(VERBOSE >=2 ):
     ninj_flat = zeros((bootstrap_sets,total_permutations,nbins*nbins*nbins),float64)
     #ninj_flat_Bayes = zeros((bootstrap_sets,permutations + 1,nbins*nbins),float64)
     for bootstrap in range(bootstrap_sets):
        my_flat = outer(outer(chi_counts1[bootstrap] + 0.0 ,chi_counts2[bootstrap] + 0.0), chi_counts3[bootstrap]).flatten() # have to add 0.0 for outer() to work reliably
        if(VERBOSE >=1):
               assert(all(my_flat >= 0))
        my_flat = resize(my_flat,(total_permutations,(my_flat.shape)[0]))
        ninj_flat[bootstrap,:,:] = my_flat[:,:]
     #    #now without the Bayes prior added into the marginal distribution
     #    my_flat_Bayes = outer(chi_counts1[bootstrap] + ni_prior,chi_counts2[bootstrap] + ni_prior).flatten() 
     #    my_flat_Bayes = resize(my_flat_Bayes,(0 + 1,(my_flat_Bayes.shape)[0]))
     #    ninj_flat_Bayes[bootstrap,:,:] = my_flat_Bayes[:,:]
     #    nbins_cor = int(nbins * FEWER_COR_BTW_BINS)
    
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix_triplet[:,:,:] == 0)) and (sum(chi_counts1) > 0) and (sum(chi_counts2) > 0) and (sum(chi_counts3) > 0):
           count_matrix_triplet[:,:,:] =  ((outer((outer(chi_counts1[bootstrap] ,chi_counts2[bootstrap] )), chi_counts3[bootstrap])).flatten()  ) / (numangles_bootstrap[0] * 1.0)
    
    if(VERBOSE >=2):
        assert(all(ninj_flat >= 0))
        Pij, PiPj = zeros((nbins, nbins), float64)  , zeros((nbins, nbins), float64)  
        Pijk, PiPjPk = zeros((nbins, nbins, nbins), float64)  , zeros((nbins, nbins, nbins), float64)  
        permutation = 0
        #PiPj[1:,1:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins,nbins))
        Pijk[:,:,:]  = (count_matrix_triplet[0,permutation,:]).reshape((nbins,nbins,nbins))  / ((numangles_bootstrap[0] / (markov_interval[0] * 1.0)) * 1.0)
        PiPjPk[:,:,:] = (ninj_flat[0,permutation,:]).reshape((nbins,nbins,nbins)) / ((numangles_bootstrap[0] / (markov_interval[0] * 1.0)) * 1.0) #convert sum of chi_counts^3 to sum of 1
        PiPjPk[:,:,:] /= (numangles_bootstrap[0] / (markov_interval[0] * 1.0)) #to prevent integer overflow
        PiPjPk[:,:,:] /= numangles_bootstrap[0] / (markov_interval[0] * 1.0) #to prevent integer overflow
        PiPj[:,:] = sum(PiPjPk[:,:,:], axis=2)
        Pij[:,:]  = sum(Pijk[:,:,:], axis=2)
    if(VERBOSE >= 2):
        print "First Pass:"
        print "Sum Pij: "+str(sum(Pij[:,:]))+" Sum PiPj: "+str(sum(PiPj[:,:]))
        print "Sum Pijk: "+str(sum(Pijk[:,:]))+" Sum PiPjPk: "+str(sum(PiPjPk[:,:]))
        print "Marginal Pij, summed over j:\n"
        print sum(Pij[:,:],axis=1)
        print "Marginal PiPj, summed over j:\n"
        print sum(PiPj[:,:],axis=1)   
        print "Marginal Pij, summed over i:\n"
        print sum(Pij[:,:],axis=0)
        print "Marginal PiPj, summed over i:\n"
        print sum(PiPj[:,:],axis=0)
    ### end redundant sanity checks
        
        
    #print floor(sum(Pij[:,:],axis=1)) == floor(sum(PiPj[:,:],axis=1))
    if(VERBOSE >=1):
           assert(abs(sum(Pij[:,:]) - sum(PiPj[:,:])) < SMALL)
           assert(all(abs(sum(Pij[:,:],axis=1) - sum(PiPj[:,:],axis=1)) < SMALL))
           assert(all(abs(sum(Pij[:,:],axis=0) - sum(PiPj[:,:],axis=0)) < SMALL))
           assert(all(abs(sum(sum(Pijk[:,:],axis=0),axis=0) - sum(sum(PiPjPk[:,:,:],axis=0),axis=0)) < SMALL)) #second axis here is actual axis number minus 1
           assert(all(abs(sum(sum(Pijk[:,:],axis=1),axis=1) - sum(sum(PiPjPk[:,:,:],axis=1),axis=1)) < SMALL)) #second axis here is actual axis number minus 1
           assert(all(abs(sum(sum(Pijk[:,:],axis=0),axis=1) - sum(sum(PiPjPk[:,:,:],axis=0),axis=1)) < SMALL)) #second axis here is actual axis number minus 1
    
    ## for missing side chains for ALA, GLY, for example, if count matrix is zero but we have chi_counts, then just stick all counts in first 2-D bin
    if(all(count_matrix_triplet[:,:,:] == 0)) and (sum(chi_counts1) > 0) and (sum(chi_counts2) > 0):
           count_matrix_triplet[:,:,:] =  (outer(chi_counts1[bootstrap] ,outer(chi_counts2[bootstrap], chi_counts3[bootstrap])).flatten()  ) / (numangles_bootstrap[0] * 1.0)

    ##################################################################

    #For coding simplicity, now just take data every markov interval -- minimum of the three lagtimes -- for the calculation of triplet mutual information
    
    #if(numangles_bootstrap[0] > 0 or nbins >= 6):
    if(True): #small sample stuff turned off for now because it's broken
    #if(numangles_bootstrap[0] > 1000 and nbins >= 6):  
          #ent1_boots = sum((chi_counts1_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts1_vector + SMALL) - (1 - 2*(chi_counts1_vector % 2)) / (chi_counts1_vector + 1.0)),axis=2) 
        
          #ent2_boots = sum((chi_counts2_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts2_vector + SMALL) - (1 - 2*(chi_counts2_vector % 2)) / (chi_counts2_vector + 1.0)),axis=2)

          #ent3_boots = sum((chi_counts3_vector * 1.0 / numangles_bootstrap_vector) * (log(numangles_bootstrap_vector) - special.psi(chi_counts3_vector + SMALL) - (1 - 2*(chi_counts3_vector % 2)) / (chi_counts3_vector + 1.0)),axis=2)
        
          #ent_1_2_boots = sum((count_matrix_1_2 * 1.0 / numangles_bootstrap_matrix) * (log(numangles_bootstrap_matrix) - special.psi(count_matrix_1_2 + SMALL) - (1 - 2*(count_matrix_1_2 % 2)) / (count_matrix_1_2 + 1.0)),axis=2) 
        
          #ent_2_3_boots = sum((count_matrix_2_3 * 1.0 / numangles_bootstrap_matrix) * (log(numangles_bootstrap_matrix) - special.psi(count_matrix_2_3 + SMALL) - (1 - 2*(count_matrix_2_3 % 2)) / (count_matrix_2_3 + 1.0)),axis=2) 
        
          #ent_1_3_boots = sum((count_matrix_1_3 * 1.0 / numangles_bootstrap_matrix) * (log(numangles_bootstrap_matrix) - special.psi(count_matrix_1_3 + SMALL) - (1 - 2*(count_matrix_1_3 % 2)) / (count_matrix_1_3 + 1.0)),axis=2) 

          #ent_1_2_3_boots = sum((count_matrix_triplet * 1.0 / numangles_bootstrap_tensor) * (log(numangles_bootstrap_tensor) - special.psi(count_matrix_triplet + SMALL) - (1 - 2*(count_matrix_triplet % 2)) / (count_matrix_triplet + 1.0)),axis=2) 

          
          mutinf_thisdof = ent_1_2_3_boots - ent_1_2_boots - ent_2_3_boots - ent_1_3_boots + ent_1_boots + ent_2_boots + ent_3_boots

          print "ent1_boots bootstrap 0: "+str(ent_1_boots[0])
          print "ent2_boots bootstrap 0: "+str(ent_2_boots[0])
          print "ent3_boots bootstrap 0: "+str(ent_3_boots[0])
          print "ent_1_2_boots bootstrap 0: "+str(ent_1_2_boots[0])
          print "ent_2_3_boots bootstrap 0: "+str(ent_2_3_boots[0])
          print "ent_1_3_boots bootstrap 0: "+str(ent_1_3_boots[0])
          print "ent_1_2_3_boots,        0: "+str(ent_1_2_3_boots[0])
          print "mutinf thisdof, boot    0: "+str(mutinf_thisdof[0,0])
       
        

    
    if (VERBOSE >=2): 
           print "Avg Descriptive Mutinf:    "+str(average(mutinf_thisdof[:,0]))
    if(permutations == 0):
           if(VERBOSE >= 2):
                  print "Avg Descriptive MI ind:    "+str(average(mutinf_triplet_multinomial,axis=1))
    else:
           if(VERBOSE >=2):
                  print "Avg Descriptive MI ind:    "+str(average(mutinf_thisdof[:,1:]))
                  print "Number of permutations:    "+str(mutinf_thisdof.shape[1] -1)
    
    #Now, if permutations==0, filter according to Bayesian estimate of distribution
    # of mutual information, M. Hutter and M. Zaffalon 2004 (or 2005).
    # Here, we will discard those MI values with p(I | data < I*) > 0.05.
    # Alternatively, we could use the permutation method or a more advanced monte carlo simulation
    # over a Dirichlet distribution to empirically determine the distribution of mutual information of the uniform
    # distribution.  The greater variances of the MI in nonuniform distributions suggest this app<roach
    # rather than a statistical test against the null hypothesis that the MI is the same as that of the uniform distribution.
    # The uniform distribution or sampling from a Dirichlet would be appropriate since we're using adaptive partitioning.

    #First, compute  ln(nij*n/(ni*nj) = logU, as we will need it and its powers shortly.
    #Here, use Perks' prior nij'' = 1/(nbins*nbins)
    
    if (markov_samples > 0): #if using markov model to get distribution under null hypothesis of independence           
           for bootstrap in range(bootstrap_sets):
                mutinf_triplet_multinomial_this_bootstrap = mutinf_triplet_multinomial[bootstrap]  # since our markov samples are in axis 1 , and we aren't averaging over bootstraps since their transition matrices are different
                if(mutinf_thisdof[bootstrap,0] > 0): #positive tail
                       num_greater_than_obs_MI = sum(1.0 * (mutinf_triplet_multinomial_this_bootstrap ) > mutinf_thisdof[bootstrap,0] )
                else: #negative tail
                       num_greater_than_obs_MI = sum(1.0 * (mutinf_triplet_multinomial_this_bootstrap ) < mutinf_thisdof[bootstrap,0] )
                pvalue_triplet_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_triplet_multinomial_this_bootstrap.shape[0]) 
                
                pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_triplet_multinomial)
                if(VERBOSE >=1):
                       print "Descriptive P(I=I_markov):"+str(pvalue_triplet_multinomial)
                       print "number of markov samples with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later
        
    elif(permutations == 0):
        for bootstrap in range(bootstrap_sets):
            if(mutinf_thisdof[bootstrap,0] > 0):
                   num_greater_than_obs_MI = sum(1.0 * (mutinf_triplet_multinomial[bootstrap]) > mutinf_thisdof[bootstrap,0] )
            else:
                   num_greater_than_obs_MI = sum(1.0 * (mutinf_triplet_multinomial[bootstrap]) < mutinf_thisdof[bootstrap,0] )
            if num_greater_than_obs_MI < 1:
                num_greater_than_obs_MI = 0
            pvalue_triplet_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_triplet_multinomial[bootstrap].shape[0])
            if (VERBOSE >= 1):
                   #print "Mutinf Triplet_Multinomial Shape:"+str(mutinf_triplet_multinomial.shape)
                   print "Num Ind Greater than Obs:"+str(num_greater_than_obs_MI)
                   print "bootstrap: "+str(bootstrap)+" Descriptive P(I=I_mult):"+str(pvalue_triplet_multinomial)
            pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_triplet_multinomial)
            Var_I = 0 
            
        if(VERBOSE >= 2):
               print "Max pvalue             :"+str(pvalue[bootstrap])
           
    else:  #use permutation test to filter out true negatives, possibly in addition to the Bayesian filter above
        
        #pvalue is the fraction of mutinf values from samples of permuted data that are greater than the observed MI
        #pvalue for false negative
        #lower pvalue is better
        if(permutations > 0): #otherwise, keep pvalue as 0 for now, use wilcoxon signed ranks test at the end
            for bootstrap in range(bootstrap_sets):
                if(mutinf_thisdof[bootstrap,0] > 0):   
                       num_greater_than_obs_MI = sum((mutinf_thisdof[bootstrap,1:]) > (mutinf_thisdof[bootstrap,0] ))
                else:
                       num_greater_than_obs_MI = sum((mutinf_thisdof[bootstrap,1:]) < (mutinf_thisdof[bootstrap,0] ))
                pvalue[bootstrap] += num_greater_than_obs_MI * 1.0 / permutations*permutations
                if(VERBOSE >=2):
                       print "number of permutations with MI > MI(observed): "+str(num_greater_than_obs_MI)
                       print "Descriptive P(avg(I) = avg(I,independent)"+str(pvalue[bootstrap])
                Var_I = 0 #will be overwritten later
        else:
            for bootstrap in range(bootstrap_sets):
                if(mutinf_thisdof[bootstrap,0] > 0):   
                       num_greater_than_obs_MI = sum(1.0 * (mutinf_triplet_multinomial[bootstrap] > mutinf_thisdof[bootstrap,0] ))
                else:
                       num_greater_than_obs_MI = sum(1.0 * (mutinf_triplet_multinomial[bootstrap] < mutinf_thisdof[bootstrap,0] ))
                pvalue_triplet_multinomial = num_greater_than_obs_MI * 1.0 / float32(mutinf_triplet_multinomial[bootstrap].shape[0])
                
                pvalue[bootstrap] = max(pvalue[bootstrap], pvalue_triplet_multinomial)
                if(VERBOSE >=2):
                       print "Descriptive P(I=I_mult):"+str(pvalue_triplet_multinomial)
                       print "number of permutations with MI > MI(observed) or MI > MI(observed): "+str(num_greater_than_obs_MI)
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
    #if(calc_mutinf_between_sims == "no" or num_pair_runs <= 1):
    #    mutinf_thisdof_different_sims= zeros((bootstrap_sets,permutations_sequential + 1),float64)
    mutinf_thisdof_different_sims = zeros((bootstrap_sets,permutations*permutations),float32)
    dKLtot_dKL1_dKL2 = 0
    Counts_ij = 0
    return mutinf_thisdof, var_mi_thisdof , mutinf_thisdof_different_sims, 0, average(mutinf_triplet_multinomial, axis=1), zeros((bootstrap_sets,permutations*permutations),float64), pvalue, dKLtot_dKL1_dKL2, Counts_ij, mutinf_triplet_multinomial

    #return mutinf_thisdof, var_mi_thisdof, mutinf_thisdof_different_sims, var_mutinf_triplet_multinomial_sequential, average(mutinf_triplet_multinomial,axis=1), average(mutinf_triplet_multinomial_sequential,axis=1), pvalue, dKLtot_dKL1_dKL2, Counts_ij




################################################################################################################################################
################ TRIPLETS #####################################

### calculation of independent information using permutations
independent_mutinf_thisdof = None

def calc_excess_triplet_mutinf(chi_counts1, chi_counts2, chi_counts3, bins1, bins2, bins3, chi_counts_sequential1, chi_counts_sequential2, chi_counts_sequential3, bins1_sequential, bins2_sequential, bins3_sequential, num_sims, nbins, numangles_bootstrap,numangles, sigalpha, permutations, bootstrap_choose, calc_variance=False, which_runs=None, pair_runs=None, calc_mutinf_between_sims = "yes", markov_samples=0, chi_counts1_markov=None, chi_counts2_markov=None, chi_counts3_markov=None, ent1_markov_boots=None, ent2_markov_boots=None, ent3_markov_boots=None, bins1_markov=None, bins2_markov=None, bins3_markov=None,file_prefix=None, plot_2d_histograms=False, adaptive_partitioning = 0, bins1_slowest_timescale = None, bins2_slowest_timescale = None, bins3_slowest_timescale = None, bins1_slowest_lagtime = None, bins2_slowest_lagtime= None, bins3_slowest_lagtime = None, lagtime_interval=None, boot_weights = None, weights = None, num_convergence_points=None):
       
    mutinf_tot_thisdof, var_mi_thisdof, mutinf_tot_thisdof_different_sims, var_ind_different_sims, mutinf_multinomial, mutinf_multinomial_sequential, pvalue, dKLtot_dKL1_dKL2, Counts_ij, mutinf_triplet_multinomial \
        = calc_triplet_mutinf_corrected(chi_counts1,chi_counts2, chi_counts3, bins1, bins2, bins3, chi_counts_sequential1, chi_counts_sequential2, chi_counts_sequential3, bins1_sequential, \
                                            bins2_sequential, bins3_sequential, num_sims, nbins, numangles_bootstrap, numangles, calc_variance=calc_variance, bootstrap_choose=bootstrap_choose, \
                                            permutations=permutations,which_runs=which_runs,pair_runs=pair_runs, calc_mutinf_between_sims=calc_mutinf_between_sims, markov_samples=markov_samples, \
                                            chi_counts1_markov=chi_counts1_markov, chi_counts2_markov=chi_counts2_markov, chi_counts3_markov=chi_counts3_markov,  ent1_markov_boots=ent1_markov_boots, ent2_markov_boots=ent2_markov_boots, ent3_markov_boots=ent3_markov_boots, bins1_markov=bins1_markov, \
                                            bins2_markov=bins2_markov, bins3_markov=bins3_markov, file_prefix=file_prefix, plot_2d_histograms=plot_2d_histograms, adaptive_partitioning = adaptive_partitioning, \
                                            bins1_slowest_timescale=bins1_slowest_timescale, bins2_slowest_timescale=bins2_slowest_timescale, bins3_slowest_timescale=bins3_slowest_timescale, bins1_slowest_lagtime = bins1_slowest_timescale, \
                                            bins2_slowest_lagtime = bins2_slowest_timescale, bins3_slowest_lagtime = bins3_slowest_timescale, lagtime_interval = lagtime_interval, boot_weights = boot_weights, weights = weights, num_convergence_points=num_convergence_points)
    
    
    #### NEED TO FIX THIS BELOW ###
    
    #need to filter using p-value: for , use p-value (descriptive) to pick which terms to discard from average.
    #for bootstrap_sets=1, use p-value (Bayes) to filter mutinf values
    
    
    bootstrap_sets = mutinf_tot_thisdof.shape[0]
    sd_ind_different_sims = sqrt(var_ind_different_sims)
    if(VERBOSE >= 2):
           print "mutinf_tot_thisdof shape:"+str(mutinf_tot_thisdof.shape)
           #print "mutinf_tot_thisdof different sims shape:"+str(mutinf_tot_thisdof_different_sims.shape)
           print "tot_mutinfs from bootstrap samples\n:"+str(mutinf_tot_thisdof[:,0])
    if(permutations > 0 ):
           if(VERBOSE >= 2):   
                  print "independent mutinf averaged over permutations\n:"+str(average(mutinf_tot_thisdof[:,1:], axis=1))
                  #print "independent mutinf different sims averaged over permutations\n:"+str(average(mutinf_tot_thisdof_different_sims[:,1:], axis=1))
    num_pair_runs = pair_runs.shape[0]
    independent_mutinf_thisdof                = zeros((bootstrap_sets),float64)
    uncorrected_mutinf_thisdof                = zeros((bootstrap_sets),float64)
    corrections_mutinf_thisdof                = zeros((bootstrap_sets),float64)
    excess_mutinf_thisdof                     = zeros((bootstrap_sets),float64)
    independent_mutinf_thisdof_different_sims = zeros((bootstrap_sets),float64)
    if(permutations == 0):
        #take either the multinomial independent mutinf or the markov independent mutinf, both of which are in this mutinf_multinomial variable
        independent_mutinf_thisdof[:] = mutinf_multinomial #average over samples not bootstraps already performed before it is returned
        #independent_mutinf_thisdof_different_sims[:] = average(mutinf_multinomial_sequential)  #replicate up to bootstraps, average over samples not set of pairs of sims already performed before it is returned ### average(mutinf_multinomial_sequential,axis=1) #average over samples not bootstraps here
    else:
        independent_mutinf_thisdof = average(mutinf_tot_thisdof[:,1:], axis=1) # average over permutations
        #independent_mutinf_thisdof_different_sims = average(mutinf_tot_thisdof_different_sims[:,1:], axis=1) #avg over permutations
    
    sd_ind_different_sims = sqrt(var_ind_different_sims)
            
    #print independent_mutinf_thisdof
    #print average(independent_mutinf_thisdof)
    #if(permutations > 0):
    #    independent_mutinf_thisdof_different_sims = average(mutinf_tot_thisdof_different_sims[:,1:],axis=1)
                
    #print "ind_mutinfs:"+str(independent_mutinf_thisdof)
    #print "tot_mutinfs_diff_sims:"+str(mutinf_tot_thisdof_different_sims)
    uncorrected_mutinf_thisdof = mutinf_tot_thisdof[:,0]
    if(sigalpha < 1.0): #if we're doing statistics at all...
        excess_mutinf_thisdof[:] = mutinf_tot_thisdof[:,0] #no subtraction for triplets as this just adds too much noise
        #excess_mutinf_thisdof = mutinf_tot_thisdof[:,0] - independent_mutinf_thisdof
        corrections_mutinf_thisdof += independent_mutinf_thisdof
    else:
        excess_mutinf_thisdof[:] = mutinf_tot_thisdof[:,0]
    
    excess_mutinf_thisdof_different_sims = zeros((bootstrap_sets),float64)
    old_excess_mutinf_thisdof = zeros((bootstrap_sets),float64)
    #excess_mutinf_thisdof_different_sims = mutinf_tot_thisdof_different_sims[:,0] - independent_mutinf_thisdof_different_sims
    #excess_mutinf_thisdof_different_sims -= sd_ind_different_sims
    ###last term is for high pass filter, will be added back later
    ###could consider having a different excess_mutinf_thisdof_different_sims for each bootstrap sample depending on the correlations between the runs it has in it
    ###print "excess_mutinf_thisdof_different_sims:"+str(excess_mutinf_thisdof_different_sims)
    #nonneg_excess_thisdof = logical_and((excess_mutinf_thisdof_different_sims > 0), (excess_mutinf_thisdof > 0))
    ###print "nonneg excess thisdof: "+str(nonneg_excess_thisdof)
    old_excess_mutinf_thisdof = excess_mutinf_thisdof.copy()
    #excess_mutinf_thisdof_different_sims += sd_ind_different_sims  #adding back the cutoff value
    #excess_mutinf_thisdof[nonneg_excess_thisdof] -= excess_mutinf_thisdof_different_sims[nonneg_excess_thisdof] # subtract out high-pass-filtered mutinf for torsions in different sims
    ###remember to zero out the excess mutinf thisdof from different sims that were below the cutoff value
    #excess_mutinf_thisdof_different_sims[excess_mutinf_thisdof_different_sims <= sd_ind_different_sims ] = 0
    ###print "corrected excess_mutinfs:"+str(excess_mutinf_thisdof)
    test_stat = 0
    mycutoff = 0
    sigtext = " "

    #now filter out those with too high of a probability for being incorrectly kept
    #print "pvalues (Bayes)"
    #print pvalue

    #for triplets we use a two-tailed test, with half of the p-value in each tail
    pvalue_toolow = pvalue > (sigalpha * 0.5)
    if(sum(pvalue_toolow) > 0):
           if(VERBOSE >= 2):   
                  print "one or more values were not significant!"
    #excess_mutinf_thisdof *= (1.0 - pvalue_toolow * 1.0) #zeros elements with pvalues that are below threshold
    excess_mutinf_thisdof[pvalue_toolow] = 0.0

    #dKLtot_dKL1_dKL2 *= (1.0 - pvalue_toolow * 1.0)      #zeros KLdiv Hessian matrix elements that aren't significant
    if(VERBOSE >= 2):   
           print "var_mi_thisdof: "+str(var_mi_thisdof)+"\n"
    if(VERBOSE >= 0): #usually want to print this anyways
           if(num_convergence_points < 2):
                  print  "   mutinf/ind_mutinf = cor: %.3f exc: %.3f ind: %.4f tot: %.3f %s " %  (average(excess_mutinf_thisdof), average(old_excess_mutinf_thisdof),  average(independent_mutinf_thisdof), average(mutinf_tot_thisdof[:,0]),sigtext )
           else:
                  print  "   mutinf/ind_mutinf = cor: %.3f exc: %.3f ind: %.4f tot: %.3f %s " %  (excess_mutinf_thisdof[-1], old_excess_mutinf_thisdof[-1],  independent_mutinf_thisdof[-1], mutinf_tot_thisdof[-1,0],sigtext )
           #print "   mutinf/ind_mutinf = cor:%.3f ex_btw:%.3f exc:%.3f ind:%.4f tot:%.3f ind_btw:%.3f tot_btw:%.3f (sd= %.3f)   %s" % (average(excess_mutinf_thisdof),  average(excess_mutinf_thisdof_different_sims), average(old_excess_mutinf_thisdof),  average(independent_mutinf_thisdof), average(mutinf_tot_thisdof[:,0]), average(independent_mutinf_thisdof_different_sims), average(mutinf_tot_thisdof_different_sims[:,0]), sqrt(average(var_mi_thisdof)),sigtext)
    #debug mutinf between sims
    #print "   mutinf/ind_mutinf = cor:%.3f ex_btw:%.3f exc:%.3f ind:%.3f tot:%.3f ind_btw:%.3f tot_btw:%.3f (sd= %.3f)   %s" % (average(excess_mutinf_thisdof),  average(excess_mutinf_thisdof_different_sims), average(old_excess_mutinf_thisdof),  average(independent_mutinf_thisdof), average(mutinf_tot_thisdof[:,0]), average(independent_mutinf_thisdof_different_sims), average(mutinf_tot_thisdof_different_sims[0,0]), var_mi_thisdof,sigtext)
    

    return excess_mutinf_thisdof, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, var_mi_thisdof, excess_mutinf_thisdof_different_sims, dKLtot_dKL1_dKL2, Counts_ij, mutinf_triplet_multinomial



#########################################################################################################################################
##### calc_triplet_stats: Mutual Information For All Triplets of Residues Given First Two are Significant ##########################################################
#########################################################################################################################################


# Calculate the mutual information between all triplets of residues.
# The MI between a triplet of residues is the sum of the MI between all combinations of res1-chi? and res2-chi? and res-chi? where the pairwise couplings were significant.
# The variance of the MI is the sum of the individual variances.
# Returns mut_info_res_matrix, mut_info_uncert_matrix
# mut_info_res_matrix here is nres x nres x 6 x 6
def calc_triplet_stats(reslist, run_params, mut_info_res_matrix):
    rp = run_params
    which_runs = rp.which_runs
    #print rp.which_runs
    bootstrap_sets = len(which_runs)
    #initialize the mut info matrix
    #check_for_free_mem()
    mut_info_triplet_res_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),len(reslist),6,6,6),float32)
    mut_info_triplet_uncorrected_res_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),len(reslist),6,6,6),float32)
    mut_info_triplet_corrections_res_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),len(reslist),6,6,6),float32)
    mut_info_triplet_uncert_matrix = zeros((bootstrap_sets, len(reslist),len(reslist),len(reslist),6,6,6),float32)
    mut_info_triplet_res_matrix_different_sims = zeros((bootstrap_sets, len(reslist),len(reslist),len(reslist), 6,6,6),float32)
    mutinfs_1_2 = []
    mutinfs_2_3 = []
    mutinfs_1_3 = []
    corrected_mutinfs = []
    independent_mutinfs = []
    uncorrected_mutinfs = []
    Counts_ijk = zeros((rp.nbins,rp.nbins,rp.nbins),float64)
    
    for res_ind1, myres1 in zip(range(len(reslist)), reslist):    
       for res_ind2, myres2 in zip(range(res_ind1 + 1 , len(reslist)), reslist[res_ind1 + 1:]):
              for res_ind3, myres3 in zip(range(res_ind2 + 1 , len(reslist)), reslist[res_ind2 + 1:]):
                if(OFF_DIAG == 1):
                  for mychi1 in range(myres1.nchi):
                     for mychi2 in range(myres2.nchi):
                       for mychi3 in range(myres3.nchi):
                              for myboot in range(bootstrap_sets):
                                    mutinfs_1_2.append(mut_info_res_matrix[myboot,res_ind1,res_ind2,mychi1,mychi2])
                                    mutinfs_2_3.append(mut_info_res_matrix[myboot,res_ind2,res_ind3,mychi2,mychi3])
                                    mutinfs_1_3.append(mut_info_res_matrix[myboot,res_ind1,res_ind3,mychi1,mychi3])


    #Loop over the residue list
    print
    print "TRIPLET STATS: "
    print "2d mutual information res matrix shape: "+str(shape(mut_info_res_matrix))
    #print average(average(average(mut_info_res_matrix,axis=0),axis=-1),axis=-1)
    for res_ind1, myres1 in zip(range(len(reslist)), reslist):    
       print "##### Working on residue %s%s  (%s) and other residues" % (myres1.num, myres1.chain, myres1.name), utils.flush()
       for res_ind2, myres2 in zip(range(res_ind1 + 1 , len(reslist)), reslist[res_ind1 + 1:]):
              for res_ind3, myres3 in zip(range(res_ind2 + 1 , len(reslist)), reslist[res_ind2 + 1:]):
                if (VERBOSE >= 0):
                       print "#### Working on residues %s%s and %s%s and %s%s (%s and %s and %s):" % (myres1.num, myres1.chain, myres2.num, myres2.chain, myres3.num, myres3.chain, myres1.name, myres2.name, myres3.name) , utils.flush()
                       print "number of torsions: %s/%s/%s : " % (myres1.nchi, myres2.nchi, myres3.nchi), utils.flush()
                max_S = 0.
                if(OFF_DIAG == 1):
                  for mychi1 in range(myres1.nchi):
                     for mychi2 in range(myres2.nchi):
                       for mychi3 in range(myres3.nchi):
                         
                         #check_for_free_mem()
                         mutinf_thisdof = var_mi_thisdof = mutinf_thisdof_different_sims = dKLtot_dKL1_dKL2 = zeros((bootstrap_sets),float64) #initialize
                         angle_str = ("%s_chi%d-%s_chi%d"%(myres1, mychi1+1, myres2, mychi2+1)).replace(" ","_")
                         #if(VERBOSE >=2):
                         #       print "twoD hist boot avg shape: " + str(twoD_hist_boot_avg.shape ) 
                         #if(((res_ind1 != res_ind2 and res_ind2 != res_ind3 and res_ind1 != res_ind3) or ((res_ind1 == res_ind2 and res_ind2 != res_ind3 and mychi1 > mychi2 ) or (res_ind2 == res_ind3 and res_ind1 != res_ind2 and mychi2 > mychi3 ) or (res_ind1 == res_ind3 and res_ind1 != res_ind2 and mychi1 > mychi3) or (res_ind1 == res_ind2 == res_ind3 and mychi1 > mychi2 > mychi3))) and (max(mut_info_res_matrix[:,res_ind1,res_ind2,mychi1,mychi2]) >= 0.001 or max(mut_info_res_matrix[:,res_ind1,res_ind3,mychi1,mychi3]) >= 0.001 or max(mut_info_res_matrix[:,res_ind2,res_ind3,mychi2,mychi3]) >= 0.001)):
                         if((res_ind1 != res_ind2 and res_ind2 != res_ind3 and res_ind1 != res_ind3) or ((res_ind1 == res_ind2 and res_ind2 != res_ind3 and mychi1 < mychi2 ) or (res_ind2 == res_ind3 and res_ind1 != res_ind2 and mychi2 < mychi3 ) or (res_ind1 == res_ind3 and res_ind1 != res_ind2 and mychi1 < mychi3) or (res_ind1 == res_ind2 == res_ind3 and mychi1 < mychi2 < mychi3))):
                             if(VERBOSE >=0):
                                print 
                                print "%s %s , %s %s, %s %s chi1/chi2/chi3: %d/%d/%d" % (myres1.name,myres1.num, myres2.name,myres2.num, myres3.name, myres3.num, mychi1+1,mychi2+1,mychi3+1)   
                             #print "slowest_lagtime: chi: "+str(mychi1)+" : "+str(max(myres1.slowest_lagtime[mychi1]))+"\n"
                             #print "slowest_lagtime: chi: "+str(mychi2)+" : "+str(max(myres2.slowest_lagtime[mychi2]))+"\n"
                             mutinf_thisdof, uncorrected_mutinf_thisdof, corrections_mutinf_thisdof, var_mi_thisdof, mutinf_thisdof_different_sims, dKLtot_dKL1_dKL2, Counts_ij, mutinf_triplet_multinomial = \
                                         calc_excess_triplet_mutinf(myres1.chi_counts[:,mychi1,:],myres2.chi_counts[:,mychi2,:],myres3.chi_counts[:,mychi3,:],\
                                                            myres1.bins[mychi1,:,:,:], myres2.bins[mychi2,:,:,:],myres3.bins[mychi3,:,:,:],\
                                                            myres1.chi_counts_sequential[:,mychi1,:],\
                                                            myres2.chi_counts_sequential[:,mychi2,:],\
                                                            myres3.chi_counts_sequential[:,mychi3,:],\
                                                            myres1.simbins[mychi1,:,:,:], myres2.simbins[mychi2,:,:,:],myres3.simbins[mychi3,:,:,:],\
                                                            rp.num_sims, rp.nbins, myres1.numangles_bootstrap,\
                                                            myres1.numangles, rp.sigalpha, rp.permutations,\
                                                            rp.bootstrap_choose, calc_variance=rp.calc_variance,\
                                                            which_runs=rp.which_runs,pair_runs=rp.pair_runs,\
                                                            calc_mutinf_between_sims=rp.calc_mutinf_between_sims, \
                                                            markov_samples = rp.markov_samples, \
                                                            chi_counts1_markov = myres1.chi_counts_markov[mychi1,:,:,:], \
                                                            chi_counts2_markov = myres2.chi_counts_markov[mychi2,:,:,:], \
                                                            chi_counts3_markov = myres3.chi_counts_markov[mychi3,:,:,:], \
                                                            ent1_markov_boots = myres1.ent_markov_boots[mychi1,:,:], \
                                                            ent2_markov_boots = myres2.ent_markov_boots[mychi2,:,:], \
                                                            ent3_markov_boots = myres3.ent_markov_boots[mychi3,:,:], \
                                                            bins1_markov = myres1.bins_markov[mychi1,:,:,:], \
                                                            bins2_markov = myres2.bins_markov[mychi2,:,:,:], \
                                                            bins3_markov = myres3.bins_markov[mychi3,:,:,:], \
                                                            file_prefix=angle_str, plot_2d_histograms=rp.plot_2d_histograms, \
                                                            adaptive_partitioning = rp.adaptive_partitioning, \
                                                            bins1_slowest_timescale = myres1.slowest_implied_timescale[mychi1], \
                                                            bins2_slowest_timescale = myres2.slowest_implied_timescale[mychi2], \
                                                            bins3_slowest_timescale = myres3.slowest_implied_timescale[mychi3], \
                                                            bins1_slowest_lagtime = myres1.slowest_lagtime[mychi1], \
                                                            bins2_slowest_lagtime = myres2.slowest_lagtime[mychi2], \
                                                            bins3_slowest_lagtime = myres3.slowest_lagtime[mychi3], \
                                                            lagtime_interval = rp.lagtime_interval, \
                                                            boot_weights = myres1.boot_weights , \
                                                            weights = myres1.weights , \
                                                            num_convergence_points = rp.num_convergence_points )               
                             
                             #print "mutinf this dof, after corrections:"
                             #print mutinf_thisdof
                             #print "uncorrected mutinf this dof"
                             #print uncorrected_mutinf_thisdof
                             #print "corrections to mutinf this dof"
                             #print corrections_mutinf_thisdof
                             
                             mut_info_triplet_res_matrix[:,res_ind1 , res_ind2, res_ind3, mychi1, mychi2, mychi3] = mutinf_thisdof
                             #print "mutinf 1 2 :",str(mut_info_res_matrix[:,res_ind1,res_ind2,mychi1,mychi2])
                             #for myboot in range(bootstrap_sets):
                             #       mutinfs_1_2.append(mut_info_res_matrix[myboot,res_ind1,res_ind2,mychi1,mychi2])
                             #       mutinfs_2_3.append(mut_info_res_matrix[myboot,res_ind2,res_ind3,mychi2,mychi3])
                             #       mutinfs_1_3.append(mut_info_res_matrix[myboot,res_ind1,res_ind3,mychi1,mychi3])

                             if rp.num_convergence_points > 1:
                                    output_mutinf_convergence(str(myres1.name)+str(myres1.num)+"chi"+str(mychi1+1)+"_"+str(myres2.name)+str(myres2.num)+"chi"+str(mychi2+1)+"_"+str(myres3.name)+str(myres3.num)+"chi"+str(mychi3+1)+"_uncorrected_convergence.txt", uncorrected_mutinf_thisdof, bootstrap_sets)
                                    output_mutinf_convergence(str(myres1.name)+str(myres1.num)+"chi"+str(mychi1+1)+"_"+str(myres2.name)+str(myres2.num)+"chi"+str(mychi2+1)+"_"+str(myres3.name)+str(myres3.num)+"chi"+str(mychi3+1)+"_independent_convergence.txt", corrections_mutinf_thisdof, bootstrap_sets)
                                    output_mutinf_convergence(str(myres1.name)+str(myres1.num)+"chi"+str(mychi1+1)+"_"+str(myres2.name)+str(myres2.num)+"chi"+str(mychi2+1)+"_"+str(myres3.name)+str(myres3.num)+"chi"+str(mychi3+1)+"_convergence.txt", mutinf_thisdof, bootstrap_sets)
                                    # get statistics from last bootstrap
                                    uncorrected_mutinfs.append(uncorrected_mutinf_thisdof[-1])
                                    corrected_mutinfs.append(mutinf_thisdof[-1])
                                    independent_mutinfs.append(corrections_mutinf_thisdof[-1])
                             else:
                                    # get statistics from average of bootstraps
                                    uncorrected_mutinfs.append(average(uncorrected_mutinf_thisdof))
                                    corrected_mutinfs.append(average(mutinf_thisdof))
                                    independent_mutinfs.append(average(corrections_mutinf_thisdof))

                             if(res_ind1 == res_ind2 == res_ind3 and (not (mychi1 < mychi2 < mychi3) )):
                                 pass   
                                 #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] = myres1.entropy[:,mychi1]
                                 #mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] = myres1.var_ent[:,mychi1]
                                 #max_S = 0

                             elif((res_ind1 == res_ind2 == res_ind3 and ( (mychi1 < mychi2 < mychi3))) or (res_ind1 == res_ind2 and  (mychi1 < mychi2 )) or (res_ind2 == res_ind3 and (mychi2 < mychi3)) or (res_ind1 == res_ind3 and (mychi1 < mychi3))  ):
                                 mut_info_triplet_res_matrix[:,res_ind1 , res_ind2, res_ind3, mychi1, mychi2, mychi3] = mutinf_thisdof
                                 mut_info_triplet_uncorrected_res_matrix[:,res_ind1 , res_ind2, res_ind3, mychi1, mychi2, mychi3] = uncorrected_mutinf_thisdof
                                 mut_info_triplet_corrections_res_matrix[:,res_ind1 , res_ind2, res_ind3, mychi1, mychi2, mychi3] = corrections_mutinf_thisdof
                                 #mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] = var_mi_thisdof
                                 mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] = mutinf_thisdof_different_sims
                                 #twoD_hist_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :] = Counts_ij
                                 #twoD_hist_boot_avg[res_ind1, res_ind2, mychi2, mychi1, :, :] = Counts_ij
                                 mut_info_triplet_res_matrix[:,res_ind1, res_ind3, res_ind2, mychi1, mychi3, mychi2] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_res_matrix[:,res_ind2, res_ind1, res_ind3, mychi2, mychi1, mychi3] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_res_matrix[:,res_ind2, res_ind3, res_ind1, mychi2, mychi3, mychi1] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_res_matrix[:,res_ind3, res_ind1, res_ind2, mychi3, mychi1, mychi2] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_res_matrix[:,res_ind3, res_ind2, res_ind1, mychi3, mychi2, mychi1] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 
                                 mut_info_triplet_uncorrected_res_matrix[:,res_ind1, res_ind3, res_ind2, mychi1, mychi3, mychi2] = uncorrected_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_uncorrected_res_matrix[:,res_ind2, res_ind1, res_ind3, mychi2, mychi1, mychi3] = uncorrected_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_uncorrected_res_matrix[:,res_ind2, res_ind3, res_ind1, mychi2, mychi3, mychi1] = uncorrected_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_uncorrected_res_matrix[:,res_ind3, res_ind1, res_ind2, mychi3, mychi1, mychi2] = uncorrected_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_uncorrected_res_matrix[:,res_ind3, res_ind2, res_ind1, mychi3, mychi2, mychi1] = uncorrected_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]

                                 mut_info_triplet_corrections_res_matrix[:,res_ind1, res_ind3, res_ind2, mychi1, mychi3, mychi2] = corrections_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_corrections_res_matrix[:,res_ind2, res_ind1, res_ind3, mychi2, mychi1, mychi3] = corrections_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_corrections_res_matrix[:,res_ind2, res_ind3, res_ind1, mychi2, mychi3, mychi1] = corrections_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_corrections_res_matrix[:,res_ind3, res_ind1, res_ind2, mychi3, mychi1, mychi2] = corrections_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 mut_info_triplet_corrections_res_matrix[:,res_ind3, res_ind2, res_ind1, mychi3, mychi2, mychi1] = corrections_mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3]
                                 #mut_info_uncert_matrix[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2]
                                 #max_S = 0



                             elif((res_ind1 != res_ind2) and (res_ind2 != res_ind3) and (res_ind1 != res_ind3)):
                                 #else:
                                 #S = 0
                                 #mychi1 = int(mychi1)
                                 #mychi2 = int(mychi2)

                                 #blah = myres1.chi_pop_hist[:,mychi1,:]
                                 #blah = myres2.chi_pop_hist[:,mychi2,:]
                                 #blah = myres1.bins[mychi1,:,:,:]
                                 #blah =  myres2.bins[mychi2,:,:,:]
                                 #blah =  myres1.chi_pop_hist_sequential[:,mychi1,:]
                                 #dKLtot_dresi_dresj_matrix[:,res_ind1, res_ind2] += dKLtot_dKL1_dKL2
                                 #dKLtot_dresi_dresj_matrix[:,res_ind2, res_ind1] += dKLtot_dKL1_dKL2 #note res_ind1 neq res_ind2 here
                                 mut_info_triplet_res_matrix[:,res_ind1 , res_ind2, res_ind3, mychi1, mychi2, mychi3] = mutinf_thisdof
                                 mut_info_triplet_uncert_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] = var_mi_thisdof
                                 #mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, mychi1, mychi2] = mutinf_thisdof_different_sims
                                 #twoD_hist_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :] = Counts_ij
                                 #max_S = max([max_S,S])
                                 mut_info_triplet_res_matrix[:,res_ind1, res_ind3, res_ind2, mychi1, mychi3, mychi2] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix[:,res_ind2, res_ind1, res_ind3, mychi2, mychi1, mychi3] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix[:,res_ind2, res_ind3, res_ind1, mychi2, mychi3, mychi1] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix[:,res_ind3, res_ind1, res_ind2, mychi3, mychi1, mychi2] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix[:,res_ind3, res_ind2, res_ind1, mychi3, mychi2, mychi1] = mutinf_thisdof #mut_info_triplet_res_matrix[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 #mut_info_uncert_matrix[res_ind1, res_ind2] = mut_info_uncert_matrix[res_ind1, res_ind2]
                                 #mut_info_uncert_matrix[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_uncert_matrix[:,res_ind1, res_ind2, mychi1, mychi2] #symmetric matrix
                                 #mut_info_triplet_res_matrix_different_sims[:,res_ind2, res_ind1, mychi2, mychi1] = mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, mychi1, mychi2] #symmetric matrix
                                 mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind3, res_ind2, mychi1, mychi3, mychi2] = mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix_different_sims[:,res_ind2, res_ind1, res_ind3, mychi2, mychi1, mychi3] = mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix_different_sims[:,res_ind2, res_ind3, res_ind1, mychi2, mychi3, mychi1] = mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix_different_sims[:,res_ind3, res_ind1, res_ind2, mychi3, mychi1, mychi2] = mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 mut_info_triplet_res_matrix_different_sims[:,res_ind3, res_ind2, res_ind1, mychi3, mychi2, mychi1] = mut_info_triplet_res_matrix_different_sims[:,res_ind1, res_ind2, res_ind3, mychi1, mychi2, mychi3] #symmetric tensor
                                 #twoD_hist_boot_avg[res_ind2, res_ind1, mychi2, mychi1, :, :] = swapaxes(twoD_hist_boot_avg[res_ind1, res_ind2, mychi1, mychi2, :, :],0,1) #symmetric matrix
                    #print "mutinf=%.3f (uncert=%.3f; max(S)=%.3f" % (average((mut_info_triplet_res_matrix[:,res_ind1, res_ind2, : ,:]).flatten()), sum((mut_info_uncert_matrix[0,res_ind1, res_ind2, :, :]).flatten()), max_S),
        #if max_S > 0.26: print "#####",
        

    
    "mut info res matrix, bootstrap 0:"
    #print mut_info_triplet_res_matrix[0]
    #print "checking for symmetry"
    mut_info_triplet_matrix_phi =  mut_info_triplet_res_matrix[0,:,:,:,0,0,0] 
    mut_info_triplet_matrix_psi =  mut_info_triplet_res_matrix[0,:,:,:,1,1,1] 
    for i in range(len(reslist)):
           for j in range(len(reslist)):
                  for k in range(len(reslist)):
                         assert mut_info_triplet_matrix_phi[i,j,k] == mut_info_triplet_matrix_phi[i,k,j]
                         assert mut_info_triplet_matrix_phi[i,j,k] == mut_info_triplet_matrix_phi[j,k,i]
                         assert mut_info_triplet_matrix_phi[i,j,k] == mut_info_triplet_matrix_phi[j,i,k]

                         assert mut_info_triplet_matrix_psi[i,j,k] == mut_info_triplet_matrix_psi[i,k,j]
                         assert mut_info_triplet_matrix_psi[i,j,k] == mut_info_triplet_matrix_psi[j,k,i]
                         assert mut_info_triplet_matrix_psi[i,j,k] == mut_info_triplet_matrix_psi[j,i,k]

    dKLtot_dresi_dresj_matrix=zeros((bootstrap_sets),float32)
    twoD_hist_boot_avg = 0
    
    #print "mutinfs_1_2: "+str(mutinfs_1_2)
    #print "mutinfs_2_3: "+str(mutinfs_2_3)
    #print "mutinfs_1_3: "+str(mutinfs_1_3)
    print "uncorrected mutinfs:"+str(uncorrected_mutinfs)
    print "independent mutinfs: "+str(independent_mutinfs)
    print "corrected mutinfs: "+str(corrected_mutinfs)
    output_mutinfs_for_hists("mutinfs_for_histograms.txt", mutinfs_1_2, mutinfs_2_3, mutinfs_1_3,  uncorrected_mutinfs ,  independent_mutinfs, corrected_mutinfs )
 
    return mut_info_triplet_res_matrix,  mut_info_triplet_uncorrected_res_matrix ,  mut_info_triplet_corrections_res_matrix , mut_info_triplet_uncert_matrix, mut_info_triplet_res_matrix_different_sims, dKLtot_dresi_dresj_matrix, twoD_hist_boot_avg

############################################################################################################################################






