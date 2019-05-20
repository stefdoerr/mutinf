

####   CONSTANTS   #####################################################################################################################
offset = 0
mycompiler = 'gcc'  #initialize options at the global level
my_extra_compile_args={'gcc':['-fopenmp', '-lgomp'], 'icc':[' -openmp -liomp5'], 'intel':[' -openmp -liomp5'], 'intelem':[' -openmp -liomp5']  }
my_extra_link_args={'gcc':['-lgomp', '-lpthread'], 'icc':['-liomp5 -lpthread'],  'intel':['-liomp5 -lpthread'],  'intelem':['-liomp5 -lpthread']}
Circular_PCA = False #by default, don't do circular PCA

SMALL = 0.000000001
SMALLER = SMALL * 0.0000001
REALLY_SMALL=SMALL*SMALL*SMALL*SMALL
PI = 3.141592653589793238462643383279502884197
TWOPI = 2 * PI
ROOTPI = 1.772453850905516027298167483341
GAMMA_EULER = 0.57721566490153286060651209
K_NEAREST_NEIGHBOR = 1
CACHE_TO_DISK = False
CORRECT_FOR_SYMMETRY = 1
VERBOSE = 1
OUTPUT_DIH_SINGLET_HISTOGRAMS = False
NO_GRASSBERGER = False
OFF_DIAG = 1 #set to "1" to look at off-diagonal terms
OUTPUT_DIAG = 1  #set to "1" to output flat files for diagonal matrix elements
EACH_BOOTSTRAP_MATRIX = 0 #set to "1" to output flat files for each bootstrapped matrix
MULT_1D_BINS = 3 #set to "3" or more for MULT_1D_BINS times the number of 1-D bins for 1-D entropy only; single and double already done by defualt
SAFE_1D_BINS = 36 # now 10 degree bins # approximately 2PI bins so that continuous h(x) = H(x, discrete) + log (binwidth)
FEWER_COR_BTW_BINS = 0.5 #set to "0.5" for half as many 2-D bins for 2-D correlation between tors in different sims, nbins must be an even number for this to be appropriate!
MAX_NEAREST_NEIGHBORS = 1 # up to k nearest neighbors
NUM_LAGTIMES = 5000 # do up to N lagtimes in markov model
OUTPUT_INDEPENDENT_MUTINF_VALUES = 0 
PAD = 1 #padding for multithreaded arrays


# Number of chi angles per residue
NumChis = { "ALA":0, "CYS":1, "CYN":1, "CYX":1, "CY2":2, "CYM":1, "ASP":2, "AS4":2, "ASH":2, "GLU":3, "GL4": 3, "GLH":3, "PHE":2, 
            "GLY":0, "HIS":2, "HIP":2, "HIE":2, "HID":2, "ILE":2, "LYS":4, "LYP":4, "LEU":2, "MET":3, "ASN":2, "GLN":3,
            "PRO":1, #GF: can't find def for chi2; rosetta uses chi1 only,
            "ARG":4, "SER":2, "THR":2, "VAL":1,"TRP":2, "TYR":2, "CTH":2, "F3G":1, 
            "TPO":1, "T2P":1, "S2P":1, "SEP":1 , "LYN":4,  #g_chi doesn't give dihedrals for phosphorylated aa
            "ACK":4, #acetyl-lysine   #ffAmber N and C termini next:
            "NALA":0, "NCYS":1, "NCYN":1, "NCYX":1, "NCY2":2, "NASP":2, "NAS4":2, "NASH":2, "NGLU":3, "NGL4": 3, "NGLH":3, "NPHE":2, 
            "NGLY":0, "NHIS":2, "NHIP":2, "NHIE":2, "NHID":2, "NILE":2, "NLYS":4, "NLYP":4, "NLEU":2, "NMET":3, "NASN":2, "NGLN":3,
            "NPRO":1, "NARG":4, "NSER":2, "NTHR":2, "NVAL":1,"NTRP":2, "NTYR":2,
            "CALA":0, "CCYS":1, "CCYN":1, "CCYX":1, "CCY2":2, "CASP":2, "CAS4":2, "CASH":2, "CGLU":3, "CGL4": 3, "CGLH":3, "CPHE":2, 
            "CGLY":0, "CHIS":2, "CHIP":2, "CHIE":2, "CHID":2, "CILE":2, "CLYS":4, "CLYP":4, "CLEU":2, "CMET":3, "CASN":2, "CGLN":3,
            "CPRO":1, "CARG":4, "CSER":2, "CTHR":2, "CVAL":1,"CTRP":2, "CTYR":2,
            "PHI":1,"PSI":1 #these last two are for treating phi and psi as belonging to different residues for alanine tripeptide for example
            } 

#next is for pdb trajectories -- no -OH chis
NumChis_Safe = { "ALA":0, "CYS":1, "CYN":1, "CYX":2, "CY2":2, "CYM":1, "ASP":2, "AS4":2, "ASH":2, "GLU":3, "GL4": 3, "GLH":3, "PHE":2, 
            "GLY":0, "HIS":2, "HIP":2, "HIE":2, "HID":2, "ILE":2, "LYS":4, "LYP":4, "LEU":2, "MET":3, "ASN":2, "GLN":3,
            "PRO":1, #GF: can't find def for chi2; rosetta uses chi1 only,
            "ARG":4, "SER":1, "THR":1, "VAL":1,"TRP":2, "TYR":2, "CTH":2, "F3G":1, 
            "TPO":1, "T2P":1, "S2P":1, "SEP":1 , "LYN":2, #g_chi doesn't give dihedrals for phosphorylated aa
            "ACK":4, #acetyl-lysine   #ffAmber N and C termini next:
            "NALA":0, "NCYS":1, "NCYN":1, "NCYX":1, "NCY2":2, "NASP":2, "NAS4":2, "NASH":2, "NGLU":3, "NGL4": 3, "NGLH":3, "NPHE":2, 
            "NGLY":0, "NHIS":2, "NHIP":2, "NHIE":2, "NHID":2, "NILE":2, "NLYS":4, "NLYP":4, "NLEU":2, "NMET":3, "NASN":2, "NGLN":3,
            "NPRO":1, "NARG":4, "NSER":1, "NTHR":1, "NVAL":1,"NTRP":2, "NTYR":2,
            "CALA":0, "CCYS":1, "CCYN":1, "CCYX":1, "CCY2":2, "CASP":2, "CAS4":2, "CASH":2, "CGLU":3, "CGL4": 3, "CGLH":3, "CPHE":2, 
            "CGLY":0, "CHIS":2, "CHIP":2, "CHIE":2, "CHID":2, "CILE":2, "CLYS":4, "CLYP":4, "CLEU":2, "CMET":3, "CASN":2, "CGLN":3,
            "CPRO":1, "CARG":4, "CSER":1, "CTHR":1, "CVAL":1,"CTRP":2, "CTYR":2,
            } 

TorsionAtoms = { "ALA":[" CB "],
                 "CYS":[" CB ", " SG "],
                 "CYX":[" CB ", " SG "],
                 "CYN":[" CB ", " SG "],
                 "CYM":[" CB ", " SG "],
                 "ASP":[" CB ", " CG "],
                 "ASH":[" CB ", " CG "],
                 "GLU":[" CB ", " CG "," CD "],
                 "GLH":[" CB ", " CG "," CD "],
                 "HIS":[" CB ", " CG "],
                 "ILE":[" CB ", " CG1"],
                 "LEU":[" CB ", " CG "],
                 "LYS":[" CB ", " CG ", " CD "," CE "],
                 "LYP":[" CB ", " CG ", " CD "," CE "],
                 "MET":[" CB ", " CG ", " SD "],
                 "PHE":[" CB ", " CG "],
                 "PRO":[" CB ", " CG "],
                 "PTR":[" CB ", " CG "],
                 "SER":[" CB ", " OG "],
                 "S2P":[" CB ", " OG "],
                 "SEP":[" CB ", " CG "],
                 "THR":[" CB ", " OG1"],
                 "TPO":[" CB ", " CG "],
                 "T2P":[" CB ", " OG1"],
                 "TRP":[" CB ", " CG "],
                 "TYR":[" CB ", " CG "],
                 "TYP":[" CB ", " CG "],
                 "VAL":[" CB ", " CG1"] 
                 }

TorsionAtoms2 = { "ALA":["_CB_"],
                 "CYS":["_CB_", "_SG_"],
                 "CYX":["_CB_", "_SG_"],
                 "CYN":["_CB_", "_SG_"],
                 "CYM":["_CB_", "_SG_"],
                 "ASP":["_CB_", "_CG_"],
                 "ASH":["_CB_", "_CG_"],
                 "GLU":["_CB_", "_CG_","_CD_"],
                 "GLH":["_CB_", "_CG_","_CD_"],
                 "HIS":["_CB_", "_CG_"],
                 "ILE":["_CB_", "_CG1"],
                 "LEU":["_CB_", "_CG_"],
                 "LYS":["_CB_", "_CG_", "_CD_","_CE_"],
                 "LYP":["_CB_", "_CG_", "_CD_","_CE_"],
                 "MET":["_CB_", "_CG_", "_SD_"],
                 "PHE":["_CB_", "_CG_"],
                 "PRO":["_CB_", "_CG_"],
                 "PTR":["_CB_", "_CG_"],
                 "SER":["_CB_", " OG "],
                 "S2P":["_CB_", " OG "],
                 "SEP":["_CB_", "_CG_"],
                 "THR":["_CB_", " OG1"],
                 "TPO":["_CB_", "_CG_"],
                 "T2P":["_CB_", " OG1"],
                 "TRP":["_CB_", "_CG_"],
                 "TYR":["_CB_", "_CG_"],
                 "TYP":["_CB_", "_CG_"],
                 "VAL":["_CB_", "_CG1"] 
                 }


my_support_code = """
      #include <math.h>
            /*************************************
      An ANSI-C implementation of the digamma-function for real arguments based
      on the Chebyshev expansion proposed in appendix E of 
      http://arXiv.org/abs/math.CA/0403344 . This is identical to the implementation
      by Jet Wimp, Math. Comp. vol 15 no 74 (1961) pp 174 (see Table 1).
      For other implementations see
      the GSL implementation for Psi(Digamma) in
      http://www.gnu.org/software/gsl/manual/html_node/Psi-_0028Digamma_0029-Function.html

     Richard J. Mathar, 2005-11-24
     **************************************/
     

     #ifndef M_PIl
     //  The constant Pi in high precision 
     #define M_PIl 3.1415926535897932384626433832795029L
     #endif
     #ifndef M_GAMMAl
     //  Euler's constant in high precision 
     #define M_GAMMAl 0.5772156649015328606065120900824024L
     #endif
     #ifndef M_LN2l
     // the natural logarithm of 2 in high precision 
     #define M_LN2l 0.6931471805599453094172321214581766L
     #endif

     // The digamma function in long double precision.
     //  @param x the real value of the argument
     // @return the value of the digamma (psi) function at that point
     // @author Richard J. Mathar
     // @since 2005-11-24
     //

     long double digammal(long double x)
     {
     /* force into the interval 1..3 */
     if( x < 0.0L )
     return digammal(1.0L-x)+M_PIl/tanl(M_PIl*(1.0L-x)) ;/* reflection formula */
     else if( x < 1.0L )
     return digammal(1.0L+x)-1.0L/x ;
     else if ( x == 1.0L)
     return -M_GAMMAl ;
     else if ( x == 2.0L)
     return 1.0L-M_GAMMAl ;
     else if ( x == 3.0L)
     return 1.5L-M_GAMMAl ;
     //else if ( x > 3.0L)
     //  duplication formula 
     //return 0.5L*(digammal(x/2.0L)+digammal((x+1.0L)/2.0L))+M_LN2l ;
     else
     {
     //  Just for your information, the following lines contain
     //  the Maple source code to re-generate the table that is
     //  eventually becoming the Kncoe[] array below
     //  interface(prettyprint=0) :
     // * Digits := 63 :
     // * r := 0 :
     // * 
     // * for l from 1 to 60 do
     // * d := binomial(-1/2,l) :
     // * r := r+d*(-1)^l*(Zeta(2*l+1) -1) ;
     // * evalf(r) ;
     // * print(%,evalf(1+Psi(1)-r)) ;
     // *o d :
     // * 
     // * for N from 1 to 28 do
     // * r := 0 :
     // * n := N-1 :
     // *
     // for l from iquo(n+3,2) to 70 do
     // *d := 0 :
     // *for s from 0 to n+1 do
     // * d := d+(-1)^s*binomial(n+1,s)*binomial((s-1)/2,l) :
     // *od :
     // *if 2*l-n > 1 then
     // *r := r+d*(-1)^l*(Zeta(2*l-n) -1) :
     // *fi :
     // *od :
     // *print(evalf((-1)^n*2*r)) ;
     // *od :
     // *quit :
     //

     static long double Kncoe[] = { .30459198558715155634315638246624251L,
     .72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
     .27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
     .17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
     .11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
     .83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
     .59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
     .42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
     .304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
     .21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
     .15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
     .11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
     .80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
     .58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
     .41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L } ;

     register long double Tn_1 = 1.0L ;// T_{n-1}(x), started at n=1 
     register long double Tn = x-2.0L ;// T_{n}(x) , started at n=1 
     register long double resul = Kncoe[0] + Kncoe[1]*Tn ;

     x -= 2.0L ;

     for(int n = 2 ; n < sizeof(Kncoe)/sizeof(long double) ;n++)
     {
     const long double Tn1 = 2.0L * x * Tn - Tn_1 ;// Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun 
     resul += Kncoe[n]*Tn1 ;
     Tn_1 = Tn ;
     Tn = Tn1 ;
     }
     return resul ;
     }
     }

////////////////////////////////////////////////////////////////////////////////
// File: digamma_function.c                                                   //
// Routine(s):                                                                //
//    DiGamma_Function                                                        //
//    xDiGamma_Function                                                       //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//  Description:                                                              //
//     The digamma function, also called the psi function, evaluated at a     //
//     x is the derivative of the log of the gamma function evaluated at x,   //
//     i.e.  psi(x) = d ln gamma(x) / dx = (1 / gamma(x)) d gamma(x) / dx.    //
////////////////////////////////////////////////////////////////////////////////
#include <math.h>                  // required for powl(), sinl(), and cosl().
#include <float.h>                 // required for DBL_MAX and LDBL_MAX.
#include <limits.h>                // required for LONG_MAX

//                         Internally Defined Routines                        //

double DiGamma_Function( double x );
long double xDiGamma_Function( long double x );

static long double xDiGamma(long double x);
static long double xDiGamma_Asymptotic_Expansion( long double x );


//                         Internally Defined Constants                       //

static double cutoff = 171.0;
static long double const pi = 3.14159265358979323846264338L;
static long double const g =  9.6565781537733158945718737389L;
static long double const a[] = { +1.144005294538510956673085217e+4L,
                                 -3.239880201523183350535979104e+4L,
                                 +3.505145235055716665660834611e+4L,
                                 -1.816413095412607026106469185e+4L,
                                 +4.632329905366668184091382704e+3L,
                                 -5.369767777033567805557478696e+2L,
                                 +2.287544733951810076451548089e+1L,
                                 -2.179257487388651155600822204e-1L,
                                 +1.083148362725893688606893534e-4L
                              };

////////////////////////////////////////////////////////////////////////////////
// double DiGamma_Function( double x )                                        //
//                                                                            //
//  Description:                                                              //
//     This function uses the derivative of the log of Lanczos's expression   //
//     for the Gamma function to calculate the DiGamma function for x > 1 and //
//     x <= cutoff = 171.  An asymptotic expression for the DiGamma function  //
//     for x > cutoff.  The reflection formula                                //
//                      DiGamma(x) = DiGamma(1+x) - 1/x                       //
//     for 0 < x < 1. and the reflection formula                              //
//                DiGamma(x) = DiGamma(1-x) - pi * cot(pi*x)                  //
//     for x < 0.                                                             //
//     The DiGamma function has singularities at the nonpositive integers.    //
//     At a singularity, DBL_MAX is returned.                                 //
//                                                                            //
//  Arguments:                                                                //
//     double x   Argument of the DiGamma function.                           //
//                                                                            //
//  Return Values:                                                            //
//     If x is a nonpositive integer then DBL_MAX is returned otherwise       //
//     DiGamma(x) is returned.                                                //
//                                                                            //
//  Example:                                                                  //
//     double x, psi;                                                         //
//                                                                            //
//     psi = DiGamma_Function( x );                                           //
////////////////////////////////////////////////////////////////////////////////
double DiGamma_Function(double x)
{
   long double psi = xDiGamma_Function((long double) x);

   if (fabsl(psi) < DBL_MAX) return (double) psi;
   return (psi < 0.0L) ? -DBL_MAX : DBL_MAX;
}


////////////////////////////////////////////////////////////////////////////////
// long double xDiGamma_Function( long double x )                             //
//                                                                            //
//  Description:                                                              //
//     This function uses the derivative of the log of Lanczos's expression   //
//     for the Gamma function to calculate the DiGamma function for x > 1 and //
//     x <= cutoff = 171.  An asymptotic expression for the DiGamma function  //
//     for x > cutoff.  The reflection formula                                //
//                      DiGamma(x) = DiGamma(1+x) - 1/x                       //
//     for 0 < x < 1. and the reflection formula                              //
//                DiGamma(x) = DiGamma(1-x) - pi * cot(pi*x)                  //
//     for x < 0.                                                             //
//     The DiGamma function has singularities at the nonpositive integers.    //
//     At a singularity, LDBL_MAX is returned.                                //
//                                                                            //
//  Arguments:                                                                //
//     long double x   Argument of the DiGamma function.                      //
//                                                                            //
//  Return Values:                                                            //
//     If x is a nonpositive integer then LDBL_MAX is returned otherwise      //
//     DiGamma(x) is returned.                                                //
//                                                                            //
//  Example:                                                                  //
//     long double x, psi;                                                    //
//                                                                            //
//     psi = xDiGamma_Function( x );                                          //
////////////////////////////////////////////////////////////////////////////////
long double xDiGamma_Function(long double x)
{
   long double sin_x, cos_x;
   long int ix;

             // For a positive argument (x > 0)                 //
             //    if x <= cutoff return Lanzcos approximation  //
             //    otherwise return Asymptotic approximation.   //

   if ( x > 0.0L )
      if (x <= cutoff)
         if ( x >= 1.0L) return xDiGamma(x);
         else return xDiGamma( x + 1.0L ) - (1.0L / x);
      else return xDiGamma_Asymptotic_Expansion(x);

                  // For a nonpositive argument (x <= 0). //
               // If x is a singularity then return LDBL_MAX. //

   if ( x > -(long double)LONG_MAX) {
      ix = (long int) x;
      if ( x == (long double) ix) return LDBL_MAX;
   }

   sin_x = sinl(pi * x);
   if (sin_x == 0.0L) return LDBL_MAX;
   cos_x = cosl(pi * x);
   if (fabsl(cos_x) == 1.0L) return LDBL_MAX;

            // If x is not a singularity then return DiGamma(x). //

   return xDiGamma(1.0L - x) - pi * cos_x / sin_x;
}


////////////////////////////////////////////////////////////////////////////////
// static long double xDiGamma( long double x )                               //
//                                                                            //
//  Description:                                                              //
//     This function uses the derivative of the log of Lanczos's expression   //
//     for the Gamma function to calculate the DiGamma function for x > 1 and //
//     x <= cutoff = 171.                                                     //
//                                                                            //
//  Arguments:                                                                //
//     double x   Argument of the Gamma function.                             //
//                                                                            //
//  Return Values:                                                            //
//     DiGamma(x)                                                             //
//                                                                            //
//  Example:                                                                  //
//     long double x;                                                         //
//     long double psi;                                                       //
//                                                                            //
//     g = xDiGamma( x );                                                     //
////////////////////////////////////////////////////////////////////////////////
static long double xDiGamma(long double x) {

   long double lnarg = (g + x - 0.5L);
   long double temp;
   long double term;
   long double numerator = 0.0L;
   long double denominator = 0.0L;
   int const n = sizeof(a) / sizeof(long double);
   int i;

   for (i = n-1; i >= 0; i--) {
      temp = x + (long double) i;
      term = a[i] / temp;
      denominator += term;
      numerator += term / temp;
   } 
   denominator += 1.0L;
   
   return logl(lnarg) - (g / lnarg) - numerator / denominator;
}


////////////////////////////////////////////////////////////////////////////////
// static long double xDiGamma_Asymptotic_Expansion( long double x )          //
//                                                                            //
//  Description:                                                              //
//     This function estimates DiGamma(x) by evaluating the asymptotic        //
//     expression:                                                            //
//         DiGamma(x) ~ ln(x) - (1/2) x +                                     //
//                        Sum B[2j] / [ 2j * x^(2j) ], summed over            //
//     j from 1 to 3 and where B[2j] is the 2j-th Bernoulli number.           //
//                                                                            //
//  Arguments:                                                                //
//     long double x   Argument of the DiGamma function. The argument x must  //
//                     be positive.                                           //
//                                                                            //
//  Return Values:                                                            //
//     DiGamma(x) where x > cutoff.                                           //
//                                                                            //
//  Example:                                                                  //
//     long double x;                                                         //
//     long double g;                                                         //
//                                                                            //
//     g = xDiGamma_Asymptotic_Expansion( x );                                //
////////////////////////////////////////////////////////////////////////////////

// Bernoulli numbers B(2j) / 2j: B(2)/2,B(4)/4,B(6)/6,...,B(20)/20.  Only     //
//  B(2)/2,..., B(6)/6 are currently used.                                    //

static const long double B[] = {   1.0L / (long double)(6 * 2 ),
                                  -1.0L / (long double)(30 * 4 ),
                                   1.0L / (long double)(42 * 6 ),
                                  -1.0L / (long double)(30 * 8 ),
                                   5.0L / (long double)(66 * 10 ),
                                -691.0L / (long double)(2730 * 12 ),
                                   7.0L / (long double)(6 * 14 ),
                               -3617.0L / (long double)(510 * 16 ),
                               43867.0L / (long double)(796 * 18 ),
                             -174611.0L / (long double)(330 * 20 ) 
                           };

static const int n = sizeof(B) / sizeof(long double);

static long double xDiGamma_Asymptotic_Expansion(long double x ) {
   const int  m = 3;
   long double term[3];
   long double sum = 0.0L;
   long double xx = x * x;
   long double xj = x;
   long double digamma = logl(xj) - 1.0L / (xj + xj);
   int i;

   xj = xx;
   for (i = 0; i < m; i++) { term[i] = B[i] / xj; xj *= xx; }
   for (i = m - 1; i >= 0; i--) sum += term[i]; 
   return digamma - sum;
}
    """
