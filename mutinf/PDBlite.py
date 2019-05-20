#!/usr/bin/python2.4
# Lite python pdb util classes & functions

import sys, os.path, commands, optparse, operator, utils
import amino_acids
import numpy as num
import math, subprocess
from itertools import izip

RAD_TO_DEG = 180.0 / math.pi

def make_pdb_atom_str(atomNum, atomName, resName, chain, resNum, x, y, z, altLoc=' ',insCode=' ',occupancy=1, bFactor=0):
        return "%-6s%5d %4s%1s%3s %1s%4s%1s   %8.3f%8.3f%8.3f%6.2f%6.2f" % \
               ("ATOM", atomNum, atomName, altLoc, resName, chain,resNum, insCode, x, y, z, occupancy, bFactor)

def get_resid(chain, res_num): return chain+"@"+str(res_num)
def parse_resid(resid): return resid.split("@")

# From Hu, biochem 03
# vectors must be size (num x 3)
def calc_S2_from_vector_array(vectors, normalized=False):
    npdb, ncol = vectors.shape
    if ncol != 3:
        print "ERROR invalid vector array shape: %s" % vectors.shape
        return None

    xx, yy, zz = num.sum(vectors**2, axis=0)/npdb
    xs, ys, zs = vectors[:,0], vectors[:,1], vectors[:,2]
    xy, xz, yz = num.sum(xs*ys)/npdb, num.sum(xs*zs)/npdb, num.sum(ys*zs)/npdb

    S2 = 3./2. * (xx**2 + yy**2 + zz**2 + 2*(xy**2 + xz**2 + yz**2)) - 1./2.
    return S2

# from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/498246
# where the input matrix shape is (nres x 3)
def calc_distance_matrix(nDimPoints):
    nDimPoints = num.array(nDimPoints)
    n,m = nDimPoints.shape
    delta = num.zeros((n,n),'d')
    for d in xrange(m):
        data = nDimPoints[:,d]
        delta += (data - data[:,num.newaxis])**2
        return num.sqrt(delta)
                    
# expects four letter atom name from a pdb file
def is_hydrogen(full_atom_name):
    c1, c2= full_atom_name[:2]
    if c2 == "H" and (c1 == " "  or c1.isdigit()): return True
    else: return False

# calc rmsd between atoms
def calc_rmsd(atoms):
    dist2s = []
    for atom1_ind in range(len(atoms)):
        for atom2_ind in range(atom1_ind+1, len(atoms)):
            dist2s.append(atoms[atom1_ind].calc_dist2(atoms[atom2_ind]))

    dist2s = num.array(dist2s)
    rmsd = math.sqrt(num.mean(dist2s)) #sum(dist2s)/len(dist2s))
    dist_std = num.std(num.sqrt(dist2s))
    return rmsd, dist_std

# calls dssp on a pdb file and parses the output
# returns dict of resid -> (chain, res_num, res_char, acc, norm_acc, ss)
# updated 1/25/06
#  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA 
#    1 1416   V              0   0  152      0, 0.0     2,-0.4     0, 0.0    26,-0.0   0.000 360.0 360.0 360.0 146.5   15.4   14.9  -31.6
#    2 1417   S        -     0   0   58      1,-0.1    24,-0.5     2,-0.1    83,-0.0  -0.748 360.0-163.3 -85.7 132.6   17.9   16.7  -33.8
#  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
#    1  566 A Q              0   0  229      0, 0.0     2,-0.4     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0-141.6  -53.2   57.2   54.1
#    2  567 A M        -     0   0   36    107,-0.0     2,-0.2   103,-0.0   100,-0.0  -0.990 360.0-106.0-135.4 139.8  -54.8   54.6   51.9
#    5  807   S  E     +A   22   0A  71     17,-1.6    17,-2.9     1,-0.2    78,-0.0  -0.649  66.7  22.5-110.5 167.8   37.4   45.8   16.6
#    6  808   Q  E    S-     0   0   99     -2,-0.2     2,-0.4    15,-0.2    -1,-0.2   0.920  72.5-173.4  43.5  59.3   38.7   49.0   18.1
#0123456789012345678901234567890123456789
# H = alpha helix
# B = residue in isolated beta-bridge
# E = extended strand, participates in beta ladder
# G = 3-helix (3/10 helix)
# I = 5 helix (pi helix)
# T = hydrogen bonded turn
# S = bend
def parse_dssp_txt(dssp_txt):
    data_started = False
    data = {}
    for line in dssp_txt.split("\n"):
        if not data_started:
            if line.find("#  RESIDUE AA STRUCTURE BP1 BP2  ACC") != -1:
                data_started = True
                continue
        else:
            res_num, chain, res_char, acc, ss = line[6:10].strip(), line[11], line[13], float(line[34:38]), line[16]
            if res_num == "": continue # true for chain breaks

            try:
                norm_acc = float(acc)/float(amino_acids.SA[res_char])
            except KeyError:
                norm_acc = -1
            resid = get_resid(chain, res_num)
            data[resid] = [chain, res_num, res_char, acc, norm_acc, ss]
    return data

# run DSSP and parse the output
def parse_dssp(pdb_fn):
    cmd = "cat %s | grep -v HETATM | ~gfriedla/bin/dssp --" % pdb_fn
    return parse_dssp_txt(commands.getoutput(cmd))

def length(u):
    """Calculates the length of u.
    """
    return num.sqrt(num.dot(u, u))

def normalize(u):
    """Returns the normalized vector along u.
    """
    return u.copy()/length(u)

# return  a restricted to [-x/2,x/2)
def periodic_range(a, x):
    if a == num.nan: return a
    
    halfx = x/2.0
    if a >= halfx or a < -halfx:
        return (a % x + x + halfx) % x - halfx
    else:
        return a
    
def calc_torsion_angle(a1, a2, a3, a4):
    """Calculates the torsion angle between the four argument atoms.
           Returns NaN if atoms can't be found, or one has all zero coordinates.
    """
    from numpy import equal, alltrue
    if a1==None or a2==None or a3==None or a4==None:
        return num.nan

#    all zero arrays are now all nans
#    zero = num.array([0,0,0])
#    if sum(a1.xyzalltrue(equal(a1.xyz, zero)) or alltrue(equal(a2.xyz, zero)) or \
#           alltrue(equal(a3.xyz, zero)) or alltrue(equal(a4.xyz, zero)):
#        return num.nan

    a12 = a2.xyz - a1.xyz
    a23 = a3.xyz - a2.xyz
    a34 = a4.xyz - a3.xyz

    n12 = num.cross(a12, a23) # vy
    n34 = num.cross(a23, a34)

    n12 = n12 / length(n12) # uy
    n34 = n34 / length(n34)

    cross_n12_n34  = num.cross(n12, n34)
    direction      = cross_n12_n34 * a23
    scalar_product = num.dot(n12, n34)

    if scalar_product > 1.0:
        scalar_product = 1.0
    if scalar_product < -1.0:
        scalar_product = -1.0

    angle = num.arccos(scalar_product) * RAD_TO_DEG
    if num.alltrue(direction < 0.0):
        angle = -angle

    # rosetta algorithm
    #uy = n12
    #vx = cross(uy, a23)
    #ux = vx / length(vx)
    #cx = Numeric.dot(a34, ux)
    #cy = Numeric.dot(a34, uy)
    #angle2 = math.atan2(cy, cx) * RAD_TO_DEG
    #print "%.0f %.0f" % (angle, angle2)

    return angle 

# Class to contain info about secondary structure
class SS_info:
    # dssp_data is the output of parse_dssp[_txt]()
    def __init__(self, dssp_fn):
        self.dssp_data = parse_dssp_txt(open(dssp_fn).read())
        self.ss_dict = {}
        for chain, res_num, res_char, acc, norm_acc, ss in self.dssp_data.values():
            self.ss_dict[int(res_num)] = ss

    def get_res_nums(self): return sorted(self.ss_dict.keys())
    def get_ss_code(self, res_num): return self.ss_dict[res_num]

    # return if the residue is structured according to the passed list of dssp codes
    def is_structured(self, res_num, dssp_codes=("H", "E", "G", "I")):
        return self.ss_dict[res_num] in dssp_codes


class PDBAtom(object):
    def __init__(self, line, extended_data=False, fast=False, load_xyz_array=True):
        self.parse(line, extended_data, load_xyz_array)
        #self.fast_parse(line, load_xyz_array)
        
    def parse(self, line, extended_data, load_xyz_array):
        self.atomName = line[12:16].strip()
        self.altLoc   = line[16:17]
        self.resName  = line[17:20]
        self.chain = line[21:22]
        self.resNum   = int(line[22:26])
        self.insCode  = line[26]
        self.x        = float(line[30:38])
        self.y        = float(line[38:46])
        self.z        = float(line[46:54])

        if self.x == 0 and self.y == 0 and self.z == 0:
            print "WARN atom '%s' has all zero coords" % self
            self.x = self.y = self.z = num.nan
            
        if load_xyz_array: self.xyz = num.array([self.x, self.y, self.z])
        else: self.xyz = None

        #if self.chain in ("", " "): self.chain = "_"

        #self.elem = self.atomName[0]
        #if self.elem.isdigit(): self.elem = self.atomName[1]

        if extended_data:
            self.recName = line[0:6]
            self.atomNum  = int(line[6:11])
            self.occupancy= float(line[54:60])
            self.bFactor  = float(line[60:66])
            l = len(line)
            if l >= 76: self.segID    = line[72:76].strip()
            if l >= 78: self.element  = line[76:78].strip()
            if l >= 80:
                chg = line[78:80].strip()
                if chg != "": self.charge = float(chg)
                else: self.charge   = -1.0

    def fast_parse(self, line, load_xyz_array=True):
        import weave
        code = """
            //weave

            int i=0;
            py::tuple results(7);
            results[i++] = line.substr(12, 4); // atomName
            results[i++] = line.substr(17, 3); // resName
            results[i++] = line.substr(21, 1); // chain
            results[i++] = line.substr(22, 4); // resNum
            results[i++] = line.substr(30, 8); // x
            results[i++] = line.substr(38, 8); // y
            results[i++] = line.substr(46, 8); // z
            return_val = results;
            """
        results = weave.inline(code, ['line'])

        self.atomName = results[0].strip()
        self.resName, self.chain = results[1:3]
        self.resNum = int(results[3])
        self.x, self.y, self.z = map(float, results[4:])
        if load_xyz_array: self.xyz = num.array([self.x, self.y, self.z])
        else: self.xyz = None
        
    def get_elem(self):
        if self.atomName[0].isdigit():
            return self.atomName[1]
        else: 
            return self.atomName[0]

    def get_xyz(self):
        if self.xyz is None: self.xyz = num.array([self.x, self.y, self.z])
        return self.xyz
    def set_xyz(self, xyz):
        self.x, self.y, self.z = xyz[0], xyz[1], xyz[2]
        self.xyz = xyz

    def __str__(self):
        return "%4s %3s %1s %5d %1s %8.3f %8.3f %8.3f" % (self.atomName, self.resName,
                                                          self.chain, self.resNum, self.altLoc,
                                                          self.x, self.y, self.z)
    def __repr__(self): return str(self)

    # calculate squared distance to another atom
    def calc_dist2(self, atom2):
        diff = [self.x - atom2.x, self.y - atom2.y, self.z - atom2.z]
        return diff[0]**2 + diff[1]**2 + diff[2]**2

    # get a pdb formatted string of this atom
    #ATOM      1  N   MET R   1      98.797  25.938  64.784  1.00 70.44      RAS  N    
    def get_pdb_str(self, atomNum):
        if self.atomName[0].isdigit():
            atom_name_str = "%-4s"%self.atomName
        else:
            atom_name_str = " %-3s"%self.atomName

        occupancy, bFactor = 1, 0
        if "occupancy" in  self.__dict__: occupancy = self.occupancy
        if "bFactor" in  self.__dict__: bFactor = self.bFactor

        return "%-6s%5d %4s%1s%3s %1s%4s%1s   %8.3f%8.3f%8.3f%6.2f%6.2f" % ("ATOM",
             atomNum, atom_name_str, self.altLoc, self.resName, self.chain,
             self.resNum, self.insCode, self.x, self.y, self.z, occupancy, bFactor)

class PDBResidue:
    def __init__(self, res_num, res_name, chain, pdb):
        self.res_num, self.res_name, self.chain = res_num, res_name, chain
        self.pdb = pdb
        self.id = get_resid(chain, res_num)
        self._atoms = {} # indexed by atom name

        try: self.resChar = amino_acids.longer_names[res_name]
        except: self.resChar = "X"

    def __str__(self): return "chain '%s'; %s%d" % (self.chain, self.resChar, self.res_num)
    def __repr__(self): return self.__str__()

    def add_atom(self, atom):
        self._atoms[atom.atomName] = atom
        #if self.res_num != atom.resNum: raise Exception("Residue & atom residue numbers don't match")
    def get_atom(self, atom_name):
        try: return self._atoms[atom_name]
        except KeyError: return None
    def iter_atoms(self):
        return self._atoms.values()
    def get(self, name):
        try: return self.__dict__[name]
        except KeyError: return None
    def set(self, name, value):
        self.__dict__[name] = value

    # return the CB atom or (if it's not there) the CA atom
    def get_CB(self):
        a = self.get_atom("CB")
        if a is None: a = self.get_atom("CA")
        return a

    # returns a tuple of N, CA, C, O, CB, next-N, next-CA, previous-N, previous-C
    def get_mainchain_atoms(self):
        aN, aCA, aC, aO, aCB = self.get_atom("N"), self.get_atom("CA"), self.get_atom("C"), self.get_atom("O"), self.get_atom("CB")
        naN = naCA = paN = paC = None
        next_res = self.pdb.get(self.chain, self.res_num+1)
        prev_res = self.pdb.get(self.chain, self.res_num-1)
        if next_res:
            naN  = next_res.get_atom("N")
            naCA = next_res.get_atom("CA")
        if prev_res:
            paN  = prev_res.get_atom("N")
            paC  = prev_res.get_atom("C")

        return aN, aCA, aC, aO, aCB, naN, naCA, paN, paC

    def calc_phi_psi_omega(self):
        """Calculates the Psi,  Phi & Omega torsion angles of the amino acid.
           Returns NaN if atoms can't be found.
        """
        aN, aCA, aC, aO, aCB, naN, naCA, paN, paC = self.get_mainchain_atoms()

        return num.array([calc_torsion_angle(paC, aN, aCA, aC),    # phi
                          calc_torsion_angle(aN, aCA, aC, naN),    # psi
                          calc_torsion_angle(aCA, aC, naN, naCA)]) # omega

    def calc_chis(self):
        """Calculates the chi torsion angles for this residue.
           Returns NaN if atoms can't be found.
        """
        chi_defs = amino_acids.chi_definitions[self.res_name]
        chis = []
        missing_atoms = None

        for i in range(4):
            chi = num.nan
            if i < len(chi_defs):
                atom_names = chi_defs[i].split()
                try:
                    atoms = map(self.get_atom, atom_names)
                    #print self.res_num, self.res_name, atom_names, 
                    chi = calc_torsion_angle(atoms[0], atoms[1], atoms[2], atoms[3])
                except KeyError:
                    missing_atoms = atom_names
                    
            chis.append(chi)

        if missing_atoms != None:
            print "ERROR can't find one of chi atoms '%s' in res '%s'" % (missing_atoms, self)            

        # make chis periodic according to Dunbrack definition (i.e. simplify symmetries)
        if self.res_name in ("PHE", "TYR"):
            chis[1] = periodic_range(chis[1]-60, 180) + 60
        elif self.res_name in ("ASP"):
            chis[1] = periodic_range(chis[1], 180)
        elif self.res_name in ("GLU"):
            chis[2] = periodic_range(chis[2], 180)

        return num.array(chis)
    
    def get_xyz_matrix(self, name):
        dat = []
	atom = res.get_atom(name)
	if atom is None: xyz = [num.nan,num.nan,num.nan]
	else: xyz = [atom.x, atom.y, atom.z]
	dat.append(xyz)
        return num.array(dat, 'd')

    def calc_rmsd(self, res2, atom_names=None):
        sum, count = 0., 0.
        for atom in self.iter_atoms():
	    if atom_names is None or atom.atomName in atom_names:
	        atom2 = res2.get_atom(atom.atomName)
		#print atom, atom2
		if atom2 is None: raise Exception("ERROR PDBlite.calc_rmsd:  can't find atom '%s %s' in pdb '%s'" % (res2, atom.atomName, pdb2))
		sum += atom.calc_dist2(atom2)
		count += 1
        rmsd = math.sqrt(sum/count)
	return rmsd

class PDB:
    # ignores hydrogen atoms by default because they slow things down and are usually not used
    def __init__(self, atom_lines, fn=None, model_num=0, heavy_only=False):
        self.fn, self.model_num = fn, model_num
        self._atoms = []
        self._residues = {}
        self._chain_names = {} # use dict as a set
        self._res_order = [] # order of the residues as loaded (contains resids)
        assert(len(atom_lines)>1)
        self._load_atom_lines(atom_lines, heavy_only=heavy_only)

    def __str__(self):
        base_fn = None
	if self.fn != None: os.path.basename(self.fn)
	return "fn: %s, model: %d, chains:'%s', numres:%s" % (base_fn, self.model_num, " ".join(self._chain_names.keys()), self.len())

    def get_from_chain1(self, res_num):
        chains = sorted(self._chain_names.keys())
        return self.get(chains[0], res_num)
    def get(self, chain, res_num):
        resid = get_resid(chain, res_num)
        try: return self._residues[resid]
        except KeyError: return None

    def get_chain_names(self): return self._chain_names.keys()
        
    def len(self): return len(self._residues.keys())
    def iter_residues(self):
        for resid in self._res_order: yield self._residues[resid]
    def iter_atoms(self):
        for res in self.iter_residues():
            for a in res.iter_atoms():
                yield a
                
    # get a string containing the pdb file data
    # optionally, only print out data for specific atom_names
    def get_pdb_str(self, atom_names=None):
        atom_strs = []
        atom_num = 1
        for a in self.iter_atoms():
            if atom_names is None or a.atomName in atom_names:
                atom_strs.append(a.get_pdb_str(atom_num))
                atom_num += 1
        return "\n".join(atom_strs)

    # takes output from dssp and puts acc, norm_acc, ss in solv_acc, norm_solv_acc, ss fields
    def load_dssp(self, dssp_fn=None, force_chain=None):
        if dssp_fn != None: dssp_data = parse_dssp_txt(open(dssp_fn).read())
        else: dssp_data =  parse_dssp(self.fn)

        for resid, res_data in dssp_data.items():
            chain, res_num, res_char, acc, norm_acc, ss = res_data
            if force_chain != None: chain = force_chain
            elif chain == " ": chain = "_"
            pdb_res = self.get(chain, res_num)
            if pdb_res != None:
                assert(pdb_res.resChar == res_char)
                pdb_res.set("solv_acc", acc)
                pdb_res.set("norm_solv_acc", norm_acc)
                pdb_res.set("ss", ss)
            else:
                raise Exception("ERROR: in '%s' unknown dssp res '%s' '%s'" % (self.fn, chain, res_num))

    # return an array of the amide bond vectors (nres x 3)
    def get_amide_bond_vectors(self, normalize=True):
        nres = self.len()
        ns = num.zeros((nres, 3))
        hs = num.zeros((nres, 3))
        vectors = num.zeros((nres, 3))
        for res_num, res in zip(range(nres), self.iter_residues()):
            n, h = res.get_atom("N"), res.get_atom("H")
	    if n != None: ns[res_num,:] = [n.x,n.y,n.z]
	    if h != None: hs[res_num,:] = [h.x,h.y,h.z]
            if n is None or h is None:
                pass #print "WARN: can't find either atom 'N' or 'H' in res '%s'" % (res)
            else:
                #vec = num.array([n.x - h.x, n.y - h.y, n.z- h.z])
                vec = hs[res_num,:] - ns[res_num,:]
                if normalize: vec = vec/length(vec)
                vectors[res_num,:] = vec
        return ns, hs, vectors

    # return an array of SC chi dihedral orienting bond vectors (nres X max_chi x 3)
    # this is the bond vector between the 3rd and 4th atoms defining a chi dihedral
    def get_chi_bond_vectors(self):
        nres = self.len()
        vectors = num.zeros((nres, 4, 3))
        for res_num, res in zip(range(nres), self.iter_residues()):
            chi_atoms = amino_acids.chi_definitions[res.res_name]
            for chi_num in range(len(chi_atoms)):
                chi_atom_names = chi_atoms[chi_num].split()
                a3_name, a4_name = chi_atom_names[2], chi_atom_names[3]

                a3, a4 = res.get_atom(a3_name), res.get_atom(a4_name)
                if res.res_name == "ILE" and a4 is None and a4_name == "CD1": a4 = res.get_atom("CD") # ILE.CD1 can also be called ILE.CD
                if a3 is None or a4 is None:
                    print "WARN: can't find either atom '%s' or '%s' in res '%s'" % (a3_name, a4_name, res)
                else:
                    vec = num.array([a4.x - a3.x, a4.y - a3.y, a4.z - a3.z])
                    vectors[res_num, chi_num, :] = vec/length(vec)
        return vectors

    # calculate all chis in this protein
    # returns array of (nres, 4)
    def calc_chis(self):
        nres = self.len()
        chis = num.zeros((nres, 4), num.float32)
        for res_num, res in zip(range(nres), self.iter_residues()):
            chis[res_num, :] = res.calc_chis()
        return chis

    def calc_phi_psi_omega(self):
        nres = self.len()
        phis = num.zeros((nres, 3), num.float32)
        for res_num, res in zip(range(nres), self.iter_residues()):
            phis[res_num, :] = res.calc_phi_psi_omega()
        return phis
    
    # calc rmsd between this pdb & pdb2 for the given residues in self & the given atom names
    # (default is all atoms); throws KeyError if residues don't have the same atom names
    def calc_rmsd(self, pdb2, atom_names=["CA"]):
        if self.len() != pdb2.len(): raise Exception("ERROR PDB.calc_rmsd: pdbs not same length: %d != %d" % (self.len(), pdb2.len()))

        sum, count = 0., 0.
        for res1, res2 in zip(self.iter_residues(), pdb2.iter_residues()):
            #print res1, res2
            for atom in res1.iter_atoms():
                if atom.atomName in atom_names:
                    atom2 = res2.get_atom(atom.atomName)
                    #print atom, atom2
                    if atom2 is None: raise Exception("ERROR PDB.calc_rmsd:  can't find atom '%s %s' in pdb '%s'" % (res2, atom.atomName, pdb2))
                    sum += atom.calc_dist2(atom2)
                    count += 1
        rmsd = math.sqrt(sum/count)
        return rmsd

    # returns matrix of xzy coords for an atom type
    def get_atom_xyz_matrix(self, atom_name):
        dat = []
        for res in self.iter_residues():
            atom = res.get_atom(atom_name)
            if atom is None: xyz = [num.nan,num.nan,num.nan]
            else: xyz = [atom.x, atom.y, atom.z]
            dat.append(xyz)
        return num.array(dat, 'd')
    # returns matrix of shape nresX3
    def get_ca_xyz_matrix(self):
        return self.get_atom_xyz_matrix("CA")

    def _load_atom_lines(self, lines, heavy_only=False):
        last_resid = None

        if heavy_only: atom_lines = filter(lambda l: l[:4] == "ATOM" and not is_hydrogen(l[12:16]), lines)
        else: atom_lines = filter(lambda l: l[:4] == "ATOM", lines)

        self._atoms = [PDBAtom(line) for line in atom_lines]

        for atom in self._atoms:
            resid = "%s@%s"%(atom.chain, atom.resNum)

            # new residue
            if last_resid != resid:
                last_resid = resid

                self._chain_names[atom.chain] = atom.chain 
                residue = PDBResidue(atom.resNum, atom.resName, atom.chain, self)
                self._residues[resid] = residue
                self._res_order.append(resid)
            else:
                residue = self._residues[resid]
                if residue.res_name != atom.resName:
                    print "ERROR: found residue atoms with different aa types in %s: %s & %s" % (self.fn, residue, atom.resName)
                    continue
                if atom.atomName in residue._atoms:
                    # atom already exists in this residue, probably alt loc; ignore
                    continue
                
            residue._atoms[atom.atomName] = atom

    # key is resid and value is 'bfact'
    def get_pdb_set_bfactor_str(self, bfact_values, default_val=0):
        s = ""
	chain = self.get_chain_names()[0]
        
	residues = self.iter_residues()
	res_map = {}
	for bfact, res in zip(bfact_values, residues): res_map[res.id] = bfact
	self.set_bfactors(res_map)
	s += self.get_pdb_str() + "\n"
        return s

    # key is resid and value is 'bfact'
    def set_bfactors(self, residue_map, default_val=0):
        for atom in self.iter_atoms():
            atom.bFactor = default_val
            
        for resid, bfact in residue_map.items():
            chain, resnum = parse_resid(resid)
            res = self.get(chain, resnum)
            for atom in res._atoms.values():
                atom.bFactor = bfact

    # key is resid and value is 'occ'
    def set_occumpancies(self, residue_map, default_val=0):
        for atom in self.iter_atoms():
            atom.occupancy = default_val
            
        for resid, occupancy in res_map.items():
            chain, resnum = parse_resid(resid)
            res = self.get(chain, resnum)
            for atom in res._atoms.values():
                atom.occupancy = occupancy

    # transform the coordinates of all atoms using the results of MAMMOTH
    # expects
    def transform(self, tmatrix, pvect, evect):
	def E_transform(v,matrix,tP,tE):
	    ans = [0.0]*3
	    for i in range(3):
		for j in range(3):
		    ans[i] = ans[i] + matrix[i][j]*(v[j]-tE[j]) 
		ans[i] = ans[i] + tP[i]
	    return ans

        #matrix = map(lambda x:map(float,string.split(x)[1:]), popen('grep -A3 "Transformation Matrix" %s'%file).readlines()[1:])
	#P_translation = map(float,string.split(popen('grep -A1 "Translation vector (Pred" %s' %file).readlines()[1])[1:])
	#E_translation = map(float,string.split(popen('grep -A1 "Translation vector (Exp" %s'  %file).readlines()[1])[1:])

	for atom in self._atoms:
	    pos = E_transform([atom.x, atom.y, atom.z], tmatrix, pvect, evect)
	    atom.x, atom.y, atom.z = pos[0], pos[1], pos[2]

class PDBTrajectory:
    # models start indexing at 1
    def __init__(self, traj_fn, start_model=1, end_model=None, model_increment=1, cache_to_disk=False):
        self.traj_fn = traj_fn
        self.name = os.path.basename(traj_fn).replace(".lst","").replace(".pdb","")
        self.start_model, self.end_model, self.model_increment = start_model, end_model, model_increment
        self.file_handle = None
        self.npdb = self.parse_len()

        if start_model < 1: raise Exception("PDBTrajectory.__init__(): Invalid start_model '%d'" % start_model)
        if end_model != None and (end_model < 1 or end_model < start_model): raise Exception("PDBTrajectory.__init__(): Invalid end_model '%d'" % end_model)
        if model_increment < 1: raise Exception("PDBTrajectory.__init__(): Invalid end_model '%d'" % model_increment)

    # get the number of pdb files in a trajectory
    def parse_len(self):
        if self.traj_fn.endswith(".pdb"): # pdb trajectory
            return int(commands.getoutput("awk '/^MODEL /' %s | wc -l" % self.traj_fn).split()[0])
        elif self.traj_fn.endswith(".pdb.gz"): # pdb trajectory
            return int(commands.getoutput("zcat %s | awk '/^MODEL /' | wc -l" % self.traj_fn).split()[0])
        else: # list of pdb files
            return int(commands.getoutput("wc -l %s" % self.traj_fn).split()[0])

    def get_fasta_str(self, res_subset=None):
        s = ""
	for pdb in self.get_next_pdb():
	    seq = []
	    for res in pdb.iter_residues():
		if res_subset is None or int(res.res_num) in res_subset: seq.append(res.resChar)
	    try: pdb_name = utils.parse_pdbname(pdb.fn)
	    except: pdb_name = ""

	    s += "> %s" % (pdb_name) + "\n"
	    s += "".join(seq) + "\n"
	return s
	    

    def get_nh_vectors(self):
        pdbs = []
        for pdb in self.get_next_pdb(): pdbs += [pdb]

        num_pdbs = len(pdbs)
        num_res = max([res.res_num for res in pdbs[0]._residues.values()])

        # extract nh_vectors
        ns = num.zeros((num_pdbs, num_res, 3))
        hs = num.zeros((num_pdbs, num_res, 3))
        nh_vectors = num.zeros((num_pdbs, num_res, 3))
        for pdb_num in range(num_pdbs):
            ns[pdb_num,:,:], hs[pdb_num,:,:], nh_vectors[pdb_num,:,:] = pdb.get_amide_bond_vectors(normalize=False)
        return ns, hs, nh_vectors

    def _model_in_range(self, model_count):
        return  model_count >= self.start_model and (self.end_model is None or model_count <= self.end_model) and ((model_count - self.start_model) % self.model_increment == 0)


    # calculate the lindemann parameter (Karplus JMB 1999)
    def calc_lindemann_param(self, include_atoms=None, exclude_atoms=None):
        num_pdbs = self.parse_len()
        atoms = num.zeros((num_pdbs, 10000, 3))
        num_atoms = -1
                                 
        for pdb_num, pdb in izip(range(num_pdbs), self.get_next_pdb()):
            heavy_atoms = filter(lambda a: a.get_elem() != "H" and
                                 (include_atoms is None or a.atomName in include_atoms) and (exclude_atoms==None or a.atomName not in exclude_atoms), pdb._atoms)
            
            if num_atoms == -1: num_atoms = len(heavy_atoms)
            else: assert(num_atoms == len(heavy_atoms))
            
            for atom_num, atom in zip(range(num_atoms), heavy_atoms):
                atoms[pdb_num, atom_num, :] = atom.get_xyz()
                #print "XXX", atoms_array[pdb_num, atom_num, :]

        atoms = atoms[:,0:num_atoms,:]

        atom_means = atoms.mean(axis=0)
        #print atom_means.shape

        for pdb_num in range(num_pdbs): atoms[pdb_num, :, :]  -= atom_means
        atoms_disp2 = num.sum(atoms*atoms, axis=2)
        #print atoms_disp2.shape
        uncorr_lp = num.mean(atoms_disp2.mean(axis=0))
        lp = num.sqrt(uncorr_lp) / 4.5
        return lp, num_atoms
    
    # get atoms from a pdb trajectory (pdb format with structures in different MODELs)
    # or a pdb list (newline separated list of pdb filenames). Names models according to their MODEL field in pdb files or their index in pdb lists.
    # Uses a generator to return a PDB object
    # if return_pdb_txt is true, then return the text rather than a PDB object
    def get_next_pdb(self, return_pdb_lines=False):
        fn = self.traj_fn

        assert(os.path.exists(fn))
        if fn.endswith(".pdb") or fn.endswith(".pdb.gz"): # pdb trajectory
            if fn.endswith(".pdb"):
                cmd = "egrep '^(ATOM|MODEL|REMARK) ' %s 2>/dev/null" % fn
            elif fn.endswith(".pdb.gz"):
                cmd = "zcat %s | egrep '^(ATOM|MODEL|REMARK) '" % fn
            self.file_handle = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE).stdout
                                       
            parsed_model = 0
	    pdb_lines = []
	    model_count = 1
	    for line in self.file_handle:
                if line[:6] in ("ATOM  ", "REMARK"):
		    pdb_lines.append(line) # ignore if MODEL hasn't been reached yet
                elif line[:5] == "MODEL":
		    if self.end_model != None and model_count > self.end_model: break
                    else:
                        if not self._model_in_range(model_count): continue

		        atom_lines = filter(lambda l: l.startswith("ATOM  "), pdb_lines)
			if len(atom_lines) == 0:
			    pdb_lines = []
			    continue
			
		        if return_pdb_lines: yield pdb_lines
			else: yield PDB(pdb_lines, model_num=parsed_model)
                        model_count += 1
                    parsed_model = int(line.split()[1])
                    pdb_lines = []
            if self._model_in_range(model_count):
                if return_pdb_lines: yield pdb_lines
                else: yield PDB(pdb_lines, model_num=parsed_model)
        else: # list of pdb files
            pdb_fns = open(fn).readlines()
            for model_count, pdb_fn in zip(range(1, len(pdb_fns)+1), pdb_fns):
	        if self.end_model != None and model_count > self.end_model: break
	        elif not self._model_in_range(model_count): continue
                pdb_fn = pdb_fn.strip()
                #print pdb_fn
		
                if not os.path.exists(pdb_fn): raise Exception("ERROR can't find file "+ pdb_fn)
                if pdb_fn.endswith(".pdb"):
                    pdb_lines = utils.run("egrep '^ATOM ' %s" % pdb_fn).split("\n")
                elif pdb_fn.endswith(".pdb.gz"):
                    pdb_lines = utils.run("zcat %s | egrep '^ATOM '" % pdb_fn).split("\n")
                else:
                    print "ERROR unrecognized pdb filetype '%s'"  % pdb_fn
                    sys.exit(1)
                #atom_lines = filter(lambda line: line[:4] == "ATOM", all_lines)

		if return_pdb_lines:
		    yield ["REMARK  99 FILE "+pdb_fn] + pdb_lines
	        else:
		    yield PDB(pdb_lines, fn=pdb_fn, model_num=model_count)

#    def close(self):
#        if self.file_handle != None: self.file_handle.close()

    def calc_S2s(self):
        # load the amide and chi vector arrays for all pdb files
        pdb1 = None
        pdb_num = 0
        amide_vectors_list, chi_vectors_list = [], []
        print "Loading trajectory: ", self.name
        for pdb in self.get_next_pdb():
            print "Loaded pdb: ", pdb
            sys.stdout.flush()
            if pdb_num == 0:
                pdb1 = pdb
                nres = pdb.len()

            if nres != pdb.len():
                print "ERROR: structures in the trajectory don't have the same number of residues (expecting '%d')" % nres
                sys.exit(1)

	    ns, hs, nh_vectors = pdb.get_amide_bond_vectors()
            amide_vectors_list.append(nh_vectors)
            chi_vectors_list.append(pdb.get_chi_bond_vectors())
            pdb_num += 1
        npdb = pdb_num

        if npdb == 0: raise Exception("ERROR calc_S2s: no pdb files loaded")

        # load the data into arrays
        amide_vectors = num.zeros((npdb, nres, 3))
        chi_vectors = num.zeros((npdb, nres, 4, 3))
        for pdb_num, amide_vectors_pdb, chi_vectors_pdb in zip(range(len(amide_vectors_list)), amide_vectors_list, chi_vectors_list):
            amide_vectors[pdb_num, :, :] = amide_vectors_pdb
            chi_vectors[pdb_num, :, :, :] = chi_vectors_pdb
        
        amide_S2s = num.zeros((nres+1)) + num.nan # 1-based indexing
        chi_S2s = num.zeros((nres+1, 4)) + num.nan
        for res_num in range(nres):
            amide_S2s[res_num+1] = calc_S2_from_vector_array(amide_vectors[:, res_num, :])
            for chi_num in range(4):
                chi_S2s[res_num+1, chi_num] = calc_S2_from_vector_array(chi_vectors[:, res_num, chi_num, :])

        amide_S2s[amide_S2s==-.5] = num.nan
        chi_S2s[chi_S2s==-.5] = num.nan
        
        return pdb1, amide_S2s, chi_S2s

    def calc_rmsds(self, ref_pdb):
        rmsds = {}
        for pdb in self.get_next_pdb():
            rmsds[pdb.model_num] = pdb.calc_rmsd(ref_pdb)
        return rmsds

    # returns list of rmsd per residue
    # chain1 is where to take residues from, chain2 is the chain id of these residues in chain 2
    def calc_rmsd_over_sequence(self, ref_pdb, atom_names=["CA"]):
        ref_residues = [res for res in ref_pdb.iter_residues()]
	residue_map = {}
	for pdb in self.get_next_pdb():
	    res_ind = 0
	    for res in pdb.iter_residues():
	        residue_map.setdefault(res_ind, []).append(res)
		res_ind += 1
	nres = len(residue_map.keys())
		
	rmsds = []
	for res_ind in sorted(residue_map.keys()):
	    rmsd = num.mean([ref_residues[res_ind].calc_rmsd(res, atom_names) for res in residue_map[res_ind]])
	    rmsds.append(rmsd)
        return rmsds

    def get_diff_dist_matrix_str(self, res_range=None, scaled=False):
        s = "# TRAJECTORY: " + self.name + "\n"
	diff_dist_matrix = self.diff_dist_matrix(res_range, scaled=False)
	s += "#" + str(diff_dist_matrix.shape) + "\n"
	s += "# 1D MEAN" + utils.fmt_floats(num.mean(diff_dist_matrix, axis=0), digits=6) + "\n"
	s += utils.arr2str2(diff_dist_matrix, precision=6) + "\n"
	return s

    # calculate the difference distance matrix
    # 1) calculate distance matrices for each structure
    # 2) for each pair of structures, take the matrix of absolute value of the difference between the distances
    # 3) average these
    def diff_dist_matrix(self, res_range=None, scaled=False):
        if res_range != None: assert(len(res_range) == 2)
        
        dist_matrices = []
        for pdb in self.get_next_pdb():
            ca_xyz = pdb.get_ca_xyz_matrix()
            if res_range != None: ca_xyz = ca_xyz[res_range[0]-1:res_range[1], :]
            dist_matrix = calc_distance_matrix(ca_xyz)
            dist_matrices.append(dist_matrix)

        scaled_diff_dist_matrix = num.zeros(dist_matrices[0].shape, 'd')
        count = 0
        for i in range(len(dist_matrices)):
            for j in range(i+1, len(dist_matrices)):
                diff_dist_matrix = num.abs(dist_matrices[i] - dist_matrices[j])
                if scaled:
                    scale = num.max(diff_dist_matrix)
                    if scale == 0: continue
                    diff_dist_matrix /= scale
                scaled_diff_dist_matrix += diff_dist_matrix
                count += 1
        #print >> sys.stderr, count
        scaled_diff_dist_matrix /= count
        if scaled:
            scaled_diff_dist_matrix /= num.max(scaled_diff_dist_matrix)
        return scaled_diff_dist_matrix
    
if __name__ == '__main__':
    # Parse the input arguments
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    parser.add_option("-t", "--traj_fns", default=None, type="string", help="filenames to load PDB trajectories from; colon separated (e.g. fn1:fn2)")
    parser.add_option("-o", "--order_params", action="store_true", default=False, help="calculate order parameters for amides and chi dihedrals over the trajectory")
    parser.add_option("-p", "--print_info", action="store_true", default=False, help="print info about the trajectory")
    parser.add_option("-s", "--start_model", type="int", default=1, help="model number to start from")
    parser.add_option("-e", "--end_model", type="int", default=None, help="model number to end at")
    parser.add_option("-i", "--model_increment", type="int", default=1, help="increment model numbers by this value")
    parser.add_option("-r", "--calc_rmsds", action="store_true", default=False, help="calculate rmsds for the trajectory relative to the reference pdb")
    parser.add_option("--diff_dist_matrix", action="store_true", default=False, help="calculate the average scaled difference distance matrix")
    parser.add_option("-f", "--ref_pdb_fn", type="string", default=None, help="the reference pdb file to load")
    parser.add_option("-l", "--plot", action="store_true", default=False, help="make plots according to actions specified by other arguments")
    parser.add_option("-d", "--dssp_fn", type="string", default=None, help="filename containing DSSP output")
    parser.add_option("-x", "--lindemann", action="store_true", default=False, help="calculate the lindemann parameter")
    parser.add_option("--res_range", type="string", default=None, help="the range of residue numbers to use in the pdb processing (e.g. '1:72')")
    parser.add_option("--set_bfactors", type="string", default=None, help="change the bfactor fields to the specified values and output the pdb (i.e. bfact_res1,bfact_res2,bfact_res3)")
    parser.add_option("--output_seq", action="store_true", default=False, help="extract the sequence from the pdb files to stdout")    
    parser.add_option("--output_fasta", action="store_true", default=False, help="extract the sequence from the pdb files to stdout in fasta format")    
    parser.add_option("--output_struct_info", type="string", default="", help="output structure info (chis)")
    parser.add_option("--res_subset", type="string", default=None, help="colon separated list of residues to process for some commands")
    parser.add_option("--transform", type="string", default=None, help="transform pdb coordinates using output of MAMMOTH; expects 'R(1,1)  R(1,2)  R(1,3)  R(2,1)  R(2,2)  R(2,3)  R(3,1)  R(3,2)  R(3,3)  Xc      Yc      Zc      Xt      Yt      Zt'")
    
    (opts, args) = parser.parse_args()
    if opts.start_model < 1: parser.error("ERROR invalid value for --start_model")
    if opts.model_increment <= 0: parser.error("ERROR invalid value for --model_increment")
    if opts.traj_fns is None: parser.error("ERROR --traj_fns option required")
    if opts.res_range != None:
        res_range = map(int, opts.res_range.split(":"))
        assert(len(res_range) == 2)
    else: res_range = None
    if opts.res_subset != None: res_subset = map(int, opts.res_subset.split(":"))
    else: res_subset = None

    traj_fns = filter(lambda fn: fn.strip()!="", opts.traj_fns.split(":"))
    trajs = [PDBTrajectory(traj_fn, opts.start_model, opts.end_model, opts.model_increment) for traj_fn in traj_fns]
    #print "Done loading PDBs"
    sys.stdout.flush()

    # Load SS info
    ss_info = None
    if opts.dssp_fn != None: ss_info = SS_info(opts.dssp_fn)

    if opts.print_info:
        for traj in trajs:
            print "TRAJECTORY: ", traj.name
            for pdb in traj.get_next_pdb():
                print pdb

    if opts.lindemann:
        for traj in trajs:
            print "TRAJECTORY: ", traj.name
            lp_all, num_all_atoms = traj.calc_lindemann_param()
            lp_bb, num_bb_atoms = traj.calc_lindemann_param(include_atoms=["N","C","CA","O"])
            lp_sc, num_sc_atoms = traj.calc_lindemann_param(exclude_atoms=["N","C","CA","O"])
        print 'lps: all=%.3f [%d], bb=%.3f [%d], sc=%.3f [%d]' % (lp_all, num_all_atoms, lp_bb, num_bb_atoms, lp_sc, num_sc_atoms)

    if opts.diff_dist_matrix:
        for traj in trajs:
	    print traj.get_diff_dist_matrix_str()
            
    if opts.order_params:
        amide_S2s_list = []
        for traj in trajs:
            print "TRAJECTORY: ", traj.name
            pdb1, amide_S2s, chi_S2s = traj.calc_S2s()
            chain = pdb1.get_chain_names()[0]
            print "RESIDUE: NH-S2  CHI1-S2 CHI2-S2 CHI3-S2 CHI4-S2"
            for res_num, res in zip(range(1,amide_S2s.shape[0]), pdb1.iter_residues()):
                ch = chi_S2s[res_num, :]
                print "S2s %3d %s: %5.2f %7.2f %7.2f %7.2f %7.2f" % (res.res_num, res.res_name, amide_S2s[res_num], ch[0], ch[1], ch[2], ch[3])
            amide_S2s_list.append(amide_S2s)

        if opts.plot:
            mean_S2s, stdev_S2s = num.zeros(amide_S2s_list[0].shape)+num.nan, num.zeros(amide_S2s_list[0].shape)+num.nan
            for res_num, res in zip(range(1,amide_S2s_list[0].shape[0]), pdb1.iter_residues()):
                res_S2s = [amide_S2s[res_num] for amide_S2s in amide_S2s_list]
                mean_S2s[res_num] = num.mean(res_S2s)
                stdev_S2s[res_num] = num.std(res_S2s)
            #print amide_S2s_list
            #print mean_S2s
            #print stdev_S2s
            import plotting
            plotting.plot_S2s_over_sequence([mean_S2s], "mean S2s", "mean_amide_S2s.png", ss_info, False, errors_list=[stdev_S2s])
            plotting.plot_S2s_over_sequence(amide_S2s_list, [traj.name for traj in trajs], "all_amide_S2s.png", ss_info, False)
            #if opts.plot:
            #    pylab.plot(amide_S2s, "-", label=traj.name)
        #if opts.plot:
            # add faded background in SS regions
        #    if opts.dssp_fn != None:
         #       dssp_data = parse_dssp_txt(open(opts.dssp_fn).read())
          #      ss_info = SS_info(dssp_data)
           #     for res_num in ss_info.get_res_nums():
            #        if ss_info.is_structured(res_num):
                        #print "Found SS:", res_num
             #           x=res_num
              #          pylab.fill([x-.5,x-.5,x+.5,x+.5], [0,1,1,0], alpha=.3, edgecolor='w')
                        
            #pylab.title("NH order parameters")
            #pylab.ylabel("Order parameter")
            #pylab.xlabel("Residue number")
            #pylab.ylim(ymax=1)
            #pylab.grid()
            #pylab.legend([traj.name for traj in trajs], prop=matplotlib.font_manager.FontProperties(size='6'), loc='lower right')
            #plot_fn = "amide_S2s.png"
            #print "Writing ", plot_fn
            #pylab.savefig(plot_fn)

    if opts.calc_rmsds:
        if opts.ref_pdb_fn is None: raise Exception("ERROR: ref_pdb needed to calculate rmsds")
        ref_pdb = PDB(open(opts.ref_pdb_fn).readlines(), fn=opts.ref_pdb_fn)
        #for res in ref_pdb.iter_residues(): print res
        
        print "Calculating RMSDs relative to " + str(ref_pdb)
	for traj in trajs:
		rmsds = traj.calc_rmsds(ref_pdb)
		for model_num, rmsd in sorted(rmsds.items()):
			print "Model %d, RMSD %.2f" % (model_num, rmsd)

    # e.g. bfact_res1,bfact_res2
    # assumes one pdb !!
    if opts.set_bfactors:
        for pdb in trajs[0].get_next_pdb():
            chain = pdb.get_chain_names()[0]
        
            bfact_data = map(float, opts.set_bfactors.split(","))
            residues = pdb.iter_residues()
            res_map = {}
            for bfact, res in zip(bfact_data, residues): res_map[res.id] = bfact

            pdb.set_bfactors(res_map)
            print pdb.get_pdb_str()
            break
            #res_info, bfact = res_data.split(":")
            #if res_info.find("@") != -1: resid = res_info
            #else: resid = make_resid(chain, res_info)
        
    if opts.output_seq:
        for traj in trajs:
            for pdb in traj.get_next_pdb():
                seq = []
                for res in pdb.iter_residues():
                    seq.append(res.resChar)
		try: pdb_name = os.path.basename(pdb.fn).replace(".pdb","").replace(".gz","")
		except: pdb_name = ""

                print ",".join([traj.name, str(pdb.model_num), pdb_name, "".join(seq)])

    if opts.output_fasta:
        for traj in trajs:
	    print traj.get_fasta_str()
	    
    # assumes one pdb!!!
    # transform the coords of a pdb using results from MAMMOTH
    # expects the transformation matrix and vectors in the format:
    #    R(1,1)  R(1,2)  R(1,3)  R(2,1)  R(2,2)  R(2,3)  R(3,1)  R(3,2)  R(3,3)  Xc      Yc      Zc      Xt      Yt      Zt
    if opts.transform != None:
        vals = map(float, opts.transform.split())
	assert(len(vals) == 15)
	tmatrix, pvect, evect = num.array(vals[:9]).reshape((3,3)), num.array(vals[9:12]), num.array(vals[12:15])

	print "REMARK 40 Transformation matrix: ", str(tmatrix).replace("\n", " ")
	print "REMARK 40 Translation vector (pred): ", pvect
	print "REMARK 40 Translation vector (exp): ", evect

        for pdb in trajs[0].get_next_pdb():
	    pdb.transform(tmatrix, pvect, evect)
	    print pdb.get_pdb_str()
	    break

    if opts.output_struct_info in ("chis"):
        for traj in trajs:
            for pdb in traj.get_next_pdb():
	        if opts.output_struct_info == "chis":
		    chis = pdb.calc_chis()
		    for res in pdb.iter_residues():
		        chain = res.chain
			if chain == " ": chain = "_"
		        print "CHIS %s #%d %3s %s %3s" % (traj.traj_fn, pdb.model_num, res.res_name, chain, res.res_num),
			print utils.fmt_floats(chis[res.res_num-1,:], digits=0, len=5)

