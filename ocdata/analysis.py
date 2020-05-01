"""
This submodule contains the scripts for VASP calculations
post-analysis:

(1) Bader Charges Analysis
(2) pDOS Analysis (TODO)
"""

__author__ = 'Javier Heras-Domingo'
__email__ = 'jherasdo@andrew.cmu.edu'

import os
import sys
import pickle
from shutil import which

import warnings
warnings.filterwarnings("ignore")



#Run Bader Analysis after vasp calculation
def run_bader(outdir='./'):
	"""
	Arguments
	--------------
	outdir [str]: Directory of the vasp job

	It returns bader.pkl file with bader analysis as a dict, 
	where charge_transfer should be the interesting feature
	per atom.

	Example
	--------------
	from ocdata.analysis import run_bader

	run_bader(outdir='./job_path')
	"""
	
	#Add bader exec into the path
	os.environ['PATH'] += os.pathsep + os.path.abspath('./ocdata')

	#Import pmg bader caller
	from pymatgen.command_line.bader_caller import bader_analysis_from_path

	#Check outdir exists
	assert os.path.exists(outdir), 'outdir does not exists!'

	#Check required files in outdir
	for r, d, f in os.walk(outdir):
		for file in f:
			chgcar_file = os.path.join(r, 'CHGCAR')
			assert os.path.exists(chgcar_file), 'CHGCAR does not exists!'
			aeccar_0 = os.path.join(r, 'AECCAR0')
			assert os.path.exists(aeccar_0), 'AECCAR0 does not exists!'
			aeccar_2 = os.path.join(r, 'AECCAR2')
			assert os.path.exists(aeccar_2), 'AECCAR2 does not exists!'

	#Bader Analysis
	summary_dict = bader_analysis_from_path(outdir)

	#Export as pickle file
	bader_analysis = open(outdir+'/bader.pkl', 'wb')
	pickle.dump(summary_dict, bader_analysis)

	return
