#C. Raithel, May 2016

import numpy as np
import os
import matplotlib
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def main():	

	filename = 'inversion_output.txt'

	matplotlib.rcParams['mathtext.fontset'] = 'stix'			#set latex font
	matplotlib.rcParams['font.family'] = 'STIXGeneral'
	matplotlib.rcParams.update({'font.size': 14})
	matplotlib.rc('xtick', labelsize=10)
	matplotlib.rc('ytick', labelsize=10)  
	matplotlib.rcParams['figure.figsize'] = 11,7
	matplotlib.rcParams['figure.dpi'] = 900

	f = open(filename,'r')
	f.readline()
	nMC = f.readline().split()[0]

	labels = ['posterior','P1','P2','P3','P4','P5']
	data = np.genfromtxt(filename,names=labels,skip_header=3)

	plotlabels = ['posterior','P$_1$','P$_2$','P$_3$','P$_4$','P$_5$']

	plt.figure(0)
	plt.plot(data['posterior'],color='k')
	plt.xlabel('N')
	plt.ylabel('Posterior Likelihood')
	plt.savefig('correlation.png')	

	#fig, ax = plt.subplots(5,5)

	fig = plt.figure(1)

	data['P1'] = [p/1.e35 for p in data['P1']]
	data['P2'] = [p/1.e35 for p in data['P2']]
	data['P3'] = [p/1.e35 for p in data['P3']]
	data['P4'] = [p/1.e35 for p in data['P4']]
	data['P5'] = [p/1.e35 for p in data['P5']]

	ax1 = plt.subplot2grid((5,5),(0,0))
	ax2 = plt.subplot2grid((5,5),(1,1))
	ax3 = plt.subplot2grid((5,5),(2,2))
	ax4 = plt.subplot2grid((5,5),(3,3))
	ax5 = plt.subplot2grid((5,5),(4,4))
	axes = [ax1, ax2, ax3, ax4, ax5]

	fig.canvas.draw()
	for k in range(5):
		for j in range(4,-1,-1):

			if j==k:
				ax = axes[k]	
				ax.plot(data[labels[k+1]],data['posterior'],'.',markersize=3,color='k')
				ax.set_ylabel('P ('+plotlabels[k+1]+')')
	
			elif j>k:
				ax = plt.subplot2grid((5,5),(j,k),sharex=axes[k])
				ax.plot(data[labels[k+1]],data[labels[j+1]],'.',markersize=3,color='k')
				ax.set_ylabel(plotlabels[j+1])
				ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
			else:
				continue

			for label in ax.yaxis.get_ticklabels()[::2]:
				label.set_visible(False)

									
  			plt.setp(ax.get_xticklabels(), visible=False)			#If not on the bottom row, get rid of xtick labels
			ax.xaxis.get_offset_text().set_visible(False)
			
			if j==4:							#Include an xlabel on bottom row
				ax.set_xlabel(plotlabels[k+1])				
				ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
				for label in ax.get_xticklabels()[1::2]:
					label.set_visible(True)

				

	fig.text(0.65,0.85,'Pressure units ($10^{35}$ g cm$^{-1}$ s$^{-2}$)')
	fig.text(0.65,0.89,'N$_{\\rm MC}$: '+str(nMC))

	fig.tight_layout()
	fig.subplots_adjust(hspace=0.055)   
	fig.subplots_adjust(wspace=.68)   
	fig.savefig('posterior.png')

	plt.close('all')
	
	'''
	plt.show(block=False)
	raw_input("<Hit enter to close>")
	plt.close('all')
	'''


if __name__=="__main__":
	main()

