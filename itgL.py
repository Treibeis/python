from radio import *

if __name__ == "__main__":
	tag = 1
	sca = 0
	ncore = 4
	ldir = ['halo1_wdm/','halo1_cdm/']#['NL4_zoom_wdm/halo1/', 'NL4_zoom_cdm/halo1/']
	rep0 = 'halo1/'
	#ldir = ['halo1_wdm/','halo1_cdm/']
	Tsh = 1e4
	mode = int(sys.argv[1])
	bins = int(sys.argv[2])
	if len(sys.argv)>4:
		low, up = int(sys.argv[3]), int(sys.argv[4])
	else:
		low, up = 1900, 2000

	if tag==0:
		out0 = []
		out1 = []
		lu0, lu1 = [], []
		for sn in range(0,24):
			#if mode==0:
			#	mesh = mesh_3d(low, up, bins, low, up, bins, low, up, bins)
			#	d0 = grid(mesh,sn,box =[[low]*3,[up]*3],ncore=ncore,Tsh=Tsh,rep=ldir[0])
			#	d1 = grid(mesh,sn,box =[[low]*3,[up]*3],ncore=ncore,Tsh=Tsh,rep=ldir[1])
			#else:
			#	d0 = read_grid(sn,nb = bins,rep=ldir[0])
			#	d1 = read_grid(sn,nb = bins,rep=ldir[1])
			#out0.append(luminosity(d0s,ncore=ncore))
			#out1.append(luminosity(d1,ncore=ncore))
			lu0.append(luminosity_particle(sn,ldir[0],[[low]*3,[up]*3],ncore=ncore))
			lu1.append(luminosity_particle(sn,ldir[1],[[low]*3,[up]*3],ncore=ncore))
		#out0 = np.array(out0).T
		#out1 = np.array(out1).T
		lu0 = np.array(lu0).T
		lu1 = np.array(lu1).T
		#totxt(rep0+'luminosity_z_'+str(bins)+'.txt',[out0[0],out0[1],out1[1]],['z', 'L(WDM)', 'L(CDM)'],1,0)
		totxt(rep0+'luminosityH2_z_'+str(low)+'_'+str(up)+'.txt',[lu0[0],lu0[1],lu0[2],lu0[3],lu1[1],lu1[2],lu1[3]],['z', 'L(WDM)_H2', 'L(WDM)_HD', 'ns(WDM)', 'L(CDM)_H2', 'L(CDM)_HD', 'ns(CDW)'],1,0)
	else:
		#out0 = np.array(retxt(rep0+'luminosity_z_'+str(bins)+'.txt',3,1,0))
		#out1 = np.array([out0[0], out0[2]])
		lu0 = np.array(retxt(rep0+'luminosityH2_z_'+str(low)+'_'+str(up)+'.txt',7,1,0))
		lu1 = np.array([lu0[0], lu0[4],lu0[5],lu0[6]])

	plt.figure()
	plt.plot(lu0[0][lu0[1]>0],lu0[1][lu0[1]>0],label='diffuse, '+lmodel[1],marker='^')
	plt.plot(lu1[0][lu1[1]>0],lu1[1][lu1[1]>0],label='diffuse, '+lmodel[0],ls='--',marker='o')
	plt.plot(lu0[0][lu0[3]>0],lu0[3][lu0[3]>0]*0.05*5e33/10,label='core, '+lmodel[1])
	plt.plot(lu1[0][lu1[3]>0],lu1[3][lu1[3]>0]*0.05*5e33/10,label='core ($\epsilon=0.05$, $M_{*}=10\ M_{\odot}$), '+lmodel[0],ls='--')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$L_{\mathrm{H_{2}}}\ [\mathrm{erg\ s^{-1}}]$')
	if sca!=0:
		plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if sca==0:
		plt.savefig(rep0+'LH2_z_'+str(bins)+'.pdf')
	else:
		plt.savefig(rep0+'logLH2_z_'+str(bins)+'.pdf')
	plt.figure()
	plt.plot(lu0[0][lu0[2]>0],lu0[2][lu0[2]>0],label=lmodel[1],marker='^')
	plt.plot(lu1[0][lu1[2]>0],lu1[2][lu1[2]>0],label=lmodel[0],ls='--',marker='o')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$L_{\mathrm{HD}}\ [\mathrm{erg\ s^{-1}}]$')
	if sca!=0:
		plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if sca==0:
		plt.savefig(rep0+'LHD_z_'+str(bins)+'.pdf')
	else:
		plt.savefig(rep0+'logLHD_z_'+str(bins)+'.pdf')
	plt.figure()
	plt.plot(lu0[0][lu0[3]>0],lu0[3][lu0[3]>0],label=lmodel[1],marker='^')
	plt.plot(lu1[0][lu1[3]>0],lu1[3][lu1[3]>0],label=lmodel[0],ls='--',marker='o')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$M_{\mathrm{sink}}\ [M_{\odot}]$')
	if sca!=0:
		plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if sca==0:
		plt.savefig(rep0+'Msink_z_'+str(bins)+'.pdf')
	else:
		plt.savefig(rep0+'logMsink_z_'+str(bins)+'.pdf')
	plt.show()

	"""
	plt.figure()
	plt.plot(out0[0][out0[1]>0],out0[1][out0[1]>0],label=lmodel[1],marker='^')
	plt.plot(out1[0][out1[1]>0],out1[1][out1[1]>0],label=lmodel[0],ls='--',marker='o')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$L\ [\mathrm{erg\ s^{-1}}]$')
	if sca!=0:
		plt.yscale('log')
	plt.legend()
	plt.tight_layout()
	if sca==0:
		plt.savefig(rep0+'L_z_'+str(bins)+'.pdf')
	else:
		plt.savefig(rep0+'logL_z_'+str(bins)+'.pdf')
	plt.figure()
	plt.plot(out0[0][out0[1]>0],out0[1][out0[1]>0]/out1[1][out0[1]>0],label=r'$L_{\mathrm{WDM}}/L_{\mathrm{CDM}}$')
	plt.plot(out0[0][out0[1]>0],[1.0 for x in range(out0[0][out0[1]>0].shape[0])],color='k',ls='--')
	plt.xlabel(r'$z$')
	plt.ylabel(r'$L_{\mathrm{WDM}}/L_{\mathrm{CDM}}$')
	if sca!=0:
		plt.yscale('log')
	if sca==0:
		plt.savefig(rep0+'Lrat_z_'+str(bins)+'.pdf')
	else:
		plt.savefig(rep0+'logLrat_z_'+str(bins)+'.pdf')
	plt.tight_layout()
	plt.show()
	"""




