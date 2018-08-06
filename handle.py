from HMC import *
#from com_red_HMC import *
#from txt import *
from scipy.interpolate import *
import txt
import imp
import matplotlib.ticker as ticker

DX = 10
EDGE = 200

linestyle = ['-','--',':','-.','-','--',':','-.','-','--',':','-.']
linewidth = [1,1,1,1,2,2,2,2,0.5,0.5,0.5,0.5]
num1 = [4, 4, 5, 7, 7, 8, 9, 5, 3]
num2 = [10, 10, 10, 12, 11, 12, 10, 8, 12, 12]

#D_Z = np.array(retxt('/home/friede/python/d_z_WMAP5.txt', 2, 0, 0))
#ZD = interp1d(D_Z[0],D_Z[1], kind='cubic')
#H_Z = np.array(retxt('/home/friede/python/H_z_WMAP5.txt', 2, 0, 0))
#HZ = interp1d(H_Z[1],H_Z[0], kind='cubic')

#def CRITICAL(z):
#	return 3*HZ(z)**2/8/np.pi/G
#def matter(z):
#	return OMEGA0*(1+z)**3/(OMEGA0*(1+z)**3+1-OMEGA0)

#zlist = [0, 0.16442351131441257, 0.3558821137017867, 0.5788210117650425, 0.8384163062564247, 1.1406951705887787, 1.4926757871907927, 1.9025302475524395, 2.3797744625513055, 2.9354888471347733]

def V_200(fcen, z, tola = 0.01, C = 3):
	num = len(fcen['info'])
	outr = []
	outm = []
	ratio1 = []
	ratio2 = []
	for i in range(num):
		m = 10**(fcen['info'][i][0]-10)
		cen = np.array([fcen['info'][i][x+1] for x in range(3)])
		part = np.array(np.matrix(fcen['part'][i][1]).transpose())
		r0 = Virial(fcen['info'][i][0], z)
		def error(r):
			M = 0
			count = 0
			for x in range(len(part)):
				dis = np.array([part[x][y]-cen[y] for y in range(3)])
				if np.linalg.norm(dis)<=r:
					M += part[x][7]
					count += 1
			#print(count/len(fcen['part'][i][1][0]))
			return M-200*CRITICAL(z)*(4*np.pi*(r/(1+z))**3/3)*(OMEGA0-OMEGA_B)/OMEGA0
		#rv = fmin_bfgs(error, r0)[0]
		#rv = brent(error)
		radius = r0*C
		er = error(r0)
		count = 0
		while abs(er)>=tola*m:
			if er>0:
				radius += tola*r0
			else:
				radius = radius - tola*r0
			er = error(radius)
			count += 1
			if count>=C/tola:
				break
		rv = radius
		mv = 200*CRITICAL(z)*(4*np.pi*(rv/(1+z))**3/3)
		outr.append(rv)
		outm.append(mv)
		R = (mv/m)**(1/3)
		#print(r0, rv, R, error(rv)/m)
		ratio1.append(rv/r0)
		ratio2.append(R)
	return [ratio1, ratio2, outr, outm]

def tree1(data1, zli = zlist1, i = 0, fs=(18,12)):
	fig = plt.figure(figsize=fs)
	plt.plot([zli[0],zli[1],zli[2]],[data1[0][0][i],data1[1][0][i],data1[2][0][i]],color='k',marker='*',label="-H'10-")
	plt.plot([zli[2],zli[3]],[data1[2][0][i],data1[3][0][i]],color='gray',marker='.')
	plt.plot([zli[2],zli[3],zli[4],zli[5],zli[6],zli[7],zli[8],zli[9]],[data1[2][0][i],data1[3][1][i],data1[4][2][i],data1[5][1][i],data1[6][1][i],data1[7][1][i],data1[8][3][i],data1[9][3][i]],color='k',marker='*',ls='--',label="-H'31-H'42-H'51-H'61-H'71-H'83-")
	plt.plot([zli[2],zli[3],zli[4],zli[5],zli[6]],[data1[2][0][i],data1[3][6][i],data1[4][5][i],data1[5][10][i],data1[6][9][i]],color='k',marker='*',ls='-.',label="-H'36-H'45-H'510-")
	plt.plot([zli[3],zli[4]],[data1[3][0][i],data1[4][5][i]],color='gray',ls='--')
	plt.plot([zli[3],zli[4]],[data1[3][0][i],data1[4][0][i]],color='gray',marker='.')
	plt.plot([zli[4],zli[5],zli[6],zli[7]],[data1[4][0][i],data1[5][0][i],data1[6][0][i],data1[7][0][i]],color='k',marker='*',ls=':',lw=2,label="-H'50-H'60-")
	plt.plot([zli[x] for x in range(4,10)],[data1[4][0][i],data1[5][3][i],data1[6][5][i],data1[7][3][i],data1[8][6][i],data1[9][5][i]],color='k',marker='^',label="-H'53-H'65-H'73-H'86-")
	plt.plot([zli[7],zli[8],zli[9]],[data1[7][0][i],data1[8][0][i],data1[9][0][i]],color='k',marker='^',ls='--',label="-H'80-")
	plt.plot([zli[7],zli[8],zli[9]],[data1[7][0][i],data1[8][1][i],data1[9][1][i]],color='k',marker='^',ls='-.',label="-H'81-")
	plt.plot([zli[7],zli[8],zli[9]],[data1[7][0][i],data1[8][2][i],data1[9][2][i]],color='k',marker='^',ls=':',lw=2,label="-H'82-")
	plt.plot([zli[7],zli[8],zli[9]],[data1[7][0][i],data1[8][5][i],data1[9][4][i]],color='k',marker='o',label="-H'85-")
	plt.plot([zli[7],zli[8]],[data1[7][0][i],data1[8][4][i]],color='gray',marker='.')
	plt.plot([zli[8],zli[9]],[data1[8][4][i],data1[9][6][i]],color='gray',marker='.')
	plt.plot([zli[8],zli[9]],[data1[8][4][i],data1[9][7][i]],color='gray',marker='.')
	
	plt.text(zli[0]+0.005,data1[0][0][i]+0.01,"H'00 (Coma)")
	plt.text(zli[2],data1[2][0][i],"H'20")
	plt.text(zli[3],data1[3][0][i],"H'30")
	plt.text(zli[4],data1[4][0][i],"H'40")
	plt.text(zli[4],data1[4][5][i],"H'45")
	plt.text(zli[6],data1[6][9][i],"H'69*")
	plt.text(zli[7],data1[7][0][i],"H'70")
	plt.text(zli[8],data1[8][4][i],"H'84")
	plt.text(zli[9],data1[9][0][i],"H'90")
	plt.text(zli[9],data1[9][1][i],"H'91")
	plt.text(zli[9],data1[9][2][i],"H'92")
	plt.text(zli[9],data1[9][3][i],"H'93")
	plt.text(zli[9],data1[9][4][i],"H'94")
	plt.text(zli[9],data1[9][5][i],"H'95")
	plt.text(zli[9],data1[9][6][i],"H'96")
	plt.text(zli[9],data1[9][7][i],"H'97")

	plt.xlabel(r'redshift $z$',size=28)
	if i==0:
		plt.ylabel(r'$\log(M\ [h^{-1}M_{\odot}])$',size=28)
		plt.axis([0,1.2,13.0,15.1])
	else:
		plt.ylabel(r'$R_{V}\ [h^{-1}\mathrm{Mpc}]$',size=28)
	plt.legend(fontsize=20)
	plt.tight_layout()
	plt.savefig('merger-tree2.pdf')
	plt.show()

def tree(data1, zli = zlist, i = 0, fs=(18,12)):
	fig = plt.figure(figsize=fs)
	plt.plot([zli[6],zli[5],zli[4],zli[3],zli[2],zli[1],zli[0]],[data1[6][6][i],data1[5][5][i],data1[4][5][i],data1[3][3][i],data1[2][2][i],data1[1][2][i],data1[0][0][i]],color='k',marker='*',label='-H12-H22-H33-H45-H55-')
	plt.plot([zli[1],zli[0]],[data1[1][0][i],data1[0][0][i]],color='gray',marker='.')
	plt.plot([zli[3],zli[2],zli[1],zli[0]],[data1[3][5][i],data1[2][1][i],data1[1][1][i],data1[0][1][i]],color='k',marker='+',label='-H11-H21-')
	plt.plot([zli[2],zli[1],zli[0]],[data1[2][3][i],data1[1][3][i],data1[0][2][i]],color='k',marker='+',label='-H13-',ls='--')
	plt.plot([zli[0]],[data1[0][3][i]],color='gray',marker='.')

	plt.plot([zli[2],zli[1]],[data1[2][0][i],data1[1][0][i]],color='gray',marker='.')
	plt.plot([zli[2],zli[1]],[data1[2][4][i],data1[1][0][i]],color='gray',marker='.')

	plt.plot([zli[3],zli[2]],[data1[3][0][i],data1[2][0][i]],color='gray',marker='.')
	plt.plot([zli[3],zli[2]],[data1[3][1][i],data1[2][0][i]],color='gray',marker='.')
	plt.plot([zli[3],zli[2]],[data1[3][6][i],data1[2][0][i]],color='gray',marker='.')
	plt.plot([zli[4],zli[3],zli[2]],[data1[4][6][i],data1[3][4][i],data1[2][0][i]],color='k',marker='*',label='-H34-',ls='--')
	plt.plot([zli[5],zli[4],zli[3],zli[2]],[data1[5][2][i],data1[4][4][i],data1[3][2][i],data1[2][0][i]],color='k',marker='*',label='-H32-H44-',ls=':',lw=2)

	plt.plot([zli[4],zli[3]],[data1[4][2][i],data1[3][0][i]],color='gray',marker='.')
	plt.plot([zli[7],zli[6],zli[5],zli[4],zli[3]],[data1[7][0][i],data1[6][0][i],data1[5][0][i],data1[4][0][i],data1[3][0][i]],color='k',marker='*',label='-H40-H50-H60-',ls='-.')
	plt.plot([zli[4],zli[3]],[data1[4][3][i],data1[3][1][i]],color='gray',marker='.')
	plt.plot([zli[7],zli[6],zli[5],zli[4],zli[3]],[data1[7][2][i],data1[6][1][i],data1[5][1][i],data1[4][1][i],data1[3][1][i]],color='k',marker='^',label='-H41-H51-H61-')
	plt.plot([zli[6],zli[5]],[data1[6][4][i],data1[5][2][i]],color='gray',marker='.')
	plt.plot([zli[7],zli[6],zli[5]],[data1[7][3][i],data1[6][3][i],data1[5][2][i]],color='k',marker='^',label='-H63-',ls='--')

	plt.plot([zli[8],zli[7]],[data1[8][0][i],data1[7][0][i]],color='gray',marker='.')
	plt.plot([zli[8],zli[7]],[data1[8][1][i],data1[7][0][i]],color='gray',marker='.')
	plt.plot([zli[7],zli[6],zli[5],zli[4]],[data1[7][4][i],data1[6][5][i],data1[5][4][i],data1[4][2][i]],color='k',marker='^',label='-H54-H65-',lw=2,ls=':')
	plt.plot([zli[6],zli[5],zli[4]],[data1[6][8][i],data1[5][7][i],data1[4][2][i]],color='k',marker='^',label='-H57-',ls='-.')
	plt.plot([zli[6],zli[5],zli[4]],[data1[6][7][i],data1[5][6][i],data1[4][3][i]],color='k',marker='o',label='-H56-')
	plt.plot([zli[8],zli[7],zli[6],zli[5],zli[4]],[data1[8][2][i],data1[7][1][i],data1[6][2][i],data1[5][3][i],data1[4][3][i]],color='k',marker='o',label='-H53-H62-H71-',ls='--')
	
	plt.text(zli[0]+0.01,data1[0][0][i]-0.01,'H00 (Coma)')
	plt.text(zli[1]+0.01,data1[1][0][i]-0.01,'H10')
	plt.text(zli[2],data1[2][4][i],'H24*')
	plt.text(zli[2],data1[2][0][i],'H20')
	plt.text(zli[3],data1[3][0][i],'H30')
	plt.text(zli[3],data1[3][1][i],'H31')
	plt.text(zli[3]+0.01,data1[3][6][i]-0.04,'H36*')
	plt.text(zli[4],data1[4][2][i],'H42')
	plt.text(zli[4]-0.09,data1[4][3][i]-0.02,'H43')
	plt.text(zli[4],data1[4][6][i],'H46*')
	plt.text(zli[5]+0.01,data1[5][2][i]-0.025,'H52')
	plt.text(zli[6]+0.01,data1[6][4][i]-0.035,'H64*')
	plt.text(zli[6]+0.01,data1[6][6][i]+0.005,'H66*')
	plt.text(zli[6]+0.01,data1[6][7][i]-0.04,'H67*')
	plt.text(zli[6]+0.01,data1[6][8][i]-0.02,'H68*')
	plt.text(zli[7]+0.01,data1[7][0][i],'H70')
	plt.text(zli[7],data1[7][2][i],'H72*')
	plt.text(zli[7],data1[7][3][i],'H73*')
	plt.text(zli[7]+0.01,data1[7][4][i]-0.04,'H74*')
	plt.text(zli[8],data1[8][0][i],'H80*')
	plt.text(zli[8]+0.005,data1[8][1][i]-0.035,'H81*')
	plt.text(zli[8]+0.01,data1[8][2][i]-0.01,'H82*')
	plt.text(zli[0]+0.01,data1[0][1][i]-0.01,'H01')
	plt.text(zli[0]+0.01,data1[0][2][i]-0.01,'H02')
	plt.text(zli[0]+0.01,data1[0][3][i]-0.01,'H03*')
	plt.text(zli[2]+0.01,data1[2][3][i]-0.01,'H23*')
	plt.text(zli[3]+0.01,data1[3][5][i]+0.005,'H35*')

	plt.xlabel(r'redshift $z$',size=28)
	if i==0:
		plt.ylabel(r'$\log(M\ [h^{-1}M_{\odot}])$',size=28)
		plt.axis([-0.05,2.55,13.0,15.1])
	else:
		plt.ylabel(r'$R_{V}\ [h^{-1}\mathrm{Mpc}]$',size=28)
	plt.legend()
	plt.tight_layout()
	plt.savefig('merger-tree1.pdf')
	plt.show()
	

def L(s):
	return r'$'+s+r'\ [h^{-1}\mathrm{Mpc}]$'

def V(s):
	return r'$'+s+r'\ [10^{3}h^{-1}\mathrm{km\ s^{-1}}]$'
	
def B():
	return r'$\log(\Sigma_{HB}\ [h^{3}\mathrm{erg\ s^{-1}cm^{-2}}] $'

def plotquiver(me, sq, label, arrow = 0.1):
	data = me[sq]
	ax = plt.axes(projection='3d',aspect='equal')
	ax.quiver(me[sq][0], me[sq][1], me[sq][2], me[sq][3], me[sq][4], me[sq][5], length=arrow, normalize=True)
	ax.set_xlabel(r'$x\ [h^{-1}\mathrm{Mpc}]$')
	ax.set_ylabel(r'$y\ [h^{-1}\mathrm{Mpc}]$')
	ax.set_zlabel(r'$z\ [h^{-1}\mathrm{Mpc}]$')
	#plt.tight_layout()
	plt.savefig(label+'quiver'+str(sq)+'.png')
	plt.show()

def plotdis(fcen, sq, label='H0',style = 0, ini = [], bk = []):
	ax = plt.axes(projection='3d',aspect='equal')
	ax.plot(fcen['part'][sq][0][0], fcen['part'][sq][0][1], fcen['part'][sq][0][2],'.',markersize=0.1)
	ax.plot(fcen['part'][sq][1][0], fcen['part'][sq][1][1], fcen['part'][sq][1][2],'.',markersize=0.1)
	if style==1:
		ax.plot(ini[0][0], ini[0][1], ini[0][2], '.', markersize=0.1,alpha=0.5)
		ax.plot(ini[0][0], ini[0][1], ini[0][2], '.', markersize=0.1,alpha=0.5)
		ax.plot(bk[0],bk[1],bk[2],'.',markersize=0.1,alpha=0.8,color='gray')
	ax.plot([fcen['info'][sq][1]], [fcen['info'][sq][2]], [fcen['info'][sq][3]],'o',color='r')
	ax.set_xlabel(r'$x\ [h^{-1}\mathrm{Mpc}]$')
	ax.set_ylabel(r'$y\ [h^{-1}\mathrm{Mpc}]$')
	ax.set_zlabel(r'$z\ [h^{-1}\mathrm{Mpc}]$')
	#plt.tight_layout()
	plt.savefig(label+str(sq)+'.png')
	plt.show()

def plotprof(p0, sn, sq, auto1 = [0.01,15], le = 'best', auto2 = []):
	if sq<=13:
		for x in range(len(p0)):
			plt.errorbar(p0[x][0][0], p0[x][0][sq], yerr=p0[x][1][sq],label="H'"+str(sn)+str(x),ls=linestyle[x],lw=linewidth[x])
	else:
		for x in range(len(p0)):
			p0[x][0][7] = np.array(p0[x][0][7])
			p0[x][0][1] = np.array(p0[x][0][1])
			p0[x][1][7] = np.array(p0[x][1][7])
			p0[x][1][1] = np.array(p0[x][1][1])
			bar = (p0[x][0][7]*OMEGA0/(p0[x][0][1]+p0[x][0][7])/OMEGA_B)*((p0[x][1][7]/p0[x][0][7])**2+(p0[x][1][1]/p0[x][0][1])**2)**0.5
			plt.errorbar(p0[x][0][0], p0[x][0][7]*OMEGA0/(p0[x][0][1]+p0[x][0][7])/OMEGA_B, yerr=bar, label='H'+str(sn)+str(x),ls=linestyle[x],lw=linewidth[x])
	if sq==1:
		plt.yscale('log')
		plt.ylabel(r'$\rho_{DM}\ [10^{10}h^{2}M_{\odot}\mathrm{Mpc^{-3}}]$')
		#plt.legend(loc='best')
	elif sq==2:
		plt.plot(auto1,[0,0],color='k')
		plt.ylabel(r'$v_{r}(1)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==3:
		plt.ylabel(r'$\sigma_{r}(1)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==4:
		plt.ylabel(r'$\sigma_{\theta}(1)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==5:
		plt.ylabel(r'$\sigma_{\phi}(1)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==6:
		plt.plot(auto1,[0,0],color='k')
		plt.ylabel(r'$\beta(1)$')
		#plt.legend(loc='best')
		plt.ylim([-2.0,2.0])
	elif sq==7:
		plt.yscale('log')
		plt.ylabel(r'$\rho_{gas}\ [10^{10}h^{2}M_{\odot}\mathrm{Mpc^{-3}}]$')
		#plt.legend(loc='best')
	elif sq==8:
		plt.plot(auto1,[0,0],color='k')
		plt.ylabel(r'$v_{r}(0)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==9:
		plt.ylabel(r'$\sigma_{r}(0)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==10:
		plt.ylabel(r'$\sigma_{\theta}(0)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==11:
		plt.ylabel(r'$\sigma_{\phi}(0)\ [\mathrm{km\ s^{-1}}]$')
		#plt.legend(loc='best')
	elif sq==12:
		plt.plot(auto1,[0,0],color='k')
		plt.ylabel(r'$\beta(0)$')
		#plt.legend(loc='best')
		plt.ylim([-2.0,2.0])
	elif sq==13:
		plt.ylabel(r'$T\ [\mathrm{K}]$')
		plt.yscale('log')
		#plt.legend(loc='best')
	else:
		plt.plot(auto1,[1,1],color='k')
		plt.ylabel(r'$\omega_{gas}$')
		plt.ylim([0.0,2.0])
	plt.xscale('log')
	plt.xlim(auto1)
	plt.legend(loc=le)
	if auto2!=[]:
		plt.ylim(auto2)
	plt.xlabel(r'$R/R_{V}$')
	plt.tight_layout()
	if sq==1:
		plt.savefig('profile/pro_DM_dens_'+str(sn)+'.pdf')
	elif sq==2:
		plt.savefig('profile/pro_DM_vr_'+str(sn)+'.pdf')
	elif sq==3:
		plt.savefig('profile/pro_DM_sr_'+str(sn)+'.pdf')
	elif sq==4:
		plt.savefig('profile/pro_DM_st_'+str(sn)+'.pdf')
	elif sq==5:
		plt.savefig('profile/pro_DM_sp_'+str(sn)+'.pdf')
	elif sq==6:
		plt.savefig('profile/pro_DM_beta_'+str(sn)+'.pdf')
	elif sq==7:
		plt.savefig('profile/pro_gas_dens_'+str(sn)+'.pdf')
	elif sq==8:
		plt.savefig('profile/pro_gas_vr_'+str(sn)+'.pdf')
	elif sq==9:
		plt.savefig('profile/pro_gas_sr_'+str(sn)+'.pdf')
	elif sq==10:
		plt.savefig('profile/pro_gas_st_'+str(sn)+'.pdf')
	elif sq==11:
		plt.savefig('profile/pro_gas_sp_'+str(sn)+'.pdf')
	elif sq==12:
		plt.savefig('profile/pro_gas_beta_'+str(sn)+'.pdf')
	elif sq==13:
		plt.savefig('profile/pro_temp_'+str(sn)+'.pdf')
	else:
		plt.savefig('profile/pro_gr_'+str(sn)+'.pdf')
	plt.show()

def fig2(l, ali=alist1, lim = 5, fs = (20,9.1), cmpos = [0.1,0.945,0.7,0.02], ori = 0, dx = 6, dy = 6):
	fig= plt.figure(figsize=fs)
	lax = [plt.subplot2grid((2,lim),(0,x), aspect='equal') for x in range(lim)]+[plt.subplot2grid((2,lim),(1,x), aspect='equal') for x in range(lim)]
	lcp=[lax[x].contourf(l[x][0]-39, l[x][1], np.log10(l[x][2]*ali[x]), np.linspace(-17,-4,101), cmap=plt.cm.Spectral) for x in range(lim*2)]
	axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	axcolor = fig.add_axes(cmpos)
	if ori==0:
		cb=fig.colorbar(lcp[0], cax=axcolor, orientation='horizontal')
	else:
		cb=fig.colorbar(lcp[0], cax=axcolor, orientation='vertical')
	cb.ax.set_title(r'$\log(y_{LoS})$')
	plt.tight_layout()
	plt.show()

def fig1(l, n, lim = 5, fs = (20,9.1), cmpos = [0.1,0.945,0.7,0.02], ori = 0, dx = 6, dy = 6):
	#for i in range(lim*2):
	#	l[i][5][0] = l[i][5][0]-39
	#	l[i][6][0] = l[i][6][0]-39
	fig= plt.figure(figsize=fs)
	lax = [plt.subplot2grid((2,lim),(0,x), aspect='equal') for x in range(lim)]+[plt.subplot2grid((2,lim),(1,x), aspect='equal') for x in range(lim)]
	if n==0:
		lcp=[lax[x].contourf(l[x][0][0], l[x][0][1], l[x][0][2]/1000, np.linspace(-1.3,1.3,101), cmap=plt.cm.bwr) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==1:
		lcp=[lax[x].contourf(l[x][1][0], l[x][1][1], l[x][1][2]/1000, np.linspace(0.0,1.5,101), cmap=plt.cm.Reds) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==2:
		lcp=[lax[x].contourf(l[x][2][0], l[x][2][1], l[x][2][2]/1000, np.linspace(-1.3,1.3,101), cmap=plt.cm.bwr) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==3:
		lcp=[lax[x].contourf(l[x][3][0], l[x][3][1], l[x][3][2]/1000, np.linspace(0.0,1.5,101), cmap=plt.cm.Reds) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==4:
		lcp=[lax[x].contourf(l[x][4][0], l[x][4][1], np.log10(l[x][4][2]), np.linspace(0,8,101),cmap=plt.cm.coolwarm) for x in range(lim*2)]
		lct=[lax[x].contour(l[x][4][0], l[x][4][1], np.log10(l[x][4][2]), [4],colors=['k']) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==5:
		lcp=[lax[x].contourf(l[x][4][0], l[x][4][1], np.log10(l[x][5][2]), np.linspace(0,8,101),cmap=plt.cm.coolwarm) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==6:
		lcp=[lax[x].contourf(l[x][4][0], l[x][4][1], np.log10(l[x][6][2]), np.linspace(36,50,101),cmap=plt.cm.hot) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	elif n==7:
		lcp=[lax[x].contourf(l[x][1][0], l[x][1][1], l[x][3][2]/l[x][1][2], np.linspace(0.0,4.5,101), cmap=plt.cm.YlGnBu) for x in range(lim*2)]
		axis = [lax[x].axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy]) for x in range(lim*2)]
	axcolor = fig.add_axes(cmpos)
	if ori==0:
		cb=fig.colorbar(lcp[0], cax=axcolor, orientation='horizontal')
	else:
		cb=fig.colorbar(lcp[0], cax=axcolor, orientation='vertical')
	if (n==0)or(n==2):
		cb.ax.set_title(r"$-v''_{LoS}\ [10^{3}\mathrm{km\ s^{-1}}]$")
	elif (n==1)or(n==3):
		cb.ax.set_title(r"$\sigma_{LoS}\ [10^{3}\mathrm{km\ s^{-1}}]$")
	elif n==4:
		cb.ax.set_title(r'$\log(T\ [\mathrm{K}])$')
	elif n==5:
		cb.ax.set_title(r'$\log(\sigma_{T}\ [\mathrm{K}])$')
	elif n==6:
		cb.ax.set_title(r'$\log(dP/d\Omega\ [h\cdot \mathrm{erg\ s^{-1}rad^{-2}}])$')
	elif n==7:
		cb.ax.set_title(r"$\sigma_{LoS}(DM)/\sigma_{LoS}(gas)$")
	plt.tight_layout()
	plt.show()

	
def fig0(l, n, zli = zlist1, lim = 5, fs = (20,9.1), cmpos = [0.1,0.94,0.7,0.02], ori = 0):
	fig= plt.figure(figsize=fs)
	if n<=7:
		lax = [plt.subplot2grid((2,lim),(0,x), aspect='equal') for x in range(lim)]+[plt.subplot2grid((2,lim),(1,x), aspect='equal') for x in range(lim)]
	else:
		lax = [plt.subplot2grid((2,lim),(0,x)) for x in range(lim)]+[plt.subplot2grid((2,lim),(1,x)) for x in range(lim)]
	if n==0:
		lcp = [lax[x].contourf(l[x][0][0], l[x][0][1], np.log10(l[x][0][2]), np.linspace(-1,5.5,101)) for x in range(lim*2)]
		lct = [lax[x].contour(l[x][0][0], l[x][0][1], np.log10(l[x][0][2]), [np.log10(EDGE*DX*CRITICAL(zli[x])*OMEGA_B*matter(zli[x])/OMEGA0/(1+zli[x])**4)],colors=['white']) for x in range(lim*2)]
	elif n==1:
		lcp = [lax[x].contourf(l[x][1][0], l[1][0][1], np.log10(l[x][1][2]), np.linspace(-1,5.5,101)) for x in range(lim*2)]
		lct = [lax[x].contour(l[x][1][0], l[x][1][1], np.log10(l[x][1][2]), [np.log10(EDGE*DX*CRITICAL(zli[x])*(OMEGA0-OMEGA_B)*matter(zli[x])/OMEGA0/(1+zli[x])**4)],colors=['white']) for x in range(lim*2)]
	elif n==2:
		lcp=[lax[x].contourf(l[x][0][0], l[x][0][1], l[x][0][2]*OMEGA0/(l[x][0][2]+l[x][1][2])/OMEGA_B, np.linspace(0,5,101), cmap=plt.cm.PuOr) for x in range(lim*2)]
		lct=[lax[x].contour(l[x][0][0], l[x][0][1], l[x][0][2]*OMEGA0/(l[x][0][2]+l[x][1][2])/OMEGA_B, [1], colors=['k']) for x in range(lim*2)]
	elif n==3:
		lcp=[lax[x].contourf(l[x][2][0], l[x][2][1], l[x][2][2]/1000, np.linspace(-1.4,1.4,101), cmap=plt.cm.bwr) for x in range(lim*2)]
		lq =[lax[x].quiver(l[x][2][0], l[x][2][1], l[x][3][2], l[x][4][2], scale=40000) for x in range(lim*2)]
	elif n==4:
		lcp=[lax[x].contourf(l[x][2][0], l[x][2][1], l[x][5][2]/1000, np.linspace(-1.4,1.4,101), cmap=plt.cm.bwr) for x in range(lim*2)]
		lq =[lax[x].quiver(l[x][2][0], l[x][2][1], l[x][6][2], l[x][7][2], scale=40000) for x in range(lim*2)]
	elif n==5:
		lcp=[lax[x].contourf(l[x][2][0], l[x][2][1], (l[x][2][2]-l[x][5][2])/1000, np.linspace(-1.4,1.4,101), cmap=plt.cm.bwr) for x in range(lim*2)]
		lq =[lax[x].quiver(l[x][2][0], l[x][2][1], l[x][3][2]-l[x][6][2], l[x][4][2]-l[x][7][2], scale=5000) for x in range(lim*2)]
	elif n==6:
		lcp=[lax[x].contourf(l[x][8][0], l[x][8][1], np.log10(l[x][8][2]), np.linspace(0,8,101),cmap=plt.cm.coolwarm) for x in range(lim*2)]
		lct=[lax[x].contour(l[x][8][0], l[x][8][1], np.log10(l[x][8][2]), [4],colors=['k']) for x in range(lim*2)]
	elif n==7:
		lcp=[lax[x].contourf(l[x][9][0], l[x][9][1], np.log10(l[x][9][2]/(2*np.pi)/(180*60/np.pi)**2), np.linspace(-23,-10,101), cmap=plt.cm.gist_ncar) for x in range(lim*2)]
	elif n==8:
		lcp = [lax[x].contourf(l[x][0][0], l[x][0][1]/1000, np.log10(l[x][0][2]), np.linspace(0,3.2,101)) for x in range(lim*2)]
	elif n==9:
		lcp = [lax[x].contourf(l[x][1][0], l[x][1][1]/1000, np.log10(l[x][1][2]), np.linspace(0,3.2,101)) for x in range(lim*2)]
	if n<=7:
		axcolor = fig.add_axes(cmpos)
		if ori==0:
			cb=fig.colorbar(lcp[0], cax=axcolor, orientation='horizontal')
		else:
			cb=fig.colorbar(lcp[0], cax=axcolor, orientation='vertical')
	if n==0:
		cb.ax.set_title(r'$\log(\Sigma_{gas}/\Sigma_{0})$')
	elif n==1:
		cb.ax.set_title(r'$\log(\Sigma_{DM}/\Sigma_{0})$')
	elif n==2:
		cb.ax.set_title(r'$\omega_{gas}$')
	elif (n==3)or(n==4):
		cb.ax.set_title(r"$-v''_{x}\ [10^{3}\mathrm{km\ s^{-1}}]$")
	elif n==5:
		cb.ax.set_title(r"$-\Delta v''_{x}\ [10^{3}\mathrm{km\ s^{-1}}]$")
	elif n==6:
		cb.ax.set_title(r'$\log(T\ [\mathrm{K}])$')
	elif n==7:
		#cb.ax.set_title(r'$\log(\Sigma_{HB}\ [h\cdot \mathrm{erg\ s^{-1}cm^{-2}}])$')
		cb.ax.set_title(r'$\log(S_{b}\ [h\cdot \mathrm{erg\ s^{-1}cm^{-2}arcmin^{-2}}])$')
	plt.tight_layout()
	plt.show()

def plotd(md, n, sq, zli = zlist, pos=111):
	plt.subplot(pos, aspect='equal')
	#cp = plt.contourf(md[0], md[1], np.log10(md[2]), np.linspace(-1,5,101))
	cp = plt.contourf(md[0], md[1], np.log10(md[2]), 101)
	cb = plt.colorbar()
	if n==0:
		cb.ax.set_title(r'$\log(\Sigma_{gas}/\Sigma_{0})$')
		plt.contour(md[0], md[1], np.log10(md[2]), [np.log10(EDGE*DX*CRITICAL(zli[sq])*OMEGA_B*matter(zli[sq])/OMEGA0/(1+zli[sq])**4)],colors=['white'])
	else:
		cb.ax.set_title(r'$\log(\Sigma_{DM}/\Sigma_{0})$')
		plt.contour(md[0], md[1], np.log10(md[2]), [np.log10(EDGE*DX*CRITICAL(zli[sq])*(OMEGA0-OMEGA_B)*matter(zli[sq])/OMEGA0/(1+zli[sq])**4)],colors=['white'])
	plt.xlabel(L("y''"))
	plt.ylabel(L("z''"))
	plt.tight_layout()
	plt.show()
	return cp

def plotgr(md0, md1, pos=111):
	plt.subplot(pos, aspect='equal')
	plt.contourf(md0[0], md0[1], md0[2]*OMEGA0/(md0[2]+md1[2])/OMEGA_B, np.linspace(0,5,101), cmap=plt.cm.PuOr)
	cb = plt.colorbar()
	cb.ax.set_title(r'gas richness')
	plt.contour(md0[0], md0[1], md0[2]*OMEGA0/(md0[2]+md1[2])/OMEGA_B, [1], colors=['k'])
	plt.xlabel(L("y''"))
	plt.ylabel(L("z''"))
	plt.tight_layout()
	plt.show()

def plotv(mx, my, mz, style=0, pos=111):
	plt.subplot(pos, aspect='equal')
	plt.contourf(mx[0], mx[1], mx[2]/1000, np.linspace(-1.3,1.3,101), cmap=plt.cm.bwr)
	cb = plt.colorbar()
	cb.ax.set_title(r"$-v''_{x}\ [10^{3}\mathrm{km\ s^{-1}}]$")
	if style==0:
		plt.quiver(mx[0], mx[1], my[2], mz[2], scale=40000, units='width')
	else:
		plt.streamplot(mx[0], mx[1], my[2], mz[2],color='k')
	plt.xlabel(L("y''"))
	plt.ylabel(L("z''"))
	plt.tight_layout()
	plt.show()

def plotdv(mx0, my0, mz0, mx1, my1, mz1, style =0, pos=111):
	plt.subplot(pos, aspect='equal')
	plt.contourf(mx0[0], mx0[1], (mx0[2]-mx1[2])/1000, np.linspace(-1.3,1.3,101), cmap=plt.cm.bwr)
	cb = plt.colorbar()
	cb.ax.set_title(r"$-\Delta v''_{x}\ [10^{3}\mathrm{km\ s^{-1}}]$")
	if style==0:
		plt.quiver(mx0[0], mx0[1], my0[2]-my1[2], mz0[2]-mz1[2], scale=5000)
	else:
		plt.streamplot(mx0[0], mx0[1], my0[2]-my1[2], mz0[2]-mz1[2],color='k')
	plt.xlabel(L("y''"))
	plt.ylabel(L("z''"))
	plt.tight_layout()
	plt.show()
	
def plott(mt,pos=111):
	plt.subplot(pos, aspect='equal')
	#plt.contourf(mt[0], mt[1], np.log10(mt[2]), np.linspace(0,8,101),cmap=plt.cm.coolwarm)
	plt.contourf(mt[0], mt[1], np.log10(mt[2]), 101,cmap=plt.cm.coolwarm)
	cb = plt.colorbar()
	cb.ax.set_title(r'$\log(T\ [\mathrm{K}])$')
	plt.contour(mt[0], mt[1], np.log10(mt[2]), [4],colors=['k'])
	plt.xlabel(L("y''"))
	plt.ylabel(L("z''"))
	plt.tight_layout()
	plt.show()
	
def plotB(mt,low=-14,up=-10,co=4,pos=111):
	plt.subplot(pos, aspect='equal')
	plt.contourf(mt[0], mt[1], np.ones((len(mt[0]),len(mt[1])))*low, np.linspace(low,(up+low)/2,101),cmap=plt.cm.gist_ncar)
	#plt.contourf(mt[0], mt[1], np.log10(mt[2]), np.linspace(-16,-3,101),cmap=plt.cm.hot)
	plt.contourf(mt[0], mt[1], np.log10(mt[2]/(2*np.pi)/(180*60/np.pi)**2), np.linspace(low,up,101),cmap=plt.cm.gist_ncar)
	cb = plt.colorbar()
	plt.contour(mt[0], mt[1], np.log10(mt[2]/(2*np.pi)/(180*60/np.pi)**2), np.linspace(low,up,co), colors=['white' for x in range(co)])
	cb.ax.set_title(r'$\log(S_{b}\ [h\cdot \mathrm{erg\ s^{-1}cm^{-2}arcmin^{-2}}])$')
	plt.xlabel(L("y''"))
	plt.ylabel(L("z''"))
	plt.tight_layout()
	plt.show()

def plotB_sky(mt,dx=1,dy=1,low = -14, up = -10, co=4,pos=111):
	ax = plt.subplot(pos, aspect='equal')
	ax.set_axis_bgcolor('k')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.grid(True)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
	#plt.contourf(mt[0], mt[1], np.ones((len(mt[0]),len(mt[1])))*low, np.linspace(low,(up+low)/2,101),cmap=plt.cm.gist_ncar)
	plt.contourf(mt[0], mt[1], np.minimum(np.log10(mt[2]/(DEGREE*UL*180/np.pi)**2/(2*np.pi)/(180*60/np.pi)**2),up), np.linspace(low,up,101),cmap=plt.cm.gist_ncar)
	#plt.contourf(mt[0], mt[1], np.log10(mt[2]/(180*60/np.pi)**2), np.linspace(np.log(0.016),np.log(0.288),101),cmap=plt.cm.gist_ncar)
	cb = plt.colorbar()
	plt.contour(mt[0], mt[1], np.log10(mt[2]/(DEGREE*UL*180/np.pi)**2/(2*np.pi)/(180*60/np.pi)**2), np.linspace(low,up,co), colors=['black' for x in range(11)])
	#plt.contour(mt[0], mt[1], np.log10(mt[2]), 4, colors=['white' for x in range(4)])
	cb.ax.set_title(r'$\log(S_{b}\ [h\cdot \mathrm{erg\ s^{-1}cm^{-2}arcmin^{-2}}])$')
	plt.axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy])
	plt.xlabel('ra')
	plt.ylabel('dec')
	plt.tight_layout()
	plt.show()

def plot_SZ(mt,dx=1,dy=1,low = -17, up = -4, pos=111):
	plt.subplot(pos, aspect='equal')
	plt.contourf(mt[0]-39, mt[1], np.log10(mt[2]), np.linspace(low-np.log10(2),up,101),cmap=plt.cm.CMRmap)
	cb = plt.colorbar()
	cb.ax.set_title(r'$\log(y_{LoS})$')
	plt.contour(mt[0]-39, mt[1], np.log10(mt[2]), np.linspace(low,up,int((up-low)/(np.log10(2)/4))), colors=['white' for x in range(int((up-low)/(np.log10(2)/4)))])
	plt.axis([360+PHI-39+dx,360+PHI-39-dx,THETA-dy,THETA+dy])
	plt.xlabel('ra')
	plt.ylabel('dec')
	plt.tight_layout()
	plt.show()

def plot_vel1(d, ran = [0,1.2,0,800], zli = zlist):
	plt.plot(zli,d[1],marker='.',label=r'$v_{x}^{in}(0)$')
	plt.plot(zli,d[2],marker='.',label=r'$v_{y}^{in}(0)$',ls='--')
	plt.plot(zli,d[3],marker='.',label=r'$v_{z}^{in}(0)$',ls='-.')
	plt.plot(zli,d[5],marker='*',label=r'$v_{x}^{in}(1)$')
	plt.plot(zli,d[6],marker='*',label=r'$v_{y}^{in}(1)$',ls='--')
	plt.plot(zli,d[7],marker='*',label=r'$v_{z}^{in}(1)$',ls='-.')
	plt.plot(zli,d[0],marker='o',label=r'$v_{in}(0)$',color='gray',lw=2)
	plt.plot(zli,d[4],marker='^',label=r'$v_{in}(1)$',color='brown',ls=':',lw=2)
	plt.axis(ran)
	plt.legend(loc=1)
	plt.xlabel(r'redshift $z$')
	plt.ylabel(r'$v\ [\mathrm{km\ s^{-1}}]$')
	plt.tight_layout()
	plt.show()

def plot_vel2(d, ran = [0,1.2,0,800], zli = zlist1):
	plt.plot(zli,d[5],marker='.',label=r'$v_{x}^{in}(0)$')
	plt.plot(zli,d[6],marker='.',label=r'$v_{y}^{in}(0)$',ls='--')
	plt.plot(zli,d[7],marker='.',label=r'$v_{z}^{in}(0)$',ls='-.')
	plt.plot(zli,d[9],marker='*',label=r'$v_{x}^{in}(1)$')
	plt.plot(zli,d[10],marker='*',label=r'$v_{y}^{in}(1)$',ls='--')
	plt.plot(zli,d[11],marker='*',label=r'$v_{z}^{in}(1)$',ls='-.')
	plt.plot(zli,d[4],marker='o',label=r'$v_{in}(0)$',color='gray',lw=2)
	plt.plot(zli,d[8],marker='^',label=r'$v_{in}(1)$',color='brown',ls=':',lw=2)
	plt.axis(ran)
	plt.legend(loc=1)
	plt.xlabel(r'redshift $z$')
	plt.ylabel(r'$v\ [\mathrm{km\ s^{-1}}]$')
	plt.tight_layout()
	plt.show()
	


