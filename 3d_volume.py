import yt
import numpy as np
import matplotlib.pyplot as plt
from cosmology import *
from txt import *
import matplotlib
#import mpl_toolkits.mplot3d
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#rep = 'test2021/'
rep = 'N32L025_test3/'
sn =  50
L = 250*KPC
mh = rhom(1) * L**3/Msun

fname = rep + 'snapshot_'+str(sn).zfill(3)+'.hdf5'

ds = yt.load(fname)
z = ds['Redshift']

#ext, thk, asp = 500, 200, 1
#cen = [1940, 1960, 1850]
#xb, yb = 800, 800
ext, thk, asp = 250, 250, 1
cen = [125, 125, 125]
xb, yb = 200, 200
cen_ = np.array(cen)
lext = np.array([ext, ext/asp, thk])
le, re = cen_ - lext/2, cen_ + lext/2

def SFhistory(sn, ind, mh, ds, ad, rep='./', dt = 1, edge=0.5, log=1, ntmax=100, up=10):
	z = ds['Redshift']
	redlab = r'$z={:.1f}$'.format(z)
	mhlab = r'$M_{\rm halo}='+'{:.2e}'.format(mh)+r'\ \rm M_{\odot}$'
	t0 = TZ(z)/YR/1e6
	m0 = np.array(ad[('PartType4', 'Masses')].to('Msun'))[0]
	sa = np.hstack([ad[('PartType4', 'StellarFormationTime')], ad[('PartType3', 'StellarFormationTime')]])
	stype = np.hstack([ad[('PartType4', 'Star Type')], ad[('PartType3', 'Star Type')]])
	sel3 = stype >= 30000
	sel2 = np.logical_not(sel3)
	t3 = np.array([TZ(x) for x in 1/sa[sel3]-1])/YR/1e6
	t2 = np.array([TZ(x) for x in 1/sa[sel2]-1])/YR/1e6
	t = np.hstack([t3, t2])
	ti = np.min(t)
	nt0 = int(abs(t0-ti)/dt)+1
	nt = min(ntmax, nt0)
	#print('Num of timebins: ', nt)
	print('Delay time: {:.1f} Myr'.format(np.min(t2)-np.min(t3)))
	ti_ = t0-nt0*dt
	if log>0:
		ed = np.geomspace(ti_, t0, nt+1)
	else:
		ed = np.linspace(ti_, t0, nt+1)
	h2, ed = np.histogram(t2, ed)
	h3, ed = np.histogram(t3, ed)
	his, ed = np.histogram(t, ed)
	base = midbin(ed)
	norm = m0/(dt*1e6)
	y1, y2 = norm*edge, np.max(his*norm)/edge*up
	plt.figure()
	plt.plot(base, his*norm, 'k', label='Total\n'+redlab+', '+mhlab, zorder=0)
	plt.plot(base, h3*norm, '--', label='Pop III', zorder=2)
	plt.plot(base, h2*norm, '-.', label='Pop II', zorder=1)
	plt.xlabel(r'$t\ [\rm Myr]$')
	plt.ylabel(r'$\dot{M}_{\star}\ [\rm M_{\odot}\ yr^{-1}]$')
	plt.yscale('log')
	if log>0:
		plt.xscale('log')
	plt.xlim(ti_, t0)
	plt.ylim(y1, y2)
	plt.legend()
	plt.tight_layout()
	plt.savefig(rep+'SFR_t_'+str(sn)+'_'+str(ind)+'.pdf')
	plt.close()
	
def tidaltensor_dis(sn, ind, ds, ad, nb=100, lind=[0,1,2,4,5,8], rep='./'):
	z = ds['Redshift']
	h = ds['HubbleParam']
	if ind==0:
		dtt = np.array(ad[('PartType0','TidalTensorPS')])
	else:
		dtt = np.array(ad[('PartType4','TidalTensorPS')])
		dtt1 = np.array(ad[('PartType3','TidalTensorPS')])
	tt = np.hstack(dtt.T[lind])*(1e5*h*(1+z)/KPC)**2*(1e9*YR)**2
	t1, t2 = np.quantile(tt, [0.01, 0.99])
	bins = np.linspace(t1, t2, nb+1)
	plt.figure()
	if ind==0:
		plt.hist(tt, bins, alpha=0.5, label='Gas')
	else:
		tt1 = np.hstack(dtt1.T[lind])*(1e5*h*(1+z)/KPC)**2*(1e9*YR)**2
		plt.hist(tt1, bins, alpha=0.5, label='BH')
		plt.hist(tt, bins, histtype='step', lw=1.5, label='Star')
	plt.xlabel(r'$T_{ij}\ [\rm Gyr^{-2}]$')
	plt.ylabel(r'Probability density [a. u.]')
	plt.legend()
	plt.xlim(t1, t2)
	plt.yscale('log')
	plt.title(r'$z={:.2f}$'.format(z))
	plt.tight_layout()
	plt.savefig(rep+'tt_dis_'+str(sn)+'_'+str(ind)+'.pdf')
	plt.close()

ad = ds.box(le, re)

#"""
prj = yt.ParticleProjectionPlot(ds, 2, ('PartType0', 'Metallicity_00'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, weight_field=('PartType0', 'Masses'))#, data_source=ad)
prj.set_zlim('all', 1e-6*0.02, 0.2)
prj.set_buff_size((xb, yb))
prj.annotate_title('$z={:.2f}$'.format(z))
prj.save(rep+'metaldis_'+str(sn)+'.png')

prj = yt.ParticleProjectionPlot(ds, 2, ('PartType1', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
prj.set_unit(('PartType1', 'Masses'), 'Msun/(kpccm/h)**2')
prj.set_buff_size((xb, yb))
prj.annotate_title('$z={:.2f}$'.format(z))
prj.save(rep+'dmdis_'+str(sn)+'.png')

prj = yt.ParticleProjectionPlot(ds, 2, ('PartType0', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
prj.set_unit(('PartType0', 'Masses'), 'Msun/(kpccm/h)**2')
prj.set_buff_size((xb, yb))
prj.annotate_title('$z={:.2f}$'.format(z))
prj.save(rep+'gasdis_'+str(sn)+'.png')

prj = yt.ParticleProjectionPlot(ds, 2, ('PartType4', 'Masses'), center=cen, width=((ext, 'kpccm/h'), (ext/asp, 'kpccm/h')), depth = thk, density=True)#, data_source=ad)
prj.set_unit(('PartType4', 'Masses'), 'Msun/(kpccm/h)**2')
prj.set_buff_size((xb, yb))
prj.annotate_title('$z={:.2f}$'.format(z))
prj.save(rep+'stardis_'+str(sn)+'.png')
#prj.save('snapshot_'+str(sn).zfill(3)+'_stellar.png')
#"""

#"""
plot = yt.ParticlePhasePlot(ad, ('PartType0', 'Density'), ('PartType0', 'InternalEnergy'), [('PartType0', 'Masses')], weight_field=None, x_bins=xb, y_bins=yb)
plot.set_unit(('PartType0', 'Density'), 'g/cm**3')
plot.set_unit(('PartType0', 'InternalEnergy'), 'km**2/s**2')
plot.set_unit(('PartType0', 'Masses'), 'Msun')
plot.set_log(('PartType0', 'Density'), True)
#plot.set_zlim('all', 9e3, 1e9)
#plot.annotate_text(xpos=0, ypos=0, text='O')
#ax = plot.plots[('PartType0', 'Masses')].axes
#ax.set_ylim(9e3, 1e9)
plot.annotate_title('$z={:.2f}$'.format(z))
plot.save(rep+'phasedg_'+str(sn)+'.png')
#"""

ad = ds.all_data()
tidaltensor_dis(sn, 0, ds, ad, rep=rep)
tidaltensor_dis(sn, 1, ds, ad, rep=rep)
SFhistory(sn, 0, mh, ds, ad, dt=10, up = 20, rep=rep)

"""
sc = yt.create_scene(ds, lens_type='perspective', field=('PartType0', 'Density'))
#im, sc = yt.volume_render(ds, ('PartType0', 'Density'), fname='rendering.png')

sc.camera.width = (1000, 'kpccm/h')

sc.camera.switch_orientation()

source = sc[0]

source.tfh.set_log(True)

source.tfh.grey_opacity = False

source.tfh.plot(rep+'transfer_function.png', profile_field=('PartType0', 'Density'))

sc.save('rendering.png')#, sigma_clip=6.0)
"""
