#include <math.h>
#include<stdio.h>

void chemsl(double * nin, double * xl, double * nout, double T, double dt0, double epsH, double J_21, int nmax, double z, double T0)
{
	int i = 0;
	double ny[17], Cr[17], Ds[17], k[50];
	double dt_cum, dt, delta_x, total = 0;
	double XNUM1, XDENOM1, XNUM2, XDENOM2;
	double xnh, xnhe, xnd, xnli;
	double LJ, NH2, xshield;
	for (i=0;i<17;i++){ ny[i]=nin[i]; }
	xnh = xl[0];
	xnhe = xl[1];
	xnd = xl[2];
	xnli =xl[3];
	dt_cum = 0.e0;
	LJ = 4.8e19*sqrt(T/xnh);
	NH2 = 0.e0;
	xshield = 0.e0;
	rates(T, xnh, k, J_21, z, T0);
	while (dt_cum<dt0){
		dt = dt0;
		Cr[5]=(k[21]+k[47])*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] + k[48]*ny[2];
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8];
		delta_x = epsH*ny[5]/fabs(Cr[5]-Ds[5]*ny[5]);
		while (dt>=delta_x){
			Cr[5]=(k[21]+k[47])*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5] + k[48]*ny[2];
			Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8];
			delta_x = epsH*ny[5]/fabs(Cr[5]-Ds[5]*ny[5]);
			dt = dt/2.e0;
		}
		if (dt>1.e5*3.14e7){ dt = 1.e5*3.14e7; }
		if (dt<dt0/((double) nmax)) {dt = dt0/((double) nmax); }
		if ((dt_cum+dt) > dt0) {
			dt=dt0-dt_cum;
			dt_cum=dt0;
		} else {
			dt_cum=dt_cum+dt;
		}
		Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+(k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5]; // +k[24]*ny[2]+k[26]*ny[3]
		Ds[5]=k[3]*ny[1]+k[4]*ny[7]+k[5]*ny[8];
		ny[5]=(ny[5]+Cr[5]*dt)/(1.e0+Ds[5]*dt);

		Cr[0]=k[3]*ny[1]*ny[5] + k[48]*ny[2] + k[49]*ny[4];
		Ds[0]=k[0]*ny[5]+k[21]+k[47];
		ny[0]=(ny[0]+Cr[0]*dt)/(1.e0+Ds[0]*dt);

		Cr[1]=k[0]*ny[5]*ny[0] + (k[21]+k[47])*ny[0] + k[49]*ny[4];
		Ds[1]=k[3]*ny[5];
		ny[1]=(ny[1]+Cr[1]*dt)/(1.e0+Ds[1]*dt);

		Cr[6]=k[4]*ny[7]*ny[5];
		Ds[6]=k[1]*ny[5]+k[22];
		ny[6]=(ny[6]+Cr[6]*dt)/(1.e0+Ds[6]*dt);

		Cr[7]=(k[1]*ny[5]+k[22])*ny[6]+k[5]*ny[5]*ny[8];
		Ds[7]=(k[2]+k[4])*ny[5]+k[23];
		ny[7]=(ny[7]+Cr[7]*dt)/(1.e0+Ds[7]*dt);

		Cr[8]=k[2]*ny[7]*ny[5]+k[23]*ny[7];
		Ds[8]=k[5]*ny[5];
		ny[8]=(ny[8]+Cr[8]*dt)/(1.e0+Ds[8]*dt);
/**** calculate equilibrium abundance for H- *********************/
		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] + k[48]; // +k[11]*ny[4];
		//if (XDENOM1>1.e-30){
		ny[2] = XNUM1/XDENOM1;//}
/**** calculate equilibrium abundance for H2+ ********************/
		XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1] +k[26]*ny[3];
		XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25] + k[49];
		//if (XDENOM2>1.e-30){
		ny[4]=XNUM2/XDENOM2;//}
/* H2 */
		NH2 = ny[3]*LJ;
		if (NH2<=1) { xshield = 1.e0; }
		else { xshield = pow(NH2, -0.75); }
		k[27] = 1.2e-12*J_21*xshield;
		Ds[3]=k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]+k[16]*ny[1]+k[17]*ny[5]+k[26]+k[27];
		Cr[3]=k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]+k[11]*ny[4]*ny[2];
		ny[3]=(ny[3]+Cr[3]*dt)/(1.e0+Ds[3]*dt);

		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0){
			ny[0]=0.e0;}
		ny[6]=xnhe-ny[7]-ny[8];
		if (ny[6] < 0.e0){
			ny[6]=0.e0;}
		ny[7]=xnhe-ny[6]-ny[8];
		if (ny[7]<0.0){
			ny[7]=0.e0;}
		ny[8] = xnhe-ny[6]-ny[7];
		if (ny[8]<0.0){
			ny[8]=0.e0;}
		ny[1]=ny[5]+ny[2]-ny[4]-ny[7]-2.e0*ny[8];
		if (ny[1] < 0.e0){
			ny[1]=0.e0;}
		ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
		if (ny[0] < 0.e0){
			ny[0]=0.e0;}

		XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
		XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]+ k[24] + k[48]; // +k[11]*ny[4];
		ny[2]=XNUM1/XDENOM1;

		for (i=0;i<9;i++){
			if (ny[i]<1e-30){
				ny[i] = 0.0;}
		}
/* HD */
		Cr[10]=k[30]*ny[9]*ny[1]+k[33]*ny[11]*ny[1];
		Ds[10]=k[3]*ny[5]+k[31]*ny[0]+k[32]*ny[3];
		ny[10]=(ny[10]+Cr[10]*dt)/(1.0+Ds[10]*dt);
		if (ny[10]>xnd){ ny[10]=xnd; }
		Cr[11]=k[32]*ny[10]*ny[3];
		Ds[11]=k[33]*ny[1];
		ny[11]=(ny[11]+Cr[11]*dt)/(1.0+Ds[11]*dt);
		if (ny[11]>xnd){ ny[11]=xnd; }
		ny[9]=xnd-ny[11]-ny[10];
/* LiH */
		Cr[12] = k[34]*ny[13]*ny[5]+k[42]*ny[15]*ny[5]+k[35]*ny[14]*ny[1]+k[39]*ny[16]*ny[0];
		Ds[12] = k[41]*ny[1]+k[36]*ny[5]+k[37]*ny[0]+(k[44]+k[45])*ny[1]+k[46]*ny[2];
		ny[12] = (ny[12]+Cr[12]*dt)/(1.0+Ds[12]*dt);
		if (ny[12]>xnli){ ny[12]=xnli; }

		Cr[13] = k[43]*ny[0]*ny[15]+(k[44]+k[45])*ny[12]*ny[1];
		Ds[13] = k[34]*ny[5]+k[40]*ny[1];
		ny[13] = (ny[13]+Cr[13]*dt)/(1.0+Ds[13]*dt);
		if (ny[13]>xnli){ ny[13]=xnli; }

		Cr[15] = k[40]*ny[0]*ny[13]+k[41]*ny[1]*ny[12];
		Ds[15] = k[43]*ny[0]+k[42]*ny[5];
		ny[15] = (ny[15]+Cr[15]*dt)/(1.0+Ds[15]*dt);
		if (ny[15]>xnli){ ny[15]=xnli; }

		Cr[16] = k[37]*ny[12]*ny[0]+k[38]*ny[14]*ny[0]+k[46]*ny[2]*ny[12];
		Ds[16] = k[39]*ny[0];
		ny[16] = (ny[16]+Cr[16]*dt)/(1.0+Ds[16]*dt);
		if (ny[16]>xnli){ ny[16]=xnli; }

		Cr[14] = k[36]*ny[12]*ny[5];
		Ds[14] = k[35]*ny[1]+k[38]*ny[0];
		ny[14] = (ny[14]+Cr[14]*dt)/(1.0+Ds[14]*dt);
		if (ny[14]>xnli){ ny[14]=xnli; }

		ny[12] = xnli-ny[15]-ny[13]-ny[14]-ny[16];

		for (i=0;i<17;i++){
			if (ny[i]<1.e-30) { ny[i]=0.e0; }
		}
		total = total+1.e0;
	}
	for (i=0;i<17;i++){
		nout[i] = ny[i];
	}
	nout[17] = dt_cum;
	nout[18] = total;
	//for (i=0;i<17;i++){
	//	printf("%f, ", ny[i]);}
	//printf("\n");
}



void rates(double tempr, double xnh, double k[50], double J_21, double z, double T0)
{
	double k5a,k5b,ncr_1,ncr_2,kh_1,kh_2,kl_1,kl_2,x,Tgas,T,S2;
	double Tcmb = T0*(1+z);
	double tt = sqrt(tempr);
	double t5fac = 1.e0/(1.e0+sqrt(tempr/1.e5));
	double logT = log10(tempr);
	double kB = 1.3806e-16;
	T = tempr;
	for (int i=0;i<50;i++){
		k[i] = 0.e0; }
	// Recombination: Hep from Black (1981, MNRAS 197, 553) Hp and Hepp from Cen (1992, ApJS 78, 341)
	k5a=1.5e-10*pow(T,-0.6353e0);
	if (T>pow(10.e0, 3.6e0)) { 
		k5b=1.9e-3*pow(tempr,-1.5e0)*exp(-4.7e5/tempr)*
      (1.e0+0.3e0*exp(-9.4e4/tempr));}
	else {
		k5b = 0.e0;}
	k[4] = k5a + k5b; // He+ + e- = He dielectronic recombination of HeII (Black 1992)
	k[3] = 8.40e-11/tt*pow(tempr/1.e3,-0.2e0)/(1.e0+pow(tempr/1.e6,0.7e0)); // H+ + e- = H + hnu
	k[5] = 3.36e-10/tt*pow(tempr/1.e3,-0.2e0)/(1.e0+pow(tempr/1.e6,0.7e0)); // He++ + e- = He+ + hnu (Cen 1992)

/*-- collisional ionization rates (from Black, with Cen fix at high-T):*/

  if (tempr > pow(10.e0,2.8e0)) {
    k[0] = 5.85e-11*tt*exp(-157809.1e0/tempr)*t5fac; //H + e- = H+ + 2e-
  } else {
    k[0]=0.e0;
  }

  if (tempr > pow(10.e0,3.e0)) {
    k[1] = 2.38e-11*tt*exp(-285335.4e0/tempr)*t5fac; //He + e- = He+ + 2e- (Cen 1992)
  } else {
    k[1] = 0.e0;
  }

  if (tempr > pow(10.e0,3.4e0)) {
    k[2] = 5.68e-12*tt*exp(-631515.e0/tempr)*t5fac; //He+ + e- = He++ + 2e- (Cen 1992)
  } else {
    k[2] = 0.e0;
  }

/*--- additional reactions from Z. Haiman: */

  if (tempr <= 4000.e0) {
    k[6] = 1.38e-23*pow(tempr,1.845e0); //H + H+ = H2+ + hnu ???
  } else {
    if (tempr < pow(10.e0,5.e0)) {
      k[6] = -6.157e-17 + 3.255e-20*tempr - 4.152e-25*tempr*tempr;
    } else {
      k[6] = 0.e0;
    }
  }
  if (k[6] <= 0.e0) k[6]=0.e0;
  k[7] = 6.4e-10; //H2+ + H = H2 + H+

/**** Try the Abel et al. 1997 rate! */     
  Tgas=tempr;
  k[8]=1.429e-18*pow(Tgas,0.762)*pow(Tgas,0.1523*log10(Tgas))*
          pow(Tgas,-3.274e-2*pow(log10(Tgas),2.e0)); //H + e- = H- + hnu

  k[9] = 1.3e-9; //H + H- = H2 + e-
  k[10] = 1.68e-8*pow(tempr/300.e0,-0.29e0); //e- + H2+ = 2H ???
  k[11] = 5.0e-6/tt; //H2+ + H- = H2 + H Too efficient???
  k[12] = 7.0e-7/tt; //H- + H+ = 2H  
  if (tempr > pow(10.e0,2.2e0)) {
    k[13] = (2.7e-8/(tempr*tt))*exp(-43000.e0/tempr)*t5fac; //e- + H2 = H- + H   ?
  } else {
    k[13] = 0.e0;
  }
  if(tempr <= pow(10.e0,2.2e0)) { //H2 + H = 3H, H2 + H2 = H2 + H + H   ???
    k[14]=0.e0;
    k[15]=0.e0;
  } else {
    x=log10(tempr/1.e4);
         
    ncr_1 = pow(10.e0,4.00e0 - 0.416e0*x - 0.327e0*x*x);
    ncr_2 = pow(10.e0,4.13e0 - 0.968e0*x + 0.119e0*x*x);
         
    kh_1=3.52e-9*exp(-43900.e0/tempr);
    kh_2=5.48e-9*exp(-53000.e0/tempr);

    if(tempr >= 7390.e0) {
      kl_1=6.11e-14*exp(-29300.e0/tempr);
    } else { 
      if (tempr > pow(10.e0,2.6e0)) {
        kl_1=2.67e-15*exp(-pow(6750.e0/tempr,2.e0));
      } else {
        kl_1 = 0.e0;
      }
    }
         
    if(tempr >= 7291.e0) {
      kl_2=5.22e-14*exp(-32200.e0/tempr);
    } else { 
      if (tempr > pow(10.e0,2.8e0)) {
        kl_2=3.17e-15*exp(-(4060.e0/tempr)-pow(7500.e0/tempr,2.e0));
      } else {
        kl_2 = 0.e0;
      }
    }
         
    k[14] = kh_1*pow(kl_1/kh_1,1.e0/(1.e0+xnh/ncr_1));
    k[15] = kh_2*pow(kl_2/kh_2,1.e0/(1.e0+xnh/ncr_2));
  }
   
/*--- More reactions from Shapiro and Kang:  */

  if (tempr > pow(10.e0,2.e0)) {
    k[16]=2.40e-9*exp(-21200.e0/tempr); //H2 + H+ = H2+ + H
  } else {
    k[16] = 0.e0;
  }

  if (tempr > pow(10.e0,2.6e0)) {
    k[17]=4.38e-10*exp(-102000.e0/tempr)*pow(tempr,0.35e0); //H2 + e- = 2H + e-
  } else {
    k[17] = 0.e0;
  }

  if (tempr > pow(10.e0,1.6e0)) {
    k[18]=4.e-12*tempr*exp(-8750.e0/tempr); //H- + e- = H + 2e-
  } else {
    k[18] = 0.e0;
  }

  if (tempr > pow(10.e0,1.6e0)) {
    k[19]=5.3e-20*pow(tempr,2.17e0)*exp(-8750.e0/tempr); //H- + H = 2H + e-
  } else {
    k[19] = 0.e0;
  }

  if (tempr > 1.e4) {
    k[20]=1.e-8*pow(tempr,-0.4e0); //H- + H+ = H2+ + e-   ?
  } else {
    if (tempr > pow(10.e0,1.8e0)) {
      k[20]=4.e-4*pow(tempr,-1.4e0)*exp(-15100.e0/tempr); 
    } else {
      k[20] = 0.e0;
    }
  }

// Radiative processes
	k[21] = 5.16e-12*J_21; // H + hnu = H+ + e-
	k[22] = 7.9e-12*J_21;
	k[23] = 1.9e-13*J_21;
	k[24] = 2.0e-11*J_21;
	k[25] = 4.8e-12*J_21;
	k[26] = 1.2e-11*J_21;

/*** 3-body-reactions from Palla et al.,1983,ApJ 271,632  */

	k[28]=5.5e-29/tempr; //H + H + H = H2 + H
	k[29]=0.125e0*k[28]; //H + H + H2 = H2 + H2

// HD from Galli 1998
	k[30]=3.7e-10*pow(T,0.28)*exp(-43.0/T); //D + H+ = D+ + H
	k[31]=3.7e-10*pow(T,0.28); //D+ + H = D + H+
	k[32]=2.1e-9; //D+ + H2 = H+ + HD
	k[33]=1.e-9*exp(-464.0/T); //HD + H+ = H2 + D+
// Li from Galli 1998
	k[34]=1.036e-11/(pow(T/107.7,0.5) * pow(1+pow(T/107.7, 0.5), 0.612) * pow(1+pow(T/1.177e7, 0.5), 1.388)); // Li+ + e- = Li + hnu
	k[35]=6.3e-6*pow(T,-0.5)-7.6e-9+2.6e-10*pow(T,0.5)+2.7e-14*T; //Li- + H+ = Li + H
	k[36]=6.1e-17*pow(T,0.58)*exp(-T/17200.e0); //Li + e- = Li- + hnu
	k[37]= 1.0/(5.6e19*pow(T,-0.15)+7.2e15*pow(T,1.21)); //Li(2S) + H = LiH + hnu
	k[38]=4.0e-10; //Li- + H = LiH + e-
	k[39]=2.0e-11; //LiH + H = Li + H2
	k[40]=pow(10, -22.4+0.999*logT-0.351*pow(logT,2) ); //Li+ + H = LiH+ + hnu
	k[41]=4.8e-14*pow(T,-0.49); //Li + H+ = LiH+ + hnu
	k[42]=3.8e-7*pow(T,-0.47); //LiH+ + e- = Li + H
	k[43]=3.0e-10; //LiH+ + H = Li+ + H2
	k[44]=0.0; //2.5e-40*T**7.9*np.exp(-T/1210)
	k[45]=0.0; //1.7e-13*T**-0.051*np.exp(-T/282000)
	k[46]=0.0; //4.e-10

	S2 = 1.0/(1.0+3.0*exp(-1.85*1.6e-12/(T*kB)));
	k[37]=k[37]*S2 + (1.9e-14*pow(T,-0.34)+2.0e-16*pow(T, 0.18)*exp(-T/5100.e0))*(1.0-S2);
	if (k[35]<=0){
		k[35] = 0.0;}

// CMB (GP98)
	k[47] = 0.e0; // 2.41e15*Tcmb**1.5 * np.exp(-39472/Tcmb)*8.76e-11*(1+z)**-0.58 # H + gamma = H+ + e-
	k[48] = 1.1e-1*pow(Tcmb,2.13)* exp(-8823.e0/Tcmb); //H- + gamma = H + e-
	k[49] = 20*pow(Tcmb, 1.59)* exp(-82000.e0/Tcmb); //H2+ + gamma = H + H+
}

/*
void cos_doubles(double * in_array, double * out_array, int size){
    int i;
    for(i=0;i<size;i++){
        out_array[i] = cos(in_array[i]);
    }
}
*/

