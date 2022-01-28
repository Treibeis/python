void chemsl(double dt_tot,double xn,double tempr,double nyin[9], 
double nydin[3],double J_21,double nyout[9],double nydout[3])  
/****************************************************************
** Solves primordial chemistry network for the 9 species : 
** H,H+,H-,H2,H2+,e-,He,He+,He++
** with approximate BDF method! 
** Deuterium network (species: D,D+,HD) is added.
**
** Species are indexed as follows:
** ny[0]: HI
** ny[1]: HII
** ny[2]: H-
** ny[3]: H2
** ny[4]: H2+
** ny[5]: e-
** ny[6]: HeI
** ny[7]: HeII
** ny[8]: HeII
** nyd[0]: DI
** nyd[1]: DII
** nyd[2]: HD
****************************************************************/
{
  double dt,dt_cum;
  double delt_x,XNUM1,XDENOM1,xnd;
  double XNUM2,XDENOM2;
  double kd1,kd3,kd4,kd8,kd10,xnh,xnhe;
  double CHD,DHD,CDp,DDp;
  int kl;
  double nyd[3],ny[9],k[30],Cr[9],Ds[9];

  for (kl=0; kl<9; kl++) {
    ny[kl]=nyin[kl];
  }

  for (kl=0; kl<3; kl++) {
    nyd[kl]=nydin[kl];
  }

  xnh=0.93e0*xn;
  xnhe=0.07e0*xn;
       
  rates(tempr,xnh,k,J_21);
  xnd=4.e-5*xnh;
  kd1=k[3];
  kd3=3.7e-10*pow(tempr,0.28e0)*exp(-43.e0/tempr);
  kd4=3.7e-10*pow(tempr,0.28e0);
  kd8=2.1e-9;
  kd10=1.e-9*exp(-464.e0/tempr);

  dt_cum=0.e0;
  while (dt_cum < dt_tot) {

    dt=dt_tot;

    delt_x=0.e0;
    while (dt >= delt_x) {
      Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+
            (k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5];
      Ds[5]=k[3]*ny[1]+k[4]*ny[7]+
            k[5]*ny[8];
      delt_x=EPS*ny[5]/fabs(Cr[5]-Ds[5]*ny[5]);
      dt =dt/2.e0;
    }

    if ((dt_cum+dt) > dt_tot) {
      dt=dt_tot-dt_cum;
      dt_cum=dt_tot;
    } else {
      dt_cum=dt_cum+dt;
    }

    Cr[5]=k[21]*ny[0]+k[22]*ny[6]+k[23]*ny[7]+
          (k[0]*ny[0]+k[1]*ny[6]+k[2]*ny[7])*ny[5];
    Ds[5]=k[3]*ny[1]+k[4]*ny[7]+
          k[5]*ny[8];
    ny[5]=(ny[5]+Cr[5]*dt)/(1.e0+Ds[5]*dt);

    Cr[0]=k[3]*ny[1]*ny[5];
    Ds[0]=k[0]*ny[5]+k[21];
    ny[0]=(ny[0]+Cr[0]*dt)/(1.e0+Ds[0]*dt);

    Cr[1]=k[0]*ny[5]*ny[0]+k[21]*ny[0];
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
    XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]
         + k[24];
    ny[2]=XNUM1/XDENOM1;

/**** calculate equilibrium abundance for H2+ ********************/
    XNUM2=(k[6]*ny[0] + k[16]*ny[3]+k[20]*ny[2])*ny[1]
        + k[26]*ny[3];
    XDENOM2=k[7]*ny[0]+k[10]*ny[5]+k[11]*ny[2]+k[25];
    ny[4]=XNUM2/XDENOM2;
/*****************************************************************/

    Ds[3]=k[13]*ny[5]+k[14]*ny[0]+k[15]*ny[3]
         +k[16]*ny[1]+k[17]*ny[5]+k[26]+k[27];
    Cr[3]=k[7]*ny[4]*ny[0]+k[9]*ny[2]*ny[0]+k[11]*ny[4]*ny[2];
    ny[3]=(ny[3]+Cr[3]*dt)/(1.e0+Ds[3]*dt);
    tform=ny[3]/Cr[3];
    tdest=1.e0/Ds[3];

    ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
    if (ny[0] < 0.e0) ny[0]=0.e0;
    ny[6]=xnhe-ny[7]-ny[8];
    if (ny[6] < 0.e0) ny[6]=0.e0;
    ny[1]=ny[5]+ny[2]-ny[4]-ny[7]-2.e0*ny[8];
    if (ny[1] < 0.e0) ny[1]=0.e0;
    ny[0]=xnh-2.e0*ny[3]-ny[1]-ny[2]-2.e0*ny[4];
    if (ny[0] < 0.e0) ny[0]=0.e0;

    XNUM1=(k[8]*ny[0] + k[13]*ny[3])*ny[5];
    XDENOM1=(k[9]+k[19])*ny[0]+(k[12]+k[20])*ny[1]+k[18]*ny[5]
        + k[24];
    ny[2]=XNUM1/XDENOM1;
    for (kl=0; kl<9; kl++) {
      if (ny[kl] < 1.e-30) ny[kl]=0.e0;
    }

/**** Solve Deuterium network for D, D+, and HD *********************
**   Galli & Palla (1998) -> 'minimal model for D chemistry!'       */

    CDp=kd3*nyd[0]*ny[1]+kd10*nyd[2]*ny[1];
    DDp=kd1*ny[5]+kd4*ny[0]+kd8*ny[3];
    CHD=kd8*nyd[1]*ny[3];
    DHD=kd10*ny[1];
    for (kl=0; kl<9; kl++) {
      if (ny[kl] < 0.e0) {
        printf("%6d  %15.6e \n",kl,ny[kl]);
        exit(0); 
      }
    }
  }

  for (kl=0; kl<9; kl++) {
    nyout[kl]=ny[kl];
  }

  for (kl=0; kl<3; kl++) {
    nydout[kl]=nydin[kl];
  }

}


void rates(double tempr,double xnh,double k[30],double J_21)
/**========================================================================
*** Rate Coefficients k[0]-k[29]:
***======================================================================*/
{
  double k5a,k5b,ncr_1,ncr_2,kh_1,kh_2,kl_1,kl_2;
  double yab[1],tt,t5fac,x,Tgas;
  yab[0]=xnh;
  tt=sqrt(tempr);
  t5fac=1.e0/(1.e0+sqrt(tempr/1.e5));

/*** recombination rate coefficients: Hep from Black (1981, MNRAS 197, 553)
----                              Hp and Hepp from Cen (1992, ApJS 78, 341)*/
  k5a=1.5e-10*pow(tempr,-0.6353e0);
  if (tempr > pow(10.e0,3.6e0)) {
    k5b=1.9e-3*pow(tempr,-1.5e0)*exp(-4.7e5/tempr)*
      (1.e0+0.3e0*exp(-9.4e4/tempr));
  } else {
    k5b = 0.e0;
  }
  k[4] = k5a + k5b;
  k[3]=8.40e-11/tt*pow(tempr/1.e3,-0.2e0)/
     (1.e0+pow(tempr/1.e6,0.7e0));
  k[5]=3.36e-10/tt*pow(tempr/1.e3,-0.2e0)/
     (1.e0+pow(tempr/1.e6,0.7e0));

/*-- collisional ionization rates (from Black, with Cen fix at high-T):*/

  if (tempr > pow(10.e0,2.8e0)) {
    k[0] = 5.85e-11*tt*exp(-157809.1e0/tempr)*t5fac;
  } else {
    k[0]=0.e0;
  }

  if (tempr > pow(10.e0,3.e0)) {
    k[1] = 2.38e-11*tt*exp(-285335.4e0/tempr)*t5fac;
  } else {
    k[1] = 0.e0;
  }

  if (tempr > pow(10.e0,3.4e0)) {
    k[2] = 5.68e-12*tt*exp(-631515.e0/tempr)*t5fac;
  } else {
    k[2] = 0.e0;
  }

/*--- additional reactions from Z. Haiman: */

  if (tempr <= 4000.e0) {
    k[6] = 1.38e-23*pow(tempr,1.845e0);
  } else {
    if (tempr < pow(10.e0,5.e0)) {
      k[6] = -6.157e-17 + 3.255e-20*tempr - 4.152e-25*tempr*tempr;
    } else {
      k[6] = 0.e0;
    }
  }
  if (k[6] <= 0.e0) k[6]=0.e0;
  k[7] = 6.4e-10;

/**** Try the Abel et al. 1997 rate! */     
  Tgas=tempr;
  k[8]=1.429e-18*pow(Tgas,0.762)*pow(Tgas,0.1523*log10(Tgas))*
          pow(Tgas,-3.274e-2*pow(log10(Tgas),2.e0));

  k[9] = 1.3e-9;
  k[10] = 1.68e-8*pow(tempr/300.e0,-0.29e0);
  k[11] = 5.0e-6/tt;
  k[12] = 7.0e-7/tt;
  if (tempr > pow(10.e0,2.2e0)) {
    k[13] = (2.7e-8/(tempr*tt))*exp(-43000.e0/tempr)*t5fac;
  } else {
    k[13] = 0.e0;
  }
  if(tempr <= pow(10.e0,2.2e0)) {
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
         
    k[14] = kh_1*pow(kl_1/kh_1,1.e0/(1.e0+yab[0]/ncr_1));
    k[15] = kh_2*pow(kl_2/kh_2,1.e0/(1.e0+yab[0]/ncr_2));
         
  }
      
/*--- More reactions from Shapiro and Kang:  */

  if (tempr > pow(10.e0,2.e0)) {
    k[16]=2.40e-9*exp(-21200.e0/tempr);
  } else {
    k[16] = 0.e0;
  }

  if (tempr > pow(10.e0,2.6e0)) {
    k[17]=4.38e-10*exp(-102000.e0/tempr)*pow(tempr,0.35e0);
  } else {
    k[17] = 0.e0;
  }

  if (tempr > pow(10.e0,1.6e0)) {
    k[18]=4.e-12*tempr*exp(-8750.e0/tempr);
  } else {
    k[18] = 0.e0;
  }

  if (tempr > pow(10.e0,1.6e0)) {
    k[19]=5.3e-20*pow(tempr,2.17e0)*exp(-8750.e0/tempr);
  } else {
    k[19] = 0.e0;
  }

  if (tempr > 1.e4) {
    k[20]=1.e-8*pow(tempr,-0.4e0);
  } else {
    if (tempr > pow(10.e0,1.8e0)) {
      k[20]=4.e-4*pow(tempr,-1.4e0)*exp(-15100.e0/tempr);
    } else {
      k[20] = 0.e0;
    }
  }

/*** Radiative processes
**      CALL photo: not implemented here! */

  k[21]=0.e0;
  k[22]=0.e0;
  k[23]=0.e0;
  k[24]=0.e0;
  k[25]=0.e0;
  k[26]=0.e0;
  k[27]=0.e0;

/*** 3-body-reactions from Palla et al.,1983,ApJ 271,632  */

  k[28]=5.5e-29/tempr;
  k[29]=0.125e0*k[28];


  return;
}
