#### Program FourierSPH
#Considers band bowing effects on valence band
#Does not consider valence band degeneracy

import os
import numpy as np
from scipy.linalg import solve, eig, eigh

file_path = os.path.abspath('fouriersph_v1.py')
index = [ind for ind, char in enumerate(file_path) if char == '\\']
file_path = file_path[:index[-1] + 1]

def SB0(X):
    SB0 = np.sin(X)/X
    return SB0

def SB1(X):
    SB1=np.sin(X)/X**2-np.cos(X)/X
    return SB1

Z0 = np.zeros(300, dtype = np.float32)
Z1 = np.zeros(100, dtype = np.float32)
Z2 = np.copy(Z1)
ZP = np.copy(Z1)
V = np.zeros((50, 50), dtype = np.float32)
T = np.copy(V)
T1 = np.copy(V)
AN0 = np.zeros(200, dtype = np.float32)
AN1 = np.copy(AN0)
A0 = np.copy(AN0)
ACN = np.copy(AN0)
C0 = np.zeros(500, dtype = np.float32)
C1 = np.copy(C0)
CBAND = np.zeros(400, dtype = np.float32)
VBAND = np.copy(CBAND)
VELEC = np.copy(CBAND)
VHOLE = np.copy(CBAND)
AHMASS = np.copy(CBAND)
AEMASS = np.copy(CBAND)
FE = np.zeros((400, 10), dtype = np.float32)
FE1 = np.copy(FE)
FH = np.copy(FE)
EE = np.zeros(150, dtype = np.float32)
EH = np.copy(EE)
FELEC = np.copy(C0)
FHOLE = np.copy(C0)
RS = np.copy(C0)
APARAM = np.copy(C0)
EPSV = np.copy(C0)
EVAL = np.zeros(50, dtype = np.float32)
EVEC = np.copy(V)
FKE = np.copy(Z0)
FKE1 = np.copy(Z0)
FKE2 = np.copy(Z0)
FKE3 = np.copy(Z0)
FKET = np.copy(Z0)
FKH = np.copy(C0)
AMIX = np.zeros(10, dtype = np.float32)
AMIX2 = np.copy(AMIX)
ESTRAINR = np.copy(C0)
ESTRAINT = np.copy(C0)
EPSR1 = np.copy(C0)
P = np.copy(C0)
AM = np.zeros((500, 500), dtype = np.float32)
C = np.copy(C0)
BII = np.copy(C0)
BIO = np.copy(C0)
BOI = np.copy(C0)
BOO = np.copy(C0)
RAD = np.copy(C0)
SV = np.copy(C0)
DISP = np.copy(C0)
EPSR = np.copy(C0)
EPSTH = np.copy(C0)
SIGTT = np.copy(C0)
LDA=50
LDEVEC=50
PI=3.1415926535

ishell = input('what type of shell?  1=CdS, 2=ZnSe, 3=ZnS\n')

if ishell == '3':
    print(' CdSe/ZnS')
    ve=8870
    vh=7260
    smsh=1.3
    smse=.25
    con1=0.11
    BMS=77.5
    ashell=23870
    
elif ishell == '2':
    print(' CdSe/ZnSe')
    ve=5650
    vh=4500
    smsh=.6
    smse=.22
    con1=0.075
    BMS=64.4
    ashell=40240.
    
elif ishell == '1':
    print(' CdSe/CdS')
    ve=380
    vh=6475
    smsh=.80
    smse=.21
    con1=0.044
    BMS=63.
    ashell=23870

ianswer = 1
while ianswer == 1:
    diam = np.float32(input('  enter core diameter (nm)\n'))
    TSHELL = np.float32(input('  enter shell thickness (nm)\n'))
    D = np.float32(input(' diffusion factor, Dt (0.0=sharp interface)\n'))
    fac1 = np.float32(input(' compression fraction (0.0 to 1.0)\n'))
    CONTOT = con1*fac1


    #poisson ratio
    PR=.34
    #CdSe bulk modulus / GPa
    BMCdSe=53.3

    T11=(1.-2*PR)/(1.+PR)

    #core radius
    R1=diam/2
    print(f'\n\ncore diameter{diam:11.6f}')
    print(f'shell thickness{TSHELL:11.6f}')
    print(f'effective lattice mismatch{CONTOT:15.7e}')
    print(f'diffusion parameter{D:11.7f}')


    R2=R1+TSHELL
    R3=R2+1.51
    X1=R1/R2
    NUM=160
    NUM2=int(NUM*R3/R2)

    DX=np.float32(1./NUM)
    DX2=np.float32(1./NUM2)
    DR=np.float32(R2/np.float32(NUM))
    NT=25
    NS=5
    for I in range(1, NUM2 + 1):
        RAD[I - 1]=I*DR

    #NT is the number of Fourier terms, NS is the number of states calculated

    VAC=20000.
    DCONST=10.2
    # vacuum level and dielectric constant
    NTCOMP=25
    jflag=0
    SCALE=1.0

    #core and shell hole effective masses
    CORX=14030.
    CMASSE=0.11*SCALE
    CMASSH=0.40*SCALE
    SMASSH=smsh*SCALE
    SMASSE=smse*SCALE


    CONST=307.0
    EHCONST=11610.
    # const is simply h-bar ** 2 / (2 * electron mass) in units of wavenumbers and nanometers
    # ehconst is (electron charge) ** 2 / (4 * pi * epsilon-zero) in units of wavenumbers and nanometers

    # zeros of j1
    Z1[0]= 4.49341
    Z1[1]= 7.72525
    Z1[2]= 10.90412
    Z1[3]= 14.06619
    Z1[4]= 17.22076
    Z1[5]= 20.3713
    Z1[6]= 23.51945
    Z1[7]= 26.66605
    Z1[8]= 29.8116
    Z1[9]= 32.95639

    for j in range(8, 29):
        Z1[j - 1] = 1.36592 + 3.17771*j - 0.00218*j**2 + 0.000040105*j**3
    #    y=1.36592 + 3.17771 X - 0.00218 X^2 + 0.000040105 X^3


    # calculate normalization factors for the spherical bessel functions
    for n in range(NT):
        AN0[n]=0
        AN1[n]=0
    for N in range(1, NT + 1):
        for J in range(1, NUM2 + 1):
            X=J*DX2
            R=X*R3
            AN0[N - 1]=AN0[N - 1]+(np.sin(N*PI*X)/(N*PI*X))**2*R**2*DR
            AN1[N - 1]=AN1[N - 1]+SB1(X*Z1[N - 1])**2*R**2*DR
    for N in range(NT):
        AN0[N]=1.0/np.sqrt(AN0[N])
        AN1[N]=1.0/np.sqrt(AN1[N])

    #----------------------------------------------------------------------
    # set up initial composition step function
    X=0
    for I in range(NUM):
        X=X+DX
        if X > X1:
            C0[I]=0.0
        else:
            C0[I]=1.0
    X=0.
    AVE=0.
    TOT = 0.
    for I in range(NUM):
        X=X+DX
        AVE=AVE + X**2*DX*C0[I]
        TOT=TOT + X**2*DX
    AVE=AVE/TOT

    # average composition
    for i in range(NUM):
        A0[i]=0
        ACN[i]=0
        C1[i]=0.
    for N in range(NTCOMP):
        NM1 = N
        X=0.0
        for I in range(NUM):
            X=X+DX
            A0[N]=A0[N]+X**2*DX*C0[I]*SB0(X*Z1[N])
            ACN[N]=ACN[N]+X**2*DX*(SB0(X*Z1[N]))**2
        A0[N]=(A0[N]/ACN[N])
    
    # radially dependent composition, C1
    for N in range (NTCOMP):
        X=0.
        for I in range(NUM):
            X=X+DX
            C1[I]=C1[I]+A0[N]*SB0(X*Z1[N])*np.exp(-D*(Z1[N]/R2)**2)

    AVE0=0.
    AVE1=0.
    AVET=0.
    for I in range(1, NUM + 1):
        X=I*DX
        AVE1=AVE1 + X**2*DX*C1[I - 1]
        AVE0=AVE0 + X**2*DX*C0[I - 1]
        AVET=AVET + X**2*DX
    AVE=(AVE0-AVE1)/AVET
    for I in range(NUM):
        C1[I]=C1[I]+AVE
    for i in range(NUM):
        if C1[i] > 1.0:
            C1[i] = 1.0
        if C1[i] < 0.0:
            C1[i] = 0.0
    #AVE0=0.
    #AVE1=0.
    #AVET=0.
    #for I in range(1, NUM + 1):
    #    X=I*DX
    #    AVE1=AVE1 + X**2*DX*C1[I - 1]
    #    AVE0=AVE0 + X**2*DX*C0[I - 1]
    #    AVET=AVET + X**2*DX
    #AVE=(AVE0-AVE1)/AVET
    #for I in range(NUM):
    #    C1[I]=C1[I]+AVE

    #compression calculation
    #radial displacement equation factors

    # first calculate the r=0 and r=R end-points
    YM=3*(BMCdSe*C1[0]+BMS*(1.0-C1[0]))*(1-2*PR)
    #YM is the Young's modulus, PR is the Poisson ratio
    PF=(1+PR)/YM
    BOO[0]=-RAD[0]*PF*T11
    # T11 is (1-2v)/(1+v)
    RCP=(RAD[NUM-1]/RAD[NUM - 2])**3
    YM=3*(BMCdSe*C1[NUM-1]+BMS*(1.0-C1[NUM-1]))*(1-2*PR)
    PF=(1+PR)/YM
    BII[NUM-1]=RAD[NUM-1]*PF*(T11+0.5*RCP)/(RCP-1)
    # calculate these factors for all other r values
    for J in range(1, NUM-1):
        YM=3*(BMCdSe*C1[J]+BMS*(1.0-C1[J]))*(1-2*PR)
        PF=(1+PR)/YM
        RCP=(RAD[J]/RAD[J - 1])**3
        RCPI=(RAD[J - 1]/RAD[J])**3
        BII[J]=RAD[J - 1]*PF*(T11+.5*RCP)/(RCP-1)
        BOI[J]=RAD[J - 1]*PF*(T11+.5)/(RCPI-1)
        BIO[J]=RAD[J]*PF*(T11+.5)/(RCP-1)
        BOO[J]=RAD[J]*PF*(T11+.5*RCPI)/(RCPI-1)

    # radially dependent pressure
    # first calculate strain boundary condition, eqn 2
    for I in range(NUM-1):
        SV[I]=RAD[I]*CONTOT*(C1[I+1]-C1[I])

    for J in range(1,NUM):
    # calculate tri-diagonal displacement matrix, eqn 5.
        if J == 1:
            AM[J-1,J-1]=BOO[J-1]-BII[J]
            AM[J-1,J]=-BOI[J]
            continue
        if J == NUM - 1:
            AM[J-1,J-2]=BIO[J-1]
            AM[J-1,J-1]=BOO[J-1]-BII[J]
            continue

        AM[J-1,J-2]=BIO[J-1]
        AM[J-1,J-1]=BOO[J-1]-BII[J]
        AM[J-1,J]=-BOI[J]

    P[:NUM-1] = solve(AM[:NUM-1, :NUM-1], SV[:NUM-1])
    # calculate the displacement u(r) in DISP
    for I in range(NUM-1):
        RR3=(RAD[I]/RAD[I+1])**3
        # a/b in Saada
        RRM3=1/RR3
        # b/a in Saada
        YM=3*(BMCdSe*C1[I]+BMS*(1.0-C1[I]))*(1-2*PR)
        A=(RAD[I+1]*(1+PR)/YM)*(T11+0.5)/(RRM3-1)
        B=(RAD[I+1]*(1+PR)/YM)*(T11+0.5*RR3)/(RR3-1)
        DISP[I]=P[I]*A+P[I+1]*B
        #  eq 11.4.16 in Saada, evaluated at b
        A1=1.5/(RRM3-1)
        B1=0.5*(RR3+2)/(1-RR3)
        #tangentail component of the stress tensor
        SIGTT[I]=P[I]*A1-P[I+1]*B1
    strtot=0
    strr=0
    strt=0
    for I in range(1, NUM-1):
        # radial component of the stress tensor
        EPSR[I]=(DISP[I+1]-DISP[I]+SV[I])/DR 
        #tangential component of the stress tensor
        EPSTH[I]=DISP[I]/RAD[I+1]
        EPSV[I]=EPSR[I]+2*EPSTH[I]
        #strain energy
        ESTRAINR[I]=-EPSR[I]*P[I]
        ESTRAINT[I]=2*EPSTH[I]*SIGTT[I]
        strtot=strtot+ (ESTRAINR[I]+ESTRAINT[I])*4.0*PI*RAD[I]**2*DR
        strr=strr+ESTRAINR[I]*4.0*PI*RAD[I]**2*DR
        strt=strt+ESTRAINT[I]*4.0*PI*RAD[I]**2*DR
        #print(RAD[I], ESTRAINR[I], ESTRAINT[I]) 
    #convert to eV
    strtot=strtot*6.242
    print(f'core pressure{P[9]:11.6f}')
    print(f'total strain energy (eV){strtot:11.5f}')
    print(f'strain energy density (eV/nm^2){strtot/(4.*PI*R1**2):11.7f}')
    #press = open('press.dat', 'w')
    #for I in range(NUM):
        #press.write(RAD[i],C1[i],SV[i],P[i])
    #press.write(P[9])
    #press.close()
    SCALE =1.1
    while jflag <= 3:
        #print(CMASSE,CMASSH)
        CMASSE=SCALE*0.11
        SMASSE=SCALE*smse
        # calculate the position dependent effective masses
        for I in range(1, NUM2 + 1):
            if I <= NUM:
                AEMASS[I - 1]=CMASSE + (1.0-C1[I - 1])*(SMASSE-CMASSE)
                AHMASS[I - 1]=CMASSH + (1.0-C1[I - 1])*(SMASSH-CMASSH)
            else:
                AEMASS[I - 1]=1.
                AHMASS[I - 1]=1.
        #for ii in range(240):
            #P[ii]=0.

    #---------------------------------------------------------------------
        EELEC=CONST/R3**2
        EHOLE=CONST/R3**2
        PSHIFT=-EPSV[9]*(C1[9]*18550.+(1.-C1[9])*ashell)
        CORX=14030.
        CORX=CORX+PSHIFT

        #print(f' diffusion parameter {D}')
        #print(f' conduction and valence band offsets and compression shifts {VE} {VH} {PSHIFT}\n')
        #print('\n')
        # band bowing
        #Eg(x)=(1-x)Eg(CdX)+xEg(CdY) + b x(1-x)
        # b=1.27 eV for CdSe/CdTe 3.5 nm particles
        # b=0.028 for CdSe - CdS
        #bow=0.28*8065
        bow=0.
    # calculate the potentials

        for I in range(1, NUM2 + 1):
            if I <= NUM:
                #VELEC[I - 1]=-C1[I - 1]*ve
                PSHIFT=-EPSV[I - 1]*(C1[I - 1]*18550.+(1.-C1[I - 1])*23872.)
                VELEC[I - 1] = (1-C1[I - 1])*ve+PSHIFT
            else:
                VELEC[I - 1]=VAC

        for I in range(1, NUM2 + 1):
            if I <= NUM:
                #VHOLE[I - 1]=(1.0-C1[I - 1])*vh
                VHOLE[I - 1] = (1.0-C1[I - 1])*vh - bow*C1[I - 1]*(1.0-C1[I - 1])
            else:
                VHOLE[I - 1]=VAC

    #ELECTRON WAVE FUNCTION----------------------------------------------
    # S states
    #calculate the V matrix
        for I in range(1, NT + 1):
            for J in range(1, I + 1):
                V[I - 1,J - 1]=0.
                for IC in range(1, NUM2 + 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)
                    BSLI=AN0[I - 1]*np.sin(I*PI*X)/(I*PI*X)
                    V[I - 1,J - 1]=V[I - 1,J - 1]+BSLJ*BSLI*R**2*DR*VELEC[IC - 1]
                V[J - 1,I - 1]=V[I - 1,J - 1]

    # calculate the T matrix
        for I in range(1, NT + 1):
            for J in range(1, I + 1):
                T[I - 1,J - 1]=0.
                for IC in range(1, NUM2 + 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)
                    BSLI=AN0[I - 1]*np.sin(I*PI*X)/(I*PI*X)
                    T[I - 1, J - 1]=T[I - 1, J - 1]+BSLJ*BSLI*R**2*DR*(1.0/AEMASS[IC - 1])
                T[I - 1,J - 1]=T[I - 1,J - 1]*EELEC*(J*PI)**2
                T[J - 1, I - 1]=T[I - 1, J - 1]

    # calculate the T1 matrix
        for I in range(1, NT + 1):
            for J in range(1, I + 1):
                T1[I - 1,J - 1]=0.
                for IC in range(1, NUM2 - 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)
                    BSLI=AN0[I - 1]*(np.sin(I*PI*(X+DX2))/(I*PI*(X+DX2))-np.sin(I*PI*X)/(I*PI*X))/DR
                    T1[I - 1, J - 1]=T1[I - 1, J - 1]-(1.0/AEMASS[IC]-1.0/AEMASS[IC - 1])*BSLJ*BSLI*R**2
                T1[I - 1,J - 1]=T1[I - 1,J - 1]*EELEC
                T1[J - 1, I - 1]=T1[I - 1, J - 1]

        for I in range(NT):
            for J in range(NT):
                V[I,J]=V[I,J]+T[I,J]+T1[I,J]

        EVAL, EVEC = eigh(V)
        EE=EVAL[EVAL > 0]
        EVEC=EVEC[:, ::-1]

    # FE is the electron wavefunction
        for JS in range(NS):
            for IC in range(NUM2 + 1):
                FE[IC,JS]=0.0
        for JS in range(NS):
            for IC in range(1,NUM2 + 2):
                X=IC*DX2
                for J in range(1,NT + 1):
                    FE[IC - 1,JS]=FE[IC - 1,JS]+EVEC[J - 1,NT-JS-1]*AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)

        for JS in range(NS):
            TOT=0.
            for IC in range(1,NUM2 + 1):
                X=IC*DX2
                R=X*R3
                TOT=TOT+FE[IC - 1,JS]**2*R**2*DR
            for IC in range(NUM2):
                FE[IC,JS]=FE[IC,JS]/np.sqrt(TOT)

        #print the amplitude of the wavefunction at the particle surface
        #for JS in range(NS):
            #print(FE[NUM - 1,JS]**2)

    #P states
    #calculate the V matrix
        for I in range(NT):
            for J in range(I + 1):
                V[I,J]=0.
                for IC in range(1, NUM2 + 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN1[J]*SB1(X*Z1[J])
                    BSLI=AN1[I]*SB1(X*Z1[I])
                    V[I,J]=V[I,J]+BSLJ*BSLI*R**2*DR*VELEC[IC - 1]
                V[J,I]=V[I,J]

    # calculate the T matrix
        for I in range(NT):
            for J in range(I + 1):
                T[I,J]=0.
                for IC in range(1, NUM2 + 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN1[J]*SB1(X*Z1[J])
                    BSLI=AN1[I]*SB1(X*Z1[I])
                    T[I,J]=T[I,J]+BSLJ*BSLI*R**2*DR*(1.0/AEMASS[IC - 1])
                T[I,J]=T[I,J]*EELEC*Z1[J]**2
                T[J,I]=T[I,J]

    # calculate the T1 matrix
        for I in range(NT):
            for J in range(I + 1):
                T1[I,J]=0.
                for IC in range(1, NUM2 - 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN1[J]*SB1(X*Z1[J])
                    BSLI=AN1[I]*(SB1((X+DX2)*Z1[I])-SB1(X*Z1[I]))/DR
                    T1[I,J]=T1[I,J]-(1.0/AEMASS[IC]-1.0/AEMASS[IC - 1])*BSLJ*BSLI*R**2
                T1[I,J]=T1[I,J]*EELEC
                T1[J,I]=T1[I,J]

        for I in range(NT):
            for J in range(NT):
                V[I,J]=V[I,J]+T[I,J]+T1[I,J]

        EVAL, EVEC = eigh(V)
        EE1=EVAL[EVAL > 0]
        EVEC=EVEC[:, ::-1]
    
    # FE is the electron wavefunction
        for JS in range(NS):
            for IC in range(NUM2 + 1):
                FE1[IC,JS]=0.0
        for JS in range(NS):
            for IC in range(1,NUM2 + 2):
                X=IC*DX2
                for J in range(NT):
                    FE1[IC - 1,JS]=FE1[IC - 1,JS]+EVEC[J,NT-JS-1]*AN0[J]*SB1(X*Z1[J])

        for JS in range(NS):
            TOT=0.
            for IC in range(1,NUM2 + 1):
                X=IC*DX2
                R=X*R3
                TOT=TOT+FE1[IC - 1,JS]**2*R**2*DR
            for IC in range(NUM2):
                FE1[IC,JS]=FE1[IC,JS]/np.sqrt(TOT)
        
    # hole wave function--------------------------------------------------

    # calculate the V matrix
        for I in range(1, NT + 1):
            for J in range(1, I + 1):
                V[I - 1,J - 1]=0.
                for IC in range(1, NUM2 + 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)
                    BSLI=AN0[I - 1]*np.sin(I*PI*X)/(I*PI*X)
                    V[I - 1,J - 1]=V[I - 1,J - 1]+BSLJ*BSLI*R**2*DR*VHOLE[IC - 1]
                V[J - 1,I - 1]=V[I - 1,J - 1]

    # calculate the T matrix
        for I in range(1, NT + 1):
            for J in range(1, I + 1):
                T[I - 1,J- 1]=0.
                for IC in range(1, NUM2 + 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)
                    BSLI=AN0[I - 1]*np.sin(I*PI*X)/(I*PI*X)
                    T[I - 1, J - 1]=T[I - 1, J - 1]+BSLJ*BSLI*R**2*(DR/AHMASS[IC - 1])
                T[I - 1,J - 1]=T[I - 1,J - 1]*EHOLE*(J*PI)**2
                T[J - 1, I - 1]=T[I - 1, J - 1]

    # calculate the T1 matrix
        for I in range(1, NT + 1):
            for J in range(1, I + 1):
                T1[I - 1,J - 1]=0.
                for IC in range(1, NUM2 - 1):
                    X=IC*DX2
                    R=X*R3
                    BSLJ=AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)
                    BSLI=AN0[I - 1]*(np.sin(I*PI*(X+DX2))/(I*PI*(X+DX2))-np.sin(I*PI*X)/(I*PI*X))/DR
                    T1[I - 1, J - 1]=T1[I - 1, J - 1]-(1.0/AHMASS[IC]-1.0/AHMASS[IC - 1])*BSLJ*BSLI*R**2
                T1[I - 1,J - 1]=T1[I - 1,J - 1]*EHOLE
                T1[J - 1, I - 1]=T1[I - 1, J - 1]

        for I in range(NT):
            for J in range(NT):
                V[I,J]=V[I,J]+T[I,J]+T1[I,J]

        EVAL, EVEC = eigh(V)
        EH=EVAL[EVAL > 0]
        EVEC=EVEC[:, ::-1]

    # FH is the hole wavefunction
        for JS in range(NS):
            for IC in range(1, NUM2 + 1):
                X=IC*DX2
                FH[IC - 1,JS]=0.
                for J in range(1,NT + 1):
                    FH[IC - 1,JS]=FH[IC - 1,JS]+EVEC[J - 1,NT-JS-1]*AN0[J - 1]*np.sin(J*PI*X)/(J*PI*X)

        for JS in range(NS):
            TOT=0.
            for IC in range(1,NUM2 + 1):
                X=IC*DX2
                R=X*R3
                TOT=TOT+FH[IC - 1,JS]**2*R**2*DR
            for IC in range(NUM2):
                FH[IC,JS]=FH[IC,JS]/np.sqrt(TOT)

    #---------------------------------------------------------------------
        #print electron and hole energies
        #print('  n       nSe energy   nPe energy    nSh energy')
        #for i in range(ns):
            #print(i,EE[i], EE1[i], Eh[i])
    
    #---------------------------------------------------------------------
    # electron-hole interaction
    #print('\n')
    # electron wavefunction
        for I in range(NUM2):
            FELEC[I]=FE[I,0]
            FHOLE[I]=FH[I,0]
        for JS in range(1, 8):
            TOTE=0
            TOTH=0
            for I in range(1,NUM2 + 1):
                for J in range(1,NUM2 + 1):
                    RE=np.float32(I*DR)
                    RH=np.float32(J*DR)
                    if RE > RH:
                        X=RE
                    else:
                        X=RH
                    TOTE=TOTE+RE**2*RH**2*FH[J - 1,0]**2*FE[I - 1,0]*FE[I - 1,JS]*DR**2*(1.0/X)
                    TOTH=TOTH+RE**2*RH**2*FE[I - 1,0]**2*FH[J - 1,0]*FH[J - 1,JS]*DR**2*(1.0/X)
            TOTE=TOTE*EHCONST/DCONST
            TOTH=TOTH*EHCONST/DCONST
            for I in range(NUM2):
                FELEC[I]= FELEC[I]+(TOTE/(EE[JS]-EE[0]))*FE[I,JS]
                FHOLE[I]= FHOLE[I]+(TOTE/(EH[JS]-EH[0]))*FH[I,JS]
        if False:
            for ic in range(1,NUM2 + 1):
                if ic < NUM:
                    FHOLE[ic]=1.
                else:
                    FHOLE[ic]=0.
    # renormalize wavefunctions
        TOTE=0
        TOTH=0
        for IC in range(1,NUM2 + 1):
            R=IC*DX2*R3
            TOTE=TOTE+FELEC[IC]**2*R**2*DR
            TOTH=TOTH+FHOLE[IC]**2*R**2*DR
        for IC in range(NUM2):
            FELEC[IC]=FELEC[IC]/np.sqrt(TOTE)
            FHOLE[IC]=FHOLE[IC]/np.sqrt(TOTH)
        
    # calculate electron-hole attraction energy of first order corrected functions
        TOT=0.
        for I in range(1,NUM2 + 1):
            for J in range(1,NUM2 + 1):
                RE=np.float32(I*DX2*R3)
                RH=np.float32(J*DX2*R3)
                if RE > RH:
                    X=RE
                else:
                    X=RH
                TOT=TOT+RE**2*RH**2*FHOLE[J - 1]**2*FELEC[I - 1]**2*(1.0/X)*DR**2
        TOT=TOT*EHCONST/DCONST
        EHREP = TOT
        QCE=EE[0]+EH[0]-TOT
    #_____________________________________________________________________

        SCALE =0.36773 + (EE[0]+EH[0])*2.7563E-4 - 8.31053E-9*(EE[0]+EH[0])**2

        #print(SCALE)

    #_____________________________________________________________________
    #calculate the electron hole overlap
        S=0
        SCE=0
        SCH=0
        for IC in range(1,NUM2 + 1):
            X=IC*DX2
            R=X*R3
            S=S+FHOLE[IC - 1]*FELEC[IC - 1]*R**2*DR
            if R>R1:
                continue
            SCE=SCE+FELEC[IC]**2*R**2*DR
            SCH=SCH+FHOLE[IC]**2*R**2*DR
        S=S**2

    # calculate electron hole overlap for unperturbed electron functions
    # 1S
        S1=0
        for IC in range(1,NUM2 + 1):
            X=IC*DX2
            R=X*R3
            S1=S1+FHOLE[IC - 1]*FELEC[IC - 1]*R**2*DR
        S1=S1**2
        #print(f'1Se/1Sh overlap ={S1}')

    # 2S
        #S1=0
        #for IC in range(1,num2 + 1):
            #X=IC*DX2
            #R=X*R3
            #S1=S1+FHOLE[IC - 1]*FELEC[IC - 1]*R**2*DR
        #S1=S1**2
        #print(f'2Se/1Sh overlap ={S1}')
        jflag=jflag+1
    
    # calculate Kane parameters
    V[0,0]=EE[0]+CORX/2
    V[1,1]=-EH[0]-CORX/2
    V[1,0]=np.sqrt(141000.*EE[0]*CMASSE)
    #V[0,1] = V[1,0]
    V[0,1]=np.sqrt(141000.*EH[0]*CMASSH)
    EVAL, EVEC = np.linalg.eig(V)
    fkane=(EVEC[0,0]**2*EVEC[0,1]**2)
    fkane2=EVEC[1,0]**2
    #print(EVAL[0], EVAL[1])
    #print(EVEC[0, 0], EVEC[1, 0])
    #print(EVEC[0, 1], EVEC[1, 1])
    #print(fkane, fkane2)
    print('')
    #if False:
    COMPOSITION = open(file_path + 'COMPOSITION.DAT', 'w')
    X=0.
    for I in range(NUM2):
        X=X+DX2*R3
        COMPOSITION.write(f'{X:15.8f}{C0[I]:15.8f}{C1[I]:15.8f}{VELEC[I]:15.8f}{VHOLE[I]:15.8f}{SV[I]:15.8f}{P[I]:15.8f}\n')
    COMPOSITION.close()

    FUNCTION = open(file_path + 'FUNCTION.DAT', 'w')
    for IC in range(1,NUM2 + 1):
        X=(IC-1)*DX2
        FUNCTION.write(f'{X*R3:15.8f}{FE[IC - 1,0]:15.8f}{FH[IC - 1,0]:15.8f}\n') #FE[IC - 1,1], FE1[IC - 1,1])
        #for I in range(NS):
            #FUNCTION.write(X*R3,FE[IC - 1,I],FH[IC - 1,I])
    FUNCTION.close()
    WAVELEN=1.e+7/(EE[0]+EH[0]-TOT+CORX)
    print(f'electron quantum confinment energy{EE[0]:11.3f}')
    print(f'hole quantum confinment energy{EH[0]:11.3f}')
    print(f'electron-hole interaction energy{EHREP:11.4f}')
    print(f'electron-hole overlap ={S:11.7f}')
    print(f'onset wavelength{WAVELEN:11.4f}')

    #print(TSHELL,WAVELEN, STRTOT, STRTOT/(4.*PI*R1**2))

    print('')
    print('')
    ianswer = eval(input('press 0 to close window, 1 to run again'))
