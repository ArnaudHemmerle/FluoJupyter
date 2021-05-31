
import numpy as np
from scipy.special import erfc

"""
Here are defined the functions for analysis.

Description of the different dictionaries of parameters:
- dparams_fit contains the fitting parameters common to the whole spectrum, such as fano, fG, etc ...
- dparams_0 contains the initial guess for the LM fit
- dparams_lm contains the current parameter set during the LM fit
- dparams_list contains lists of results from the fits (one list per fit parameter)
"""


def Fcn_spectrum(dparams, groups, eV):
    """
    Definition of the spectrum as the sum of the peaks + the background.
    Takes a dictionnary with the params values, the list of Group, the array of eV.
    """

    ct = dparams['ct']
    sl = dparams['sl']
    noise = dparams['noise']

    # The total fit for the whole spectrum
    spectrum_tot = 0.
    # The different parts composing the fit (for control)
    gau_tot = 0.
    she_tot = 0.
    tail_tot = 0.
    compton = 0.
    for group in groups:
        # The total fit for the group of peaks
        spectrum_group = 0.
        # The different parts composing the fit (for control)
        gau_group = 0.
        she_group = 0.
        tail_group = 0.   
        area = group.area
        for peak in group.peaks:
            position = peak.position
            intensity_rel = peak.intensity_rel
            if group.elem_name == 'Compton':
                compton += area*Fcn_compton_peak(position,intensity_rel,eV,dparams)
                spectrum_group += compton
            else:
                ppic, gau, she, tail = Fcn_peak(position,intensity_rel,eV,dparams)
                spectrum_group += area*ppic
                gau_group += area*gau
                she_group += area*she
                tail_group += area*tail
                compton += ppic*0.
                
        spectrum_tot += spectrum_group
        gau_tot += gau_group
        she_tot += she_group
        tail_tot += tail_group

    
    # Uncomment to cut the baseline at the end of the elastic peak
    # Do not forget to comment the line baseline = ct+sl*eV
    
    # We add a linear baseline, which cannot be < 0, and stops after the elastic peak (if there is one)
    limit_baseline = eV[-1]
    for group in groups:
        if group.elem_name == 'Elastic':
            for peak in group.peaks:
                if peak.name == 'El':
                    limit_baseline = peak.position                
    eV_tmp = np.where(eV<limit_baseline+1000*noise, eV, 0.)
    baseline = ct+sl*eV_tmp
    
    
    #baseline = ct+sl*eV
    baseline = np.where(baseline>0.,baseline,0.)
    spectrum_tot+= baseline

    return spectrum_tot, gau_tot, she_tot, tail_tot, baseline, compton


def Interpolate_scf(atom, energy):
    """
    Interpolation for the scattering factors.
    Used here to take into account absorption from Si within the detector.
    Requires the file f-si.
    """
    en2=0.
    f2=0.
    f2p=0.
    for line in open('lib/fit/f-'+str(atom),'r'):
        en1=en2
        f1=f2
        f1p=f2p
        try:
            en2=float(line.split()[0])
            f2=float(line.split()[1])
            f2p=float(line.split()[2])
            if en1<=energy and en2>energy:
                scf=f1+(energy-en1)/(en2-en1)*(f2-f1)
                scfp=f1p+(energy-en1)/(en2-en1)*(f2p-f1p)
            else:
                pass
        except:
            pass
    return scf,scfp

def Fcn_peak(pos, amp, eV, dparams):
    """
    Definition of a peak (area normalised to 1).
    Following:
    - M. Van Gysel, P. Lemberge & P. Van Espen, “Implementation of a spectrum fitting
    procedure using a robust peak model”, X-Ray Spectrometry 32 (2003), 434–441
    - J. Campbell & J.-X. Wang, “Improved model for the intensity of low-energy tailing in
    Si (Li) x-ray spectra”, X-Ray Spectrometry 20 (1991), 191–197
    The params for peak definition should be passed as a dictionnary :
    dparams = {'sl': 0.01, 'ct':-23., 'sfa0':1.3 ... }
    """

    sfa0 = dparams['sfa0']
    sfa1 = dparams['sfa1']
    tfb0 = dparams['tfb0']
    tfb1 = dparams['tfb1']
    twc0 = dparams['twc0']
    twc1 = dparams['twc1']
    noise = dparams['noise']
    fano = dparams['fano']
    epsilon = dparams['epsilon']

    # We work in keV for the peak definition
    pos_keV = pos/1000.
    keV = eV/1000.

    # Peak width after correction from detector resolution (sigmajk)
    wid = np.sqrt((noise/2.3548)**2.+epsilon*fano*pos_keV)

    # Tail width (cannot be <0)
    TW = twc0 + twc1*pos_keV
    TW = np.where(TW>0.,TW,0.)

    # Energy dependent attenuation by Si in the detector
    atwe_Si = 28.086 #atomic weight in g/mol
    rad_el = 2.815e-15 #radius electron in m
    Na = 6.02e23 # in mol-1
    llambda = 12398./pos*1e-10 # in m
    # mass attenuation coefficient of Si in cm^2/g
    musi = 2.*llambda*rad_el*Na*1e4*float(Interpolate_scf('si',pos)[1])/atwe_Si

    # Shelf fraction SF (cannot be <0)
    SF = (sfa0 + sfa1*pos_keV)*musi
    SF = np.where(SF>0.,SF,0.)
    
    # Tail fraction TF (cannot be <0)
    TF = tfb0 + tfb1*musi
    TF = np.where(TF>0.,TF,0.)

    # Definition of gaussian
    arg = (keV-pos_keV)**2./(2.*wid**2.)
    farg = (keV-pos_keV)/wid
    gau = amp/(np.sqrt(2.*np.pi)*wid)*np.exp(-arg)
    # Avoid numerical instabilities
    gau = np.where(gau>1e-10,gau, 0.)

    # Function shelf S(i, Ejk)
    she = amp/(2.*pos_keV)*erfc(farg/np.sqrt(2.))
    # Avoid numerical instabilities
    she = np.where(she>1e-10,she, 0.)

    # Function tail T(i, Ejk)
    tail = amp/(2.*wid*TW)*np.exp(farg/TW+1/(2*TW**2))*erfc(farg/np.sqrt(2.)+1./(np.sqrt(2.)*TW))
   
    # Avoid numerical instabilities
    tail = np.where(tail>1e-10,tail, 0.)

    # Function Peak
    ppic = np.array(gau+SF*she+TF*tail)

    return ppic, np.array(gau), np.array(SF*she), np.array(TF*tail)


def Fcn_compton_peak(pos, amp, eV, dparams):
    """
    The function used to fit the compton peak, inspired by  M. Van Gysel, P. Lemberge & P. Van Espen,
    “Description of Compton peaks in energy-dispersive  x-ray fluorescence spectra”,
    X-Ray Spectrometry 32 (2003), 139–147
    The params for peak definition should be passed as a dictionnary :
    dparams = {'fG': 0.01, 'fA':2., ...}
    """

    fG = dparams['fG']
    noise = dparams['noise']
    fano = dparams['fano']
    epsilon = dparams['epsilon']

    # We work in keV for the peak definition
    pos_keV = pos/1000.
    keV = eV/1000.

    # Peak width after correction from detector resolution (sigmajk)
    wid = np.sqrt((noise/2.3548)**2.+epsilon*fano*pos_keV)

    # Definition of gaussian
    arg = (keV-pos_keV)**2./(2.*(fG*wid)**2.)
    gau = amp/(np.sqrt(2.*np.pi)*fG*wid)*np.exp(-arg)
  
    fA = dparams['fA']
    fB = dparams['fB']
    gammaA = dparams['gammaA']
    gammaB = dparams['gammaB']

    #Low energy tail TA
    farg = (keV-pos_keV)/wid
    TA = amp/(2.*wid*gammaA)*np.exp(farg/gammaA+1/(2*gammaA**2))*erfc(farg/np.sqrt(2.)+1./(np.sqrt(2.)*gammaA))

    #High energy tail TB
    TB = amp/(2.*wid*gammaB)*np.exp(-farg/gammaB+1/(2*gammaB**2))*erfc(-farg/np.sqrt(2.)+1./(np.sqrt(2.)*gammaB))

    ppic = np.array(gau+fA*TA+fB*TB)
    
    # Avoid numerical instabilities
    ppic = np.where(ppic>1e-10,ppic, 0.)
    
    return ppic