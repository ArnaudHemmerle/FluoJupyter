# JupyFluo

JupyFluo is a Jupyter Notebook to analyze X-Ray Fluorescence (XRF) experiments on the beamline SIRIUS at the synchrotron SOLEIL.  
The notebook should be first set up by an Expert following the instructions in the "Expert" section. User can then follow the guidelines in the "User" section to start using the notebook. Please note that the notebook is currently in development. As such, be skeptical about any unexpected results.  Any feedback on the notebook or the code is welcome.

## Latest version
v0.5

## User

### Getting Started
1. Once the notebook is open, check that the following parameters in the second cell are correct:
```
# Name of the notebook: expt.notebook_name
# Directory where the data will be saved: expt.working_dir
# Directory where the nexus files are: expt.recording_dir
```  

2. Run the two first cells, check that no missing file is reported.

3. Use the dropdown list to choose the scan. Click on ```OK```.  

4. You can enter information about the scan by clicking on ```Insert comment```. 

### Extraction

1. Click on ```Extract the scan```.

2. Here you can define the channels of interest, and the spectrums you want to extract. Put first, for example, ```First Spectrum=0``` and ```Last Spectrum=1```. Click on ```Extract``` .

3. Use the figures to help you choose your subsets of channels and spectrums. 

4. Click again on ```Extract``` to update the parameters or click on ```Continue```.

5. A csv file ```Parameters_extraction.csv``` is created in the folder ```working_directory/filename/```.

### Set the fit parameters

1. Click on ```Set parameters```.

2. The parameters here should be already set by the expert user. You can still play with them and tick/untick the parameters you would like to fit.

3. Click on ```Validate```.

### Choose the peaks
1. Click on ```Set peaks```.  

2. A few peaks are already displayed (typically the Rayleigh and Compton peaks). Check their energies.

3. Use the tool ```Add peaks from database``` to import new peaks.

4. Modify the sheet to remove peaks or change their parameters. You can leave a peak in the list and do not include it in the analysis by writting ```no``` in the column ```#Fit Peak?```. 

5. **The strength of a line is its X-ray fluorescence production cross section with full cascade (in cm^2/g).** It is extracted from the database but can be obtained here as well http://lvserver.ugent.be/xraylib-web/.

6. Keep the peak/line names ```Elastic/El``` and ```Compton/Co```for the elastic (Rayleigh) and Compton peaks. You can add a Compton peak for an element by naming it ```Compton``` as well, and adding the element in the line column. For example, if you want to add a Compton peak for the Au La1 line, name the peak ```Compton``` and the line ```CoAuLa1```.

7. If you add an escape peak, do not name it with the name of its corresponding element! For example, if you add the escape peak of Au La1, name it 'EscAuLa1', name the line 'Esc', and put a strength of 1.

8. You can use the plot below the sheet to find where the peaks are. 

9. When you think you are done with the peaks, validate the sheet by clicking on ```Validate Peaks```.

10. Check the peaks on the spectrum. Modify them, or click on ```Start Fit``` to start the fit.

11. Note: You can also directly edit the Excel file ```Parameters_peaks.csv``` in your folder ```working_directory/filename/``` if you prefer.
 

### Fit the spectrums

When you click on ```Start Fit``` , you can follow the fit in real time. The results are continuously updated in your folder ```working_directory/filename/```. Check the files ```FitResult.csv``` and  the folder ```FitSpectrums/``` that you can open with you prefered software (Excel, Origin, Matlab, vim ...).


Once the fit is done, the fitted parameters will be displayed and saved as png in your folder ```working_directory/filename/```.
 
**The control panel will appear at the bottom of the plots when the fit is done.** 

### Add a plot to the PDF report

1. Until now, nothing will be rendered in the final PDF. Click on the button ```Add a plot to report```.

2. Choose a spectrum (if you were not working on the spectrums sum). Click on ```Preview the selected plot``` to preview it.

3. Choose the spectrum that you want to add to the PDF. Click on ```Add the selected plot```.

4. The output of the cell ```FF.Load_results(expt, spectrum_index=XXX)```, which is automatically generated and executed, will appear in the PDF.

5. Click on ```Export to pdf``` in the panel to generate the PDF, or ```Analyze a new scan``` to continue with the next scan. 

## Expert

### Getting Started

Start with a fresh download from the last main version on GitHub. If you rename the notebook, do not forget to change the corresponding parameter inside its first cell. Note that the notebook does not work with JupyterLab in its current version.

The aim of the Expert part is to determine the general and fitting parameters. It can be quite painful, but those parameters should not vary during an experiment. 

### General parameters for extraction

1. Click on  ```Extract```.

2. Fill in  ```First/Last channel```  and click on the right fluo elements.

3. Choose a range of channels covering from the lowest peak you want to fit to the end of the elastic peak (if included).

4. Extract about 10 spectrums from the file, and try to choose good ones (without any weird behaviour). 

### Fit parameters

**You should always start with a set of parameters already completed for the same energy as yours.** To do so, click on ```Load fit params``` after extracting the spectrums.   
Do not start from zero! Taking a previous set of parameters should directly give you reasonnable results.


### Determine the fit parameters
See inside ```lib/fit/```  and the section ```Quick description of the parameters``` of this file to have an explanation on the peak fitting. Here we detail a procedure which seems to be robust for determining the fit parameters, **when starting from a previous configuration.**

1. So far these parameters could be kept at zero and did not have to be fitted:

```
tfb0 = 0. (do not cancel twc0 at the same time)
fB = 0.
gammaB = 0.
```
 
2. Linear background:  
Put ```sl=0.```. Start by finding manually a reasonnable value for ```ct```, which should be close to the value of the background in a region without peaks.

3. Step function at low energy (<8 keV):  
Adjust ```sfa0``` manually (or you can try to fit it). It will change the step function at low energy.

4. Compton peaks:  
- Fit only ```gammaA```, which sets the slope of the foot of the Compton peaks at low energy. When the fit is done, get the average value using the button ```Extract averages```.
- Fit only ```fA```, which sets the transition between a linear slope to a Gaussian in the Compton foot, at low energy.

5. Gaussians width:  
Fit only ```noise``` and ```fG```.

6. Other parameters:  
If the fit is still not good enough, you can try to fit simulatenously ```sl``` and ```ct```.

### Peak definition

Determine the peaks to be used (see guidelines in the User section if needed).

### Finishing
1. Try to keep only ```ct``` as a fitting parameters for the User (or no fit parameters at all). If the Compton peak varies a lot with time, add ```gammaA```.

2. **Make the list of peaks the default one by clicking on ```Save peaks params default``` in the control panel.**

3. **Make the list of parameters the default one by clicking on ```Save fit params as default``` in the control panel.**

4. Delete the cells you have generated (keep only the first one), and the file is ready for the User.


## Description of the parameters

### References
- The spectrum function and peak definition are adapted from results in the following papers (please cite these references accordingly):
    -  M. Van Gysel, P. Lemberge & P. Van Espen,
    “Description of Compton peaks in energy-dispersive  x-ray fluorescence spectra”,
    X-Ray Spectrometry 32 (2003), 139–147
    -   M. Van Gysel, P. Lemberge & P. Van Espen, “Implementation of a spectrum fitting
    procedure using a robust peak model”, X-Ray Spectrometry 32 (2003), 434–441
    - J. Campbell & J.-X. Wang, “Improved model for the intensity of low-energy tailing in
    Si (Li) x-ray spectra”, X-Ray Spectrometry 20 (1991), 191–197  
    

- See this reference as well for results using these peak/spectrum definitions:
    - F. Malloggi, S. Ben Jabrallah, L. Girard, B. Siboulet, K. Wang, P. Fontaine, and J. Daillant, "X-ray Standing Waves and Molecular Dynamics Studies of Ion Surface Interactions in Water at a Charged Silica Interface", J. Phys. Chem. C, 123 (2019), 30294–30304  
    

- The  X-ray fluorescence production cross section are extracted from the database xraylib (https://github.com/tschoonj/xraylib/wiki):
    - T. Schoonjans et al. "The xraylib library for X-ray-matter interactions. Recent developments" Spectrochimica Acta Part B: Atomic Spectroscopy 66 (2011), pp. 776-784
(https://github.com/xraypy/XrayDB).

### Quick description of the parameters
Here we give a quick description of each parameters and typical values. For more details, see corresponding publications and the code itself.  

Note that for synchrotron-based experiments some parameters can be significantly different than parameters used with lab X-ray sources (most of them can actually be cancelled). 

- ```First/last channel```: the subset of channels to be extracted. Between 0 and 2047.
- ```First/last spectrum```: the subset of spectrums to be extracted. The first spectrum in the file has the index 0.


- ```Elements XXX```: check the box corresponding the detector elements. Typical value: ```Element 4``` for the single-element detector, ```Element 0,1,2,3``` for the four-elements detector.

- ```Energy```: the beamline energy in eV, for calculating the fluorescence cross sections. 

- ```sl, ct```: linear baseline ```ct+sl*eV```.
- ```noise```: electronic noise (FWHM in keV). Typical value: ```noise=0.1```.
- ```fG```: broadening factor of the gaussian width for the Compton peak. Typical value:  ```fG=1.-1.5```.


- ```sfa0```:  shelf fraction. Define the step function at low energy (~<8 keV). Typical value: ```sfa0=0.1-1.```.

- ```tfb0```: tail fraction. Can be cancelled. Typical value: ```tb0=0.```.  

- ```twc0```:  tail widths. Can be cancelled. Typical values: ```twc0=0.```.

- ```epsilon```: energy to create a charge carrier pair in the detector (in keV). Typical value for Si: ```epsilon=0.0036```.

- ```fano```: fano factor. Typical value for Si: ```fano=0.115```. 


- ```fA, fB```: Tail fractions fA (low energy) and fB (high energy) for the Compton peak.  Set the transition between a linear slope to a Gaussian in the Compton feet. Typical values: ```fA=0.05-0.3, fB=0.```.

- ```gammaA, gammaB```: Tail gammas gammaA (low energy) and gammaB (high energy). Set the slope of the feet of Compton peaks. Typical values: ```gammaA=1.-5., gammaB=0.```.  


- ```gain, ev0```: linear conversion channels/eVs through ```eVs = gain*channels + eV0```. Typical values: ```gain=9.9, ev0=0```.


- ```Limit iter.```: number of iterations at which a fit is considered as stuck and returns NaN values. Typical value: ```1000``` (but can be increased if needed).


- ```Use ipysheet```: use ipysheet or only Peak.csv to define the peaks. Typical value: ```True```.
- ```Fast extract```: use the fast extract option of PyNexus. Typical value: ```True```.
- ```Transmit fit params```: when ```True```, the results of a fit are the initial guess of the next fit. Can trigger a bad behaviour when there are sudden jumps in the spectrum.

- ```Set peaks on sum```: use the integration of all the sprectrums to define the peaks.

- ```Show zooms?```: show the zoom of each peak (takes time).  

- ```Show peaks?```: show the peaks or not in the plots.  

- ```Strength min```: minimal strength to appear in the peak database (in cm^2/g). 0.05 seems to be a reasonnable value. Put to zero to see all the peaks.

- The peak strengths are the X-ray fluorescence production cross section with full cascade (in cm^2/g). They are used to fix the relative intensity of each peak from the same atom.   

## Contributions
Contributions are more than welcome. Please report any bug or submit any new ideas to arnaud.hemmerle@synchrotron-soleil.fr or directly in the Issues section of the GitHub.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
