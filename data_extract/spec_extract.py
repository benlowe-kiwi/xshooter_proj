import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
import bokeh.plotting as bok
import pandas as pd
from astropy.table import Table
from multiprocessing import Pool
from functools import partial


def mkdir(dir_path):
    """Makes a directory at a given path. If it already exists, then it will skip.

    Parameters
    ----------
    dir_name : str
        Pathway to directory.
    """
    
    os.makedirs(dir_path, exist_ok=True)
    
    return None


def html_plotter(x_data, y_data, x_label, y_label, title, filename, path):
    """Creates and saves interactive .html plots using the bokeh package.

    Parameters
    ----------
    x_data : ndarray
        Data on the x-axis
    y_data : ndarray
        Data on the y-axis
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    title : str
        Title to be put at the top of the plot
    filename : str
        Name of the saved .html file.
    path : str
        Location for where to save the plot

    Returns
    -------
    None
    """
    # Defining the bokeh figure
    p = bok.figure(title=title, x_axis_label=x_label, y_axis_label=y_label, width = 1000)
    p.sizing_mode = 'scale_height'
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "12pt"
    p.yaxis.major_label_text_font_size = "12pt"
    p.title.text_font_size = "16pt"

    # Plotting the spectra
    p.line(x_data, y_data, line_width=0.9, color='black')

    # Saving the spectra to the given directory
    bok.save(obj=p, filename=path + filename, title=title)
    
    return None



def spec_extract(fits_files, sci_path, anc_path):
    """Opens up all the spectra, then returns the spectral wavelengths and fluxes.

    Parameters
    ----------
    fits_files : list
        List of .fits files containing X-Shooter spectra.
    sci_path : str
        Path to the SCIENCE.SPECTRUM directory
    anc_path : str
        Path to the ANCILLARY.2DSPECTRUM directory 
        
    Returns
    -------
    None
    
    """
    # Opening up each .fits file in fits_files
    for file in fits_files:
        star_fits = fits.open(path + file)
        star_name = star_fits[0].header['OBJECT']
        
        # fits_table = Table.read(path + file)
        # print(fits_table.to_pandas())
        
        print("Extracting file {0} for star {1}".format(file, star_name))
        
        # Determining if the file is ancillary or science, then storing the data in that
        star_prodcatg = star_fits[0].header['PRODCATG']
        
        # SCIENCE.SPECTRUM data extraction
        if star_prodcatg == 'SCIENCE.SPECTRUM':
            
            # Extracting spectral data
            wave = star_fits[1].data['WAVE'][0].byteswap().newbyteorder()
            flux = star_fits[1].data['FLUX_REDUCED'][0].byteswap().newbyteorder()
            flux_err = star_fits[1].data['ERR_REDUCED'][0].byteswap().newbyteorder()
            snr = star_fits[0].header['SNR']

            # Making directory based on the star name
            obj_path = sci_path + star_name + '/'
            mkdir(obj_path)
            
            # Within the directory, store the UVB, VIS and NIR data. Will also make .html plots.
            star_dispelem = star_fits[0].header['DISPELEM']
            
            # UVB data
            if star_dispelem == 'UVB':
                uvb_path = obj_path + 'uvb/'
                mkdir(uvb_path)
                
                wave_index = np.where((wave >= 300) & (wave <= 556))[0]
                wave = wave[wave_index]
                flux = flux[wave_index]
                flux_err = flux_err[wave_index]
                
                html_plotter(x_data=wave, 
                             y_data=flux, 
                             x_label='Wavelength (nm)', 
                             y_label='Flux', 
                             title='{0} ({1}, SNR = {2:.2f})'.format(star_name, star_dispelem, snr), 
                             filename=file[:-5] + '_plot.html', 
                             path=uvb_path)
                star_fits.writeto(uvb_path + file, overwrite=True)
                uvb_df = pd.DataFrame({'wave': wave * 10, 'flux': flux, 'flux_err': flux_err})
                uvb_df.to_csv(uvb_path + file + '.csv', index=False)
                
            # VIS data
            if star_dispelem == 'VIS':
                vis_path = obj_path + 'vis/'
                mkdir(vis_path)
                
                wave_index = np.where(wave >= 556)[0]
                wave = wave[wave_index]
                flux = flux[wave_index]
                flux_err = flux_err[wave_index]
                
                html_plotter(x_data=wave, 
                             y_data=flux, 
                             x_label='Wavelength (nm)', 
                             y_label='Flux', 
                             title='{0} ({1}, SNR = {2:.2f})'.format(star_name, star_dispelem, snr), 
                             filename=file[:-5] + '_plot.html', 
                             path=vis_path)
                star_fits.writeto(vis_path + file, overwrite=True)
                vis_df = pd.DataFrame({'wave': wave * 10, 'flux': flux, 'flux_err': flux_err})
                vis_df.to_csv(vis_path + file + '.csv', index=False)
            
            # NIR data
            if star_dispelem == 'NIR':
                nir_path = obj_path + 'nir/'
                mkdir(nir_path)
                html_plotter(x_data=wave, 
                             y_data=flux, 
                             x_label='Wavelength (nm)', 
                             y_label='Flux', 
                             title='{0} ({1}, SNR = {2:.2f})'.format(star_name, star_dispelem, snr), 
                             filename=file[:-5] + '_plot.html', 
                             path=nir_path)
                star_fits.writeto(nir_path + file, overwrite=True)
                nir_df = pd.DataFrame({'wave': wave * 10, 'flux': flux, 'flux_err': flux_err})
                nir_df.to_csv(nir_path + file + '.csv', index=False)
                
                
        # # ANCILLARY.2DSPECTRUM data extraction  
        # if star_prodcatg == 'ANCILLARY.2DSPECTRUM':
            
        #     # Making directory based on the star name, then saving
        #     obj_path = anc_path + star_name + '/'
        #     mkdir(obj_path)
        #     star_fits.writeto(obj_path + file, overwrite=True)
            
    return None
            
            
        
if __name__ == '__main__':
    
    # Loading in the data
    path = '/data/misfit/benlowe/xshooter/reduced_data/p115_data/original/'
    xshoot_fits = [file for file in os.listdir(path) if file[-5:] == '.fits']

    # Paths to the SCIENCE_SPECTRUM and ANCILLARY_2DSPECTRUM directories
    sci_spec_path = '/data/misfit/benlowe/xshooter/reduced_data/p115_data/science_spec/'
    # tomasz_spec_path = '/data/misfit/benlowe/xshooter/reduced_data/tomasz/'
    anc_2dspec_path = '/data/misfit/benlowe/xshooter/reduced_data/p115_data/ancillary_2dspec/'
    
    # Running the data extraction code
    spec_extract(fits_files=xshoot_fits, sci_path=sci_spec_path, anc_path=anc_2dspec_path)    