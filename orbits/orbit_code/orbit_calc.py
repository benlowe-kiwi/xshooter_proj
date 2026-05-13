import os
import pickle
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
import numpy as np
import pandas as pd
from numpy.random import normal,rand
from astropy.coordinates import Galactic
from galpy.orbit import Orbit
from galpy import potential 
from galpy.potential.mwpotentials import McMillan17
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from galpy.actionAngle import actionAngleStaeckel, actionAngleSpherical
from scipy.stats import iqr
from astropy.table import Table

def standard_error(data):
    """Calculate the standard error."""
    
    # Removing nans from the data
    data = data[~np.isnan(data)]
    
    # Taking the length of the dataset (with nans removed)
    data_len = len(data)
    
    # Calculating the standard error
    if data_len > 0:
        return (1.57 * iqr(data))/ np.sqrt(data_len)
    else:
        return np.nan


def calc_orbit_mwpotential(data_df, savepath, ts, nrand=100, use_plx=True, save_type='fits', plot=True):
    """Calculate orbits for the given set of stars using the MWPotential2014 potential.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the star data with required columns.
        Must contain the following columns:
        'star', 'ra', 'dec', 'plx', 'plx_err',
        'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
        'rv', 'rv_err'.
        Or contain the following if using distances:
        'star', 'ra', 'dec', 'dist', 'dist_err',
        'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
        'rv', 'rv_err'.
    savepath : str
        Path to save the orbit results.
    ts : array_like
        Time steps for the orbit calculation in Gyr. Recommend to be in the following format:
        e.g. np.arange(-4, 4, 0.01) * u.Gyr i.e. ±4 Gyr with 10 million time steps.
    nrand : int, optional
        Number of iterations, by default 100
    use_plx : bool, optional
        Whether parallax is included in the orbit calculation, by default True. Use False if
        using distances instead (measured in kpc)
    save_type : str, optional
        Type of file to save the results, either 'csv' or 'fits', by default 'fits'.
    plot : bool, optional
        Whether to plot the orbits, by default True.

    Returns
    -------
    res_df : DataFrame
        DataFrame containing the calculated orbits for each star.
        Contains the following columns:
        'star', 'ra', 'dec', 'plx', 'plx_err',
        'pmRA', 'pmRA_err', 'pmdec', 'pmdec_err',
        'Vlos', 'eVlos', 'rmax', 'rapo', 'e',
        'zmax', 'E', 'jp', 'jr', 'jz',
        'xpos', 'ypos', 'zpos', 'vx',
        'vy', 'vz', 'vphi', 'vR',
        'vT', 'vbb', 'vll'.

    Raises
    ------
    ValueError
        If any of the required columns are missing from the input DataFrame.
    """
    
    j = 1
    process_id = os.getpid()
    
    # ---CHECKING INPUTS--- #
    # Checking if the required columns are present in the dataframe
    if use_plx:
        required_columns = [
            'star', 'ra', 'dec', 'plx', 'plx_err',
            'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
            'rv', 'rv_err'
        ]
        for col in required_columns:
            if col not in data_df.columns:
                raise ValueError(f"Missing required column: {col}")
            else:
                pass
    else:
        required_columns = [
            'star', 'ra', 'dec', 'dist', 'dist_err',
            'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
            'rv', 'rv_err'
        ]
        for col in required_columns:
            if col not in data_df.columns:
                raise ValueError(f"Missing required column: {col}")
            else:
                pass
            
    # Checking if the save_type is valid
    if save_type not in ['csv', 'fits']:
        raise ValueError("save_type must be either 'csv' or 'fits'")
    
    # Ensuring that ts is in the correct format
    if not isinstance(ts, (np.ndarray, list)):
        raise ValueError("ts must be a numpy array or a list of time steps in Gyr. Try e.g. np.arange(-4, 4, 0.01) * u.Gyr, where 4 is the number of Gyr to calculate the orbit for, and 0.01 is the time step in Gyr.")

    # ---LOADING IN THE DATA--- #
    # Unpacking the data from the dataframe
    name = data_df['star'].values
    ra = data_df['ra'].values
    dec = data_df['dec'].values
    
    if use_plx:
        plx = data_df['plx'].values 
        plx_err = data_df['plx_err'].values
    else:
        dist = data_df['dist'].values 
        dist_err = data_df['dist_err'].values
    pmRA = data_df['pmra'].values
    pmRA_err = data_df['pmra_err'].values 
    pmdec = data_df['pmdec'].values
    pmdec_err = data_df['pmdec_err'].values
    Vlos = data_df['rv'].values 
    eVlos = data_df['rv_err'].values
        
        
    # ---SETTING UP ORBIT CALCULATION--- #
    # Setting up the potential
    lp = potential.MWPotential2014
    solarmotion = 'schoenrich'
    integr = 'symplec4_c'  # Integrator to use for orbit integration
    
    # Setting up the solar motion
    ro = 8.2 # Radius of Sun from Galactic Centre in kpc
    vo = 220. # Circular velocity of the Sun in km/s
    sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0], radec=True, ro=ro, vo=vo, solarmotion=solarmotion)
    
    # # ts = np.linspace(-2, 2, 1000) * u.Gyr # Time steps for the orbit calculation in Gyr
    # sun.integrate(ts, lp, method=integr)
    
    # Number of stars
    n = min(len(ra), len(Vlos))
    
    
    # ---CALCULATING ORBITS FOR EACH STAR--- #
    res = np.zeros((19, nrand, n))
    
    # Looping over the stars to calculate orbits
    orb_res = []
    for i in range(n):
        ith_star = str(name[i]) 
        ith_ra = ra[i] 
        ith_dec = dec[i]
        
        print('Running {0}\n- Process {1} \n- Star {2}/{3}\n'.format(ith_star, process_id, j, len(data_df)))
        j += 1
        
        if use_plx: 
            ith_plx = plx[i] 
            ith_plx_err = plx_err[i]
        else:
            ith_dist = dist[i]
            ith_dist_err = dist_err[i]
        ith_pmRA = pmRA[i] 
        ith_pmRA_err = pmRA_err[i]
        ith_pmdec = pmdec[i] 
        ith_pmdec_err = pmdec_err[i]
        ith_Vlos = Vlos[i]
        ith_eVlos = eVlos[i]
        
        # print(f'Running star {ith_star} ({i+1}/{n})')
        
        # Making save directory for the star
        star_savepath = os.path.join(savepath, ith_star)
        os.makedirs(star_savepath, exist_ok=True)
        
        # Running initial orbit calculation
        if use_plx:
            otmp = Orbit(vxvv=[ith_ra, ith_dec, 1/ith_plx, ith_pmRA, ith_pmdec, ith_Vlos], 
                            radec=True, 
                            ro=ro, 
                            vo=vo, 
                            solarmotion=solarmotion)
        else:
            otmp = Orbit(vxvv=[ith_ra, ith_dec, ith_dist, ith_pmRA, ith_pmdec, ith_Vlos], 
                            radec=True, 
                            ro=ro, 
                            vo=vo, 
                            solarmotion=solarmotion)
        otmp.integrate(ts, lp, method=integr)
        orbits = [otmp]  
        
        # Saving the results for the first orbit
        res[:, 0, i] = (otmp.rperi(), otmp.rap(), otmp.e(), otmp.zmax(), otmp.E(), otmp.jp(),otmp.jr(),otmp.jz(), 
                                otmp.x(),otmp.y(),otmp.z(), otmp.vx(), otmp.vy(), otmp.vz(), 
                                otmp.vphi(),otmp.vR(),otmp.vT(), otmp.vbb(),otmp.vll())
        
        # Calculating nrand orbits by randomly varying the parameters
        for r in range(1, nrand):
            if use_plx:
                rand_plx = max(normal(ith_plx, ith_plx_err), 1e-6)
            else:
                rand_dist = normal(ith_dist, ith_dist_err)
            rand_pmRA = normal(ith_pmRA, ith_pmRA_err)
            rand_pmdec = normal(ith_pmdec, ith_pmdec_err)
            rand_Vlos = normal(ith_Vlos, max(ith_eVlos, 1))
            
            print(f'Orbit {r+1}/{nrand}')
            
            if use_plx:
                otmp = Orbit(vxvv=[ith_ra, dec[i], 1/rand_plx, rand_pmRA, rand_pmdec,rand_Vlos], 
                                radec=True, ro=ro, vo=vo, solarmotion=solarmotion)
            else:
                otmp = Orbit(vxvv=[ith_ra, dec[i], rand_dist, rand_pmRA, rand_pmdec,rand_Vlos], 
                                radec=True, ro=ro, vo=vo, solarmotion=solarmotion)
            otmp.integrate(ts, lp, method=integr)
            
            # Saving the results for each random orbit
            res[:, r, i] = (otmp.rperi(), otmp.rap(), otmp.e(), otmp.zmax(), otmp.E(), otmp.jp(),otmp.jr(),otmp.jz(), 
                                    otmp.x(),otmp.y(),otmp.z(), otmp.vx(), otmp.vy(), otmp.vz(), 
                                    otmp.vphi(),otmp.vR(),otmp.vT(), otmp.vbb(),otmp.vll())
    
    
        # ---SAVING THE RESULTS--- #
        # Save file for the orbits
        filename = f'{ith_star}_orbits.pkl'
        filepath = os.path.join(star_savepath, filename)
        
        # # Storing the orbits into a pickle file
        # with open(filepath, 'wb') as fh:
        #     pickle.dump(res, fh, pickle.HIGHEST_PROTOCOL)
        #     pickle.dump(orbits, fh, pickle.HIGHEST_PROTOCOL)
            
        # Unpacking the orbit results
        rmax, rapo, eccentricity, zmax, E, jp, jr, jz, xpos, ypos, zpos, vx, vy, vz, vphi, vR, vT, vbb, vll = res
        xhelio = 8 - xpos
        yhelio = ypos
        zhelio = zpos
        ravg = 0.5 * (rmax + rapo)
        
        # Taking the median of the results
        med_rmax = np.nanmedian(rmax, axis=0)[i]
        med_rapo = np.nanmedian(rapo, axis=0)[i]
        med_eccentricity = np.nanmedian(eccentricity, axis=0)[i]
        med_zmax = np.nanmedian(zmax, axis=0)[i]
        med_E = np.nanmedian(E, axis=0)[i]
        med_jp = np.nanmedian(jp, axis=0)[i]
        med_jr = np.nanmedian(jr, axis=0)[i]
        med_jz = np.nanmedian(jz, axis=0)[i]
        med_xpos = np.nanmedian(xpos, axis=0)[i]
        med_ypos = np.nanmedian(ypos, axis=0)[i]
        med_zpos = np.nanmedian(zpos, axis=0)[i]
        med_vx = np.nanmedian(vx, axis=0)[i]
        med_vy = np.nanmedian(vy, axis=0)[i]
        med_vz = np.nanmedian(vz, axis=0)[i]
        med_vphi = np.nanmedian(vphi, axis=0)[i]
        med_vR = np.nanmedian(vR, axis=0)[i]
        med_vT = np.nanmedian(vT, axis=0)[i]
        med_vbb = np.nanmedian(vbb, axis=0)[i]
        med_vll = np.nanmedian(vll, axis=0)[i]
        med_xhelio = np.nanmedian(xhelio, axis=0)[i]
        med_yhelio = np.nanmedian(yhelio, axis=0)[i]
        med_zhelio = np.nanmedian(zhelio, axis=0)[i]
        
        # Calculating the standard errors for the results
        err_rmax = standard_error(data=rmax)
        err_rapo = standard_error(data=rapo)
        err_eccentricity = standard_error(data=eccentricity)
        err_zmax = standard_error(data=zmax)
        err_E = standard_error(data=E)
        err_jp = standard_error(data=jp)
        err_jr = standard_error(data=jr)
        err_jz = standard_error(data=jz)
        err_xpos = standard_error(data=xpos)
        err_ypos = standard_error(data=ypos)
        err_zpos = standard_error(data=zpos)
        err_vx = standard_error(data=vx)
        err_vy = standard_error(data=vy)
        err_vz = standard_error(data=vz)
        err_vphi = standard_error(data=vphi)
        err_vR = standard_error(data=vR)
        err_vT = standard_error(data=vT)
        err_vbb = standard_error(data=vbb)
        err_vll = standard_error(data=vll)
        err_xhelio = standard_error(data=xhelio)
        err_yhelio = standard_error(data=yhelio)
        err_zhelio = standard_error(data=zhelio)
        
        if use_plx:
            res_dict = {
                'star': ith_star,
                'ra': ith_ra,
                'dec': ith_dec,
                'plx': ith_plx,
                'plx_err': ith_plx_err,
                'pmRA': ith_pmRA,
                'pmRA_err': ith_pmRA_err,
                'pmdec': ith_pmdec,
                'pmdec_err': ith_pmdec_err,
                'Vlos': ith_Vlos,
                'eVlos': ith_eVlos,
                'rmax': med_rmax,
                'rmax_err': err_rmax,
                'rapo': med_rapo,
                'rapo_err': err_rapo,
                'e': med_eccentricity,
                'e_err': err_eccentricity,
                'zmax': med_zmax,
                'zmax_err': err_zmax,
                'E': med_E,
                'E_err': err_E,
                'jp': med_jp,
                'jp_err': err_jp,
                'jr': med_jr,
                'jr_err': err_jr,
                'jz': med_jz,
                'jz_err': err_jz,
                'xpos': med_xpos,
                'xpos_err': err_xpos,
                'ypos': med_ypos,
                'ypos_err': err_ypos,
                'zpos': med_zpos,
                'zpos_err': err_zpos,
                'vx': med_vx,
                'vx_err': err_vx,
                'vy': med_vy,
                'vy_err': err_vy,
                'vz': med_vz,
                'vz_err': err_vz,
                'vphi': med_vphi,
                'vphi_err': err_vphi,
                'vR': med_vR,
                'vR_err': err_vR,
                'vT': med_vT,
                'vT_err': err_vT,
                'vbb': med_vbb,
                'vbb_err': err_vbb,
                'vll': med_vll,
                'vll_err': err_vll,
                'xhelio': med_xhelio,
                'xhelio_err': err_xhelio,
                'yhelio': med_yhelio,
                'yhelio_err': err_yhelio,
                'zhelio': med_zhelio,
                'zhelio_err': err_zhelio
            }
            orb_res.append(res_dict)
        else:
            res_dict = {
                'star': ith_star,
                'ra': ith_ra,
                'dec': ith_dec,
                'dist': ith_dist,
                'dist_err': ith_dist_err,
                'pmRA': ith_pmRA,
                'pmRA_err': ith_pmRA_err,
                'pmdec': ith_pmdec,
                'pmdec_err': ith_pmdec_err,
                'Vlos': ith_Vlos,
                'eVlos': ith_eVlos,
                'rmax': med_rmax,
                'rmax_err': err_rmax,
                'rapo': med_rapo,
                'rapo_err': err_rapo,
                'e': med_eccentricity,
                'e_err': err_eccentricity,
                'zmax': med_zmax,
                'zmax_err': err_zmax,
                'E': med_E,
                'E_err': err_E,
                'jp': med_jp,
                'jp_err': err_jp,
                'jr': med_jr,
                'jr_err': err_jr,
                'jz': med_jz,
                'jz_err': err_jz,
                'xpos': med_xpos,
                'xpos_err': err_xpos,
                'ypos': med_ypos,
                'ypos_err': err_ypos,
                'zpos': med_zpos,
                'zpos_err': err_zpos,
                'vx': med_vx,
                'vx_err': err_vx,
                'vy': med_vy,
                'vy_err': err_vy,
                'vz': med_vz,
                'vz_err': err_vz,
                'vphi': med_vphi,
                'vphi_err': err_vphi,
                'vR': med_vR,
                'vR_err': err_vR,
                'vT': med_vT,
                'vT_err': err_vT,
                'vbb': med_vbb,
                'vbb_err': err_vbb,
                'vll': med_vll,
                'vll_err': err_vll,
                'xhelio': med_xhelio,
                'xhelio_err': err_xhelio,
                'yhelio': med_yhelio,
                'yhelio_err': err_yhelio,
                'zhelio': med_zhelio,
                'zhelio_err': err_zhelio
            }
            orb_res.append(res_dict)

        # Saving the median results to a CSV file or FITS file
        star_res_df = pd.DataFrame([res_dict])
        if save_type == 'csv':
            star_res_df.to_csv(os.path.join(star_savepath, f'{ith_star}_orbit_medparams.csv'), index=False)
        if save_type == 'fits':
            star_res_table = Table.from_pandas(star_res_df)
            star_res_table.write(os.path.join(star_savepath, f'{ith_star}_orbit_medparams.fits'), overwrite=True)

        # Saving the full nrand results to a CSV file or FITS file 
        # rmax, rapo, eccentricity, zmax, E, jp, jr, jz, xpos, ypos, zpos, vx, vy, vz, vphi, vR, vT, vbb, vll = res      
        full_res_df = pd.DataFrame({
            'rmax': [i[0] for i in rmax],
            'rapo': [i[0] for i in rapo],
            'e': [i[0] for i in eccentricity],
            'zmax': [i[0] for i in zmax],
            'E': [i[0] for i in E],
            'jp': [i[0] for i in jp],
            'jr': [i[0] for i in jr],
            'jz': [i[0] for i in jz],
            'xpos': [i[0] for i in xpos],
            'ypos': [i[0] for i in ypos],
            'zpos': [i[0] for i in zpos],
            'vx': [i[0] for i in vx],
            'vy': [i[0] for i in vy],
            'vz': [i[0] for i in vz],
            'vphi': [i[0] for i in vphi],
            'vR': [i[0] for i in vR],
            'vT': [i[0] for i in vT],
            'vbb': [i[0] for i in vbb],
            'vll': [i[0] for i in vll]
        })
        
        if save_type == 'csv':
            full_res_df.to_csv(os.path.join(star_savepath, f'{ith_star}_orbit_fullparams.csv'), index=False)   
        if save_type == 'fits':
            full_res_table = Table.from_pandas(full_res_df)
            full_res_table.write(os.path.join(star_savepath, f'{ith_star}_orbit_fullparams.fits'), overwrite=True)     
            
        
        # ---PLOTTING THE ORBITS--- #
        # 2D plot of the orbit
        if plot:
            
            try:
                plot_filename = f'{ith_star}_orbit_2d.png'
                plt.figure(i, figsize=(8,6))
                otmp.plot(d1='x', d2='y', color='firebrick', lw=0.8)
                plt.plot([otmp.x(ts[0])], [otmp.y(ts[0])], marker='x', color='black')  
                plt.xlabel('X (kpc)')
                plt.ylabel('Y (kpc)')
                plt.title(f'{ith_star}')
                plt.savefig(os.path.join(star_savepath, plot_filename), dpi=300, transparent=False, facecolor="white")
                
                # 3D plot of the orbit
                plot_filename = f'{ith_star}_orbit_3d.html'
                
                x = otmp.x(ts)
                y = otmp.y(ts)
                z = otmp.z(ts)
                
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        line=dict(color='firebrick', width=3.5)
                    )
                ])
                        
                fig.add_trace(go.Scatter3d(
                    x=[otmp.x(ts[0])], y=[otmp.y(ts[0])], z=[otmp.z(ts[0])],
                    mode='markers',
                    marker=dict(
                        symbol='cross',
                        size=10,
                        color='black'
                    )
                ))        

                fig.update_layout(
                    scene=dict(
                        xaxis_title='X (kpc)',
                        yaxis_title='Y (kpc)',
                        zaxis_title='Z (kpc)'
                    ),
                    title=ith_star,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                
                fig.write_html((os.path.join(star_savepath, plot_filename)))
            except ValueError:
                print(f'Could not plot the orbit for {ith_star} due to a ValueError. Skipping the plot.')
                continue


            # # 3D animation of the orbit
            # plot_filename = f'{ith_star}_orbit_animation.mp4'
            
            # x = otmp.x(ts)
            # y = otmp.y(ts)
            # z = otmp.z(ts)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # ax.set_xlim(min(x), max(x))
            # ax.set_ylim(min(y), max(y))
            # ax.set_zlim(min(z), max(z))

            # line, = ax.plot([], [], [], 'b-', lw=2)
            # point, = ax.plot([], [], [], 'ro')

            # def update(num):
            #     line.set_data(x[:num], y[:num])
            #     line.set_3d_properties(z[:num])
            #     point.set_data(x[num-1:num], y[num-1:num])
            #     point.set_3d_properties(z[num-1:num])
            #     return line, point

            # ani = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)

            # ani.save(os.path.join(savepath, plot_filename), writer='ffmpeg', fps=30)

            # plt.close(fig)
        else:
            pass
        print()
        
    # Creating dataframe from the median results
    res_df = pd.DataFrame(orb_res)
    # res_df.to_csv(os.path.join(star_savepath, 'orbits_results.csv'), index=False)
    
    return res_df




def calc_orbit_mcmillian(data_df, savepath, ts, nrand=100, use_plx=True, save_type='fits', plot=True):
    """ Calculate orbits for the given set of stars using the McMillan17 potential.

    Parameters
    ----------
    data_df : DataFrame
        DataFrame containing the star data with required columns.
        Must contain the following columns:
        'star', 'ra', 'dec', 'plx', 'plx_err',
        'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
        'rv', 'rv_err'.
        Or contain the following if using distances:
        'star', 'ra', 'dec', 'dist', 'dist_err',
        'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
        'rv', 'rv_err'.
    savepath : str
        Path to save the orbit results.
    ts : array_like
        Time steps for the orbit calculation in Gyr. Recommend to be in the following format:
        e.g. np.arange(-4, 4, 0.01) * u.Gyr i.e. ±4 Gyr with 10 million time steps.
    nrand : int, optional
        Number of iterations, by default 100
    use_plx : bool, optional
        Whether parallax is included in the orbit calculation, by default True. Use False if
        using distances instead (measured in kpc)
    save_type : str, optional
        Type of file to save the results, either 'csv' or 'fits', by default 'fits'.
    plot : bool, optional
        Whether to plot the orbits, by default True.

    Returns
    -------
    res_df : DataFrame
        DataFrame containing the calculated orbits for each star.
        Contains the following columns:
        'star', 'ra', 'dec', 'plx', 'plx_err',
        'pmRA', 'pmRA_err', 'pmdec', 'pmdec_err',
        'Vlos', 'eVlos', 'rmax', 'rapo', 'e',
        'zmax', 'E', 'jp', 'jr', 'jz',
        'xpos', 'ypos', 'zpos', 'vx',
        'vy', 'vz', 'vphi', 'vR',
        'vT', 'vbb', 'vll'.

    Raises
    ------
    ValueError
        If any of the required columns are missing from the input DataFrame.
    """

    j = 1
    process_id = os.getpid()
    
    # ---CHECKING INPUTS--- #
    # Checking if the required columns are present in the dataframe
    if use_plx:
        required_columns = [
            'star', 'ra', 'dec', 'plx', 'plx_err',
            'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
            'rv', 'rv_err'
        ]
        for col in required_columns:
            if col not in data_df.columns:
                raise ValueError(f"Missing required column: {col}")
            else:
                pass
    else:
        required_columns = [
            'star', 'ra', 'dec', 'dist', 'dist_err',
            'pmra', 'pmra_err', 'pmdec', 'pmdec_err',
            'rv', 'rv_err'
        ]
        for col in required_columns:
            if col not in data_df.columns:
                raise ValueError(f"Missing required column: {col}")
            else:
                pass
            
    # Checking if the save_type is valid
    if save_type not in ['csv', 'fits']:
        raise ValueError("save_type must be either 'csv' or 'fits'")
    
    # Ensuring that ts is in the correct format
    if not isinstance(ts, (np.ndarray, list)):
        raise ValueError("ts must be a numpy array or a list of time steps in Gyr. Try e.g. np.arange(-4, 4, 0.01) * u.Gyr, where 4 is the number of Gyr to calculate the orbit for, and 0.01 is the time step in Gyr.")
    
    
    # ---LOADING IN THE DATA--- #
    # Unpacking the data from the dataframe
    name = data_df['star'].values
    ra = data_df['ra'].values
    dec = data_df['dec'].values
    if use_plx:
        plx = data_df['plx'].values 
        plx_err = data_df['plx_err'].values 
    else:
        dist = data_df['dist'].values 
        dist_err = data_df['dist_err'].values
    pmRA = data_df['pmra'].values
    pmRA_err = data_df['pmra_err'].values 
    pmdec = data_df['pmdec'].values
    pmdec_err = data_df['pmdec_err'].values
    Vlos = data_df['rv'].values 
    eVlos = data_df['rv_err'].values
        
        
    # ---SETTING UP ORBIT CALCULATION--- #
    # Setting up the potential
    lp = McMillan17
    solarmotion = [-11.1, 15.17, 7.25]*u.km/u.s
    integr = 'symplec4_c'  # Integrator to use for orbit integration
    
    # Setting up the solar motion
    r_galactic_centre = 8.21*u.kpc # McMillan Potential, 2017
    z_galactic_plane = 25.0*u.pc # Bland-Hawthorn & Gerhard, 2016, ARA&A, 54, 529

    v_total_sun = (np.tan(6.379*u.mas)*r_galactic_centre/u.yr).to(u.km/u.s) # pm_l by Reid & Brunthaler 2004, ApJ, 616, 872
    v_peculiar = np.array([11.1, 15.17, 7.25]) *u.km/u.s # Peculiar motion of the Sun in the McMillan potential
    v_circular = np.round(v_total_sun-v_peculiar[1],1) # Circular velocity of the Sun in km/s
    # sun = Orbit(vxvv=[0, 0, 0, 0, 0, 0], radec=True, ro=ro, vo=vo, solarmotion=solarmotion)
    # ts = np.linspace(-2, 2, 1000) * u.Gyr # Time steps for the orbit calculation in Gyr
    # sun.integrate(ts, lp, method=integr)
    
    # Setting up the Staeckel potential
    aAS = actionAngleStaeckel(
        pot   = lp,        #potential                                                                                                                                                                      
        delta = 0.45,       #focal length of confocal coordinate system                                                                                                                            
        c     = True        #use C code (for speed)                                                                                                                                                         
        )
    
    # Number of stars
    n = min(len(ra), len(Vlos))
    
    
    # ---CALCULATING ORBITS FOR EACH STAR--- #
    res = np.zeros((26, nrand, n))
    
    # Looping over the stars to calculate orbits
    orb_res = []
    for i in range(n):
        ith_star = str(name[i]) 
        ith_ra = ra[i] 
        ith_dec = dec[i]
        
        print('Running {0}\n- Process {1} \n- Star {2}/{3}\n'.format(ith_star, process_id, j, len(data_df)))
        j += 1
        
        if use_plx:
            ith_plx = plx[i] 
            ith_plx_err = plx_err[i]
        else:
            ith_dist = dist[i]
            ith_dist_err = dist_err[i]
        ith_pmRA = pmRA[i] 
        ith_pmRA_err = pmRA_err[i]
        ith_pmdec = pmdec[i] 
        ith_pmdec_err = pmdec_err[i]
        ith_Vlos = Vlos[i]
        ith_eVlos = eVlos[i]
        
        # print(f'Running star {ith_star} ({i+1}/{n})')
        
        # Making save directory for the star
        star_savepath = os.path.join(savepath, ith_star)
        os.makedirs(star_savepath, exist_ok=True)
        
        # Running initial orbit calculation
        print(f'Orbit 1/{nrand}')
        if use_plx:
            otmp = Orbit(vxvv=[ith_ra, ith_dec, 1/ith_plx, ith_pmRA, ith_pmdec, ith_Vlos], 
                         radec=True, 
                         ro=r_galactic_centre, 
                         vo=v_circular, 
                         zo=z_galactic_plane,
                         solarmotion=solarmotion)
        else:
            otmp = Orbit(vxvv=[ith_ra, ith_dec, ith_dist, ith_pmRA, ith_pmdec, ith_Vlos], 
                         radec=True, 
                         ro=r_galactic_centre, 
                         vo=v_circular, 
                         zo=z_galactic_plane,
                         solarmotion=solarmotion)
        otmp.integrate(ts, lp, method=integr)
        orbits = [otmp]  
        
        xhelio = otmp.helioX()        
        yhelio = otmp.helioY()
        zhelio = otmp.helioZ()
        U = otmp.U()
        V = otmp.V()
        W = otmp.W()
        R = otmp.R()
        phi = otmp.phi()
        z = otmp.z()
        vR = otmp.vR()
        vT = otmp.vT()
        vz = otmp.vz()
        
        # Calculating the actions, frequencies, and angles
        actionsFreqsAngles = aAS.actionsFreqsAngles(otmp,
                                                    ro=r_galactic_centre,
                                                    vo=v_circular)
        jr = actionsFreqsAngles[0][0]
        jp = actionsFreqsAngles[1][0]
        jz = actionsFreqsAngles[2][0]
        omega_R = actionsFreqsAngles[3][0]
        omega_phi = actionsFreqsAngles[4][0]
        omega_z = actionsFreqsAngles[5][0]
        angle_R = actionsFreqsAngles[6][0]
        alpha_phi = actionsFreqsAngles[7][0]
        angle_z = actionsFreqsAngles[8][0]
                
        EccZmaxRperiRap = aAS.EccZmaxRperiRap(otmp,
                                              ro=r_galactic_centre,
                                              vo=v_circular)
        e = EccZmaxRperiRap[0][0]
        zmax = EccZmaxRperiRap[1][0]
        R_peri = EccZmaxRperiRap[2][0]
        R_ap = EccZmaxRperiRap[3][0]
        
        E = otmp.E(pot=lp,
                        ro=r_galactic_centre,
                        vo=v_circular,
                        zo=z_galactic_plane)
                
        # Saving the results for the first orbit
        res[:, 0, i] = (xhelio, yhelio, zhelio, U, V, W, R, phi, z, vR, vT, vz, jr, jp, 
                        jz, omega_R, omega_phi, omega_z, angle_R, alpha_phi, angle_z,
                        e, zmax, R_peri, R_ap, E)
        
        
        # Calculating nrand orbits by randomly varying the parameters
        
        for r in range(1, nrand):
            if use_plx:
                rand_plx = max(normal(ith_plx, ith_plx_err), 1e-6)
            else:
                rand_dist = normal(ith_dist, ith_dist_err)
            rand_pmRA = normal(ith_pmRA, ith_pmRA_err)
            rand_pmdec = normal(ith_pmdec, ith_pmdec_err)
            rand_Vlos = normal(ith_Vlos, max(ith_eVlos, 1))
            
            print(f'Orbit {r+1}/{nrand}')   
                     
            if use_plx:
                otmp = Orbit(vxvv=[ith_ra, ith_dec, 1/rand_plx, rand_pmRA, rand_pmdec, rand_Vlos],
                            radec=True, 
                            ro=r_galactic_centre, 
                            vo=v_circular, 
                            zo=z_galactic_plane,
                            solarmotion=solarmotion)
            else:
                otmp = Orbit(vxvv=[ith_ra, ith_dec, rand_dist, rand_pmRA, rand_pmdec, rand_Vlos],
                            radec=True, 
                            ro=r_galactic_centre, 
                            vo=v_circular, 
                            zo=z_galactic_plane,
                            solarmotion=solarmotion)
            otmp.integrate(ts, lp, method=integr) 
            
            xhelio = otmp.helioX()        
            yhelio = otmp.helioY()
            zhelio = otmp.helioZ()
            U = otmp.U()
            V = otmp.V()
            W = otmp.W()
            R = otmp.R()
            phi = otmp.phi()
            z = otmp.z()
            vR = otmp.vR()
            vT = otmp.vT()
            vz = otmp.vz()
            
            # Calculating the actions, frequencies, and angles
            actionsFreqsAngles = aAS.actionsFreqsAngles(otmp,
                                                        ro=r_galactic_centre,
                                                        vo=v_circular)
            jr = actionsFreqsAngles[0][0]
            jp = actionsFreqsAngles[1][0]
            jz = actionsFreqsAngles[2][0]
            omega_R = actionsFreqsAngles[3][0]
            omega_phi = actionsFreqsAngles[4][0]
            omega_z = actionsFreqsAngles[5][0]
            angle_R = actionsFreqsAngles[6][0]
            alpha_phi = actionsFreqsAngles[7][0]
            angle_z = actionsFreqsAngles[8][0]
            
            EccZmaxRperiRap = aAS.EccZmaxRperiRap(otmp,
                                                ro=r_galactic_centre,
                                                vo=v_circular)
            e = EccZmaxRperiRap[0][0]
            zmax = EccZmaxRperiRap[1][0]
            R_peri = EccZmaxRperiRap[2][0]
            R_ap = EccZmaxRperiRap[3][0]
            
            E = otmp.E(pot=lp,
                       ro=r_galactic_centre,
                       vo=v_circular,
                       zo=z_galactic_plane)
            
            
            # Saving the results for each random orbit
            res[:, r, i] = (xhelio, yhelio, zhelio, U, V, W, R, phi, z, vR, vT, vz, jr, jp, 
                            jz, omega_R, omega_phi, omega_z, angle_R, alpha_phi, angle_z,
                            e, zmax, R_peri, R_ap, E)    
    
    
        # ---SAVING THE RESULTS--- #
        # Save file for the orbits
        filename = f'{ith_star}_orbits.pkl'
        filepath = os.path.join(star_savepath, filename)
        
        # # Storing the orbits into a pickle file
        # with open(filepath, 'wb') as fh:
        #     pickle.dump(res, fh, pickle.HIGHEST_PROTOCOL)
        #     # pickle.dump(orbits, fh, pickle.HIGHEST_PROTOCOL)
            
        # Unpacking the orbit results
        xhelio, yhelio, zhelio, U, V, W, R, phi, z, vR, vT, vz, jr, jp, jz, omega_R, omega_phi, omega_z, angle_R, alpha_phi, angle_z, e, zmax, R_peri, R_ap, E = res
        
        # Taking the median of the results
        med_xhelio = np.nanmedian(xhelio, axis=0)[i]
        med_yhelio = np.nanmedian(yhelio, axis=0)[i]
        med_zhelio = np.nanmedian(zhelio, axis=0)[i]
        med_U = np.nanmedian(U, axis=0)[i]
        med_V = np.nanmedian(V, axis=0)[i]
        med_W = np.nanmedian(W, axis=0)[i]
        med_R = np.nanmedian(R, axis=0)[i]
        med_phi = np.nanmedian(phi, axis=0)[i]
        med_z = np.nanmedian(z, axis=0)[i]
        med_vR = np.nanmedian(vR, axis=0)[i]
        med_vT = np.nanmedian(vT, axis=0)[i]
        med_vz = np.nanmedian(vz, axis=0)[i]
        med_jr = np.nanmedian(jr, axis=0)[i]
        med_jp = np.nanmedian(jp, axis=0)[i]
        med_jz = np.nanmedian(jz, axis=0)[i]
        med_omega_R = np.nanmedian(omega_R, axis=0)[i]
        med_omega_phi = np.nanmedian(omega_phi, axis=0)[i]
        med_omega_z = np.nanmedian(omega_z, axis=0)[i]
        med_angle_R = np.nanmedian(angle_R, axis=0)[i]
        med_alpha_phi = np.nanmedian(alpha_phi, axis=0)[i]
        med_angle_z = np.nanmedian(angle_z, axis=0)[i]
        med_e = np.nanmedian(e, axis=0)[i]
        med_zmax = np.nanmedian(zmax, axis=0)[i]
        med_R_peri = np.nanmedian(R_peri, axis=0)[i]
        med_R_ap = np.nanmedian(R_ap, axis=0)[i]
        med_E = np.nanmedian(E, axis=0)[i]
        
        # Calculating the standard errors for the results
        err_xhelio = standard_error(data=xhelio)
        err_yhelio = standard_error(data=yhelio)
        err_zhelio = standard_error(data=zhelio)
        err_U = standard_error(data=U)
        err_V = standard_error(data=V)
        err_W = standard_error(data=W)
        err_R = standard_error(data=R)
        err_phi = standard_error(data=phi)
        err_z = standard_error(data=z)
        err_vR = standard_error(data=vR)
        err_vT = standard_error(data=vT)
        err_vz = standard_error(data=vz)
        err_jr = standard_error(data=jr)
        err_jp = standard_error(data=jp)
        err_jz = standard_error(data=jz)
        err_omega_R = standard_error(data=omega_R)
        err_omega_phi = standard_error(data=omega_phi)
        err_omega_z = standard_error(data=omega_z)
        err_angle_R = standard_error(data=angle_R)
        err_alpha_phi = standard_error(data=alpha_phi)
        err_angle_z = standard_error(data=angle_z)
        err_e = standard_error(data=e)
        err_zmax = standard_error(data=zmax)
        err_R_peri = standard_error(data=R_peri)
        err_R_ap = standard_error(data=R_ap)
        err_E = standard_error(data=E)

        
        if use_plx:
            res_dict = {
                'star': ith_star,
                'ra': ith_ra,
                'dec': ith_dec,
                'plx': ith_plx,
                'plx_err': ith_plx_err,
                'pmRA': ith_pmRA,
                'pmRA_err': ith_pmRA_err,
                'pmdec': ith_pmdec,
                'pmdec_err': ith_pmdec_err,
                'Vlos': ith_Vlos,
                'eVlos': ith_eVlos,
                'xhelio': med_xhelio,
                'xhelio_err': err_xhelio,
                'yhelio': med_yhelio,
                'yhelio_err': err_yhelio,
                'zhelio': med_zhelio,
                'zhelio_err': err_zhelio,
                'U': med_U,
                'U_err': err_U,
                'V': med_V,
                'V_err': err_V,
                'W': med_W,
                'W_err': err_W,
                'R': med_R,
                'R_err': err_R,
                'phi': med_phi,
                'phi_err': err_phi,
                'z': med_z,
                'z_err': err_z,
                'vR': med_vR,
                'vR_err': err_vR,
                'vT': med_vT,
                'vT_err': err_vT,
                'vz': med_vz,
                'vz_err': err_vz,
                'jr': med_jr,
                'jr_err': err_jr,
                'jp': med_jp,
                'jp_err': err_jp,
                'jz': med_jz,
                'jz_err': err_jz,
                'omega_R': med_omega_R,
                'omega_R_err': err_omega_R,
                'omega_phi': med_omega_phi,
                'omega_phi_err': err_omega_phi,
                'omega_z': med_omega_z,
                'omega_z_err': err_omega_z,
                'angle_R': med_angle_R,
                'angle_R_err': err_angle_R,
                'alpha_phi': med_alpha_phi,
                'alpha_phi_err': err_alpha_phi,
                'angle_z': med_angle_z,
                'angle_z_err': err_angle_z,
                'e': med_e,
                'e_err': err_e,
                'zmax': med_zmax,
                'zmax_err': err_zmax,
                'R_peri': med_R_peri,
                'R_peri_err': err_R_peri,
                'R_ap': med_R_ap,
                'R_ap_err': err_R_ap,
                'E': med_E,
                'E_err': err_E}
            orb_res.append(res_dict)
        else:  
            res_dict = {
                'star': ith_star,
                'ra': ith_ra,
                'dec': ith_dec,
                'dist': ith_dist,
                'dist_err': ith_dist_err,
                'pmRA': ith_pmRA,
                'pmRA_err': ith_pmRA_err,
                'pmdec': ith_pmdec,
                'pmdec_err': ith_pmdec_err,
                'Vlos': ith_Vlos,
                'eVlos': ith_eVlos,
                'xhelio': med_xhelio,
                'xhelio_err': err_xhelio,
                'yhelio': med_yhelio,
                'yhelio_err': err_yhelio,
                'zhelio': med_zhelio,
                'zhelio_err': err_zhelio,
                'U': med_U,
                'U_err': err_U,
                'V': med_V,
                'V_err': err_V,
                'W': med_W,
                'W_err': err_W,
                'R': med_R,
                'R_err': err_R,
                'phi': med_phi,
                'phi_err': err_phi,
                'z': med_z,
                'z_err': err_z,
                'vR': med_vR,
                'vR_err': err_vR,
                'vT': med_vT,
                'vT_err': err_vT,
                'vz': med_vz,
                'vz_err': err_vz,
                'jr': med_jr,
                'jr_err': err_jr,
                'jp': med_jp,
                'jp_err': err_jp,
                'jz': med_jz,
                'jz_err': err_jz,
                'omega_R': med_omega_R,
                'omega_R_err': err_omega_R,
                'omega_phi': med_omega_phi,
                'omega_phi_err': err_omega_phi,
                'omega_z': med_omega_z,
                'omega_z_err': err_omega_z,
                'angle_R': med_angle_R,
                'angle_R_err': err_angle_R,
                'alpha_phi': med_alpha_phi,
                'alpha_phi_err': err_alpha_phi,
                'angle_z': med_angle_z,
                'angle_z_err': err_angle_z,
                'e': med_e,
                'e_err': err_e,
                'zmax': med_zmax,
                'zmax_err': err_zmax,
                'R_peri': med_R_peri,
                'R_peri_err': err_R_peri,
                'R_ap': med_R_ap,
                'R_ap_err': err_R_ap,
                'E': med_E,
                'E_err': err_E}
            orb_res.append(res_dict)
            
        # Saving the median results to a CSV file or FITS file
        star_res_df = pd.DataFrame([res_dict])
        if save_type == 'csv':
            star_res_df.to_csv(os.path.join(star_savepath, f'{ith_star}_orbit_medparams.csv'), index=False)
        if save_type == 'fits':
            star_res_table = Table.from_pandas(star_res_df)
            star_res_table.write(os.path.join(star_savepath, f'{ith_star}_orbit_medparams.fits'), overwrite=True)
        
        # Saving the full nrand results to a CSV file or FITS file       
        full_res_df = pd.DataFrame({
            'xhelio': [i[0] for i in xhelio],
            'yhelio': [i[0] for i in yhelio],
            'zhelio': [i[0] for i in zhelio],
            'U': [i[0] for i in U],
            'V': [i[0] for i in V],
            'W': [i[0] for i in W],
            'R': [i[0] for i in R],
            'phi': [i[0] for i in phi],
            'z': [i[0] for i in z],
            'vR': [i[0] for i in vR],
            'vT': [i[0] for i in vT],
            'vz': [i[0] for i in vz],
            'jr': [i[0] for i in jr],
            'jp': [i[0] for i in jp],
            'jz': [i[0] for i in jz],
            'omega_R': [i[0] for i in omega_R],
            'omega_phi': [i[0] for i in omega_phi],
            'omega_z': [i[0] for i in omega_z],
            'angle_R': [i[0] for i in angle_R],
            'alpha_phi': [i[0] for i in alpha_phi],
            'angle_z': [i[0] for i in angle_z],
            'e': [i[0] for i in e],
            'zmax': [i[0] for i in zmax],
            'R_peri': [i[0] for i in R_peri],
            'R_ap': [i[0] for i in R_ap],
            'E': [i[0] for i in E]
            })
        
        if save_type == 'csv':
            full_res_df.to_csv(os.path.join(star_savepath, f'{ith_star}_orbit_fullparams.csv'), index=False)   
        if save_type == 'fits':
            full_res_table = Table.from_pandas(full_res_df)
            full_res_table.write(os.path.join(star_savepath, f'{ith_star}_orbit_fullparams.fits'), overwrite=True)     
        
        # ---PLOTTING THE ORBITS--- #
        if plot:
            # 2D plot of the orbit
            plot_filename = f'{ith_star}_orbit_2d.png'
            plt.figure(i, figsize=(8,6))
            otmp.plot(d1='x', d2='y', color='firebrick', lw=0.8)
            plt.plot([otmp.x(ts[0])], [otmp.y(ts[0])], marker='x', color='black')  
            plt.xlabel('X (kpc)')
            plt.ylabel('Y (kpc)')
            plt.title(f'{ith_star}')
            plt.savefig(os.path.join(star_savepath, plot_filename), dpi=300, transparent=False, facecolor="white")
            
            # 3D plot of the orbit
            plot_filename = f'{ith_star}_orbit_3d.html'
            
            x = otmp.x(ts)
            y = otmp.y(ts)
            z = otmp.z(ts)
            
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='firebrick', width=3.5)
                )
            ])
                    
            fig.add_trace(go.Scatter3d(
                x=[otmp.x(ts[0])], y=[otmp.y(ts[0])], z=[otmp.z(ts[0])],
                mode='markers',
                marker=dict(
                    symbol='cross',
                    size=10,
                    color='black'
                )
            ))        

            fig.update_layout(
                scene=dict(
                    xaxis_title='X (kpc)',
                    yaxis_title='Y (kpc)',
                    zaxis_title='Z (kpc)'
                ),
                title=ith_star,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            fig.write_html((os.path.join(star_savepath, plot_filename)))


            # # 3D animation of the orbit
            # plot_filename = f'{ith_star}_orbit_animation.mp4'
            
            # x = otmp.x(ts)
            # y = otmp.y(ts)
            # z = otmp.z(ts)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # ax.set_xlim(min(x), max(x))
            # ax.set_ylim(min(y), max(y))
            # ax.set_zlim(min(z), max(z))

            # line, = ax.plot([], [], [], 'b-', lw=2)
            # point, = ax.plot([], [], [], 'ro')

            # def update(num):
            #     line.set_data(x[:num], y[:num])
            #     line.set_3d_properties(z[:num])
            #     point.set_data(x[num-1:num], y[num-1:num])
            #     point.set_3d_properties(z[num-1:num])
            #     return line, point

            # ani = FuncAnimation(fig, update, frames=len(x), interval=20, blit=True)

            # ani.save(os.path.join(savepath, plot_filename), writer='ffmpeg', fps=30)

            # plt.close(fig)
        else:
            pass
        print()
        
    # Creating dataframe from the median results
    res_df = pd.DataFrame(orb_res)
    # res_df.to_csv(os.path.join(star_savepath, 'orbits_results.csv'), index=False)
    
    return res_df