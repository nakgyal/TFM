"""
This file defines the `make_figure` function, which can be useful for
debugging and/or examining the morphology of a source in detail.
"""
# Author: Vicente Rodriguez-Gomez <vrodgom.astro@gmail.com>
# Licensed under a 3-Clause BSD License.

import numpy as np
import sys
import skimage.transform
import statmorph
from astropy.visualization import simple_norm

__all__ = ['make_figure']

def _get_ax(fig, row, col, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig):
    x_ax = (col+1)*eps + col*wpanel
    y_ax = eps + (nrows-1-row)*(hpanel+htop)
    return fig.add_axes([x_ax/wfig, y_ax/hfig, wpanel/wfig, hpanel/hfig])

def make_figure(morph,z,id):
    """
    Creates a figure analogous to Fig. 4 from Rodriguez-Gomez et al. (2019)
    for a given ``SourceMorphology`` object.
    
    Parameters
    ----------
    morph : ``statmorph.SourceMorphology``
        An object containing the morphological measurements of a single
        source.
    z : ``float``
        Redshift of the source.
    id : ``int``
        ID of the source.

    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        The figure.

    """
    from astropy.cosmology import Planck18
    from astropy import units as u
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.cm

    if not isinstance(morph, statmorph.SourceMorphology):
        raise TypeError('Input must be of type SourceMorphology.')

    if morph.flag == 4:
        raise Exception('Catastrophic flag (not worth plotting)')

    # I'm tired of dealing with plt.add_subplot, plt.subplots, plg.GridSpec,
    # plt.subplot2grid, etc. and never getting the vertical and horizontal
    # inter-panel spacings to have the same size, so instead let's do
    # everything manually:
    nrows = 2
    ncols = 3
    wpanel = 4.0  # panel width
    hpanel = 4.0  # panel height
    htop = 0.05*nrows*hpanel  # top margin and vertical space between panels
    eps = 0.005*nrows*hpanel  # all other margins
    wfig = ncols*wpanel + (ncols+1)*eps  # total figure width
    hfig = nrows*(hpanel+htop) + eps  # total figure height
    fig = plt.figure(figsize=(wfig, hfig))

    # For drawing circles/ellipses
    theta_vec = np.linspace(0.0, 2.0*np.pi, 200)

    # Add black to pastel colormap
    cmap_orig = matplotlib.cm.Pastel1
    colors = ((0.0, 0.0, 0.0), *cmap_orig.colors)
    cmap = matplotlib.colors.ListedColormap(colors)

    # Get some general info about the image
    image = np.float64(morph._cutout_stamp_maskzeroed)  # skimage wants double
    ny, nx = image.shape
    xc, yc = morph._xc_stamp, morph._yc_stamp  # centroid
    xca, yca = morph._asymmetry_center  # asym. center
    xcs, ycs = morph._sersic_model.x_0.value, morph._sersic_model.y_0.value  # Sersic center

    ##################
    # Original image #
    ##################
    ax = _get_ax(fig, 0, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    R = np.sqrt(nx**2 + ny**2)
    theta = morph.orientation_centroid
    # Some text
    text = 'ID: %.0f \n z = %.2f' % (id,z)
    ax.text(0.034, 0.966, text,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text2 = 'Ellip. (Centroid) = %.3f\nEllip. (Asym.) = %.3f\nEllip. (Sérsic) = %.3f' % (
        morph.ellipticity_centroid, morph.ellipticity_asymmetry, morph.sersic_ellip)
    ax.text(0.55, 0.15, text2,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_title('Original Image (Log Stretch)', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##############
    # Sersic fit #
    ##############
    ax = _get_ax(fig, 0, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_model = morph._sersic_model(x, y)
    # Add background noise (for realism)
    if morph.sky_sigma > 0:
        sersic_model += np.random.normal(scale=morph.sky_sigma, size=(ny, nx))
    ax.imshow(sersic_model, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    # Some text
    rhalf_arcsec=morph.sersic_rhalf*0.03 #scale: 0.03 arcsec/px
    d_A = Planck18.angular_diameter_distance(z)
    scale_pc_arcsec = (d_A).to(u.pc) / (206265.0 * u.arcsec)
    rhalf_kpc = rhalf_arcsec * scale_pc_arcsec / 1e3 / u.pc * u.arcsec
    text = ('$R_e = $ %.2f' % (rhalf_arcsec,) + ' arcsec\n' +
            r'$R_e = %.3f$' % (rhalf_kpc,) + ' kpc\n' +
            r'$n = %.2f$' % (morph.sersic_n,))
    ax.text(0.034, 0.966, text,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.set_title('Sérsic Model + Noise', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Sersic residual #
    ###################
    ax = _get_ax(fig, 0, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_res = morph._cutout_stamp_maskzeroed - morph._sersic_model(x, y)
    sersic_res[morph._mask_stamp] = 0.0
    ax.imshow(sersic_res, cmap='gray', origin='lower',
              norm=simple_norm(sersic_res, stretch='linear'))

    ## Contour lines
    ### Calculate the gradient of the image using np.gradient
    dx, dy = np.gradient(sersic_res)

    ### Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    ### Draw contours:
    levels=np.percentile(sersic_res,[95.48,99])
    ax.contour(sersic_res, levels, colors=['rosybrown','firebrick'], linewidths=1.5)
    
    # Create dummy lines for the legend
    contour_line1 = plt.Line2D((0,1),(0,0), color='rosybrown')
    contour_line2 = plt.Line2D((0,1),(0,0), color='firebrick')

    # Add legend
    ax.legend([contour_line1, contour_line2], ['2$\sigma$', '$3\sigma$'],
        loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')

    ax.set_title('Sérsic Residual, ' + r'$I - I_{\rm model}$', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    ######################
    # Asymmetry residual #
    ######################
    ax = _get_ax(fig, 1, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    # Rotate image around asym. center
    # (note that skimage expects pixel positions at lower-left corners)
    image_180 = skimage.transform.rotate(image, 180.0, center=(xca, yca))
    image_res = image - image_180
    # Apply symmetric mask
    mask = morph._mask_stamp.copy()
    mask_180 = skimage.transform.rotate(mask, 180.0, center=(xca, yca))
    mask_180 = mask_180 >= 0.5  # convert back to bool
    mask_symmetric = mask | mask_180
    image_res = np.where(~mask_symmetric, image_res, 0.0)
    ax.imshow(image_res, cmap='gray', origin='lower',
              norm=simple_norm(image_res, stretch='linear'))
    ax.set_title('Asymmetry Residual, ' + r'$I - I_{180}$', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    ##### Segmaps #####
    ###################
    ax = _get_ax(fig, 1, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(image, cmap='gray', origin='lower',
              norm=simple_norm(image, stretch='log', log_a=10000))
    # Show original segmap
    contour_levels = [0.5]
    contour_colors = [(0, 0, 0)]
    segmap_stamp = morph._segmap.data[morph._slice_stamp]
    Z = np.float64(segmap_stamp == morph.label)
    contour1=ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5, label='Original segmap')
    # Show skybox
    xmin = morph._slice_skybox[1].start
    ymin = morph._slice_skybox[0].start
    xmax = morph._slice_skybox[1].stop - 1
    ymax = morph._slice_skybox[0].stop - 1
    skybox=ax.plot(np.array([xmin, xmax, xmax, xmin, xmin]),
            np.array([ymin, ymin, ymax, ymax, ymin]),
            'b', lw=1.5, label='Skybox')
    # Show gini segmap
    contour_levels = [0.5]
    Z = np.float64(morph._segmap_gini)
    contour2=ax.contour(Z, contour_levels, colors='r', linewidths=1.5, label='Gini segmap')
    #Some text
    text = r'$\left\langle {\rm S/N} \right\rangle = %.0f$' % (morph.sn_per_pixel,)
    ax.text(0.034, 0.966, text, fontsize=12,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Create dummy lines for the legend
    contour_line1 = plt.Line2D((0,1),(0,0), color='k')
    contour_line2 = plt.Line2D((0,1),(0,0), color='r')
    skybox_line=plt.Line2D((0,1),(0,0), color='b')

    # Add legend
    ax.legend([contour_line1, contour_line2,skybox_line], ['Original segmap', 'Gini segmap','Skybox'],
	loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    # Finish plot
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.set_title('Segmaps', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ####################
    # Watershed segmap #
    ####################
    ax = _get_ax(fig, 1, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    labeled_array, peak_labels, xpeak, ypeak = morph._watershed_mid
    labeled_array_plot = (labeled_array % (cmap.N-1)) + 1
    labeled_array_plot[labeled_array == 0] = 0.0  # background is black
    ax.imshow(labeled_array_plot, cmap=cmap, origin='lower',
              norm=matplotlib.colors.NoNorm())
    sorted_flux_sums, sorted_xpeak, sorted_ypeak = morph._intensity_sums
    if len(sorted_flux_sums) > 0:
        ax.plot(sorted_xpeak[0], sorted_ypeak[0], 'bo', markersize=2,
                label='First Peak')
    if len(sorted_flux_sums) > 1:
        ax.plot(sorted_xpeak[1], sorted_ypeak[1], 'ro', markersize=2,
                label='Second Peak')
    # Some text
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Watershed Segmap (' + r'$I$' + ' statistic)', fontsize=14)
    ax.set_xlim(-0.5, nx-0.5)
    ax.set_ylim(-0.5, ny-0.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig
