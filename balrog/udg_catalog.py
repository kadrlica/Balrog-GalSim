#----------------------------------------------------------------------------
# Classes for new input type `udgCatalog`; intended for use with Balrog.
# Based on ngmix_catalog.py from Spencer Everett
#
# Contributors:
# Alex Drlica-Wagner (FNAL)
#----------------------------------------------------------------------------

import galsim
import galsim.config
import numpy as np
import logging
import warnings
from past.builtins import basestring # Python 2&3 compatibility
# import pudb

# TODO: Include noise, pixscale
# TODO: Handle case of gparams=None

class udgCatalog(object):
    """ Class that handles catalogs of UDG profiles. 

    Much of this class as well as its corresponding loader/builder are designed by inspection of
    `des_psfex.py` and `scene.py`. Credit to the GalSim team.

    @param file_name       The file name to be read in, or a pyfits HDU in which case it is used
                           directly instead of being opened.
    @param bands           A string of the desired bands to simulate from (only griz allowed). For
                           example, selecting only the 'g' and 'r' bands would be done by setting
                           bands='gr'. If none are passed, the g-band is selected by default.
    @param dir             Optionally a directory name can be provided if the file_name does not
                           already include it.  (The image file is assumed to be in the same
                           directory.) (Default `dir = None`).  Cannot pass an HDU with this option.
    @param catalog_type    The type of the input catalog. Only those in `valid_catalog_types`
                           are currently supported. If none is passed, the type is attempted to be
                           inferred from the filename.
    @param _nobjects_only  This is only passed if GalSim wants to know how many input objects will
                           be used without processing the whole input catalog.
    """

    _req_params = { 'file_name' : str, 'bands' : str}
    _opt_params = { 'dir' : str, 'catalog_type' : str}
    _single_params = []
    _takes_rng = False

    # Only these catalog types currently supported
    _valid_catalog_types = ['sersic']

    # Only these color bands are currently supported for a catalog
    _valid_band_types = ['g','r','i','z']

    # Dictionary of color band flux to array index in catalogs
    _band_index = {'g' : 0, 'r' : 1, 'i' : 2, 'z' : 3}

    # The catalog column name prefix doens't always match the catalog type (e.g. 'mof' has a prefix
    # of 'cm' for most columns). Set this for each new supported catalog type.
    _cat_col_prefix = {'sersic' : 'sersic'}

    def __init__(self, file_name, bands, dir=None, catalog_type='sersic',  _nobjects_only=False):

        if dir:
            if not isinstance(file_name, basestring):
                raise ValueError("Cannot provide dir and an HDU instance!")
            import os
            file_name = os.path.join(dir,file_name)
        if not isinstance(file_name, basestring):
            raise ValueError("The input filename must be a string!")
        self.file_name = file_name

        if catalog_type not in self._valid_catalog_types:
            raise ValueError("Invalid catalog type: {}".format(catalog_type))
        else:
            self.cat_type = catalog_type

        # Catalog column name prefixes
        self.col_prefix = self._cat_col_prefix[self.cat_type]

        if isinstance(bands, basestring):
            # Strip all whitespace
            bands = bands.replace(" ", "")
            # More useful as a list of individual bands
            bands_list = list(bands)
            if len(bands_list) > 1:
                warnings.warn('WARNING: Passed more than one band - injections will contain ' + \
                              'flux from multiple bands!')
            if set(bands_list).issubset(self._valid_band_types):
                self.bands = bands_list
            else:
                raise ValueError("The only valid color bands for a udg catalog are {}!".format(self._valid_band_types))
        else:
            # TODO: Wouldn't be a bad idea to allow a list of individual bands as well
            raise ValueError("Must enter desired color bands as a string! (For example, `bands : \'gr\'`)")

        self.read()

        return

    #------------------------------------------------------------------------------------------------

    def read(self):
        """Read in relevant catalog information"""

        from galsim._pyfits import pyfits

        if isinstance(self.file_name, basestring):
            # If a filename is passed:
            hdu_list = pyfits.open(self.file_name)
            model_fits = hdu_list[1]
        else:
            # If a fits HDU is directly passed:
            hdu_list = None
            model_fits = self.file_name

        self.catalog = model_fits.data

        # NB: As discussed in `scene.py`, there is a bug in the pyfits FITS_Rec class that leads to memory leaks.
        # The simplest workaround seems to be to convert it to a regular numpy recarray.
        self.catalog = np.array(self.catalog, copy=True)

        # The input logger needs to know the original catalog size
        self.ntotal = len(self.catalog)

        # Close file!
        if hdu_list: hdu_list.close()

        # Galaxy indices in original catalog
        self.orig_index = np.arange(self.ntotal)

        # Get flags and create mask
        self.getFlags()
        self.makeMask()

        # Do mask cut
        self.maskCut()

        # pudb.set_trace()

        return

    #------------------------------------------------------------------------------------------------

    def getFlags(self):
        """Retrieve object flags, where implementation depends on catalog type."""
        # General flags
        self.flags = self.catalog['flags']

    #------------------------------------------------------------------------------------------------

    def makeMask(self):
        """Add a masking procedure, if desired."""
        # For now, remove objects with any flags present
        mask = np.ones(len(self.orig_index), dtype=bool)
        mask[self.flags != 0] = False
        self.mask = mask

    #------------------------------------------------------------------------------------------------

    def maskCut(self):
        """Do mask cut defined in `makeMask()`."""
        self.catalog = self.catalog[self.mask]
        self.orig_index = self.orig_index[self.mask]
        self.nobjects = len(self.orig_index)

    #------------------------------------------------------------------------------------------------

    def makeGalaxies(self, index=None, n_random=None, rng=None, gsparams=None):
        """
        Construct GSObjects from a list of galaxies in the catalog specified by `index` (or a randomly generated one).
        This is done using Erin's code.

        @param index            Index of the desired galaxy in the catalog for which a GSObject
                                should be constructed.  You can also provide a list or array of
                                indices, in which case a list of objects is returned. If None,
                                then a random galaxy (or more: see n_random kwarg) is chosen,
                                correcting for catalog-level selection effects if weights are
                                available. [default: None]
        @param n_random         The number of random galaxies to build, if 'index' is None.
                                [default: 1 (set below)]
        @param rng              A random number generator to use for selecting a random galaxy
                                (may be any kind of BaseDeviate or None) and to use in generating
                                any noise field when padding.  [default: None]
        @param gsparams         An optional GSParams argument.  See the docstring for GSParams for
                                details. [default: None]
        """

        # Make rng if needed
        if index is None:
            if rng is None:
                rng = galsim.BaseDeviate()
            elif not isinstance(rng, galsim.BaseDeviate):
                raise TypeError("The rng provided to makeGalaxies is not a BaseDeviate")

        # Select random indices if necessary (no index given).
        if index is None:
            if n_random is None: n_random = 1
            index = self.selectRandomIndex(n_random, rng=rng)
        else:
            # n_random is set to None by default instead of 1 for this check
            if n_random is not None:
                warnings.warn("Ignoring input n_random, since indices were specified!")

        # Format indices correctly
        if hasattr(index, '__iter__'):
            indices = index
        else:
            indices = [index]

        # print('\nIndices = {}'.format(indices))

        # Now convert galaxy to GSObject, with details dependent on type
        # TODO: Should we add a more general scheme for types?

        galaxies = []

        for index in indices:
            gal = self.cat2gs(index,gsparams)
            galaxies.append(gal)

        # Store the orig_index as gal.index
        for gal, idx in zip(galaxies, indices):
            gal.index = self.orig_index[idx]

        # Only return a list if there are multiple GSObjects
        if hasattr(index, '__iter__'):
            return galaxies
        else:
            return galaxies[0]

    #------------------------------------------------------------------------------------------------

    def cat2gs(self, index, gsparams):
        """
        This function handles the conversion of a catalog object to a GS object. The required conversion is
        different for each catalog type.
        @ param index       The catalog index of the galaxy to be converted.
        @ param gsparams    The GalSim parameters.
        """

        # TODO: Make sure that the index usage is consistent with original/ indices!!
        cp = 'sersic'

        cp = self.col_prefix
        ct = self.cat_type

        n = self.catalog[cp+'_n'][index]
        reff = self.catalog[cp+'_reff'][index]
        g1, g2 = self.catalog[cp+'_g'][index]

        # List of individual band GSObjects
        gsobjects = []

        # Convert from dict to actual GsParams object
        # TODO: Currently fails if gsparams isn't passed!
        if gsparams: gsp = galsim.GSParams(**gsparams)
        else: gsp = None

        # Iterate over all desired bands
        for band in self.bands:
            # Grab current band flux
            flux = self.catalog['flux'][index][self._band_index[band]]

            # DEBUGGING
            #print('n={}; half_light_radius={}; flux={}, band={}'.format(n,reff,flux,band))
            gal = galsim.Sersic(n=n, half_light_radius=reff, flux=flux, gsparams=gsp)
            gal = gal.shear(g1=g1, g2=g2)

            # Add galaxy in given band to list
            gsobjects.append(gal)

        # NOTE: If multiple bands were passed, the fluxes are simply added together.
        gs_gal = galsim.Add(gsobjects)

        return gs_gal

    #------------------------------------------------------------------------------------------------
    @staticmethod
    def _makeSingleGalaxy(catalog, index, rng=None, gsparams=None):
        """ A static function that mimics the functionality of makeGalaxes() for single index.
        The only point of this class is to circumvent some serialization issues. This means it can be used
        through a proxy udgCatalog object, which is needed for the config layer.
        """

        # TODO: Write the static version of makeGalaxies! (We don't need it for prototyping Balrog, however)
        # NB: I have noticed an occasional memory issue with N>~200 galaxies. This may be related to the serialization
        # issues Mike talks about in scene.py
        pass

    #------------------------------------------------------------------------------------------------

    def selectRandomIndex(self, n_random=1, rng=None, _n_rng_calls=False):
        """
        Routine to select random indices out of the catalog.  This routine does a weighted random
        selection with replacement (i.e., there is no guarantee of uniqueness of the selected
        indices).  Weighting uses the weight factors available in the catalog, if any; these weights
        are typically meant to remove any selection effects in the catalog creation process.
        @param n_random     Number of random indices to return. [default: 1]
        @param rng          A random number generator to use for selecting a random galaxy
                            (may be any kind of BaseDeviate or None). [default: None]
        @returns A single index if n_random==1 or a NumPy array containing the randomly-selected
        indices if n_random>1.
        """

        # Set up the random number generator.
        if rng is None:
            rng = galsim.BaseDeviate()

        # QSTN: What is the weighting scheme for catalogs? Will need to adjust below code to match (or exclude entierly)
        if hasattr(self.catalog, 'weight'):
            use_weights = self.catalog.weight[self.orig_index]
        else:
            warnings.warn('Selecting random object without correcting for catalog-level selection effects.')
            use_weights = None

        # By default, get the number of RNG calls. Then decide whether or not to return them
        # based on _n_rng_calls.
        index, n_rng_calls = galsim.utilities.rand_with_replacement(
                n_random, self.nobjects, rng, use_weights, _n_rng_calls=True)

        if n_random>1:
            if _n_rng_calls:
                return index, n_rng_calls
            else:
                return index
        else:
            if _n_rng_calls:
                return index[0], n_rng_calls
            else:
                return index[0]

    #------------------------------------------------------------------------------------------------

    def getNObjects(self):
        # Used by input/logger methods
        return self.nobjects

    def getNTot(self):
        # Used by input/logger methods
        return self.ntotal

    def getCatalog(self):
        return self.catalog

    def getBands(self):
        return self.bands

    # TODO: Write remaining `get` methods once saved columns are determined

    #------------------------------------------------------------------------------------------------

    # Since makeGalaxies is a function, not a class, it needs to use an unconventional location for defining
    # certain config parameters.
    makeGalaxies._req_params = {}
    makeGalaxies._opt_params = { "index" : int,
                               "n_random": int
                             }
    makeGalaxies._single_params = []
    makeGalaxies._takes_rng = True

#####------------------------------------------------------------------------------------------------

class udgCatalogLoader(galsim.config.InputLoader):
    """ The udgCatalogLoader doesn't need anything special other than registration as a valid input type.
        These additions are only used for logging purposes.
    """

    def setupImage(self, catalog, config, base, logger):
        # This method is blank for a general InputLoader, and a convenient place to put the logger
        if logger: # pragma: no cover
            # Only report as a warning the first time.  After that, use info.
            first = not base.get('_udgCatalogLoader_reported_as_warning',False)
            base['_udgCatalogLoader_reported_as_warning'] = True
            if first:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO
            if 'input' in base:
                if 'udg_catalog' in base['input']:
                    cc = base['input']['udg_catalog']
                    if isinstance(cc,list): cc = cc[0]
                    out_str = ''
                    if 'dir' in cc:
                        out_str += '\n  dir = %s'%cc['dir']
                    if 'file_name' in cc:
                        out_str += '\n  file_name = %s'%cc['file_name']
                    # TODO: Add any desired additional catalog inputs
                    if out_str != '':
                        logger.log(log_level, 'Using user-specified udgCatalog: %s',out_str)
            logger.info("file %d: Udg catalog has %d total objects; %d passed initial cuts.",
                        base['file_num'], catalog.getNTot(), catalog.getNObjects())

# Need to add the udgCatalog class as a valid input_type.
galsim.config.RegisterInputType('udg_catalog', udgCatalogLoader(udgCatalog, has_nobj=True))

#####------------------------------------------------------------------------------------------------

def BuildUdgGalaxy(config, base, ignore, gsparams, logger):
    """ Build a UdgGalaxy type GSObject from user input."""

    catalog = galsim.config.GetInputObj('udg_catalog', config, base, 'UdgGalaxy')

    # If galaxies are selected based on index, and index is Sequence or Random, and max
    # isn't set, set it to nobjects-1.
    if 'index' in config:
        galsim.config.SetDefaultIndex(config, catalog.getNObjects())

    # Grab necessary parameters
    req = udgCatalog.makeGalaxies._req_params
    opt = udgCatalog.makeGalaxies._opt_params
    single = udgCatalog.makeGalaxies._single_params
    ignore = ignore + ['num']

    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt, single=single, ignore=ignore)

    # Convert gsparams from a dict to an actual GSParams object
    if gsparams:
        kwargs['gsparams'] = galsim.GSParams(**gsparams)
    else:
        gsparams = None

    # This handles the case of no index passed in config file
    # Details are in udgCatalog
    rng = None
    if 'index' not in kwargs:
        rng = galsim.config.GetRNG(config, base, logger, 'UdgGalaxy')
        kwargs['index'], n_rng_calls = catalog.selectRandomIndex(1, rng=rng, _n_rng_calls=True)

        # Make sure this process gives consistent results regardless of the number of processes
        # being used.
        if not isinstance(catalog, udgCatalog) and rng is not None:
            # Then catalog is really a proxy, which means the rng was pickled, so we need to
            # discard the same number of random calls from the one in the config dict.
            rng.discard(int(n_rng_calls))

    # Check that inputted/set index is valid
    index = kwargs['index']
    if index >= catalog.getNObjects():
        raise IndexError("%s index has gone past the number of entries in the catalog"%index)

    logger.debug('obj %d: UdgGalaxy kwargs = %s',base.get('obj_num',0),kwargs)

    kwargs['udg_catalog'] = catalog

    # NB: This uses a static method of udgCatalog to save memory. Not needed for the moment, but
    # worth looking into when trying to save resources for large Balrog runs
    # gal = catalog._makeSingleGalaxy(**kwargs)

    # Create GSObject galaxies from the catalog
    gal = catalog.makeGalaxies(index=index,gsparams=gsparams)

    # The second item is "safe", a boolean that declares whether the returned value is
    # safe to save and use again for later objects (which is not the case for udgGalaxies).
    return gal, False

# Register this builder with the config framework:
galsim.config.RegisterObjectType('udgGalaxy', BuildUdgGalaxy, input_type='udg_catalog')

