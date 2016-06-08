# RadioAstronomyThings
This contains algorithms and tools that I have thought of and had the time to code up, that relate to radio astronomy.

## BeamDeconvolution
Tools that deconvololve gaussian beams, find common beams, and a few other things.

## UniformSelection
Tools for selecting uniformly distributed calibrators from available calibrators.
Requires pybdsm to be sourced, and to change the variables at the bottom. It will generate a gaussian source list of the brightest sources, and a factor_directions.txt in the cwd. It essentially functions as follows:
1) produce a list of gaussians with snr>100 (can be set as desired)
2) filter: select only gaussian sources with 3 or fewer islands
3) filter: combine sources closer than minSep in arcsec
4) selection: find the combination of calibrators that minimizes an L2 non-uniformity criterion. Performs in parallel (set by ncpu).
  I make the approximation that we can search chunks of flux sorted calbrators at a time (instead of permutating all calbrators)
  The number to search through is dynamically allotcated to perform the search within a give number of user defined minutes.
  If you allow full search, then it may take days and permutating over the faint calibrators is a waste of time anyways so my approxmiation is valid. Possible issues: If snr thresh for pybdsm is too high, or minFlux is too high then not enough calibrators may be found, and we may return less than the requested numClusters.

## CombineImages
Tool for combining (stacking) images. Regrids, convoles to common beamsize, then weighted averages them. 
Requires beam doconvolution.
