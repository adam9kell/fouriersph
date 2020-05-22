# fouriersph
absorption energy of quantum dots

based on fortran source code written by david f kelley (UC Merced)
see https://pubs.acs.org/doi/10.1021/jp4002753
https://pubs.acs.org/doi/10.1021/acsnano.6b00370

calls numpy and scipy

calculates electron and hole wavefunctions and optical bandgap for cdse-based core/shell particles (cdse/cds, cse/znse and cdse/zns). spherical bessel functions are used as basis set for wavefunctions and the electron effective mass is scaled empirically to reproduce cdse quantum dot sizing curves.
