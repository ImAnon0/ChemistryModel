# Grambow optimised-valence failure diagnosis

Rows: 200. Score format detected: csv.

Bond labels below are geometry-derived diagnostics, not ChemistryModel bond declarations.

## Broad failure families

| Family | Count | Barrier mean | Barrier MAE | Reaction mean | Reaction MAE | Sign failures |
| --- | --- | --- | --- | --- | --- | --- |
| bond formation/dissociation | 14 | -6.373 | 8.191 | -8.475 | 10.289 | 7 |
| bond rearrangement | 41 | 3.237 | 4.983 | -0.366 | 4.137 | 2 |
| hydrogen transfer | 144 | 2.302 | 4.049 | 0.723 | 3.928 | 17 |
| O-O/peroxide chemistry | 1 | 1.927 | 1.927 | 1.865 | 1.865 | 0 |

## Detailed failure classes

| Failure class | Count | Barrier mean | Barrier MAE | Reaction mean | Reaction MAE | Sign failures |
| --- | --- | --- | --- | --- | --- | --- |
| multi-bond dissociation | 6 | -12.488 | 12.488 | -15.510 | 15.510 | 5 |
| H transfer (N-H -> O-H) | 1 | 12.381 | 12.381 | -2.232 | 2.232 | 0 |
| C-N formation | 1 | 10.687 | 10.687 | 10.013 | 10.013 | 0 |
| H transfer (C-H -> N-H) | 17 | 3.719 | 6.003 | 3.755 | 6.806 | 2 |
| H transfer (C/O-H -> H-H) | 3 | 5.889 | 5.889 | 3.708 | 3.708 | 0 |
| H transfer (C/O-H -> O-H) | 1 | 5.467 | 5.467 | 4.430 | 4.430 | 0 |
| C-C dissociation | 4 | -4.412 | 5.431 | -6.314 | 7.656 | 1 |
| bond rearrangement | 41 | 3.237 | 4.983 | -0.366 | 4.137 | 2 |
| H transfer (C/N-H -> H-H) | 3 | -2.847 | 4.933 | -2.828 | 7.048 | 1 |
| H transfer (C-H -> C-H) | 59 | 1.954 | 4.094 | -0.774 | 3.608 | 10 |
| H transfer (C-H -> O-H) | 27 | 3.356 | 3.979 | 1.674 | 4.040 | 1 |
| H transfer (N-H -> N-H) | 1 | -3.187 | 3.187 | 0.528 | 0.528 | 1 |
| H transfer (C-H -> H-H) | 16 | 1.693 | 3.132 | 1.887 | 3.607 | 1 |
| N-N dissociation | 1 | -2.753 | 2.753 | -2.512 | 2.512 | 1 |
| H transfer (O-H -> C-H) | 2 | -0.949 | 2.446 | -1.417 | 1.417 | 1 |
| C-N dissociation | 2 | -2.292 | 2.292 | -3.921 | 3.921 | 0 |
| H transfer (C/O-H -> C/H-H) | 1 | 2.122 | 2.122 | 1.321 | 1.321 | 0 |
| H transfer (N-H -> C-H) | 5 | 1.192 | 2.076 | -0.872 | 1.389 | 0 |
| O-O formation | 1 | 1.927 | 1.927 | 1.865 | 1.865 | 0 |
| H transfer (N-H -> H-H) | 2 | 1.867 | 1.867 | 0.602 | 0.920 | 0 |
| H transfer (O-H -> O-H) | 4 | -0.055 | 1.375 | 2.230 | 3.127 | 0 |
| H transfer (C-H -> C/H-H) | 1 | 0.916 | 0.916 | 3.090 | 3.090 | 0 |
| H transfer (O-H -> N-H) | 1 | 0.818 | 0.818 | -2.229 | 2.229 | 0 |

## Worst 50 barrier errors

| Reaction | Class | Reactants | Products | Model | Reference | Error |
| --- | --- | --- | --- | --- | --- | --- |
| rxn006559 | multi-bond dissociation | C5H7NO | C5H7NO | -33.218 | 2.284 | -35.502 |
| rxn011804 | bond rearrangement | C5H5NO | C5H5NO | 28.541 | 2.531 | 26.010 |
| rxn004353 | H transfer (C-H -> C-H) | C4H6O2 | C4H6O2 | 26.919 | 4.154 | 22.765 |
| rxn000096 | multi-bond dissociation | C5H7N | C5H7N | -18.542 | 3.426 | -21.968 |
| rxn010742 | C-C dissociation | C6H6O | C6H6O | -17.710 | 1.862 | -19.573 |
| rxn009023 | bond rearrangement | C6H8O | C6H8O | 18.389 | 3.696 | 14.693 |
| rxn007636 | H transfer (C-H -> C-H) | C6H9N | C6H9N | 16.767 | 2.630 | 14.137 |
| rxn006048 | H transfer (C-H -> C-H) | C5H8O | C5H8O | 15.877 | 3.354 | 12.523 |
| rxn005120 | H transfer (N-H -> O-H) | C3H7NO2 | C2H3NO + CH4O | 14.862 | 2.482 | 12.381 |
| rxn011394 | bond rearrangement | C6H8O | C6H8O | -8.494 | 3.789 | -12.283 |
| rxn001081 | H transfer (C-H -> N-H) | C4H6N2 | C4H6N2 | 15.723 | 3.983 | 11.740 |
| rxn001662 | H transfer (C-H -> O-H) | C4H6O2 | C4H6O2 | 16.840 | 5.370 | 11.470 |
| rxn011223 | H transfer (C/N-H -> H-H) | C5H7NO | C5H5NO + H2 | -5.762 | 5.704 | -11.465 |
| rxn008341 | bond rearrangement | C6H6O | C6H6O | 13.934 | 2.532 | 11.402 |
| rxn008195 | C-N formation | C5H5NO | C5H5NO | 14.770 | 4.083 | 10.687 |
| rxn009797 | bond rearrangement | C6H13N | C6H13N | 14.521 | 4.130 | 10.391 |
| rxn003547 | bond rearrangement | C4H7NO | C4H7NO | 12.833 | 2.708 | 10.125 |
| rxn003071 | bond rearrangement | C3H4N2O | C3H4N2O | 12.669 | 2.713 | 9.956 |
| rxn008452 | H transfer (C-H -> C-H) | C6H8O | C6H8O | 15.455 | 5.783 | 9.672 |
| rxn000920 | H transfer (C-H -> N-H) | C3H5N3 | C3H5N3 | 13.237 | 3.752 | 9.485 |
| rxn005403 | bond rearrangement | C5H10O | C5H10O | 14.368 | 5.135 | 9.233 |
| rxn005670 | H transfer (C-H -> N-H) | C4H6N2 | C4H6N2 | 12.152 | 3.291 | 8.861 |
| rxn005094 | bond rearrangement | C4H5NO | C4H5NO | 14.034 | 5.211 | 8.822 |
| rxn007313 | H transfer (C-H -> N-H) | C3H5N3O | C2H3NO + CH2N2 | -3.920 | 4.874 | -8.794 |
| rxn008525 | H transfer (C-H -> C-H) | C5H6O2 | C5H6O2 | 11.634 | 3.154 | 8.480 |
| rxn003901 | bond rearrangement | C3H7NO | C2H4O + CH3N | 13.131 | 4.681 | 8.450 |
| rxn010199 | H transfer (C-H -> H-H) | C4H6O3 | C4H4O3 + H2 | 11.438 | 3.129 | 8.309 |
| rxn011246 | H transfer (C/O-H -> H-H) | C5H9NO | C5H7NO + H2 | 12.083 | 3.829 | 8.254 |
| rxn005619 | H transfer (C-H -> C-H) | C2H5NO | CH3N + CH2O | 12.215 | 4.060 | 8.155 |
| rxn011847 | H transfer (C-H -> N-H) | C4H5N3 | C4H5N3 | 10.666 | 2.567 | 8.098 |
| rxn005576 | H transfer (C-H -> H-H) | C5H8O | C5H6O + H2 | -1.054 | 6.851 | -7.905 |
| rxn011929 | H transfer (C-H -> C-H) | C7H10 | C7H10 | 10.769 | 3.036 | 7.733 |
| rxn000941 | H transfer (C-H -> O-H) | C4H6O2 | C4H4O + H2O | 11.261 | 3.726 | 7.535 |
| rxn008551 | H transfer (C-H -> N-H) | C4H6N2O | C4H6N2O | 11.464 | 4.040 | 7.424 |
| rxn004031 | bond rearrangement | C5H8O | C5H8O | 10.665 | 3.304 | 7.360 |
| rxn009328 | H transfer (C-H -> C-H) | C5H8N2 | C5H8N2 | -2.893 | 4.429 | -7.322 |
| rxn011522 | H transfer (C-H -> N-H) | C5H10N2 | C5H10N2 | 10.239 | 3.024 | 7.215 |
| rxn001291 | H transfer (C-H -> O-H) | C3H7NO2 | C3H5NO + H2O | 10.096 | 2.994 | 7.102 |
| rxn003095 | multi-bond dissociation | CHN3O2 | CHNO + N2O | -5.093 | 1.940 | -7.033 |
| rxn004419 | H transfer (C-H -> C-H) | C4H8O2 | C4H8O2 | 9.593 | 2.582 | 7.011 |
| rxn003773 | H transfer (C-H -> N-H) | C2H3N3 | C2H3N3 | -3.257 | 3.699 | -6.957 |
| rxn011863 | H transfer (C-H -> O-H) | C6H12O | C6H10 + H2O | 10.185 | 3.366 | 6.819 |
| rxn005999 | H transfer (C-H -> N-H) | C4H7NO | C4H7NO | 9.850 | 3.063 | 6.787 |
| rxn007125 | H transfer (C-H -> C-H) | C5H7NO | C5H7NO | 9.710 | 2.937 | 6.773 |
| rxn000206 | H transfer (C-H -> O-H) | C3H4O2 | C3H4O2 | 9.980 | 3.335 | 6.645 |
| rxn008955 | H transfer (C-H -> O-H) | C5H6O2 | C5H6O2 | 9.774 | 3.291 | 6.482 |
| rxn000501 | H transfer (C-H -> C-H) | C4H6O | C4H6O | -3.024 | 3.398 | -6.421 |
| rxn004738 | H transfer (C-H -> O-H) | C4H10O2 | C4H8O + H2O | 9.630 | 3.224 | 6.406 |
| rxn007038 | H transfer (C-H -> H-H) | C6H10O | C6H8O + H2 | 12.041 | 5.717 | 6.325 |
| rxn011123 | H transfer (C-H -> N-H) | C6H9N | C6H9N | 10.931 | 4.778 | 6.153 |

## Worst 50 reaction-energy errors

| Reaction | Class | Reactants | Products | Model | Reference | Error |
| --- | --- | --- | --- | --- | --- | --- |
| rxn006559 | multi-bond dissociation | C5H7NO | C5H7NO | -51.118 | -1.743 | -49.375 |
| rxn011394 | bond rearrangement | C6H8O | C6H8O | -33.937 | -1.426 | -32.511 |
| rxn000096 | multi-bond dissociation | C5H7N | C5H7N | -28.103 | 0.147 | -28.250 |
| rxn010742 | C-C dissociation | C6H6O | C6H6O | -27.430 | 0.511 | -27.941 |
| rxn011123 | H transfer (C-H -> N-H) | C6H9N | C6H9N | 21.550 | 2.229 | 19.321 |
| rxn009023 | bond rearrangement | C6H8O | C6H8O | -15.862 | 1.251 | -17.112 |
| rxn005625 | H transfer (C-H -> C-H) | C5H6O | C5H6O | -14.554 | 1.888 | -16.442 |
| rxn011223 | H transfer (C/N-H -> H-H) | C5H7NO | C5H5NO + H2 | -13.066 | 1.749 | -14.814 |
| rxn007809 | H transfer (C-H -> O-H) | C5H8O2 | C5H8O2 | -12.809 | 0.977 | -13.786 |
| rxn007313 | H transfer (C-H -> N-H) | C3H5N3O | C2H3NO + CH2N2 | -11.495 | 1.061 | -12.557 |
| rxn005576 | H transfer (C-H -> H-H) | C5H8O | C5H6O + H2 | -9.907 | 0.884 | -10.791 |
| rxn002413 | H transfer (C-H -> O-H) | C4H6O | C4H6O | -7.935 | 2.533 | -10.468 |
| rxn009384 | H transfer (C-H -> N-H) | C5H3NO | C5H3NO | 12.546 | 2.502 | 10.043 |
| rxn008195 | C-N formation | C5H5NO | C5H5NO | 13.988 | 3.975 | 10.013 |
| rxn009328 | H transfer (C-H -> C-H) | C5H8N2 | C5H8N2 | -8.611 | 0.560 | -9.171 |
| rxn002365 | H transfer (C-H -> C-H) | C5H8 | C5H8 | -7.239 | 1.711 | -8.950 |
| rxn000908 | bond rearrangement | C3H5N3 | C3H5N3 | -4.368 | 4.483 | -8.851 |
| rxn000941 | H transfer (C-H -> O-H) | C4H6O2 | C4H4O + H2O | 10.248 | 1.459 | 8.788 |
| rxn011847 | H transfer (C-H -> N-H) | C4H5N3 | C4H5N3 | 9.476 | 1.024 | 8.452 |
| rxn000920 | H transfer (C-H -> N-H) | C3H5N3 | C3H5N3 | 11.552 | 3.279 | 8.273 |
| rxn009321 | H transfer (C-H -> C-H) | C5H7NO | C3H3NO + C2H4 | 11.236 | 3.076 | 8.159 |
| rxn008341 | bond rearrangement | C6H6O | C6H6O | 8.459 | 0.447 | 8.012 |
| rxn011492 | H transfer (C-H -> C-H) | C5H8O2 | C5H8O2 | -5.891 | 1.757 | -7.649 |
| rxn005670 | H transfer (C-H -> N-H) | C4H6N2 | C4H6N2 | 10.369 | 2.809 | 7.560 |
| rxn000354 | H transfer (C-H -> C-H) | C2H3N3 | C2H3N3 | -3.947 | 3.483 | -7.431 |
| rxn001081 | H transfer (C-H -> N-H) | C4H6N2 | C4H6N2 | 10.847 | 3.432 | 7.416 |
| rxn008551 | H transfer (C-H -> N-H) | C4H6N2O | C4H6N2O | 10.630 | 3.476 | 7.153 |
| rxn002775 | C-N dissociation | C2H2N2O | C2H2O + N2 | -2.157 | 4.971 | -7.127 |
| rxn000408 | bond rearrangement | C2H4N4 | C2H4N2 + N2 | -6.125 | 0.996 | -7.121 |
| rxn008398 | H transfer (C-H -> C-H) | C6H12O | C6H12O | -5.376 | 1.660 | -7.037 |
| rxn010376 | H transfer (C-H -> C-H) | C5H6N2 | C5H6N2 | -4.076 | 2.934 | -7.010 |
| rxn005065 | H transfer (C-H -> C-H) | C4H8N2 | C3H4N2 + CH4 | 10.649 | 3.683 | 6.966 |
| rxn001662 | H transfer (C-H -> O-H) | C4H6O2 | C4H6O2 | -7.121 | -0.261 | -6.860 |
| rxn010264 | bond rearrangement | C5H4N2 | C5H4N2 | -3.020 | 3.718 | -6.739 |
| rxn002059 | H transfer (C-H -> C-H) | C4H4O | C4H4O | -5.125 | 1.564 | -6.688 |
| rxn010316 | H transfer (C-H -> C-H) | C5H10O2 | C5H10O2 | -5.007 | 1.669 | -6.676 |
| rxn007038 | H transfer (C-H -> H-H) | C6H10O | C6H8O + H2 | 9.779 | 3.138 | 6.640 |
| rxn000746 | bond rearrangement | C4H5NO | C4H5NO | 9.078 | 2.460 | 6.618 |
| rxn001855 | H transfer (C-H -> O-H) | C5H8O | C5H8O | 7.375 | 0.909 | 6.466 |
| rxn011568 | H transfer (C-H -> O-H) | C5H11NO | C5H11NO | 7.557 | 1.151 | 6.405 |
| rxn004948 | H transfer (O-H -> O-H) | C3H6O2 | C3H6O2 | 6.415 | 0.023 | 6.392 |
| rxn003773 | H transfer (C-H -> N-H) | C2H3N3 | C2H3N3 | -4.005 | 2.366 | -6.371 |
| rxn004311 | H transfer (C-H -> O-H) | C5H10O | C5H8 + H2O | 7.092 | 0.732 | 6.360 |
| rxn007149 | H transfer (C-H -> O-H) | C5H8O2 | C3H4O + C2H4O | 8.314 | 2.472 | 5.842 |
| rxn002799 | bond rearrangement | C4H6O2 | C4H6O2 | 5.653 | -0.027 | 5.680 |
| rxn003494 | bond rearrangement | C4H6O2 | C3H4O + CH2O | -1.601 | 4.065 | -5.666 |
| rxn009555 | H transfer (C-H -> C-H) | C6H10O | C6H10O | -2.876 | 2.776 | -5.652 |
| rxn007485 | H transfer (C/N-H -> H-H) | C4H6N2O | C4H4N2O + H2 | 8.939 | 3.460 | 5.478 |
| rxn003095 | multi-bond dissociation | CHN3O2 | CHNO + N2O | -5.578 | -0.226 | -5.352 |
| rxn000322 | H transfer (C-H -> H-H) | C3H5NO | C2H3N + CO + H2 | 7.599 | 2.384 | 5.215 |

## Barrier-sign failures

| Reaction | Class | Reactants | Products | Model | Reference | Error |
| --- | --- | --- | --- | --- | --- | --- |
| rxn000096 | multi-bond dissociation | C5H7N | C5H7N | -18.542 | 3.426 | -21.968 |
| rxn000354 | H transfer (C-H -> C-H) | C2H3N3 | C2H3N3 | -0.383 | 3.741 | -4.124 |
| rxn000409 | N-N dissociation | C2H4N4 | C2H4N4 | -1.166 | 1.587 | -2.753 |
| rxn000445 | H transfer (O-H -> C-H) | C3H4N2O | C3H4N2O | -0.096 | 3.299 | -3.395 |
| rxn000501 | H transfer (C-H -> C-H) | C4H6O | C4H6O | -3.024 | 3.398 | -6.421 |
| rxn001097 | bond rearrangement | C4H5NO | C4H5NO | -1.909 | 3.046 | -4.955 |
| rxn002209 | multi-bond dissociation | C2H3N3O | CH2N2 + CHNO | -1.496 | 3.521 | -5.018 |
| rxn002413 | H transfer (C-H -> O-H) | C4H6O | C4H6O | -0.089 | 3.342 | -3.431 |
| rxn003095 | multi-bond dissociation | CHN3O2 | CHNO + N2O | -5.093 | 1.940 | -7.033 |
| rxn003577 | multi-bond dissociation | C4H8O2 | C3H6O + CH2O | -1.991 | 2.975 | -4.966 |
| rxn003773 | H transfer (C-H -> N-H) | C2H3N3 | C2H3N3 | -3.257 | 3.699 | -6.957 |
| rxn005576 | H transfer (C-H -> H-H) | C5H8O | C5H6O + H2 | -1.054 | 6.851 | -7.905 |
| rxn006559 | multi-bond dissociation | C5H7NO | C5H7NO | -33.218 | 2.284 | -35.502 |
| rxn007313 | H transfer (C-H -> N-H) | C3H5N3O | C2H3NO + CH2N2 | -3.920 | 4.874 | -8.794 |
| rxn008209 | H transfer (C-H -> C-H) | C6H8O | C6H8O | -0.253 | 4.361 | -4.614 |
| rxn008398 | H transfer (C-H -> C-H) | C6H12O | C6H12O | -1.081 | 3.162 | -4.243 |
| rxn008404 | H transfer (C-H -> C-H) | C6H12O | C5H8O + CH4 | -0.318 | 4.023 | -4.342 |
| rxn009328 | H transfer (C-H -> C-H) | C5H8N2 | C5H8N2 | -2.893 | 4.429 | -7.322 |
| rxn009555 | H transfer (C-H -> C-H) | C6H10O | C6H10O | -1.780 | 3.701 | -5.481 |
| rxn010316 | H transfer (C-H -> C-H) | C5H10O2 | C5H10O2 | -1.764 | 3.032 | -4.796 |
| rxn010376 | H transfer (C-H -> C-H) | C5H6N2 | C5H6N2 | -0.307 | 4.071 | -4.378 |
| rxn010742 | C-C dissociation | C6H6O | C6H6O | -17.710 | 1.862 | -19.573 |
| rxn011223 | H transfer (C/N-H -> H-H) | C5H7NO | C5H5NO + H2 | -5.762 | 5.704 | -11.465 |
| rxn011394 | bond rearrangement | C6H8O | C6H8O | -8.494 | 3.789 | -12.283 |
| rxn011450 | H transfer (N-H -> N-H) | C2H4N4O | C2H4N4O | -1.020 | 2.167 | -3.187 |
| rxn011492 | H transfer (C-H -> C-H) | C5H8O2 | C5H8O2 | -2.761 | 3.165 | -5.926 |
