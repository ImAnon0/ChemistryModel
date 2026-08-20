# Grambow component-bias audit

Correlations and frozen-endpoint term removals identify candidate causes only; they are not parameter fits or acceptable production changes.

## Barrier

Current MAE 4.519547 eV; RMSE 6.483076 eV; signed mean +1.884328 eV.

### Signed-error correlations

| Component | r(error, component) | r(abs error, abs component) | Mean abs contribution |
| --- | --- | --- | --- |
| base_overcoordination | +0.8434 | +0.7615 | 6.3960 eV |
| valence_topology_correction | -0.8147 | +0.6781 | 1.6227 eV |
| base_angle | +0.8021 | +0.6802 | 2.8050 eV |
| base_bond | -0.3317 | +0.0771 | 2.6545 eV |
| effective_angle_total | +0.3240 | +0.1911 | 1.6145 eV |
| h_state_correction | -0.1781 | +0.0423 | 1.4397 eV |

### Frozen-endpoint sensitivity (diagnostic only)

| Scenario | MAE | RMSE | Signed mean | Sign agreement |
| --- | --- | --- | --- | --- |
| current | 4.5195 | 6.4831 | +1.8843 | 87.0% |
| remove_heavy_base_overcoordination_diagnostic | 2.2452 | 2.8274 | +0.2502 | 95.0% |
| remove_all_base_overcoordination_diagnostic | 4.1733 | 5.1962 | -2.6154 | 59.5% |
| remove_effective_angle_diagnostic | 3.8010 | 5.9167 | +0.7413 | 86.5% |
| remove_h_state_correction_diagnostic | 5.5058 | 7.6599 | +3.2473 | 92.0% |
| remove_heavy_over_and_effective_angle_diagnostic | 2.0957 | 2.6261 | -0.8928 | 92.5% |

## Reaction

Current MAE 4.405767 eV; RMSE 7.068841 eV; signed mean -0.138715 eV.

### Signed-error correlations

| Component | r(error, component) | r(abs error, abs component) | Mean abs contribution |
| --- | --- | --- | --- |
| valence_topology_correction | -0.8928 | +0.8515 | 1.7713 eV |
| base_overcoordination | +0.8870 | +0.8841 | 3.5817 eV |
| base_angle | +0.8713 | +0.8370 | 3.2141 eV |
| effective_angle_total | +0.3828 | +0.2230 | 1.8023 eV |
| base_bond | -0.3086 | +0.4129 | 3.1438 eV |
| h_state_correction | -0.1018 | +0.0719 | 0.0000 eV |

### Frozen-endpoint sensitivity (diagnostic only)

| Scenario | MAE | RMSE | Signed mean | Sign agreement |
| --- | --- | --- | --- | --- |
| current | 4.4058 | 7.0688 | -0.1387 | 73.5% |
| remove_heavy_base_overcoordination_diagnostic | 3.0499 | 3.9592 | +1.0459 | 72.5% |
| remove_all_base_overcoordination_diagnostic | 3.0499 | 3.9592 | +1.0459 | 72.5% |
| remove_effective_angle_diagnostic | 3.8382 | 6.5332 | -0.1908 | 70.0% |
| remove_h_state_correction_diagnostic | 4.4058 | 7.0688 | -0.1387 | 73.5% |
| remove_heavy_over_and_effective_angle_diagnostic | 2.6050 | 3.5134 | +0.9938 | 75.5% |
