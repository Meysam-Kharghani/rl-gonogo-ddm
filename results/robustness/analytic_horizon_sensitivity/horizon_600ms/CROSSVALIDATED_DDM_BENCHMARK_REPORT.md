# Strict stacked RL-to-DDM benchmark

## Main cumulative-horizon summary

|   horizon_min_prior_trials | variant   | variant_name                      |   n_folds |   total_elpd |   mean_elpd_per_trial |   mean_observed_press |   mean_predicted_press |   mean_observed_fa |   mean_predicted_fa |   mean_observed_hit |   mean_predicted_hit |   mean_abs_press_error |   mean_abs_fa_error |   mean_abs_hit_error |
|---------------------------:|:----------|:----------------------------------|----------:|-------------:|----------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|---------------------:|
|                          0 | D15       | policy_plus_compact_lagged_history        |         5 |     -521.923 |            -0.0362543 |              0.625157 |               0.624511 |           0.352442 |            0.357069 |            0.713506 |             0.711192 |            0.000645346 |          0.00462751 |          0.00231406  |
|                          0 | D9        | actor_control_plus_compact_lagged_history |         5 |     -522.645 |            -0.0363045 |              0.625157 |               0.623584 |           0.352442 |            0.355689 |            0.713506 |             0.710411 |            0.00157271  |          0.00324739 |          0.00309449  |
|                          0 | D14       | actor_control_plus_compact_lagged_history   |         5 |     -572.075 |            -0.0396496 |              0.625157 |               0.626257 |           0.352442 |            0.359286 |            0.713506 |             0.712784 |            0.00110025  |          0.00684371 |          0.000721528 |
|                          0 | D6        | online_behavioral_summary_control |         5 |     -574.183 |            -0.0397759 |              0.625157 |               0.626291 |           0.352442 |            0.359396 |            0.713506 |             0.712793 |            0.001134    |          0.00695405 |          0.000712369 |

## Planned subject-clustered contrasts

| model   | reference   | contrast                                     |   n_subjects |   mean_subject_delta_elpd |   bootstrap_ci_low |   bootstrap_ci_high |   positive_subjects |   total_delta_elpd |
|:--------|:------------|:---------------------------------------------|-------------:|--------------------------:|-------------------:|--------------------:|--------------------:|-------------------:|
| D14     | D6          | actor/control beyond history without p_press |           40 |                 0.0527102 |        -0.737329   |            0.857405 |                  23 |           2.10841  |
| D9      | D14         | p_press beyond actor/control plus history    |           40 |                 1.23575   |        -0.00392126 |            2.38254  |                  26 |          49.43     |
| D9      | D15         | actor/control beyond p_press plus history    |           40 |                -0.0180445 |        -0.863191   |            0.716788 |                  23 |          -0.721779 |
| D15     | D6          | p_press beyond history                       |           40 |                 1.30651   |         0.321062   |            2.29052  |                  25 |          52.2602   |
| D9      | D6          | total RL information beyond history          |           40 |                 1.28846   |         0.255798   |            2.30431  |                  27 |          51.5384   |

## Optimizer summary

|   fold | variant   | variant_name                      | success   | message                                              |   nll_train |   n_params |   n_successful_starts |   n_starts |   best_start_nll |   worst_start_nll |
|-------:|:----------|:----------------------------------|:----------|:-----------------------------------------------------|------------:|-----------:|----------------------:|-----------:|-----------------:|------------------:|
|      0 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   522.746   |         18 |                     6 |          6 |        522.746   |         522.746   |
|      0 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   509.163   |         20 |                     6 |          6 |        509.163   |         509.163   |
|      0 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   474.222   |         19 |                     6 |          6 |        474.222   |         474.222   |
|      0 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   463.033   |         21 |                     6 |          6 |        463.033   |         463.033   |
|      1 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    24.3818  |         18 |                     6 |          6 |         24.3818  |          24.3819  |
|      1 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     6.45016 |         20 |                     6 |          6 |          6.45016 |           6.45017 |
|      1 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   -19.5607  |         19 |                     6 |          6 |        -19.5607  |         -19.5607  |
|      1 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   -24.6012  |         21 |                     6 |          6 |        -24.6012  |         -24.6011  |
|      2 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   190.809   |         18 |                     6 |          6 |        190.809   |         190.809   |
|      2 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   175.383   |         20 |                     6 |          6 |        175.383   |         175.384   |
|      2 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   127.479   |         19 |                     6 |          6 |        127.479   |         127.479   |
|      2 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   112.875   |         21 |                     6 |          6 |        112.875   |         112.875   |
|      3 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   305.23    |         18 |                     6 |          6 |        305.23    |         305.23    |
|      3 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   297.095   |         20 |                     6 |          6 |        297.095   |         297.095   |
|      3 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   251.137   |         19 |                     6 |          6 |        251.137   |         251.137   |
|      3 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   228.474   |         21 |                     6 |          6 |        228.474   |         228.474   |
|      4 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   123.228   |         18 |                     6 |          6 |        123.228   |         123.228   |
|      4 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |   106.236   |         20 |                     6 |          6 |        106.236   |         106.236   |
|      4 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    69.6376  |         19 |                     6 |          6 |         69.6376  |          69.6376  |
|      4 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    57.0332  |         21 |                     6 |          6 |         57.0332  |          57.0332  |

## Interpretation rules

- D14 > D6 tests actor-value/control information beyond online behavioral history without p_press.
- D9 > D14 tests policy probability beyond actor-value/control plus history.
- D9 > D15 tests actor-value/control beyond policy probability plus history.
- D15 > D6 tests policy probability beyond history.
- D11 > D10 tests strictly lagged empirical updating beyond instructed cue probabilities.
- Use horizon 0 as confirmatory; later cumulative horizons are warm-start sensitivities.
- Inspect optimizer success and multi-start stability before interpreting ELPD contrasts.