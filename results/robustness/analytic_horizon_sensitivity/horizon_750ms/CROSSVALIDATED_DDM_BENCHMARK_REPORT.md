# Strict stacked RL-to-DDM benchmark

## Main cumulative-horizon summary

|   horizon_min_prior_trials | variant   | variant_name                      |   n_folds |   total_elpd |   mean_elpd_per_trial |   mean_observed_press |   mean_predicted_press |   mean_observed_fa |   mean_predicted_fa |   mean_observed_hit |   mean_predicted_hit |   mean_abs_press_error |   mean_abs_fa_error |   mean_abs_hit_error |
|---------------------------:|:----------|:----------------------------------|----------:|-------------:|----------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|---------------------:|
|                          0 | D9        | actor_control_plus_compact_lagged_history |         5 |     -1305.83 |            -0.0893619 |              0.674086 |               0.691743 |           0.358502 |            0.355334 |            0.776334 |             0.800772 |              0.0176575 |         0.00316861  |            0.0244381 |
|                          0 | D15       | policy_plus_compact_lagged_history        |         5 |     -1314.55 |            -0.089945  |              0.674086 |               0.692475 |           0.358502 |            0.356743 |            0.776334 |             0.801284 |              0.0183893 |         0.00175936  |            0.0249498 |
|                          0 | D14       | actor_control_plus_compact_lagged_history   |         5 |     -1374.88 |            -0.0940175 |              0.674086 |               0.694215 |           0.358502 |            0.359111 |            0.776334 |             0.80282  |              0.020129  |         0.000608174 |            0.0264858 |
|                          0 | D6        | online_behavioral_summary_control |         5 |     -1377.86 |            -0.0942025 |              0.674086 |               0.694178 |           0.358502 |            0.359055 |            0.776334 |             0.802789 |              0.0200918 |         0.000552181 |            0.0264551 |

## Planned subject-clustered contrasts

| model   | reference   | contrast                                     |   n_subjects |   mean_subject_delta_elpd |   bootstrap_ci_low |   bootstrap_ci_high |   positive_subjects |   total_delta_elpd |
|:--------|:------------|:---------------------------------------------|-------------:|--------------------------:|-------------------:|--------------------:|--------------------:|-------------------:|
| D14     | D6          | actor/control beyond history without p_press |           40 |                 0.0744292 |          -0.677181 |            0.844749 |                  22 |            2.97717 |
| D9      | D14         | p_press beyond actor/control plus history    |           40 |                 1.72641   |           0.376928 |            2.95815  |                  29 |           69.0565  |
| D9      | D15         | actor/control beyond p_press plus history    |           40 |                 0.218033  |          -0.713203 |            1.02113  |                  24 |            8.7213  |
| D15     | D6          | p_press beyond history                       |           40 |                 1.58281   |           0.496763 |            2.65936  |                  25 |           63.3124  |
| D9      | D6          | total RL information beyond history          |           40 |                 1.80084   |           0.651383 |            2.94142  |                  27 |           72.0337  |

## Optimizer summary

|   fold | variant   | variant_name                      | success   | message                                              |   nll_train |   n_params |   n_successful_starts |   n_starts |   best_start_nll |   worst_start_nll |
|-------:|:----------|:----------------------------------|:----------|:-----------------------------------------------------|------------:|-----------:|----------------------:|-----------:|-----------------:|------------------:|
|      0 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1244.51  |         18 |                     6 |          6 |         1244.51  |          1244.51  |
|      0 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1232.41  |         20 |                     6 |          6 |         1232.41  |          1232.41  |
|      0 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1183.11  |         19 |                     6 |          6 |         1183.11  |          1183.11  |
|      0 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1168.23  |         21 |                     6 |          6 |         1168.23  |          1168.23  |
|      1 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     696.161 |         18 |                     6 |          6 |          696.161 |           696.161 |
|      1 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     678.324 |         20 |                     6 |          6 |          678.324 |           678.324 |
|      1 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     641.869 |         19 |                     6 |          6 |          641.869 |           641.869 |
|      1 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     634.032 |         21 |                     6 |          6 |          634.032 |           634.032 |
|      2 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     871.692 |         18 |                     6 |          6 |          871.692 |           871.692 |
|      2 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     858.975 |         20 |                     6 |          6 |          858.975 |           858.975 |
|      2 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     787.862 |         19 |                     6 |          6 |          787.862 |           787.862 |
|      2 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     765.505 |         21 |                     6 |          6 |          765.505 |           765.505 |
|      3 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     959.573 |         18 |                     6 |          6 |          959.573 |           959.573 |
|      3 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     952.3   |         20 |                     6 |          6 |          952.3   |           952.3   |
|      3 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     896.325 |         19 |                     6 |          6 |          896.325 |           896.325 |
|      3 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     869.519 |         21 |                     6 |          6 |          869.519 |           869.52  |
|      4 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     843.222 |         18 |                     6 |          6 |          843.222 |           843.222 |
|      4 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     825.218 |         20 |                     6 |          6 |          825.218 |           825.218 |
|      4 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     772.956 |         19 |                     6 |          6 |          772.956 |           772.956 |
|      4 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     753.831 |         21 |                     6 |          6 |          753.831 |           753.831 |

## Interpretation rules

- D14 > D6 tests actor-value/control information beyond online behavioral history without p_press.
- D9 > D14 tests policy probability beyond actor-value/control plus history.
- D9 > D15 tests actor-value/control beyond policy probability plus history.
- D15 > D6 tests policy probability beyond history.
- D11 > D10 tests strictly lagged empirical updating beyond instructed cue probabilities.
- Use horizon 0 as confirmatory; later cumulative horizons are warm-start sensitivities.
- Inspect optimizer success and multi-start stability before interpreting ELPD contrasts.