# Strict stacked RL-to-DDM benchmark

## Main cumulative-horizon summary

|   horizon_min_prior_trials | variant   | variant_name                      |   n_folds |   total_elpd |   mean_elpd_per_trial |   mean_observed_press |   mean_predicted_press |   mean_observed_fa |   mean_predicted_fa |   mean_observed_hit |   mean_predicted_hit |   mean_abs_press_error |   mean_abs_fa_error |   mean_abs_hit_error |
|---------------------------:|:----------|:----------------------------------|----------:|-------------:|----------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|---------------------:|
|                          0 | D9        | actor_control_plus_compact_lagged_history |         5 |     -1496.43 |             -0.102279 |              0.678612 |               0.699533 |           0.359334 |            0.355137 |            0.782059 |             0.81115  |              0.0209207 |         0.00419711  |            0.0290909 |
|                          0 | D15       | policy_plus_compact_lagged_history        |         5 |     -1505.58 |             -0.102891 |              0.678612 |               0.700185 |           0.359334 |            0.356457 |            0.782059 |             0.811585 |              0.0215732 |         0.00287689  |            0.0295264 |
|                          0 | D14       | actor_control_plus_compact_lagged_history   |         5 |     -1566.54 |             -0.107003 |              0.678612 |               0.701822 |           0.359334 |            0.358715 |            0.782059 |             0.81302  |              0.0232095 |         0.000619457 |            0.0309612 |
|                          0 | D6        | online_behavioral_summary_control |         5 |     -1569.37 |             -0.107178 |              0.678612 |               0.701771 |           0.359334 |            0.358632 |            0.782059 |             0.81298  |              0.0231592 |         0.000702    |            0.0309219 |

## Planned subject-clustered contrasts

| model   | reference   | contrast                                     |   n_subjects |   mean_subject_delta_elpd |   bootstrap_ci_low |   bootstrap_ci_high |   positive_subjects |   total_delta_elpd |
|:--------|:------------|:---------------------------------------------|-------------:|--------------------------:|-------------------:|--------------------:|--------------------:|-------------------:|
| D14     | D6          | actor/control beyond history without p_press |           40 |                 0.0708269 |          -0.672841 |            0.829255 |                  22 |            2.83307 |
| D9      | D14         | p_press beyond actor/control plus history    |           40 |                 1.75276   |           0.396985 |            3.00313  |                  29 |           70.1105  |
| D9      | D15         | actor/control beyond p_press plus history    |           40 |                 0.228842  |          -0.680033 |            1.02565  |                  24 |            9.1537  |
| D15     | D6          | p_press beyond history                       |           40 |                 1.59475   |           0.492636 |            2.67706  |                  24 |           63.7899  |
| D9      | D6          | total RL information beyond history          |           40 |                 1.82359   |           0.669037 |            2.9767   |                  27 |           72.9436  |

## Optimizer summary

|   fold | variant   | variant_name                      | success   | message                                              |   nll_train |   n_params |   n_successful_starts |   n_starts |   best_start_nll |   worst_start_nll |
|-------:|:----------|:----------------------------------|:----------|:-----------------------------------------------------|------------:|-----------:|----------------------:|-----------:|-----------------:|------------------:|
|      0 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1405     |         18 |                     6 |          6 |         1405     |          1405     |
|      0 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1393.2   |         20 |                     6 |          6 |         1393.2   |          1393.2   |
|      0 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1343.85  |         19 |                     6 |          6 |         1343.85  |          1343.85  |
|      0 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1329.52  |         21 |                     6 |          6 |         1329.52  |          1329.52  |
|      1 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     857.846 |         18 |                     6 |          6 |          857.846 |           857.846 |
|      1 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     839.956 |         20 |                     6 |          6 |          839.956 |           839.956 |
|      1 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     802.772 |         19 |                     6 |          6 |          802.772 |           802.772 |
|      1 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     795.034 |         21 |                     6 |          6 |          795.034 |           795.034 |
|      2 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1028.08  |         18 |                     6 |          6 |         1028.08  |          1028.08  |
|      2 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1015.63  |         20 |                     6 |          6 |         1015.63  |          1015.63  |
|      2 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     944.157 |         19 |                     6 |          6 |          944.157 |           944.158 |
|      2 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     922.451 |         21 |                     6 |          6 |          922.451 |           922.451 |
|      3 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1120.08  |         18 |                     6 |          6 |         1120.08  |          1120.08  |
|      3 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1112.99  |         20 |                     6 |          6 |         1112.99  |          1112.99  |
|      3 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1057.26  |         19 |                     6 |          6 |         1057.26  |          1057.26  |
|      3 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1031.24  |         21 |                     6 |          6 |         1031.24  |          1031.24  |
|      4 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |    1004.47  |         18 |                     6 |          6 |         1004.47  |          1004.47  |
|      4 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     986.705 |         20 |                     6 |          6 |          986.705 |           986.705 |
|      4 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     932.817 |         19 |                     6 |          6 |          932.817 |           932.817 |
|      4 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     912.965 |         21 |                     6 |          6 |          912.965 |           912.965 |

## Interpretation rules

- D14 > D6 tests actor-value/control information beyond online behavioral history without p_press.
- D9 > D14 tests policy probability beyond actor-value/control plus history.
- D9 > D15 tests actor-value/control beyond policy probability plus history.
- D15 > D6 tests policy probability beyond history.
- D11 > D10 tests strictly lagged empirical updating beyond instructed cue probabilities.
- Use horizon 0 as confirmatory; later cumulative horizons are warm-start sensitivities.
- Inspect optimizer success and multi-start stability before interpreting ELPD contrasts.