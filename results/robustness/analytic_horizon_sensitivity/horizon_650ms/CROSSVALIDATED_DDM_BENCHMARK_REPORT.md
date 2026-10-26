# Strict stacked RL-to-DDM benchmark

## Main cumulative-horizon summary

|   horizon_min_prior_trials | variant   | variant_name                      |   n_folds |   total_elpd |   mean_elpd_per_trial |   mean_observed_press |   mean_predicted_press |   mean_observed_fa |   mean_predicted_fa |   mean_observed_hit |   mean_predicted_hit |   mean_abs_press_error |   mean_abs_fa_error |   mean_abs_hit_error |
|---------------------------:|:----------|:----------------------------------|----------:|-------------:|----------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|---------------------:|
|                          0 | D9        | actor_control_plus_compact_lagged_history |         5 |     -731.611 |            -0.0504916 |              0.648907 |               0.654059 |           0.355494 |            0.356893 |            0.743967 |             0.750371 |             0.00515202 |          0.00139956 |           0.00640385 |
|                          0 | D15       | policy_plus_compact_lagged_history        |         5 |     -736.795 |            -0.0508395 |              0.648907 |               0.655008 |           0.355494 |            0.358383 |            0.743967 |             0.751144 |             0.00610062 |          0.00288909 |           0.00717674 |
|                          0 | D14       | actor_control_plus_compact_lagged_history   |         5 |     -791.066 |            -0.0545087 |              0.648907 |               0.6568   |           0.355494 |            0.360678 |            0.743967 |             0.752773 |             0.00789288 |          0.00518413 |           0.0088056  |
|                          0 | D6        | online_behavioral_summary_control |         5 |     -792.739 |            -0.0546053 |              0.648907 |               0.656811 |           0.355494 |            0.360752 |            0.743967 |             0.752764 |             0.00790443 |          0.00525798 |           0.00879723 |

## Planned subject-clustered contrasts

| model   | reference   | contrast                                     |   n_subjects |   mean_subject_delta_elpd |   bootstrap_ci_low |   bootstrap_ci_high |   positive_subjects |   total_delta_elpd |
|:--------|:------------|:---------------------------------------------|-------------:|--------------------------:|-------------------:|--------------------:|--------------------:|-------------------:|
| D14     | D6          | actor/control beyond history without p_press |           40 |                 0.0418322 |          -0.725693 |            0.815152 |                  23 |            1.67329 |
| D9      | D14         | p_press beyond actor/control plus history    |           40 |                 1.48638   |           0.169074 |            2.68846  |                  27 |           59.4552  |
| D9      | D15         | actor/control beyond p_press plus history    |           40 |                 0.12961   |          -0.782808 |            0.9142   |                  25 |            5.18441 |
| D15     | D6          | p_press beyond history                       |           40 |                 1.3986    |           0.36681  |            2.42573  |                  27 |           55.9441  |
| D9      | D6          | total RL information beyond history          |           40 |                 1.52821   |           0.44061  |            2.59937  |                  29 |           61.1285  |

## Optimizer summary

|   fold | variant   | variant_name                      | success   | message                                              |   nll_train |   n_params |   n_successful_starts |   n_starts |   best_start_nll |   worst_start_nll |
|-------:|:----------|:----------------------------------|:----------|:-----------------------------------------------------|------------:|-----------:|----------------------:|-----------:|-----------------:|------------------:|
|      0 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     732.401 |         18 |                     6 |          6 |          732.401 |           732.401 |
|      0 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     720.241 |         20 |                     6 |          6 |          720.241 |           720.241 |
|      0 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     678.398 |         19 |                     6 |          6 |          678.398 |           678.398 |
|      0 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     664.823 |         21 |                     6 |          6 |          664.823 |           664.823 |
|      1 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     201.589 |         18 |                     6 |          6 |          201.589 |           201.589 |
|      1 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     185.105 |         20 |                     6 |          6 |          185.105 |           185.105 |
|      1 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     154.645 |         19 |                     6 |          6 |          154.645 |           154.645 |
|      1 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     147.63  |         21 |                     6 |          6 |          147.63  |           147.63  |
|      2 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     382.927 |         18 |                     6 |          6 |          382.927 |           382.927 |
|      2 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     369.626 |         20 |                     6 |          6 |          369.626 |           369.627 |
|      2 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     311.217 |         19 |                     6 |          6 |          311.217 |           311.217 |
|      2 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     292.552 |         21 |                     6 |          6 |          292.552 |           292.553 |
|      3 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     489.142 |         18 |                     6 |          6 |          489.142 |           489.142 |
|      3 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     482.24  |         20 |                     6 |          6 |          482.24  |           482.24  |
|      3 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     431.439 |         19 |                     6 |          6 |          431.439 |           431.439 |
|      3 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     404.498 |         21 |                     6 |          6 |          404.498 |           404.499 |
|      4 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     315.104 |         18 |                     6 |          6 |          315.104 |           315.104 |
|      4 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     297.572 |         20 |                     6 |          6 |          297.572 |           297.572 |
|      4 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     254.997 |         19 |                     6 |          6 |          254.997 |           254.997 |
|      4 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     238.463 |         21 |                     6 |          6 |          238.463 |           238.463 |

## Interpretation rules

- D14 > D6 tests actor-value/control information beyond online behavioral history without p_press.
- D9 > D14 tests policy probability beyond actor-value/control plus history.
- D9 > D15 tests actor-value/control beyond policy probability plus history.
- D15 > D6 tests policy probability beyond history.
- D11 > D10 tests strictly lagged empirical updating beyond instructed cue probabilities.
- Use horizon 0 as confirmatory; later cumulative horizons are warm-start sensitivities.
- Inspect optimizer success and multi-start stability before interpreting ELPD contrasts.