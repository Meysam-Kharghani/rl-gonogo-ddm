# Strict stacked RL-to-DDM benchmark

## Main cumulative-horizon summary

|   horizon_min_prior_trials | variant   | variant_name                      |   n_folds |   total_elpd |   mean_elpd_per_trial |   mean_observed_press |   mean_predicted_press |   mean_observed_fa |   mean_predicted_fa |   mean_observed_hit |   mean_predicted_hit |   mean_abs_press_error |   mean_abs_fa_error |   mean_abs_hit_error |
|---------------------------:|:----------|:----------------------------------|----------:|-------------:|----------------------:|----------------------:|-----------------------:|-------------------:|--------------------:|--------------------:|---------------------:|-----------------------:|--------------------:|---------------------:|
|                          0 | D9        | actor_control_plus_compact_lagged_history |         5 |     -991.319 |            -0.0680713 |              0.664874 |               0.676445 |           0.357942 |            0.357174 |            0.764317 |             0.779921 |              0.0115714 |         0.000768152 |            0.0156036 |
|                          0 | D15       | policy_plus_compact_lagged_history        |         5 |     -998.952 |            -0.0685828 |              0.664874 |               0.67728  |           0.357942 |            0.358617 |            0.764317 |             0.780557 |              0.0124057 |         0.00067559  |            0.0162399 |
|                          0 | D14       | actor_control_plus_compact_lagged_history   |         5 |    -1057.11  |            -0.072511  |              0.664874 |               0.679069 |           0.357942 |            0.360988 |            0.764317 |             0.782158 |              0.0141955 |         0.00304654  |            0.0178411 |
|                          0 | D6        | online_behavioral_summary_control |         5 |    -1059.77  |            -0.0726747 |              0.664874 |               0.679049 |           0.357942 |            0.36099  |            0.764317 |             0.782131 |              0.014175  |         0.00304838  |            0.0178137 |

## Planned subject-clustered contrasts

| model   | reference   | contrast                                     |   n_subjects |   mean_subject_delta_elpd |   bootstrap_ci_low |   bootstrap_ci_high |   positive_subjects |   total_delta_elpd |
|:--------|:------------|:---------------------------------------------|-------------:|--------------------------:|-------------------:|--------------------:|--------------------:|-------------------:|
| D14     | D6          | actor/control beyond history without p_press |           40 |                 0.0666197 |          -0.698765 |            0.844351 |                  24 |            2.66479 |
| D9      | D14         | p_press beyond actor/control plus history    |           40 |                 1.64472   |           0.306415 |            2.87746  |                  28 |           65.7886  |
| D9      | D15         | actor/control beyond p_press plus history    |           40 |                 0.19084   |          -0.737611 |            0.996037 |                  25 |            7.63358 |
| D15     | D6          | p_press beyond history                       |           40 |                 1.5205    |           0.447769 |            2.56965  |                  26 |           60.8198  |
| D9      | D6          | total RL information beyond history          |           40 |                 1.71134   |           0.587014 |            2.82312  |                  27 |           68.4534  |

## Optimizer summary

|   fold | variant   | variant_name                      | success   | message                                              |   nll_train |   n_params |   n_successful_starts |   n_starts |   best_start_nll |   worst_start_nll |
|-------:|:----------|:----------------------------------|:----------|:-----------------------------------------------------|------------:|-----------:|----------------------:|-----------:|-----------------:|------------------:|
|      0 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     967.78  |         18 |                     6 |          6 |          967.78  |           967.78  |
|      0 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     955.691 |         20 |                     6 |          6 |          955.691 |           955.691 |
|      0 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     909.71  |         19 |                     6 |          6 |          909.71  |           909.71  |
|      0 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     895.184 |         21 |                     6 |          6 |          895.184 |           895.184 |
|      1 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     425.057 |         18 |                     6 |          6 |          425.057 |           425.057 |
|      1 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     407.678 |         20 |                     6 |          6 |          407.678 |           407.678 |
|      1 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     373.723 |         19 |                     6 |          6 |          373.723 |           373.723 |
|      1 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     366.121 |         21 |                     6 |          6 |          366.121 |           366.121 |
|      2 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     608.307 |         18 |                     6 |          6 |          608.307 |           608.307 |
|      2 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     595.351 |         20 |                     6 |          6 |          595.351 |           595.351 |
|      2 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     530.407 |         19 |                     6 |          6 |          530.407 |           530.407 |
|      2 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     509.751 |         21 |                     6 |          6 |          509.751 |           509.751 |
|      3 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     706.927 |         18 |                     6 |          6 |          706.927 |           706.927 |
|      3 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     699.606 |         20 |                     6 |          6 |          699.606 |           699.606 |
|      3 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     645.347 |         19 |                     6 |          6 |          645.347 |           645.347 |
|      3 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     618.314 |         21 |                     6 |          6 |          618.314 |           618.314 |
|      4 | D6        | online_behavioral_summary_control | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     562.163 |         18 |                     6 |          6 |          562.163 |           562.163 |
|      4 | D14       | actor_control_plus_compact_lagged_history   | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     544.155 |         20 |                     6 |          6 |          544.155 |           544.155 |
|      4 | D15       | policy_plus_compact_lagged_history        | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     495.93  |         19 |                     6 |          6 |          495.93  |           495.931 |
|      4 | D9        | actor_control_plus_compact_lagged_history | True      | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |     477.902 |         21 |                     6 |          6 |          477.902 |           477.902 |

## Interpretation rules

- D14 > D6 tests actor-value/control information beyond online behavioral history without p_press.
- D9 > D14 tests policy probability beyond actor-value/control plus history.
- D9 > D15 tests actor-value/control beyond policy probability plus history.
- D15 > D6 tests policy probability beyond history.
- D11 > D10 tests strictly lagged empirical updating beyond instructed cue probabilities.
- Use horizon 0 as confirmatory; later cumulative horizons are warm-start sensitivities.
- Inspect optimizer success and multi-start stability before interpreting ELPD contrasts.