# M6 hierarchical RL parameter recovery

## Parameter summary

| parameter                 |   n_elements |   coverage_95 |   mean_absolute_error |   source_sd |   recovered_sd |   mean_recovery_r |   min_recovery_r |   n_reps |
|:--------------------------|-------------:|--------------:|----------------------:|------------:|---------------:|------------------:|-----------------:|---------:|
| alpha_actor               |          400 |      0.955    |            0.0725887  |    0.10143  |     0.0697566  |          0.556185 |         0.424527 |       10 |
| alpha_ctrl                |          400 |      0.9575   |            0.0941804  |    0.144309 |     0.0900323  |          0.704083 |         0.594447 |       10 |
| b_cue                     |           30 |      0.966667 |            0.0363831  |    0.128621 |     0.121567   |        nan        |       nan        |      nan |
| b_nogo                    |           10 |      1        |            0.0338019  |    0        |     0.0410831  |        nan        |       nan        |      nan |
| b_prev_error              |           10 |      0.9      |            0.0658751  |    0        |     0.0836126  |        nan        |       nan        |      nan |
| beta_actor                |          400 |      0.955    |            0.292167   |    0.680638 |     0.572361   |          0.824876 |         0.683575 |       10 |
| bias                      |          400 |      0.945    |            0.170557   |    0.58135  |     0.575202   |          0.942414 |         0.914122 |       10 |
| delta_alpha_actor_group   |           10 |      1        |            0.0824585  |    0        |     0.091688   |        nan        |       nan        |      nan |
| delta_alpha_ctrl_group    |           10 |      1        |            0.117341   |    0        |     0.111649   |        nan        |       nan        |      nan |
| delta_bias_group          |           10 |      1        |            0.0507779  |    0        |     0.0600592  |        nan        |       nan        |      nan |
| delta_logbeta_actor_group |           10 |      1        |            0.0645909  |    0        |     0.0790031  |        nan        |       nan        |      nan |
| lapse_eps                 |           10 |      1        |            0.00762369 |    0        |     0.00752088 |        nan        |       nan        |      nan |
| mu_alpha_actor            |           10 |      0.9      |            0.288988   |    0        |     0.231726   |        nan        |       nan        |      nan |
| mu_alpha_ctrl             |           10 |      0.9      |            0.470355   |    0        |     0.296439   |        nan        |       nan        |      nan |
| mu_bias                   |           10 |      1        |            0.0688316  |    0        |     0.0905372  |        nan        |       nan        |      nan |
| mu_logbeta_actor          |           10 |      1        |            0.138155   |    0        |     0.17974    |        nan        |       nan        |      nan |
| sigma_alpha_actor         |           10 |      0.5      |            0.482713   |    0        |     0.228909   |        nan        |       nan        |      nan |
| sigma_alpha_ctrl          |           10 |      0.3      |            0.582097   |    0        |     0.179552   |        nan        |       nan        |      nan |
| sigma_bias                |           10 |      1        |            0.0388835  |    0        |     0.0408531  |        nan        |       nan        |      nan |
| sigma_logbeta_actor       |           10 |      0.9      |            0.134613   |    0        |     0.103513   |        nan        |       nan        |      nan |
| w_control                 |           10 |      0.9      |            0.192824   |    0        |     0.228442   |        nan        |       nan        |      nan |

## Sampler diagnostics

|   rep |   rhat_max |   rhat_frac_gt_1_05 |   ess_bulk_min |   divergences |
|------:|-----------:|--------------------:|---------------:|--------------:|
|     0 |       1.01 |                   0 |            662 |             0 |
|     1 |       1.01 |                   0 |            643 |             0 |
|     2 |       1.01 |                   0 |            806 |             0 |
|     3 |       1.01 |                   0 |            753 |             0 |
|     4 |       1.01 |                   0 |            544 |             0 |
|     5 |       1.01 |                   0 |            846 |             0 |
|     6 |       1.01 |                   0 |            744 |             0 |
|     7 |       1.01 |                   0 |            744 |             0 |
|     8 |       1    |                   0 |            550 |             0 |
|     9 |       1.01 |                   0 |            774 |             0 |

## Interpretation guardrails

- Recovery is strongest when subject-level correlations are high and 95% interval coverage is near 0.95.
- Poor recovery of a parameter means it should not carry a strong mechanistic claim, even if predictive performance is good.
- This simulation conditions on the observed cue/target sequence and source posterior-mean parameters.