# Richer-history joint-response robustness analysis

## Planned contrasts

| model   | reference   | label                                      |   n_subjects |   mean_subject_delta_elpd |    ci_low |   ci_high |   positive_subjects |   total_delta_elpd |
|:--------|:------------|:-------------------------------------------|-------------:|--------------------------:|----------:|----------:|--------------------:|-------------------:|
| HR14    | HR6         | actor/control beyond rich history          |           40 |                  0.554059 | -0.198179 |  1.29659  |                  25 |           22.1623  |
| HR15    | HR6         | policy beyond rich history                 |           40 |                  0.439287 | -0.165083 |  1.06101  |                  26 |           17.5715  |
| HR9     | HR14        | policy beyond actor/control + rich history |           40 |                  0.181561 | -0.260009 |  0.630566 |                  25 |            7.26245 |
| HR9     | HR15        | actor/control beyond policy + rich history |           40 |                  0.296333 | -0.222482 |  0.799997 |                  25 |           11.8533  |
| HR9     | HR6         | total RL beyond rich history               |           40 |                  0.73562  | -0.135565 |  1.61815  |                  25 |           29.4248  |

Interpretation: this is a prospective robustness family. Historical D6/D14/D15/D9 remain the historical primary models.
