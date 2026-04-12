# deepseek_bench1_eval summary

Model run directory: `logs/run_evaluation/deepseek_bench1_eval/deepseek__deepseek-chat`

Generated from `report.json` files with:

```powershell
python scripts\check_eval_results.py logs\run_evaluation\deepseek_bench1_eval\deepseek__deepseek-chat
```

## Overall

| Metric | Value |
| --- | ---: |
| Total instances | 100 |
| Resolved | 61 |
| Failed | 39 |
| Resolve rate | 61.0% |
| Cases with regressions | 11 |
| Failed but fixed some tests | 9 |

## Highest-regression cases

| Instance ID | Resolved | Fixed | Not Fixed | Regressions |
| --- | --- | ---: | ---: | ---: |
| django__django-15400 | FAIL | 0 | 2 | 62 |
| django__django-13401 | FAIL | 0 | 1 | 32 |
| django__django-12747 | FAIL | 3 | 0 | 7 |
| django__django-11630 | FAIL | 0 | 2 | 2 |
| django__django-13660 | FAIL | 2 | 0 | 2 |
| django__django-11742 | FAIL | 0 | 2 | 1 |
| django__django-11905 | FAIL | 1 | 1 | 1 |
| django__django-12589 | FAIL | 0 | 1 | 1 |
| django__django-14667 | FAIL | 0 | 1 | 1 |
| django__django-14997 | FAIL | 3 | 0 | 1 |

## Failed but partially fixed

| Instance ID | Fixed | Not Fixed | Regressions |
| --- | ---: | ---: | ---: |
| django__django-13321 | 337 | 18 | 0 |
| django__django-14997 | 3 | 0 | 1 |
| django__django-12747 | 3 | 0 | 7 |
| django__django-13220 | 2 | 2 | 0 |
| django__django-15213 | 2 | 1 | 0 |
| django__django-13660 | 2 | 0 | 2 |
| django__django-13925 | 1 | 1 | 0 |
| django__django-14534 | 1 | 1 | 0 |
| django__django-11905 | 1 | 1 | 1 |

## Full results

| Instance ID | Resolved | Fixed | Not Fixed | Regressions |
| --- | --- | ---: | ---: | ---: |
| astropy__astropy-12907 | PASS | 2 | 0 | 0 |
| astropy__astropy-14182 | PASS | 1 | 0 | 0 |
| astropy__astropy-14365 | PASS | 1 | 0 | 0 |
| astropy__astropy-14995 | PASS | 1 | 0 | 0 |
| astropy__astropy-6938 | PASS | 2 | 0 | 0 |
| astropy__astropy-7746 | FAIL | 0 | 1 | 0 |
| django__django-10914 | PASS | 1 | 0 | 0 |
| django__django-10924 | FAIL | 0 | 1 | 0 |
| django__django-11001 | PASS | 2 | 0 | 0 |
| django__django-11019 | FAIL | 0 | 16 | 0 |
| django__django-11039 | PASS | 1 | 0 | 0 |
| django__django-11049 | PASS | 1 | 0 | 0 |
| django__django-11099 | PASS | 3 | 0 | 0 |
| django__django-11133 | PASS | 1 | 0 | 0 |
| django__django-11179 | PASS | 1 | 0 | 0 |
| django__django-11283 | FAIL | 0 | 1 | 0 |
| django__django-11422 | PASS | 1 | 0 | 0 |
| django__django-11564 | FAIL | 0 | 2 | 0 |
| django__django-11583 | PASS | 2 | 0 | 0 |
| django__django-11620 | PASS | 1 | 0 | 0 |
| django__django-11630 | FAIL | 0 | 2 | 2 |
| django__django-11742 | FAIL | 0 | 2 | 1 |
| django__django-11797 | PASS | 1 | 0 | 0 |
| django__django-11815 | PASS | 2 | 0 | 0 |
| django__django-11848 | PASS | 2 | 0 | 0 |
| django__django-11905 | FAIL | 1 | 1 | 1 |
| django__django-11910 | FAIL | 0 | 1 | 0 |
| django__django-11964 | FAIL | 0 | 2 | 0 |
| django__django-11999 | PASS | 1 | 0 | 0 |
| django__django-12113 | FAIL | 0 | 1 | 0 |
| django__django-12125 | PASS | 2 | 0 | 0 |
| django__django-12184 | PASS | 1 | 0 | 0 |
| django__django-12284 | PASS | 1 | 0 | 0 |
| django__django-12286 | PASS | 1 | 0 | 0 |
| django__django-12308 | FAIL | 0 | 2 | 0 |
| django__django-12453 | PASS | 1 | 0 | 0 |
| django__django-12470 | PASS | 1 | 0 | 0 |
| django__django-12497 | PASS | 2 | 0 | 0 |
| django__django-12589 | FAIL | 0 | 1 | 1 |
| django__django-12700 | PASS | 1 | 0 | 0 |
| django__django-12708 | PASS | 1 | 0 | 0 |
| django__django-12747 | FAIL | 3 | 0 | 7 |
| django__django-12856 | PASS | 3 | 0 | 0 |
| django__django-12908 | PASS | 2 | 0 | 0 |
| django__django-12915 | PASS | 3 | 0 | 0 |
| django__django-12983 | PASS | 1 | 0 | 0 |
| django__django-13028 | PASS | 2 | 0 | 0 |
| django__django-13033 | PASS | 1 | 0 | 0 |
| django__django-13158 | PASS | 1 | 0 | 0 |
| django__django-13220 | FAIL | 2 | 2 | 0 |
| django__django-13230 | PASS | 1 | 0 | 0 |
| django__django-13265 | FAIL | 0 | 4 | 0 |
| django__django-13315 | PASS | 1 | 0 | 0 |
| django__django-13321 | FAIL | 337 | 18 | 0 |
| django__django-13401 | FAIL | 0 | 1 | 32 |
| django__django-13447 | PASS | 1 | 0 | 0 |
| django__django-13448 | FAIL | 0 | 1 | 0 |
| django__django-13551 | PASS | 2 | 0 | 0 |
| django__django-13590 | PASS | 1 | 0 | 0 |
| django__django-13658 | PASS | 1 | 0 | 0 |
| django__django-13660 | FAIL | 2 | 0 | 2 |
| django__django-13710 | PASS | 1 | 0 | 0 |
| django__django-13757 | PASS | 1 | 0 | 0 |
| django__django-13768 | FAIL | 0 | 1 | 0 |
| django__django-13925 | FAIL | 1 | 1 | 0 |
| django__django-13933 | PASS | 1 | 0 | 0 |
| django__django-13964 | PASS | 1 | 0 | 0 |
| django__django-14016 | FAIL | 0 | 2 | 0 |
| django__django-14017 | PASS | 2 | 0 | 0 |
| django__django-14155 | FAIL | 0 | 3 | 0 |
| django__django-14238 | PASS | 2 | 0 | 0 |
| django__django-14382 | PASS | 1 | 0 | 0 |
| django__django-14411 | FAIL | 0 | 1 | 0 |
| django__django-14534 | FAIL | 1 | 1 | 0 |
| django__django-14580 | PASS | 1 | 0 | 0 |
| django__django-14608 | PASS | 4 | 0 | 0 |
| django__django-14667 | FAIL | 0 | 1 | 1 |
| django__django-14672 | PASS | 168 | 0 | 0 |
| django__django-14730 | FAIL | 0 | 1 | 0 |
| django__django-14752 | PASS | 1 | 0 | 0 |
| django__django-14787 | PASS | 1 | 0 | 0 |
| django__django-14855 | PASS | 1 | 0 | 0 |
| django__django-14915 | PASS | 1 | 0 | 0 |
| django__django-14997 | FAIL | 3 | 0 | 1 |
| django__django-14999 | PASS | 1 | 0 | 0 |
| django__django-15061 | FAIL | 0 | 3 | 0 |
| django__django-15202 | FAIL | 0 | 2 | 0 |
| django__django-15213 | FAIL | 2 | 1 | 0 |
| django__django-15252 | FAIL | 0 | 2 | 0 |
| django__django-15320 | FAIL | 0 | 1 | 0 |
| django__django-15347 | PASS | 1 | 0 | 0 |
| django__django-15388 | FAIL | 0 | 1 | 1 |
| django__django-15400 | FAIL | 0 | 2 | 62 |
| django__django-15498 | PASS | 1 | 0 | 0 |
| django__django-15695 | FAIL | 0 | 1 | 0 |
| django__django-15738 | FAIL | 0 | 2 | 0 |
| django__django-15781 | FAIL | 0 | 1 | 0 |
| django__django-15789 | PASS | 1 | 0 | 0 |
| django__django-15790 | PASS | 1 | 0 | 0 |
| django__django-15814 | PASS | 1 | 0 | 0 |
