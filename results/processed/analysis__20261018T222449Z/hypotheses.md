# Hypothesis assessment (generated from results/raw by scripts/analyze.py)

## H1: supported
DOGE has higher mean accuracy than TA and TIES in each experiment.

```json
{
  "experiments": 7,
  "doge_wins": [
    "F01",
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06"
  ],
  "values": {
    "F01": {
      "task-arithmetic": 83.9732974767685,
      "ties": 79.0583536028862,
      "doge-ta": 93.44216734170914
    },
    "P01": {
      "task-arithmetic": 93.06228458881378,
      "ties": 83.90543162822723,
      "doge-ta": 96.66520953178406
    },
    "P02": {
      "task-arithmetic": 90.78408479690552,
      "ties": 79.43853437900543,
      "doge-ta": 97.90749847888947
    },
    "P03": {
      "task-arithmetic": 94.67296302318573,
      "ties": 85.60962975025177,
      "doge-ta": 98.56444299221039
    },
    "P04": {
      "task-arithmetic": 87.73449659347534,
      "ties": 72.77111411094666,
      "doge-ta": 95.54109573364258
    },
    "P05": {
      "task-arithmetic": 86.80594563484192,
      "ties": 73.13516139984131,
      "doge-ta": 94.9734628200531
    },
    "P06": {
      "task-arithmetic": 86.79295480251312,
      "ties": 75.34784972667694,
      "doge-ta": 96.16288244724274
    }
  }
}
```

## H2: supported
DOGE has higher worst-task accuracy than the baselines.

```json
{
  "experiments": 7,
  "doge_wins": [
    "F01",
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06"
  ],
  "values": {
    "F01": {
      "task-arithmetic": 72.66666889190674,
      "ties": 70.57799100875854,
      "doge-ta": 91.34600162506104
    },
    "P01": {
      "task-arithmetic": 88.1645679473877,
      "ties": 74.30086135864258,
      "doge-ta": 94.03042197227478
    },
    "P02": {
      "task-arithmetic": 85.36816835403442,
      "ties": 69.89706754684448,
      "doge-ta": 96.63499593734741
    },
    "P03": {
      "task-arithmetic": 92.92592406272888,
      "ties": 82.25926160812378,
      "doge-ta": 97.88888692855835
    },
    "P04": {
      "task-arithmetic": 86.53998374938965,
      "ties": 70.92636823654175,
      "doge-ta": 94.77182030677795
    },
    "P05": {
      "task-arithmetic": 81.42670392990112,
      "ties": 66.01106524467468,
      "doge-ta": 93.354332447052
    },
    "P06": {
      "task-arithmetic": 80.58590888977051,
      "ties": 67.80681014060974,
      "doge-ta": 95.39983868598938
    }
  }
}
```

## H3: partly supported
Every method retains less when merging four models than two.

```json
{
  "overall_lower_in_four": {
    "task-arithmetic": true,
    "ties": false,
    "doge-ta": true
  },
  "tasks_lower_in_four_of_4": {
    "task-arithmetic": 2,
    "ties": 1,
    "doge-ta": 4
  }
}
```

## H4: partly supported
DOGE's retention drop from two to four models is smaller than the baselines'.

```json
{
  "drop_points": {
    "task-arithmetic": 6.05338571087529,
    "ties": -0.7283940280755701,
    "doge-ta": 3.2366720072928956
  },
  "smaller_than": [
    "task-arithmetic"
  ]
}
```

## H5: not supported: no consistent rank relationship (descriptive)
Pairs with higher task-vector cosine similarity retain more (descriptive).

```json
{
  "n_pairs": 6,
  "spearman_rho": {
    "task-arithmetic": 0.028571428571428574,
    "ties": -0.2,
    "doge-ta": -0.028571428571428574
  },
  "rule": "supported if rho >= 0.5 for every method; no relationship if |rho| < 0.5 for every method"
}
```

## H6: supported
DOGE loses less performance than the baselines on dissimilar pairs.

```json
{
  "pairs": [
    "P03",
    "P05",
    "P06"
  ],
  "retention_gap_points": {
    "P03": {
      "task-arithmetic": 4.6894674121192566,
      "ties": 13.818377048540157,
      "doge-ta": 0.7688490213490695
    },
    "P05": {
      "task-arithmetic": 11.620073308487676,
      "ties": 25.563017564649968,
      "doge-ta": 3.2645001658874264
    },
    "P06": {
      "task-arithmetic": 12.327345508886793,
      "ties": 23.890284694295474,
      "doge-ta": 2.8572471082938478
    }
  },
  "doge_wins": [
    "P03",
    "P05",
    "P06"
  ]
}
```
