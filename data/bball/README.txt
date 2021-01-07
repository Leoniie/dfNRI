## Overview

This dataset contains tracking data from the 2012-2013 NBA season.

This version of the dataset was used in the following publications:

1) Zhan et al., Generating Multi-Agent Trajectories using Programmatic Weak Supervision, ICLR 2019.
2) Liu et al., NAOMI: Non-Autoregressive Multiresolution Sequence Imputation, NeurIPS 2019.
3) Zhan et al., Learning Calibratable Policies using Programmatic Style-Consistency, ICML 2020.
 
## Details

Each sequence contains XY-coordinates (in feet) of the ball, 5 offensive, and 5 defensive players (11 trajectories in total) over 8 seconds.

The coordinates are unnormalized, with (0,0) at the bottom-left corner of the court. 

Sequence length: 50
Frequency: 6.25 Hz
Num trajectories per sequence: 11 (ordered by ball, then 5 offense, then 5 defense)
Num train sequences: 104,003
Num test sequences: 13,464

## Files

Using numpy:

> np.load('train.npz')['data'] will have shape (104003, 50, 22)
> np.load('test.npz')['data'] will have shape (13464, 50, 22)
