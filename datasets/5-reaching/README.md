

This dataset is a single subject (subject 1) from experiment 1 of [Haith et al., PLOS Comp Biol, 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004171). Provided files are:

- `...xy.npy` is a `2 x ntrials x ntime` numpy array. `xy[0]` are x positions, and `xy[1]` y positions in meters from the center position. The sampling rate of the timeseries is 130 Hz.
- `...targets.p` is a `ntrials x 9` pandas dataframe, with columns:
  `['jump_time', 'target_id_pre', 'target_id_post', 'target_angle_pre', 'target_angle_post', 'target_x_pre', 'target_y_pre', 'target_x_post', 'target_y_post']`
  - jump_time indicates the time (in s) of the target jump
  - target_id is a number from 0-7, indexing the target position before and after target jump
  - target_angle expresses the same variables as angles (in radians)
  - target_x and target_y encode x and y positions of targets before and after the jump, calculated as x = radius * cos(angle) and y = radius * sin(angle)

Moreover, you can find the script to extract the same data from other subjects, if you want: `Haith_data.py`, which reads `Expt1_Data.mat` that you can download [here](https://datadryad.org/stash/dataset/doi:10.5061/dryad.63k6q).
