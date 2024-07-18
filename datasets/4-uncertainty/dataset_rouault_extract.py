# Script to extract the data from the source files of Rouault et al 2022
# Paper: https://elifesciences.org/articles/75038
# Source files: https://github.com/marionrouault/actobscom/tree/main/SCRIPTS/DATA
# Script by Max Shinn for BAMB 2023

import numpy as np
import pandas
import scipy.io
import glob
import re

FIELDS = {"blkind": "block_index",
          "mu": "mu",
          "kappa": "kappa",
          "cbef": "confidence_before",
          "caft": "confidence_after",
          "rbef": "response_before",
          "raft": "response_after",
          "seqdir": "sequence_direction",
          "seqind": "sequence_index",
          "smpang": "sample_angle",
          "smpllr": "sample_log_likelihood_ratio",
          "taskid": "task_id",
}

_data = []
for f in glob.glob("*.mat"):
    d = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)['dat']
    m = re.match(".*ACTOBS_([CD])(?:_rule([12]))?_S([0-9]*)_task([12])_expdata.mat", f)
    _session_data = {}
    exp, rule, subj, task = m.groups()
    _session_data["exp"] = exp
    _session_data["rule"] = rule
    _session_data["subj"] = subj
    _session_data["task"] = task
    for field in FIELDS.keys():
        _session_data[FIELDS[field]] = getattr(d, field)
    _data.append(pandas.DataFrame(_session_data))

df = pandas.concat(_data).reset_index()
df.to_csv("rouault2022_data.csv")
