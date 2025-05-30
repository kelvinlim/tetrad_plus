import os
import pandas as pd
import numpy as np

import jpype
import jpype.imports

# REPLACE WITH YOUR PATH TO JAR
classpath="tetrad_plus/jars/tetrad-gui-7.6.3-launch.jar"
res = jpype.startJVM("-Xmx8g", classpath=classpath)

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts


# HELPER FUNCTION TO CONVERT A PANDAS DF
# TO A JAVA DATASET; I USUALLY COPY PASTE THIS
# INTO ALL CDA PROJECTS
def df_to_data(df):
    cols = df.columns
    values = df.values
    n, p = df.shape

    # JITTER THE DATA; FEEL FREE TO REMOVE
    values += 1e-3 * np.random.randn(n, p)

    variables = util.ArrayList()
    for col in cols:
        variables.add(td.ContinuousVariable(str(col)))

    databox = td.DoubleDataBox(n, p)
    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    return td.BoxDataSet(databox, variables)


### START SEARCH PARAMETERS ###
penalty_discount = 1
boss_bes = False
boss_starts = 10
boss_threads = 8
### END SEARCH PARAMETERS ###


# IN THIS EXAMPLE DATA CONTAINS MANY DATASETS
path = "data/"
for fname in ['boston_data_raw.csv']:
    print(fname)
    df = pd.read_csv(path + fname)
    data = df_to_data(df)

    # MAKE TIERED KNOWLEDGE BASED ON LAGS
    knowledge = td.Knowledge()
    for col in df.columns:
        # new lag variable
        lag_var = col + "_lag"
        # add lag variable to the knowledge base
        knowledge.addToTier(0, lag_var)
        # add the original variable to the knowledge base
        knowledge.addToTier(0, col)
    #knowledge.setTierForbiddenWithin(0, True)

    score = ts.score.SemBicScore(data, True)
    score.setPenaltyDiscount(penalty_discount)
    score.setStructurePrior(0)

    boss = ts.Boss(score)
    boss.setUseBes(boss_bes)
    boss.setNumStarts(boss_starts)
    boss.setNumThreads(boss_threads)
    boss.setUseDataOrder(False)
    boss.setResetAfterBM(False)
    boss.setResetAfterRS(False)
    boss.setVerbose(False)

    search = ts.PermutationSearch(boss)
    search.setKnowledge(knowledge)

    # SAVE THE OUTPUT GRAPHS
    graph = search.search().toString()
    # remove csv from fname
    output_file = fname.replace('.csv','_graph.txt')
    with open(output_file, "w") as f: f.write(str(graph))
    pass