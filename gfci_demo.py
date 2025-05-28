
import os
import pandas as pd
import numpy as np

import jpype
import jpype.imports

classpath="jars/tetrad-gui-7.6.3-launch.jar"
jpype.startJVM("-Xmx8g", classpath=classpath)

import java.util as util
import edu.cmu.tetrad.data as td
import edu.cmu.tetrad.graph as tg
import edu.cmu.tetrad.search as ts

#td = jpype.JPackage("edu.cmu.tetrad.data")

def df_to_data(df):
    cols = df.columns
    values = df.values
    n, p = df.shape

    # OPTIONAL
    values += 1e-3 * np.random.randn(n, p)

    variables = util.ArrayList()
    for col in cols:
        variables.add(td.ContinuousVariable(str(col)))

    databox = td.DoubleDataBox(n, p)
    for col, var in enumerate(values.T):
        for row, val in enumerate(var):
            databox.set(row, col, val)

    return td.BoxDataSet(databox, variables)


### SEARCH PARAMETERS ###

alpha = 0.01
penalty_discount = 1

# boss_bes = False
# boss_starts = 10
# boss_threads = 8

### SEARCH PARAMETERS ###

path = "data/"
for fname in ['boston_data_raw.csv']:
    print(fname)
    df = pd.read_csv(path + fname)
    data = df_to_data(df)

    test = ts.test.IndTestFisherZ(data, alpha)
    score = ts.score.SemBicScore(data, True)
    score.setPenaltyDiscount(penalty_discount)
    score.setStructurePrior(0)

    # boss = ts.Boss(score)
    # boss.setUseBes(boss_bes)
    # boss.setNumStarts(boss_starts)
    # boss.setNumThreads(boss_threads)
    # boss.setUseDataOrder(False)
    # boss.setResetAfterBM(False)
    # boss.setResetAfterRS(False)
    # boss.setVerbose(False)
    # search = ts.PermutationSearch(boss)
    # graph = search.search().toString()

    # FOR THE MOST PART, DONT CHANGE ANY OF THESE
    # UNLESS COMPUTATION IS TAKING TOO LONG
    gfci = ts.GFci(test, score)
    gfci.setCompleteRuleSetUsed(True)
    gfci.setDepth(-1)
    gfci.setDoDiscriminatingPathRule(True)
    gfci.setFaithfulnessAssumed(True)
    gfci.setMaxDegree(-1)
    gfci.setMaxPathLength(-1)
    gfci.setPossibleMsepSearchDone(True)
    gfci.setVerbose(False)
    graph = gfci.search().toString()

    with open(f"graphs/{fname}", "w") as f: f.write(str(graph))