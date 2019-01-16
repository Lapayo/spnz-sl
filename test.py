from experiment.lspn import run_incremental_lspn, run_lspn
import pandas as pd
import numpy as np

import spn.product_node

spn.product_node.mult = False

# create dataset
# test_df = pd.read_csv("../VAE/test.csv")
# test_data = test_df.values
# np.savetxt("test.csv", test_data, delimiter=",")
# test_df = pd.read_csv("../VAE/train_small.csv")
# test_data = test_df.values
# np.savetxt("train_small.csv", test_data, delimiter=",")
test_df = pd.read_csv("../VAE/train.csv")
test_data = test_df.values
np.random.shuffle(test_data)
np.savetxt("train_shuffle.csv", test_data, delimiter=",")
0/0 # exit


#model = run_lspn("../VAE/train_small.csv")#ä, batch_size=100)
# model = run_incremental_lspn(["train_small.csv"], batch_size=100)
model = run_incremental_lspn(["train.csv"], batch_size=1000)

# eval
data = np.loadtxt("train.csv", delimiter=",", dtype=np.float32)
print(np.mean(model.evaluate(data, mpe=False)))