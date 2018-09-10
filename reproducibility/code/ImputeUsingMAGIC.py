import magic
import os

scdata = magic.mg.SCData.from_csv("../data/chu/chu_original.csv", cell_axis="columns", data_type='sc-seq')
scdata.run_magic()
mdata = scdata.magic.data
mdata=mdata.transpose()
mdata.to_csv("../data/chu/chu_magic.csv")

scdata = magic.mg.SCData.from_csv("../data/francesconi/francesconi_original.csv", cell_axis="columns", data_type='sc-seq')
scdata.run_magic()
mdata = scdata.magic.data
mdata=mdata.transpose()
mdata.to_csv("../data/francesconi/francesconi_magic.csv")

scdata = magic.mg.SCData.from_csv("../data/stoeckius/stoeckius_original.csv", cell_axis="columns", data_type='sc-seq')
scdata.run_magic()
mdata = scdata.magic.data
mdata=mdata.transpose()
mdata.to_csv("../data/stoeckius/stoeckius_magic.csv")
