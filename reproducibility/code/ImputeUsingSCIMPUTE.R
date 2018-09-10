library(scImpute)

scimpute(count_path = "../data/francesconi/francesconi_withDropout.csv", infile = "csv", outfile = "csv", out_dir = "../data/francesconi/scimpute", Kcluster = 1)
scimpute(count_path = "../data/chu/chu_original.csv", infile = "csv", outfile = "csv", out_dir = "../data/chu/scimpute", Kcluster = 2)
scimpute(count_path = "../data/stoeckius/stoeckius_original.csv", infile = "csv", outfile = "csv", out_dir = "../data/stoeckius/scimpute", Kcluster = 13)
