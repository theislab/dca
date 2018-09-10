#!/bin/bash

# downloads 722M dataset file
wget https://hmgubox.helmholtz-muenchen.de/f/1a014dc377f64b2b964c/?dl=1 -O datasets.zip
mkdir data; cd data
unzip ../datasets.zip
