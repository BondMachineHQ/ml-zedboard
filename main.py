from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.regularizers import l1
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ssl
import sys
import argparse
from sklearn.metrics import accuracy_score
import hls4ml
import matplotlib.pyplot as plt
import json
from os.path import exists
import networkx as nx
import pylab
from networkx.drawing.nx_agraph import graphviz_layout
from bmtrain import *

parser = argparse.ArgumentParser(description="Arguments for training nn", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", help="dataset name")
parser.add_argument("-b", "--fpga_board_number", help="fpga board number")
parser.add_argument("-f", "--fpga_part_number", help="fpga part number")
parser.add_argument("-m", "--nn_model_type", help="neural network architecture")
args = vars(parser.parse_args())
dataset_name = args["dataset"]
fpga_part_number = args["fpga_part_number"]
fpga_board_number = args["fpga_board_number"]
nn_model_type = args["nn_model_type"]

if dataset_name == None or len(dataset_name.replace(" ", "")) == 0:
    print(" # ERROR: No dataset name has been specified. ")
    sys.exit(1)

use_part = True
if fpga_part_number == None and fpga_board_number == None:
    print(bcolors.OKGREEN+" # INFO: FPGA part number not specified, using default xc7z010clg400-1"+bcolors.WHITE)
    fpga_part_number = "xc7z010clg400-1"
elif fpga_board_number != None:
    fpga_part_number = fpga_board_number
    use_part = False

if nn_model_type == None:
    nn_model_type = "MLP"

t = Trainer(fpga_part_number, nn_model_type)
t.use_part = use_part
t.initialize()
try:
    t.setup_data(dataset_name)
except Exception as e:
    print(" # An error occurred during setup data:", e)
    sys.exit(1)

t.exec_train()
t.exec_test()
t.build_model_fpga()
sys.exit()
