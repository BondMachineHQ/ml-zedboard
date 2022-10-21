import xlrd
import matplotlib.pyplot as plt
from tabulate import tabulate
import plotly.express as px
import pandas as pd
import sys

reports = [
    "banknote-authentication_nf_4",
    "shuttle-landing-control_nf_6",
    "monks-problems-2_nf_6" ,
    "diabetes_nf_8" ,
    "hls4ml_lhc_jets_hlf_nf_15", 
    "climate-model-simulation-crashes_nf_20",
    "higgs_nf_28",
    "PhishingWebsites_nf_30",
    "pc4_nf_37",
    "optdigits_nf_64",
    "dna_nf_180",
    "scene_nf_299",
    "madelon_nf_500",
    "mnist_784_nf_784"]

features = []
luts = []
luts_ram  = []
ffs = []
brams =  []

resources = ["LUT", "LUTRAM", "FF", "BRAM"]

board = "ebaz4205"
part = "xc7z010clg400-1"


lut_thresholds = {
    "ebaz4205": {
        "LUT": 17700,
        "LUTRAM": 6000,
        "FF": 35200,
        "BRAM": 60
    },
    "pynqz2": {
        "LUT": 53200
    },
}

model = "nn_mod_1"

for r in reports:

    workbook = xlrd.open_workbook("reports/"+model+"/"+r+"/xc7z010clg400-1/Table.xlsx")
    worksheet = workbook.sheet_by_index(0)

    lut = 0
    lut_ram = 0
    ff = 0
    bram = 0

    for i in range(0, 5):
        for j in range(0, 4):
            if worksheet.cell_value(i, j) in resources:
                if worksheet.cell_value(i, j) == "LUT":
                    lut = worksheet.cell_value(i, j+1)
                elif worksheet.cell_value(i, j) == "LUTRAM":
                    lut_ram = worksheet.cell_value(i, j+1)
                elif worksheet.cell_value(i, j) == "FF":
                    ff = worksheet.cell_value(i, j+1)
                elif worksheet.cell_value(i, j) == "BRAM":
                    bram = worksheet.cell_value(i, j+1)

    features.append(r[r.rindex("_")+1:len(r)])
    luts.append(lut)
    luts_ram.append(lut_ram)
    ffs.append(ff)
    brams.append(bram)

table = []
table.append(["FEATURES", "LUTS", "LUTRAM", "FF", "BRAM"])
for i in range(0, len(features)):
    dataset_name = reports[i][0:reports[i].rfind("_")-3]
    table.append([dataset_name, features[i], luts[i], luts_ram[i], ffs[i], brams[i]])

print(tabulate(table, headers='firstrow', tablefmt='fancy_grid', showindex=range(1,len(table))))

for r in resources:

    if r == "LUT":
        # fig, ax = plt.subplots()
        # ax.plot(features, luts, linewidth=2.0)
        # plt.axhline(y = lut_thresholds[board][r], color = 'r', linestyle = '-')
        # plt.xlabel("Luts", fontsize=20)
        
        df = pd.DataFrame(dict(
            features = features,
            luts = luts
        ))
        fig = px.line(df, x="features", y="luts", markers=True)
        fig.add_hline(y=lut_thresholds[board][r], line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=20
        )
        fig.update_xaxes(title_font_family="Arial")
        fig.show()
        fig.write_image("reports/"+model+"/images/luts.png")
    elif r == "LUTRAM":
        df = pd.DataFrame(dict(
            features = features,
            lutsram = luts_ram
        ))
        fig = px.line(df, x="features", y="lutsram", markers=True)
        fig.add_hline(y=lut_thresholds[board][r], line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=20
        )
        fig.update_xaxes(title_font_family="Arial")
        fig.show()
        fig.write_image("reports/"+model+"/images/lutsram.png")
        # fig, ax = plt.subplots()
        # ax.plot(features, luts_ram, linewidth=2.0)
        # plt.axhline(y = lut_thresholds[board][r], color = 'r', linestyle = '-')
        # plt.xlabel("Lut ram", fontsize=20)
    elif r == "FF":
        df = pd.DataFrame(dict(
            features = features,
            flipflop = ffs
        ))
        fig = px.line(df, x="features", y="flipflop", markers=True)
        fig.add_hline(y=lut_thresholds[board][r], line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=20
        )
        fig.update_xaxes(title_font_family="Arial")
        fig.show()
        fig.write_image("reports/"+model+"/images/flipflop.png")
        # fig, ax = plt.subplots()
        # ax.plot(features, ffs, linewidth=2.0)
        # plt.axhline(y = lut_thresholds[board][r], color = 'r', linestyle = '-')
        # plt.xlabel("flip flop", fontsize=20)
    elif r == "BRAM":
        df = pd.DataFrame(dict(
            features = features,
            bram = brams
        ))
        fig = px.line(df, x="features", y="bram", markers=True)
        fig.add_hline(y=lut_thresholds[board][r], line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(
            font_family="Courier New",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="red",
            legend_title_font_color="green",
            font_size=20
        )
        fig.update_xaxes(title_font_family="Arial")
        fig.show()
        fig.write_image("reports/"+model+"/images/bram.png")
        # fig, ax = plt.subplots()
        # ax.plot(features, brams, linewidth=2.0)
        # plt.axhline(y = lut_thresholds[board][r], color = 'r', linestyle = '-')
        # plt.xlabel("block ram", fontsize=20)

    
    # plt.ylabel("Features", fontsize=20)
    # plt.show()