import pandas as pd
import sklearn
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output
import ipywidgets as widgets
from ipywidgets.widgets import Dropdown
from ipywidgets import HBox, Label, Layout
from ipywidgets import Button, HBox, VBox


class Interactive_Plot:
    def __init__(self):
        #load and clean dataset
        data=pd.read_csv("auto-mpg.csv", na_values = "?")
        car_name=data["car name"]
        car_year=data["model year"]
        car_origin=data["origin"]
        data.rename(columns={"mpg":"MPG","cylinders":"Cylinders","displacement":"Displacement","horsepower":"Horsepower","weight":"Weight","acceleration":"Acceleration"}, inplace=True)
        data.fillna(data.mean(), inplace=True)
        data=data[['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration']]
        data["Horsepower"]=data.Horsepower.astype(int)
        #add year to the name of repeated cars
        repeated=set(car_name[car_name.duplicated()])
        for i, (name, year) in enumerate(zip(car_name, car_year)):
            if name in repeated:
                car_name.iloc[i]=name+", "+str(year)
        for name in data.columns:
            name_col=name+"_class"
            data[name_col]=pd.qcut(data[name], q=3, labels=["Low","Medium","High"])

        self.data=data
        self.car_name=car_name
    #method that runs the visualization
    def run(self):
        data=self.data
        #set layout and buttoms
        style = {'description_width': '100px'}
        layout=Layout(width="30%")
        car_bt=Dropdown(options=self.car_name.sort_values().values, description="Choose your car", style=style, layout=Layout(width="50%"))
        MPG_bt=Dropdown(options=["All","Low","Medium","High"], description="MPG", style=style, layout=layout)
        Cylinders_bt=Dropdown(options=["All","Low","Medium","High"], description="Cylinders", style=style, layout=layout)
        Displacement_bt=Dropdown(options=["All","Low","Medium","High"], description="Displacement", style=style, layout=layout)
        Horsepower_bt=Dropdown(options=["All","Low","Medium","High"], description="Horsepower", style=style, layout=layout)
        Weight_bt=Dropdown(options=["All","Low","Medium","High"], description="Weight", style=style, layout=layout)
        Acceleration_bt=Dropdown(options=["All","Low","Medium","High"], description="Acceleration", style=style, layout=layout)
        #function that plots the comparissons
        def compare(car_, MPG, Cylinders, Displacement, Horsepower, Weight, Acceleration):
            import matplotlib.gridspec as gridspec
            dict_data={}
            #set figure
            fig = plt.figure(figsize=(20,10))
            fig.suptitle(car_.title(), fontsize=20)
            ax1=fig.add_subplot(1, 2, 1)   #top and bottom left
            ax2=fig.add_subplot(2, 2, 2)   #top right
            ax3=fig.add_subplot(2, 2, 4)   #bottom right 
            #find car features
            car=np.where(self.car_name==car_)[0][0]
            list_var=[MPG, Cylinders, Displacement, Horsepower, Weight, Acceleration]
            #build boolean:
            data_boolean=pd.DataFrame()
            for name, val in zip(data.columns, list_var):
                data_boolean[name]=data[name+"_class"].apply(lambda x: True if x==val or val=="All" else False)
            data_boolean=data_boolean.all(axis=1)
            booleans=(data.index.values!=(car))&data_boolean.values
            values=((data.loc[(car),data.columns[:6]]-data.loc[booleans,data.columns[:6]].mean())/data.loc[booleans,data.columns[:6]].mean())*100
            
            #plot 1, left side
            sns.barplot(values.index, values.values, palette="Blues_d", ax=ax1)
            ax1.set_xticklabels(data.columns.values[:6],rotation=45, fontsize=16)
            ax1.yaxis.set_ticks(np.arange(-100, 160, 20))
            ax1.set_title("Performance over average(%)", fontsize=18)
            ax1.set_xlabel(r'Features', fontsize=16)
            ax1.set_ylabel("Performance over average(%)", fontsize=16)
            ax1.grid(True, axis='y')
            ax1.set_ylim(-100,150)

            #plot 2, top-right
            data_dict={}
            for name in data.columns[:6]:
                data_dict[name]=data[name].rank()[(car)]/len(data)*100
            sns.barplot(list(data_dict.values()),list(data_dict.keys()), palette="Blues_d", orient="h", ax=ax3)
            ax3.set_yticklabels(data.columns.values[:6],rotation=45, fontsize=14)
            ax3.xaxis.set_ticks(np.arange(0, 110, 10))
            ax3.set_title(" Ranking Position (%)", fontsize=16)
            ax3.set_xlabel("Top percent (%)", fontsize=14)
            ax3.set_xlim(0,100)
            ax3.set_ylabel("Features", fontsize=14)
            ax3.grid(True, axis='x')
            col_labels = ["Your car", "Type"]
            row_labels = data.columns[:6]
            table_vals =np.transpose(np.array([data.iloc[car,:6].tolist(), data.iloc[car,6:12].tolist()]))

            # Draw table
            the_table = ax2.table(cellText=table_vals,
                                  colWidths=[0.1] * 2,
                                  rowLabels=row_labels,
                                  colLabels=col_labels,loc='center',rowColours=sns.color_palette("Blues_d", 6))
            the_table.auto_set_font_size(True)
            the_table.set_fontsize(15)
            the_table.scale(2, 3)

            # Removing ticks and spines enables you to get the figure only with table
            ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax2.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
            ax2.grid(False)
            ax2.axis('off')
            for pos in ['right','top','bottom','left']:
                plt.gca().spines[pos].set_visible(False)
            plt.show()


        #iteractive features
        w=interactive_output(compare, {"car_":car_bt, "MPG":MPG_bt, "Cylinders":Cylinders_bt, "Displacement":Displacement_bt, "Horsepower":Horsepower_bt, "Weight":Weight_bt, "Acceleration":Acceleration_bt})

        hbox1 = HBox([car_bt])
        hbox2 = HBox([MPG_bt, Cylinders_bt, Displacement_bt])
        hbox3 = HBox([Horsepower_bt, Weight_bt, Acceleration_bt])
        ui = VBox([hbox1, hbox2, hbox3])
        display(ui, w)