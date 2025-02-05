
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


# # Mise en place de l'environnement de travail

# In[353]:


# Define root directory
root = "/Users/DELL/Documents/CODING_training/School Projects/School_Projects"

# Define sub-paths
base_snwork = os.path.join(root, "Projet Python KC-TN")
base_snpath = os.path.join(base_snwork, "base_sn")
codes = os.path.join(base_snwork, "PY_scripts")
outputs = os.path.join(base_snwork, "outputs")



um_base_sn = pd.read_excel(os.path.join(datapath, "donnes_UEMOA.xlsx"), sheet_name="base_complete")
events = base_sn = pd.read_csv(os.path.join(datapath,"Points_data.csv" ))




## On visualise les 5 premières variables.
events.head(5)



events.info()


um_base_sn.head(25)



# Nom des variables et infos
um_base_sn.info()



um_base_sn['VARIABLES'] = um_base_sn['VARIABLES'].ffill() ## On veut voir aaparaitre les noms des variables
## On fait un fill na 


um_base_sn.head(7)



## Voyons les différentes variables 
print(um_base_sn['VARIABLES'].unique().tolist())



print(events['country'].unique().tolist())



## Nos pays 
liste_pays = ["Mali", "Senegal", "Niger", "Togo", "Benin", "Burkina Faso", "Guinea-Bissau", "Ivory Coast"]



# Dictionnaire pour stocker les base_snFrames pivots par pays : on va faire un remaniement de la base 
pays_events = {}

for pays in liste_pays:
    # Filtrer les données pour le pays en cours
    df_pays = events[events['country'] == pays]
    
    # Créer la table pivot : index = 'annee', colonnes = 'type'
    df_pivot = df_pays.pivot_table(
        index='year', ## Le reférence, valeur unique
        columns='event_type', ## les colonnes
        aggfunc='size', #" on compte le nb d'events
        fill_value=0
    ).reset_index()
    
    ##On somme pour avoir le total des évènments
    df_pivot["total_events"] = df_pivot.iloc[:, 1:].sum(axis=1)
    
    pays_events[pays] = df_pivot
    



sn_events = pays_events["Senegal"]
sn_events.head(7)


# In[366]:


sn_events.columns 



liste_vars = um_base_sn.iloc[:, 2:-1].columns.tolist()
print(liste_vars)



## On réordonne la liste comme sur la base
liste_pays = ["Ivory Coast",
                 "Benin",
                "Burkina Faso",
                 "Mali",
                 "Niger",
                 "Senegal", 
                 "Guinea-Bissau",
                "Togo"]


#  On procède au pivot. Pour chaque pays, on a une base qui contient, dans le temps, ses indicateurs.

# Dictionnaire pour stocker les base_snFrames pivots par pays : on va faire un remaniement de la base 
pays_indic = {}

for pays,i in zip(liste_vars, liste_pays):
    # On extrait uniquement les colonnes nécessaires : ANNEE, VARIABLE et celle du pays en cours
    df_temp = um_base_sn[['ANNEE', 'VARIABLES', pays]]
    
    # On pivote la base :

    df_pivot = df_temp.pivot(index='ANNEE', 
                             columns='VARIABLES', 
                             values=pays).reset_index() 
    
    ## On enlève la variable dette/PIB -- son nom pose problème
    df_pivot.drop("DETTES/PIB", axis=1, inplace=True)
    
    # Stockage dans le dico, avec le nom dans la list pays
    pays_indic[i] = df_pivot


sn_indic =pays_indic["Senegal"]
sn_indic.head(7)




base_pays= {}

for pays in liste_pays:
    base_pays[pays]=pd.merge(pays_indic[pays], pays_events[pays], left_on='ANNEE', right_on='year',
                            how='inner') ## on ne prend que les années dans les deux bases
    ## On force le type numérique
    base_pays[pays] = base_pays[pays].apply(pd.to_numeric, errors='coerce')



base_sn = base_pays["Senegal"]
base_sn


# <p style="font-size:20px; font-family: 'Calisto MT'; letter-spacing: 0.01em;">  Nous avons enfin la base pour le Sénégal, place au véritable nettoyage !</p>

# ## Gestion des valeurs abbérentes

# <p style="font-size:20px; font-family: 'Calisto MT'; letter-spacing: 0.01em;">  Ici, nous allons faire des box plots.</p>


base_sn.drop("year", axis=1, inplace= True)
our_vars = base_sn.iloc[:, 1:].columns.tolist()
print(our_vars)



for var in our_vars:
    vars_s = var[:5]
    plt.figure(figsize=(6, 3))  # Crée une nouvelle figure
    plt.boxplot(base_sn[var].dropna())
    plt.xlabel(var)
    plt.ylabel('Valeur')
    plt.title(f'Boxplot_{var}')
    plt.tight_layout()
    plt.savefig(f'{outputs}/Boxplot_{vars_s}')


#  O notera que la variable year ne nous interesse pas.

# _**En supposant que le boxplot utilise la méthode de Turquey, nous allons remplacer les valeurs aberrantes par les bornes du boxplt qui sont q3 + écart_interquartiles et q1 - écart_interquartiles.**_


## On remplace les valeurs ectrêmes par nos bornes.

for var in our_vars:
    inter_q=(base_sn[var].dropna().quantile(0.75) - base_sn[var].quantile(0.25))/2
    lower_bound = base_sn[var].dropna().quantile(0.25) - inter_q
    upper_bound = base_sn[var].dropna().quantile(0.75) + inter_q
    
    # Remplacement des valeurs 
    base_sn[var] = np.where(base_sn[var] < lower_bound, lower_bound, base_sn[var])
    
    base_sn[var] = np.where(base_sn[var] > upper_bound, upper_bound, base_sn[var])


# Voyons le résultat après.


for var in our_vars:
    plt.figure(figsize=(6, 3))  # Crée une nouvelle figure
    plt.boxplot(base_sn[var].dropna())
    plt.xlabel(var)
    plt.ylabel('Valeur')
    plt.title(f'Boxplot_{var}')
    plt.tight_layout()



# ## Gestion des valeurs manquantes



## Imputation par la médiane pour les colonnes Numériques
base_sn.fillna(base_sn.median(), inplace=True)



## Supprimons la variable year en même temps
table = base_sn.describe()

table.to_excel(f'{outputs}/description_des_vars.xlsx', index=True)
table


# ## Plotting : représentations graphiques

# ### Histogrammes 

# Création des subplots pour chaque variable quantitative
for i, var in enumerate(our_vars):
    plt.figure(figsize=(7, 4))
    plt.hist(base_sn[var], bins=20)
    plt.xlabel(var, fontsize= 10)
    plt.ylabel('Fréquence', fontsize= 10)
    plt.savefig(f'{outputs}/Histogrammmes_{var[:5]}.png')
    plt.tight_layout()

plt.show()


# ### Evolution dans le temps


# Evolution dans le temps
# Appliquons un style pour des graphiques plus jolis
sns.set(style="whitegrid", context="talk")
for var in our_vars:
    plt.figure(figsize=(11, 5))  # Crée une nouvelle figure
    plt.plot(base_sn['ANNEE'],base_sn[var], marker='o', linestyle='-', color='steelblue', 
             linewidth=3, markersize=4)
    
    plt.title(var)
    plt.ylabel(var)
    plt.grid(True)
    plt.tight_layout() 
    
    # Enregistrement du graphique dans un fichier image
    plt.savefig(f'{outputs}/{var[:5]}_plot.png')
    


# _______________________

# ### Superposons l'évolution des différents évènemts politiques dans le temps


# Appliquons un style pour des graphiques plus jolis
sns.set(style="whitegrid", context="talk")
plt.figure(figsize=(11, 5))

for var in base_sn.columns[7:]:
    plt.plot(base_sn['ANNEE'], base_sn[var], marker='o', linestyle='-', linewidth=3, markersize=4, label=var)

plt.xlabel("ANNEE")
plt.ylabel("Valeur par type d'évènement")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{outputs}/all_curves.png")
plt.show()
    


# ### Nombre total d'events par type


# Nombre total d'events par type
events_tab= base_sn.iloc[:, 7:-1].sum(axis=0).round(0)


# <p style="font-size:20px; font-family: 'Calisto MT'; letter-spacing: 0.01em;"> 
# Nous n'avons malheureusement pas de variables qualitatives dans notre base. Ainsi, nous allons considérer les variables de type d'évènement comme nos catégories et faire un pie chart (diagramme circulaire). Allons-y!</p>


events_tab



## Faisons un pie chart

# de jolies couleurs!
colors = sns.color_palette("PuBu", n_colors=len(events_tab))

plt.figure(figsize=(6, 6))
plt.pie(
    events_tab.values,
    labels=events_tab.index,
    autopct='%1.2f%%',
    startangle=90,
    textprops={'fontsize': 10 },
    colors=colors
)
plt.title("Part des événements par type", fontsize=10)
plt.tight_layout()
plt.savefig(f"{outputs}/pie_plot_events_type.png", dpi=300)
plt.show()


# <hr>

# <center><h1 style="font-size:40px; font-family: 'Calisto MT'; letter-spacing: 0.1em;"> Partie 3 : Statistique descriptive avancée </h1></center>

# <hr>

# ### pairplots


## faisons des pair plots
data = base_sn.drop("ANNEE", axis= 1)
sns.pairplot(data)
plt.show()
plt.savefig(f'{outputs}/pairplot.png')


# ### Heatmap


# Corrélation entre les variables numériques
corr_matrix = data.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation entre les variables numériques')
plt.savefig(f'{outputs}/Corr_Matrix')
plt.show()


for var in base_sn.columns[1:7] :
    plt.figure(figsize=(8, 4))
    plt.scatter(base_sn['total_events'], base_sn[var], color='steelblue', alpha=0.7)
    plt.xlabel('total_events', fontsize=8)
    plt.ylabel(var, fontsize=8)
    plt.title(f"Nuage de points : total_events et {var}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{outputs}/scatter_total_events_{var}.png", dpi=300)
    plt.show()


# # Partie 4 : Un peu de cartographie



import geopandas as gpd

sn_regions = gpd.read_file(
    os.path.join(datapath, "sen_admbnda_adm1_anat_20240520.shp")
).to_crs(epsg=4326) ## système de projection : on met celui pour le Sénégal.



sn_regions.head(5)


data_points_sn= events[events['country']=="Senegal"]



count_region = data_points_sn['admin1'].value_counts().reset_index()
count_region = count_region.sort_values(by="admin1").reset_index()
## on trie comme dans le shp
count_region


# Place au merging.


sn_merged = pd.concat([sn_regions, count_region.iloc[:,1:]], axis=1)


# On vérifie bien...
sn_merged




fig, ax = plt.subplots(figsize=(8, 8))
sn_merged.plot(column="count", cmap="Blues", legend=True, ax=ax, edgecolor="black")
plt.title("Nombre d'événements par région")
plt.tight_layout()
plt.savefig(f"{outputs}/mapping_total_events_par_region.png")
plt.show()

