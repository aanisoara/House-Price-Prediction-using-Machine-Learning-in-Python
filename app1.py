import time
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import seaborn as sns
sns.set_style("whitegrid")
import base64
import datetime
from matplotlib import rcParams
from  matplotlib.ticker import PercentFormatter
import itertools
import numpy as np
from sklearn import preprocessing
from streamlit_folium import folium_static
import folium
import pandas as pd
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import predictor 


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from yellowbrick.regressor.alphas import alphas
import statsmodels.api as sm
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap
from predictor import *
import clustering

cat=pd.read_csv(f"Cat.csv")
df_gps=pd.read_csv(f"df_gps.csv")
df_clean=pd.read_csv(f"df_clean.csv").drop(['Unnamed: 0',"Neighbourhood Cleansed"],axis=1) 
train, test = train_test_split(df_clean, test_size=0.3,  random_state=42)
x_train = train.drop(['Price'],axis = 1 )
x_test = test.drop(['Price'],axis = 1 ) 
y_train = train["Price"].values
y_test = test['Price'].values
df_city=pd.read_csv(f"gps_use_data.csv",usecols=['Bathrooms', 'Bedrooms', 'Beds', 'Price','City'])

     #   import streamlit as st
    #    from streamlit_folium import folium_static
    #    import folium
    #    from folium.plugins import HeatMap
World = folium.Map()
heat_data = [[row['lat'],row['lng']] for index, row in df_gps.iterrows()]
HeatMap(heat_data).add_to(World)
      #  folium_static(World)

df_dum_cities=['Alcúdia', 'Alexandria', 'Algaida', 'Alhambra', 'Altadena', 'Amsterdam-Zuidoost', 'Anderlecht', 'Andratx', 'Annandale', 'Antwerpen', 'Arcadia', 'Artà', 'Astoria', 'Athens', 'Austin', 'Avalon Beach', 'Balgowlah', 'Ballsbridge', 'Balmain', 'Barcelona', 'Bellevue Hill', 'Berlin', 'Beverly Hills', 'Bondi', 'Boston', 'Brighton', 'Bromley', 'Bronte', 'Bronx', 'Brooklyn', 'Brunswick East', 'Brussels', 'Burbank', 'Byron Bay', 'Cala Ratjada', 'Calvià', 'Camperdown', 'Campos', 'Can Pastilla', 'Can Picafort', 'Capdepera', 'Carlton', 'Chatswood', 'Chicago', 'Chippendale', 'Chula Vista', 'Clovelly', 'Collingwood', 'Colònia de Sant Jordi', 'Coogee', 'Copenhagen', 'Cremorne', 'Cronulla', 'Croydon', 'Culver City', 'Darlinghurst', 'Dee Why', 'Denver', 'Docklands', 'Double Bay', 'Drumcondra', 'Dublin', 'Edgware', 'Edinburgh', 'Elizabeth Bay', 'Elwood', 'Enfield', 'Erskineville', 'Etterbeek', 'Fairlight', 'Felanitx', 'Fitzroy North', 'Footscray', 'Forest', 'Frederiksberg', 'Freshwater', 'Genève', 'Glebe', 'Glendale', 'Greater London', 'Harrow', 'Hawthorn', 'Haymarket', 'Hermosa Beach', 'Hong Kong', 'Hounslow', 'Ilford', 'Illes Balears', 'Inca', 'Inglewood', 'Ixelles', 'Kensington', 'Kingsford', 'Kingston upon Thames', 'Kirribilli', 'La Jolla', 'Leichhardt', 'Lennox Head', 'Lido', 'Lido di Ostia', 'Llucmajor', 'London', 'Long Beach', 'Long Island City', 'Los Angeles', 'Madrid', 'Malibu', 'Manacor', 'Manchester', 'Manhattan Beach', 'Manly', 'Marina del Rey', 'Maroubra', 'Marrickville', 'Mascot', 'Melbourne', 'Monterey Park', 'Montreal', 'Mosman', 'Mullumbimby', 'Muro', 'Nashville', 'Neutral Bay', 'New Orleans', 'New York', 'Newport', 'Newtown', 'North Melbourne', 'North Sydney', 'Northcote', 'Oakland', 'Paddington', 'Palma de Mallorca', 'Paris', 'Pasadena', 'Pollença', 'Port Melbourne', 'Portland', 'Porto Cristo', 'Potts Point', 'Prahran', 'Pyrmont', 'Quebec', 'Queens', 'Queenscliff', 'Randwick', 'Rathmines', 'Redfern', 'Redondo Beach', 'Richmond', 'Ringsend', 'Riva del Garda', 'Rome', 'Rose Bay', 'Rosebery', 'Rowland Heights', 'Rozelle', 'Rushcutters Bay', 'Sa Pobla', 'Sa Ràpita', 'Saint Kilda', 'Saint-Gilles', 'Saint-Josse-ten-Noode', 'San Diego', 'San Francisco', 'San Gabriel', 'Sant Llorenç des Cardassar', 'Santa Cruz', 'Santa Margalida', 'Santa Monica', 'Santanyí', 'Schaerbeek', 'Seattle', 'Selva', 'Son Servera', 'South Melbourne', 'South Yarra', 'Southbank', 'St Kilda', 'Stanmore', 'Staten Island', 'Suffolk Park', 'Surry Hills', 'Sutton', 'Sydney', 'Sóller', 'Tamarama', 'Topanga', 'Toronto', 'Torrance', 'Trento', 'Tsim Sha Tsui', 'Twickenham', 'Uccle', 'Ultimo', 'Valldemossa', 'Vancouver', 'Vaucluse', 'Venice', 'Vienna', 'Walnut', 'Washington', 'Waterloo', 'Waverley', 'Wembley', 'West Hollywood', 'West Melbourne', 'Wien', 'Windsor', 'Woollahra', 'Zetland']
cities_dum=[]
df_dum_countries=['Austria', 'Belgium', 'Canada', 'China', 'Denmark', 'France', 'Germany', 'Greece', 'Ireland', 'Italy', 'Netherlands', 'Spain', 'Switzerland', 'United Kingdom', 'United States']
countries_dum=[]
df_dum_amenities=['pets allowed','long term stays allowed','air conditioning', 'elevator in building','suitable for events','bathtub', 'essentials','gym','baby-friendly','security', 'secure_access', 'outdoor','bathroom_essentials','heating', 'pool_jacuzzi','clothes_stuff','not specified','kitchen','cooking basics','tv','family/kid friendly', 'internet','easy_to_check_in','free parking', 'laptop friendly workspace','old_handicap_people_friendly']
amenities_dum=[]
df_dum_host=['Host_has profile pic',
 'Host_is superhost',
 'Host_identity verified',
 'Host_location exact',
 'Host_instant bookable',
 'Host_require guest phone verification',
 'Host_require guest profile picture',
 'Host_no specified',
 'Host_requires license']
host_dum=[]
df_dum_cancel=['moderate', 'strict']
cancel_dum=[]
df_dum_roomtype=['Private room', 'Shared room']
roomtype_dum=[]
df_dum_bedtype=['Couch']
bedtype_dum=[]
df_dum_propertytype=['Atypical housing', 'Hotel', 'House', 'Other', 'Villa']
propertyty=[]



def get_cat(df,var):

    return sorted(df[var].unique().tolist())

def get_city(df,pays):
        return sorted(list(df["City"][df["Country"]==pays].unique()))

def get_city_dum(ville):
    for city in df_dum_cities:
        if ville in city:
            cities_dum.append(1)
        else:
            cities_dum.append(0)
    return cities_dum
def get_country_dum(pays):
    for country in df_dum_countries:
        if pays in country:
            countries_dum.append(1)
        else:
            countries_dum.append(0)
    return countries_dum
def get_amenities_dum(amenities):
    for equip in df_dum_amenities:
        if  equip in amenities:
            amenities_dum.append(1)
        else :
            amenities_dum.append(0)
    return amenities_dum
def get_host_dum(host):
    for host_elem in df_dum_host:
            if host_elem in host:
                host_dum.append(1)
            else :
                host_dum.append(0)
    return host_dum
def get_cancellation_dum(cancel):
    for cancel_d in df_dum_cancel:
        if cancel in cancel_d:
            cancel_dum.append(1)
        else:
            cancel_dum.append(0)
    return cancel_dum

def get_roomtype_dum(roomtype):
    for roomtype_d in df_dum_roomtype:
        if roomtype in roomtype_d:
            roomtype_dum.append(1)
        else:
            roomtype_dum.append(0)
    return roomtype_dum

def get_bedtype_dum(bedtype):
    for bedty in df_dum_bedtype:
        if bedtype in bedty:
            bedtype_dum.append(1)
        else:
            bedtype_dum.append(0)
    return bedtype_dum
def get_property_dum(propertytype):
    for propertytype_dum in df_dum_propertytype:
        if propertytype in propertytype_dum:
            propertyty.append(1)
        else:
            propertyty.append(0)
    return propertyty








def main():
    import streamlit as st
    st.sidebar.header("Airbnb Price Predictor")
    pays=st.sidebar.selectbox('Country',get_cat(cat,"Country"))
    ville=st.sidebar.selectbox("City", get_city(cat,pays))
    propertytype=st.sidebar.selectbox("Property Type", get_cat(cat,"Property Type"))
    bedtype=st.sidebar.selectbox("Bed Type", get_cat(cat,"Bed Type"))
    roomtype=st.sidebar.selectbox("Room Type", get_cat(cat,"Room Type"))
    bedrooms=st.sidebar.slider("Bedrooms",0,15,key=0)
    bathrooms=st.sidebar.slider("Bathrooms", 0,15,key=1)
    beds=st.sidebar.slider("Beds", 0,15,key=2)
    min_nights=st.sidebar.selectbox("Minimum Nights", get_cat(cat,"Minimum Nights"))
    max_nights=st.sidebar.selectbox("Maximum Nights", get_cat(cat,"Maximum Nights"))
    cancel=st.sidebar.selectbox("Cancellation policy", get_cat(cat,"Cancellation Policy"))
    amenities=st.sidebar.multiselect("Amenities",['pets allowed','long term stays allowed','air conditioning', 'elevator in building','suitable for events','bathtub', 'essentials','gym','baby-friendly','security', 'secure_access', 'outdoor','bathroom_essentials','heating', 'pool_jacuzzi','clothes_stuff','not specified','kitchen','cooking basics','tv','family/kid friendly', 'internet','easy_to_check_in','free parking', 'laptop friendly workspace','old_handicap_people_friendly'])
    amenities = ' '.join([str(elem) for elem in amenities])
    host=st.sidebar.multiselect("host",["has profile pic","is superhost","identity verified","location exact","instant bookable","require guest phone verification","require guest profile picture","no specified","requires license"])
    host= ' '.join([str(elem) for elem in host])
    default_values=['Host Response Rate','Calculated host listings count','Reviews per Month','How long Host', 'Availability 30','Availability 60','Availability 90','Availability 365']
    default_dico={}
    for elem in default_values:
        default_dico[elem]=cat[elem].median()
    st.header("🌴 🏠 Welcome to AIRBNBoost! 💶")
    st.write("Our aim is to help you to find the best estimation of your listing. As seen below, our price predictor is available for few destinations for the moment (but only for the moment 😝)")
    folium_static(World)
    st.subheader("How determine the best estimator? 🧐" )
    feat=Image.open("feature_importance.png")
    rmse= Image.open("RMSE.png") 
    pred=Image.open("predvsreal.png")
    st.image(feat) 
    st.image(rmse) 
    st.image(pred) 
    


    if st.sidebar.button("Submit here 👈"):
        st.subheader("Let's see together what should be the optimal price for your listing 🧐" )
        X = pd.DataFrame([[8.283996001114758,default_dico["Host Response Rate"], bathrooms,bedrooms,beds,min_nights,max_nights,default_dico['Availability 30'],default_dico['Availability 60'],default_dico["Availability 90"], default_dico['Availability 365'],default_dico["Calculated host listings count"],default_dico['Reviews per Month'],default_dico['How long Host']]], 
                    columns = ['mean_rates','Host Response Rate','Bathrooms','Bedrooms','Beds','Minimum Nights','Maximum Nights', 'Availability 30', 'Availability 60','Availability 90','Availability 365','Calculated host listings count','Reviews per Month','How long Host'])
        cities_df=pd.DataFrame([get_city_dum(ville)], columns=df_dum_cities)
        countries_df=pd.DataFrame([get_country_dum(pays)], columns=df_dum_countries)
        amenities_df=pd.DataFrame([get_amenities_dum(amenities)],columns=df_dum_amenities)
        host_df=pd.DataFrame(data=[get_host_dum(host)],columns=df_dum_host)
        cancel_df=pd.DataFrame([get_cancellation_dum(cancel)],columns=df_dum_cancel)
        roomtype_df=pd.DataFrame([get_roomtype_dum(roomtype)],columns=df_dum_roomtype)
        bedtype_df=pd.DataFrame([get_bedtype_dum(bedtype)],columns=df_dum_bedtype)
        propertytype_df=pd.DataFrame([get_property_dum(propertytype)],columns=df_dum_propertytype)
        resp_time_df=pd.DataFrame([[0,0,1]],columns=['within a day', 'within a few hours', 'within an hour'])
        df_list = [X, amenities_df,host_df,cancel_df,resp_time_df,roomtype_df,bedtype_df,propertytype_df,cities_df,countries_df]
        df1=pd.concat(df_list, axis = 1).drop_duplicates().reset_index(drop=True)
        st.write(df1)
        y_hat_pred = predictor.predict_(df1,x_train,y_train)
        y_hat_pred=y_hat_pred[0]
        st.metric(label="Estimated Price", value=f"{y_hat_pred}€")
        target_predict,df_city2= clustering.cluster(ville,df_city,bathrooms,bedrooms,beds,y_hat_pred)
        df_predict=df_city2[df_city2["cluster"]==target_predict]
        st.write("Other listings like yours:")
        st.dataframe(df_predict)
        cluster_means=(df_city2.groupby("cluster")['Bathrooms', 'Bedrooms', 'Beds',"Price"].mean())
        cluster_means.drop("Price",axis=1).plot(kind="bar",cmap="spring", legend = True)
        plt.title("Caracteristics depending on clusters",fontsize=16)
        st.pyplot(fig=plt)
        st.title(f"Your listing is in class {target_predict}")
        st.write(f'The median price for its cluster is {df_predict["Price"].median()}')
    
        st.balloons()



if __name__ == '__main__':
	main()

