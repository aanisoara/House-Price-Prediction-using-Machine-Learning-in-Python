def cluster(ville,df_city,bathrooms,bedrooms,beds,price):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.cluster import KMeans
    import pandas as pd
    df_city_scale=pd.DataFrame()
    df_city=df_city[(df_city["City"]==ville)] #A changer en str.contains
    df_city=df_city.drop(columns=["City"])
    df_city.loc[len(df_city.index)] = [bathrooms,bedrooms,beds,price]  
    min_max_scaler = MinMaxScaler()#StandardScaler()
    df_city_scale[['Bathrooms', 'Bedrooms', 'Beds', 'Price']] = min_max_scaler.fit_transform(df_city[['Bathrooms', 'Bedrooms', 'Beds', 'Price']])
    kmeans = KMeans(n_clusters= 3)
    label = kmeans.fit_predict(df_city_scale)
    df_city["cluster"]=label
    target_predict=label[-1]
    
    return target_predict,df_city


    
