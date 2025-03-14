import pandas as pd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from OApolys_todb import upload_geometries_to_postgis, update_clusters



from sqlalchemy import MetaData, Table
from sqlalchemy.exc import SQLAlchemyError

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
# Database connection info using the provided format
db_info = {
    'dbname': 'gis',
    'user': 'gis',
    'password': 'password',
    'host': 'localhost',
    'port': '5432'
}

# Construct the database URL using the db_info dictionary
db_url = f'postgresql+psycopg2://{db_info["user"]}:{db_info["password"]}@{db_info["host"]}:{db_info["port"]}/{db_info["dbname"]}'
DATABASE_URL = db_url
# Create the database engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
from sqlalchemy import MetaData, Table, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

import pandas as pd
import geopandas as gpd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Function to upload geometries
def upload_geometries_to_postgis(gdf):
    try:
        # Ensure the geometries are in a valid CRS (preferably EPSG:4326)
        # gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        # Upload the geometries table, only the 'geom' column will be uploaded
        #OA as the primary key
      
        gdf[['OA', 'geometry','cluster']].to_postgis("geometries", engine, if_exists="replace", index=False)
        # gdf[['geometry']].to_postgis("geometries", engine, if_exists="replace", index=False)
        print("Geometries uploaded successfully!")

    except SQLAlchemyError as e:
        print(f"Error uploading geometries: {e}")
        raise

#select the input vars
def select_input_vars(selected_vars=None):
    df = pd.read_csv("data/OAC_output/variable_data_normed.csv")
    #set outout area as the index
    df.set_index('OA', inplace=True)
    if selected_vars is not None:
        df = df[selected_vars]
    return df

def remove_corrolated_vars(df):
        # Compute the correlation matrix
    corr_matrix = df.corr()

    # Set a correlation threshold, e.g., 0.9
    threshold = 0.9

    # Find the pairs of highly correlated columns
    # Create a mask to exclude the diagonal (correlation of a column with itself)
    highly_corr = np.where((corr_matrix > threshold) & (corr_matrix != 1))

    # List to keep track of columns to remove
    columns_to_remove = set()

    # Loop through the pairs and add one column to remove for each pair of high correlation
    for i, j in zip(*highly_corr):
        colname_i = corr_matrix.columns[i]
        colname_j = corr_matrix.columns[j]
        
        # Keep the column that has the most information or less missing data
        if colname_i not in columns_to_remove:
            columns_to_remove.add(colname_j)  # Add the second column of the pair for removal

    # Remove the columns with high correlation
    df_cleaned = df.drop(columns=columns_to_remove)

    
    #remove columns with any missing values
    df_cleaned = df_cleaned.dropna(axis=1)
    # Remove the problematic variables in python
    ex_variables = ["v07","v09","v11","v23","v42","v60","v32","v40","v41"]
    df_cleaned = df_cleaned.drop(columns=ex_variables, errors='ignore')

    return df_cleaned


def transform_data(df_cleaned):
    
    ## or arcsinh transform
    df_cleaned = np.arcsinh(df_cleaned)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_cleaned)

    return df_scaled


#run cluster
def run_clustering(df_cleaned, df_scaled, num_clusters=8):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=44)
    df_cleaned['cluster'] = kmeans.fit_predict(df_scaled)
    #add one to each cluster to avoid zero
    df_cleaned['cluster'] = df_cleaned['cluster'] + 1
    # Update the clusters in the database
    #print(uploading now)
    cluster_data = df_cleaned[['cluster']].reset_index()
    # Load OA boundaries (adjust file path)
    oa_boundaries = gpd.read_parquet("data/OAbounds/oabounds.parquet")
    #save to 

    # Merge cluster data with OA boundaries
    merged = oa_boundaries.merge(cluster_data, left_on='OA', right_on='OA')  # Replace 'OA_column_name' with the correct column name

    upload_geometries_to_postgis(merged)

    return df_cleaned

def run_geodem(num_clusters=1,selected_vars=None):
    # Load the input variables
    df = select_input_vars(selected_vars=selected_vars)
    # Remove highly correlated variables
    df_cleaned = remove_corrolated_vars(df)
    # Transform the data
    df_scaled = transform_data(df_cleaned)
    # Run clustering
    run_clustering(df_cleaned, df_scaled, num_clusters=num_clusters)


def get_corr(selected_vars=None):
    # Load the input variables
    df = select_input_vars(selected_vars=selected_vars)
    # Remove highly correlated variables
    return df.corr()

def get_index_table(num_clusters=1,selected_vars=None):
    # Load the input variables
    df = select_input_vars(selected_vars=selected_vars)
    # Remove highly correlated variables
    df_cleaned = remove_corrolated_vars(df)
    # Transform the data
    df_scaled = transform_data(df_cleaned)
    # Run clustering
    df_cleaned = run_clustering(df_cleaned, df_scaled, num_clusters=num_clusters)

        # Compute the mean of each cluster
    cluster_means = df_cleaned.groupby('cluster').mean()
    #compute the overall mean
    overall_mean = df_cleaned.mean()
    # Compute the ratio of each cluster mean to the overall mean *100

    cluster_means_ratio = (cluster_means - overall_mean)/overall_mean
    cluster_means_ratio.drop(columns='cluster', inplace=True)

    return cluster_means_ratio

#read possible input vars
#load input df
def load_input_vars():
    #data/OAC_output/oac_variable_matches.csv
    df = pd.read_csv("data/OAC_output/oac_variable_matches.csv")
    df=df[["OACcode","OACname"]]
    return df

from clustergram import Clustergram
import matplotlib.pyplot as plt
import io
import base64

def make_clustergram_plot(input_vars=None):
    # Load the input variables
    df = select_input_vars(selected_vars=input_vars)
    # Remove highly correlated variables
    df_cleaned = remove_corrolated_vars(df)
    # Transform the data
    df_scaled = transform_data(df_cleaned)
    # Create a clustergram
    cgram = Clustergram(range(1, 12))
    cgram.fit(df_scaled)
    
    # Plot the clustergram using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    cgram.plot(
        ax=ax
    )
    
    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)
    
    # Encode the buffer to a base64 string
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    
    return image_base64