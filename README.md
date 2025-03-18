# Geodemographic Python Example  

This repository contains the workflow for producing a geodemographic classification in Python using k-means clustering. It follows a simplified process, similar to that described in the [2021 OAC Paper](https://rgs-ibg.onlinelibrary.wiley.com/doi/full/10.1111/geoj.12550).  

## Files  
- **Main notebook:** `1_geodemographic_example.ipynb`  
- **Requirements:** Dependencies are listed in `requirements.txt`  
- **Example data:** `example_oacdata.csv`  

## Setup  

### Using `pip` and a virtual environment  
Create and activate a virtual environment:  
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

Alternatively if using a cloud enviroment (eg google collab) the dependencies can be installed from inside the notebook.
