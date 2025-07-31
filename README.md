# NestQuest - Personalized Real Estate Search

HomeMatch is a real estate search tool that leverages vector databases and language models to find home listings tailored to a buyer’s preferences. It embeds property descriptions and buyer preferences into a vector space, enabling semantic search and personalized recommendations.

---

## Project Structure

- **data/listings.json**  
  Contains all the home listings with details like neighborhood, price, bedrooms, bathrooms, size, and descriptions.

- **data/preferences.py**  
  Contains predefined buyer preference questions and example answers to capture what a user is looking for in a home.

- **db/**  
  Directory where the vector database files are stored after creation. This folder persists the indexed embeddings and metadata for efficient querying.

- **app/create_vector_db.py**  
  Script to generate embeddings from the listings and create the vector database. Run this once or whenever the listings data is updated.

- **app/main.py**  
  The main script that loads the vector database and queries it based on buyer preferences. This returns personalized home listings matching the user’s criteria.

- **examples/**
  Contains text files with examples of the code run on different questions/answers   


## Getting Started

1. **Prepare your environment and install dependencies**  
Set up a conda environment and run.

conda env create -f environment.yml

2. **Set your OpenAI API key**  
Create a `.env` file with your API key and endpoint URL as below

OPENAI_API_KEY=YOUR_API_KEY
BASE_URL=YOUR_LLM_ENDPOINT



3. **Populate the data/listings.json file as well as the preferences.py file (or use the ones provided)**  
If creating a new listings.json file, please match the format in the current file.  


4. **Create the vector database**  
Run the database creation script to process listings and build the vector index:

python app/create_vector_db.py

This creates a vector db and persists it to disk. 
Note that this repo comes with an-already created database. Delete this if starting from scratch with a new listings file. 
If using the provided listings.json file, this step can be skipped.


5. **Run the main application**  
Query the database with buyer preferences and get matched listings output to the screen:
python app/main.py



---

## How it Works

- The **create_vector_db.py** script reads the listings from `data/listings.json`, generates vector embeddings for each listing using OpenAI embeddings, and stores them in a LanceDB vector database inside the `db/` directory.

- The **main.py** script loads buyer preferences (from `data/preferences.py`), generates an embedding for those preferences, then performs a similarity search on the vector database to find listings that best match the buyer’s criteria.

---

## Notes

- The project uses **LanceDB** for efficient vector similarity search and persistence.

- Buyer preferences are modeled as natural language prompts, enabling semantic matching beyond exact keyword matches.

- Ensure your OpenAI API key has access to the embedding and chat models used.

---

Feel free to contribute or customize the preference questions and listings data to better suit your use case!

---


