# vector_db_lancedb.py
import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import lancedb
import logging
import pandas as pd # <-- IMPORTANT: Added this import for DataFrame operations

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths
db_persist_directory = os.path.join(script_dir, "../db", "lance_db")
listings_file_path = os.path.join(script_dir, "../data", "listings.json")

# Ensure the database directory exists
try:
    os.makedirs(db_persist_directory, exist_ok=True)
    logging.info(f"LanceDB will persist to: {db_persist_directory}")
    logging.info(f"Loading listings from: {listings_file_path}")
except OSError as e:
    logging.error(f"Error creating directory {db_persist_directory}: {e}")
    exit(1)

# Load environment variables
load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

if not api_key:
    logging.error("OPENAI_API_KEY not found in environment variables.")
    exit(1)

# Initialize OpenAI client
try:
    client = OpenAI(
        base_url= base_url,
        api_key=api_key
    )
    logging.info("OpenAI client initialized.")
except Exception as e:
    logging.error(f"Error initializing OpenAI client or connecting: {e}")
    exit(1)

# Load listings from JSON first, as we need data to create the table
try:
    with open(listings_file_path, "r", encoding="utf-8") as f:
        listings = json.load(f)
    logging.info(f"Loaded {len(listings)} listings from {listings_file_path}.")
except FileNotFoundError:
    logging.error(f"Listings file not found at: {listings_file_path}")
    exit(1)
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON from {listings_file_path}: {e}")
    exit(1)


# --- Initialize LanceDB client and table ---
db = lancedb.connect(db_persist_directory)
table_name = "real_estate_listings"
collection = None

# We need all records with embeddings to create the DataFrame for insertion.
all_records_for_lancedb = []

logging.info("Starting to generate embeddings and prepare data for LanceDB...")
for idx, listing in enumerate(listings):
    text_content = (
        f"Neighborhood: {listing['neighborhood']}\n"
        f"Price: ${listing['price']}\n"
        f"Bedrooms: {listing['bedrooms']}, Bathrooms: {listing['bathrooms']}\n"
        f"Size: {listing['size_sqft']} sqft\n"
        f"Description: {listing['description']}\n"
        f"Neighborhood Description: {listing['neighborhood_description']}"
    )

    try:
        response = client.embeddings.create(
            input=[text_content],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        record = {
            "id": f"listing_{idx}",
            "text": text_content,
            "vector": embedding, 
            "neighborhood": listing["neighborhood"],
            "price": listing["price"],
            "bedrooms": listing["bedrooms"],
            "bathrooms": listing["bathrooms"],
            "size_sqft": listing["size_sqft"]
        }
        all_records_for_lancedb.append(record)

    except Exception as e:
        logging.error(f"Error processing listing {idx} (ID: {listing.get('id', 'N/A')}): {e}", exc_info=True)
        logging.error(f"Problematic text content: {text_content[:200]}...")

if not all_records_for_lancedb:
    logging.error("No records processed to add to LanceDB. Exiting.")
    exit(1)

# Convert all records to a Pandas DataFrame for efficient insertion
df_to_insert = pd.DataFrame(all_records_for_lancedb)

try:
    # Check if table exists. If not, create it. If it does, open it.
    if table_name in db.table_names():
        collection = db.open_table(table_name)
        logging.info(f"Opened existing LanceDB table '{table_name}'.")

        # Check if new data needs to be appended
        current_row_count = collection.count_rows()
        if current_row_count < len(df_to_insert):
            logging.info(f"Table '{table_name}' has {current_row_count} rows. Appending {len(df_to_insert) - current_row_count} new rows.")
            # For simplicity, we add all. If IDs are unique, LanceDB will handle appending
            collection.add(df_to_insert)
            logging.info(f"Total rows after append: {collection.count_rows()}")
        else:
            logging.info(f"Table '{table_name}' already contains {current_row_count} rows. No new data appended.")

    else:
        # Create the table using the DataFrame. This also inserts the data.
        collection = db.create_table(table_name, df_to_insert)
        logging.info(f"Created new LanceDB table '{table_name}' and inserted all {len(df_to_insert)} records.")

except Exception as e:
    logging.error(f"Error initializing LanceDB table or inserting data: {e}", exc_info=True)
    exit(1)

logging.info("Listings and embeddings successfully stored/updated in LanceDB.")




# Test Query LanceDB using approximate nearest neighbors search

"""# Prepare query text and embedding
query_text = "home with garden"
try:
    query_response = client.embeddings.create(
        input=[query_text],
        model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding
    logging.info("Query embedding generated.")
except Exception as e:
    logging.error(f"Error generating query embedding: {e}")
    exit(1)

try:
    if collection is None:
        logging.error("LanceDB collection is None, can't perform search. This indicates a critical error during table initialization.")
        exit(1)

    # Correct LanceDB search syntax: chain methods
    # Since the embedding column is named 'vector', no need to specify it in search()
    results = collection.search(query_embedding).limit(3).select(["text", "neighborhood", "price", "bedrooms", "bathrooms", "size_sqft"]).to_list()

    logging.info("🔍 Top matches:")
    if results:
        for res in results:
            print("\n---")
            print(f"Document: {res.get('text', 'N/A')}") # Access 'text' column for document content
            # Extract and print other metadata fields
            metadata = {k: res.get(k, 'N/A') for k in ["neighborhood", "price", "bedrooms", "bathrooms", "size_sqft"]}
            print("Metadata:", metadata)
            print(f"Distance: {res.get('_distance', 'N/A'):.4f}") # LanceDB often includes _distance
    else:
        logging.info("No results found in LanceDB for the query.")

except Exception as e:
    logging.error(f"Error querying LanceDB: {e}", exc_info=True) """