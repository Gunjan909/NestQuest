from openai import OpenAI
import os
import sys
from dotenv import load_dotenv, find_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preferences import questions, answers  #load the users answers to questions
import lancedb
from openai import OpenAI


# Loads .env from parent directories too
load_dotenv(find_dotenv())

script_dir = os.path.dirname(os.path.abspath(__file__))


#function which augments returned listings with LLM
def augment_listing(listing_text, buyer_preferences):
    prompt = f"""
You are a real estate assistant.

Listing description:
\"\"\"
{listing_text}
\"\"\"

Buyer preferences:
\"\"\"
{buyer_preferences}
\"\"\"

Rewrite the listing description to subtly emphasize the features that match the buyer’s preferences,
making it more appealing while keeping all factual information intact.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()




#Define and load OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")

# Initialize OpenAI client
client = OpenAI(
    base_url = base_url,
    api_key = api_key
)

# Connect to existing vector database. run vector_db.py first to create it, if it does not already exist
db_path = os.path.abspath(os.path.join(script_dir, "../db/lance_db"))
db = lancedb.connect(db_path)
collection = db.open_table("real_estate_listings")

#a set of user preferences defined in data/preferences.py
user_pref_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]) # Combine and parse preferences


# Generate an embedding from the combined preferences
response = client.embeddings.create(
    input=[user_pref_text],
    model="text-embedding-3-small"
)
user_pref_embedding = response.data[0].embedding



#Query LanceDB using this embedding
results = (
    collection.search(user_pref_embedding, "vector")  # 🔄 was "embedding"
    .limit(5)
    .select(["text", "neighborhood", "price", "bedrooms", "bathrooms", "size_sqft"])
    .to_list()
)


# Example:
for listing in results:
    augmented_text = augment_listing(listing["text"], user_pref_text)
    print("Augmented Description:\n", augmented_text)
    print("\n")