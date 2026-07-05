"""
mongodb.py
----------
This file sets up the connection to MongoDB Atlas.

Key terms:
  - DATABASE   → Like a folder that groups related data (e.g. "xoodrip_intelligence")
  - COLLECTION → Like a table in SQL (e.g. "categories")
  - DOCUMENT   → Like a row in SQL, but it's a JSON object (e.g. {"name": "Cricket", ...})
  - _id        → Every document gets a unique `_id` (like a primary key)

SQL vs MongoDB:
  SQL table     →  MongoDB collection
  SQL row       →  MongoDB document
  SQL column    →  MongoDB field
  SQL schema    →  No fixed schema! Documents can have different fields

WHAT IS MOTOR?
--------------
Motor is the ASYNC driver for MongoDB in Python.
  - "Async" means it doesn't block your server while waiting for the database.
  - FastAPI is async, so we use Motor (not regular pymongo) to avoid blocking.
  - Think of it like: pymongo is a phone call (you wait on the line),
    Motor is a text message (you send it and continue doing other things).

HOW THIS FILE WORKS:
--------------------
  - `client`   → The connection to your MongoDB Atlas cluster
  - `db`       → The specific database inside that cluster
  - `connect_to_mongo()`  → Opens the connection (called once at server startup)
  - `close_mongo()`       → Closes it cleanly (called at server shutdown)
  - `get_db()`            → FastAPI dependency that gives routes access to `db`
"""

from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

# These start as None and are set when the server starts up
client: AsyncIOMotorClient | None = None
db = None


async def connect_to_mongo():
    """
    Open the connection to MongoDB Atlas.

    Called once in main.py's lifespan startup.
    After this runs, `db` points to our database and is ready for queries.
    """
    global client, db

    # Create the client — Motor handles connection pooling automatically
    # (it keeps a pool of connections open so we don't reconnect every request)
    client = AsyncIOMotorClient(settings.MONGO_URI)

    # Select our database — if it doesn't exist, MongoDB creates it
    # automatically when we first insert data (no CREATE DATABASE needed!)
    db = client[settings.MONGO_DB_NAME]

    # Quick check: try to reach the server
    # (this will raise an error if the connection string is wrong)
    await client.admin.command("ping")
    print(f"✅ Connected to MongoDB Atlas — database: {settings.MONGO_DB_NAME}")


async def close_mongo():
    """
    Close the MongoDB connection cleanly.

    Called once in main.py's lifespan shutdown.
    """
    global client
    if client:
        client.close()
        print("🔌 MongoDB connection closed.")


def get_db():
    """
    FastAPI dependency that provides the database handle to route functions.

    Usage in a route:
        async def my_route(db = Depends(get_db)):
            await db.categories.find_one({"domain": "sports"})

    Note: Unlike SQLAlchemy's get_db(), this is NOT a generator.
    Motor's client handles connection pooling internally, so we don't
    need to open/close anything per-request. We just return the
    database handle directly.
    """
    return db
