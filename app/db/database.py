"""
database.py
-----------
This file sets up the connection to our SQLite database.

Think of this like setting up a phone line between our Python code
and the database file on disk (xoodrip.db).

- `engine`       → the actual connection to the database file
- `SessionLocal` → a factory that creates temporary "work sessions"
                   (like opening and closing a notebook each time you write)
- `Base`         → a base class that all our database table models will inherit from
- `get_db()`     → a FastAPI helper that opens a session per request, then closes it cleanly
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# This tells SQLAlchemy to use a SQLite file called `xoodrip.db`
# located in the current working directory (the project root).
# The three slashes `///` mean "relative to the current directory".
DATABASE_URL = "sqlite:///./xoodrip.db"

# Create the engine — this is the actual connection to the DB file.
# `check_same_thread=False` is required for SQLite when used with FastAPI,
# because FastAPI can handle multiple requests at once.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# SessionLocal is a class. Every time we call SessionLocal(), we get
# a fresh database "session" — a temporary workspace for reading/writing.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Base is the foundation class for all our ORM models (database tables).
class Base(DeclarativeBase):
    pass


def get_db():
    """
    FastAPI dependency that yields a database session per request.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)):
            ...

    This ensures the session is always closed after each request,
    even if something goes wrong (the `finally` block guarantees cleanup).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
