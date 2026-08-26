"""Ground-truth question set for the rag-kit vs LlamaIndex benchmark.

Every expected phrase has been verified to appear verbatim in the corpus
(benchmark/corpus/sqlite3.rst) — see the corpus-check section in
run_benchmark.py. Scoring is pure string matching (no LLM judges).
"""

# (question, [expected phrases]) — answer is correct if ANY phrase appears
# in the normalized answer text.
QUESTIONS = [
    ("What is the name of the Python module documented here?",
     ["sqlite3"]),
    ("Which version of the DB-API does the sqlite3 module implement?",
     ["2.0"]),
    ("SQLite itself is written in what programming language?",
     ["c library"]),
    ("What special string do you pass to sqlite3.connect() to create an in-memory database?",
     [":memory:"]),
    ("What is the default placeholder style used for parameters in sqlite3?",
     ["qmark"]),
    ("What prefix or syntax marks named placeholders in SQL statements?",
     [":name", "colon"]),
    ("What object does Connection.execute() return?",
     ["cursor"]),
    ("Which Connection method executes a script containing multiple SQL statements at once?",
     ["executescript"]),
    ("Which Cursor method executes the same statement repeatedly for a sequence of parameter sets?",
     ["executemany"]),
    ("What is the base exception class defined by the sqlite3 module?",
     ["sqlite3.error"]),
    ("What exception is raised when a database operation fails because the database is locked?",
     ["operationalerror"]),
    ("Which row factory returns rows that can be accessed by column name?",
     ["sqlite3.row"]),
    ("Which Connection attribute reports the total number of rows modified, inserted, or deleted?",
     ["total_changes"]),
    ("Which sqlite3 module function registers a callable that adapts a Python object to an SQLite value?",
     ["register_adapter"]),
    ("Which sqlite3 module function registers a callable that converts SQLite values into custom Python types?",
     ["register_converter"]),
    ("Which parameter of sqlite3.connect() enables detection of declared column types?",
     ["detect_types"]),
    ("Which Connection attribute controls whether the connection may be used from other threads?",
     ["check_same_thread"]),
    ("What language construct does the documentation recommend for managing connections and transactions?",
     ["context manager", "with statement"]),
    ("Which Cursor attribute holds the row id of the last inserted row?",
     ["lastrowid"]),
    ("Which Connection attribute controls autocommit behavior?",
     ["autocommit"]),
]

# Short ids used in output tables
QUESTION_IDS = [f"Q{i+1}" for i in range(len(QUESTIONS))]
