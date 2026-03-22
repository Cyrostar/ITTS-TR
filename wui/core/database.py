import os
import sqlite3
import logging

class SQLiteManager:
    def __init__(self, db_path: str):
        """Initializes the SQLite manager with a specific database path."""
        self.db_path = db_path
        # Ensure the directory exists before connecting
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_connection(self):
        """Establishes and returns a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Returns rows as dictionary-like objects
        return conn

    def create_table(self, table_name: str, schema: str) -> bool:
        """
        Creates a new table if it doesn't exist.
        Example schema: 'id INTEGER PRIMARY KEY, name TEXT, audio_path TEXT'
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        return self.execute_write(query)

    def execute_write(self, query: str, params: tuple = ()) -> bool:
        """Executes single INSERT, UPDATE, or DELETE queries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return True
        except sqlite3.Error as e:
            logging.error(f"SQLite Write Error: {e} | Query: {query}")
            return False

    def fetch_all(self, query: str, params: tuple = ()) -> list:
        """Executes SELECT queries and returns all matching rows as dictionaries."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"SQLite Fetch Error: {e} | Query: {query}")
            return []

    def fetch_one(self, query: str, params: tuple = ()) -> dict:
        """Executes a SELECT query and returns the first matching row as a dictionary."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                return dict(row) if row else {}
        except sqlite3.Error as e:
            logging.error(f"SQLite FetchOne Error: {e} | Query: {query}")
            return {}

    def execute_many(self, query: str, params_list: list) -> bool:
        """Executes bulk INSERT or UPDATE queries efficiently."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return True
        except sqlite3.Error as e:
            logging.error(f"SQLite Bulk Write Error: {e} | Query: {query}")
            return False

    def update_record(self, table_name: str, set_clause: str, condition: str, params: tuple = ()) -> bool:
        """
        Updates existing records in the database.
        Example: update_record('audio_corpus', 'transcript = ?', 'id = ?', ('New text', 5))
        """
        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
        return self.execute_write(query, params)

    def insert_or_replace(self, table_name: str, columns: str, placeholders: str, params: tuple = ()) -> bool:
        """
        Inserts a new row, or replaces it entirely if a UNIQUE constraint causes a collision.
        Example: insert_or_replace('config', 'key, value', '?, ?', ('wui_lang', 'tr'))
        """
        query = f"REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})"
        return self.execute_write(query, params)

    def count_rows(self, table_name: str, condition: str = "", params: tuple = ()) -> int:
        """Returns the total number of rows in a table, with an optional WHERE condition."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
            
        result = self.fetch_one(query, params)
        return result.get("count", 0) if result else 0
        
    def truncate_table(self, table_name: str, reset_autoincrement: bool = True) -> bool:
        """
        Deletes all rows from a table.
        Optionally resets the AUTOINCREMENT counter for the table.
        """
        success = self.execute_write(f"DELETE FROM {table_name}")
        if success and reset_autoincrement:
            # Reset sequence. It's safe to run even if the table doesn't use AUTOINCREMENT
            self.execute_write("DELETE FROM sqlite_sequence WHERE name=?", (table_name,))
        return success

    def delete_database(self) -> bool:
        """Safely closes any implicit connections and deletes the database file."""
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                return True
            except OSError as e:
                logging.error(f"Error deleting database at {self.db_path}: {e}")
                return False
        return True