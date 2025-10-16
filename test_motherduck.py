#!/usr/bin/env python3
"""
Quick test to verify MotherDuck connection
"""
import os

# Load token from secrets or environment
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")
if not MOTHERDUCK_TOKEN:
    try:
        import streamlit as st
        MOTHERDUCK_TOKEN = st.secrets.get("MOTHERDUCK_TOKEN")
    except:
        pass

if not MOTHERDUCK_TOKEN:
    print("❌ MOTHERDUCK_TOKEN not found in environment or secrets")
    exit(1)

print("✅ MotherDuck token found!")
print(f"   Token starts with: {MOTHERDUCK_TOKEN[:50]}...")

try:
    import duckdb
    print("✅ DuckDB module installed")

    # Test connection
    connection_string = f"md:?motherduck_token={MOTHERDUCK_TOKEN}"
    print("🔌 Connecting to MotherDuck...")
    con = duckdb.connect(connection_string)

    # List databases
    dbs = con.execute("SHOW DATABASES").fetchall()
    print(f"✅ Connected successfully!")
    print(f"   Your databases: {[db[0] for db in dbs]}")

    con.close()
    print("\n🎉 MotherDuck is ready to use!")

except ImportError:
    print("⚠️  DuckDB not installed yet. Run: pip install duckdb")
except Exception as e:
    print(f"❌ Connection error: {e}")

