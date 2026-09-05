"""
Supabase client for RevenueRescue AI.

Loads Supabase configuration from environment variables and exposes
a small reusable client factory.

This module does not perform any database operations on import.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Create and cache the Supabase client.

    Required environment variables:
        SUPABASE_URL
        SUPABASE_KEY
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is not configured.")

    if not supabase_key:
        raise RuntimeError("SUPABASE_KEY is not configured.")

    return create_client(
        supabase_url,
        supabase_key,
    )


if __name__ == "__main__":
    client = get_supabase_client()
    print("Supabase client initialized successfully.")