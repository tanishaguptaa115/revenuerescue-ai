from backend.app.core.database import supabase


def main() -> None:
    response = (
        supabase
        .table("merchants")
        .select("id")
        .limit(1)
        .execute()
    )

    print("Supabase connection successful.")
    print("Response:", response.data)


if __name__ == "__main__":
    main()