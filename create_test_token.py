"""Explain how to obtain a real test token without minting synthetic users."""

if __name__ == "__main__":
    print(
        "The AI service no longer signs test JWTs. Authenticate against the local "
        "Express backend and use the returned bearer token; the AI service only "
        "verifies tokens issued by the platform."
    )
