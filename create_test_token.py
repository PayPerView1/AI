"""
Utility script to generate valid test JWT Bearer Tokens for testing endpoints.
"""
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.core.security import create_access_token

if __name__ == "__main__":
    print("=" * 60)
    print("[+] PAYPERVIEW AI SERVICE - TEST JWT TOKEN GENERATOR")
    print("=" * 60)

    # 1. Clipper Token
    clipper_token = create_access_token(
        user_id="usr_clipper_123",
        role="CLIPPER",
        email="clipper@example.com",
        name="Ahmad Creator",
    )

    # 2. Brand Token
    brand_token = create_access_token(
        user_id="usr_brand_456",
        role="BRAND",
        email="brand@example.com",
        name="Tech Brand Marketing",
    )

    print("\n1. CLIPPER ROLE TOKEN (Content Creator):")
    print(f"Bearer {clipper_token}\n")

    print("2. BRAND ROLE TOKEN (Advertiser):")
    print(f"Bearer {brand_token}\n")

    print("=" * 60)
    print("[*] Copy the Bearer token above and use it in HTTP headers:")
    print("Authorization: Bearer <TOKEN>")
    print("Or input it into Swagger UI at: http://localhost:8001/docs")
    print("=" * 60)
