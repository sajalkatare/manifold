
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
print(str(Path(__file__).resolve().parent))
from shared.failure_simulator import fetch_product
from shared.failure_simulator import simulate_failure as trigger_failure


def get_product_details_unreliable(product_id: int, simulate_failure: str | None = None) -> dict:
    """Fetch product details with no timeout, handling, or safe envelope."""
    print(trigger_failure(simulate_failure))
    return fetch_product(product_id)


def main() -> None:
    print("=" * 64)
    print("DEMO 1 — Unreliable tool")
    print("=" * 64)

    for failure in (None, "timeout", "server_error", "invalid_response"):
        print(f"\n--- simulate_failure={failure!r} ---")
        try:
            result = get_product_details_unreliable(1, simulate_failure=failure)
            print(result)
        except Exception as exc:
            print(f"RAISED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
