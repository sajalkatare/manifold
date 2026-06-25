
POLICY_WINDOW_DAYS= 30
KNOWN_TIERS = ['premium', 'standard']

def calculate_refund_eligibility(days_since_purchase: int, customer_tier: str) -> dict:
    """
    Decide whether a purchase is eligible for a refund based 
    on the number of days since purchase and the customer tier.
    Args:
        days_since_purchase: The number of days since the purchase
        customer_tier: The tier of the customer - 'premium' or 'standard'
    Returns:
        {
            "eligible": bool,
            "reason": str,
            "policy_window_days": int,
        }
    """
    # Boundry check 1 - number of days since purchase must be a non-negative integer
    problems = []
    if not isinstance(days_since_purchase, int) or days_since_purchase < 0:
        problems.append("days_since_purchase must be a non-negative integer")
    
    # Boundry check 2 - customer tier must be a valid tier
    tier = customer_tier.lower() if isinstance(customer_tier, str) else ""
    if tier not in KNOWN_TIERS:
        problems.append(f"customer_tier must be one of: {', '.join(KNOWN_TIERS)}")
    
    if problems:
        return {
            "eligible": False,
            "reason": "Invalid input: " + ", ".join(problems),
            "policy_window_days": POLICY_WINDOW_DAYS,
        }
    
    # Happy Path
    eligible = days_since_purchase <= POLICY_WINDOW_DAYS
    if eligible:
        return {
            "eligible": True,
            "reason": f"Purchase is eligible for a refund within the last {POLICY_WINDOW_DAYS} days",
            "policy_window_days": POLICY_WINDOW_DAYS,
        }
    else:
        return {
            "eligible": False,
            "reason": f"Purchase is not eligible for a refund because it is outside the last {POLICY_WINDOW_DAYS} days",
            "policy_window_days": POLICY_WINDOW_DAYS,
        }

def main():
    print("Demo 1: Basic Tool")
    print("==================")
    print("Running with input: days_since_purchase=20, customer_tier='premium'")
    result = calculate_refund_eligibility(20, 'premium')
    print(result)
    print("--------------------------------")
    print("Running with input: days_since_purchase=40, customer_tier='standard'")
    result = calculate_refund_eligibility(40, 'standard')
    print(result)
    print("--------------------------------")
    print("Running with input: days_since_purchase=-10, customer_tier='premium'")
    result = calculate_refund_eligibility(-10, 'premium')
    print(result)

if __name__ == "__main__":
    main()