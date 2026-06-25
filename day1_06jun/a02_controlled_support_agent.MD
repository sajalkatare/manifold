# FAQ

## 1. What capability did I give the agent?

I added a new tool that allows the agent to calculate and provide shipping charges based on the customer's membership tier and selected payment method.

---

## 2. What tool did I create and when should it be used?

I created the `calculate_shipping_charge` tool. This tool should be used whenever a customer asks about shipping fees based on their payment method and customer tier.

---

## 3. What state fields did my graph track?

The graph tracks the following state fields:

* `customer_id`
* `tier`
* `payment_type`
* `shipping_charges`
* `issue_category`

---

## 4. What routing decision did my graph make?

The graph changes its routing logic based on the customer's tier. Different customer tiers can follow different processing paths.

---

## 5. Why is this more controlled than a simple agent?

This approach is more controlled than a simple agent because the workflow follows predefined conditions and routing rules. The agent cannot make arbitrary decisions and instead routes requests based on specific business logic, making the behavior more predictable and reliable.
