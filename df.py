import pandas as pd

data = {
    "question": [
        "What is your return policy?",
        "How long does shipping take?",
        # Add more questions here
    ],
    "answer": [
        "Our return policy allows returns within 30 days of purchase.",
        "Shipping usually takes 5-7 business days.",
        # Add more answers here
    ]
}

df = pd.DataFrame(data)
df.to_csv("faqs.csv", index=False)