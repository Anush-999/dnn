"""Generate a small synthetic spam/ham dataset so the project runs out of the box.
For your real submission, replace data/spam.csv with a real dataset (see README)."""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

spam_templates = [
    "Congratulations! You have won a {prize}. Click {link} to claim now",
    "URGENT: Your account has been suspended. Verify your details at {link} immediately",
    "Get {pct}% off {product} today only! Buy now at {link}",
    "You are selected for a free {prize}. Reply YES to claim",
    "Earn ${amt} per day working from home. No experience needed. Visit {link}",
    "Your loan of ${amt} is pre-approved. Apply now {link}",
    "Final notice: claim your {prize} before it expires. Call now",
    "Hot singles in your area want to meet you. Click {link}",
    "Limited offer! Cheap {product} no prescription needed. Order at {link}",
    "Your lottery ticket won ${amt}! Send your bank details to receive the money",
    "FREE entry to win a {prize}. Text WIN to 80085 now",
    "Dear customer, your password expires today. Update it here {link}",
]
ham_templates = [
    "Hi {name}, are we still meeting for lunch tomorrow?",
    "Please find attached the report for the {topic} meeting",
    "Can you send me the notes from yesterday's {topic} class?",
    "Reminder: {topic} assignment is due on Friday",
    "Thanks {name}, I will review the document and get back to you",
    "Let's catch up this weekend, I am free after 5 pm",
    "The {topic} project review is scheduled for 3 pm in the seminar hall",
    "Happy birthday {name}! Have a great day",
    "Could you share the lab record for {topic}? I missed the session",
    "Mom called, please call her back when you are free",
    "Team standup moved to 10 am tomorrow. Please join on time",
    "I have pushed the latest code to the repo, please check the {topic} module",
]
prizes = ["iPhone 15", "free vacation", "gift voucher", "cash prize", "new laptop"]
links = ["http://bit.ly/win123", "www.claim-prize.biz", "http://secure-login.xyz", "http://free-offer.top"]
products = ["watches", "pills", "sunglasses", "software", "shoes"]
names = ["Arun", "Priya", "Karthik", "Divya", "Ravi", "Meena"]
topics = ["DBMS", "machine learning", "Java", "IoT", "statistics", "web dev"]

def make(kind):
    t = random.choice(spam_templates if kind == "spam" else ham_templates)
    s = t.format(
        prize=random.choice(prizes), link=random.choice(links),
        pct=random.choice([30, 50, 70, 90]), product=random.choice(products),
        amt=random.choice([500, 1000, 5000, 100000]),
        name=random.choice(names), topic=random.choice(topics),
    )
    if kind == "spam" and random.random() < 0.3:
        s = s.upper()
    return s

rows = [(("spam", make("spam"))) for _ in range(1500)] + [(("ham", make("ham"))) for _ in range(2500)]
random.shuffle(rows)
Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows, columns=["label", "text"]).to_csv("data/spam.csv", index=False)
print("Wrote data/spam.csv with", len(rows), "rows")
