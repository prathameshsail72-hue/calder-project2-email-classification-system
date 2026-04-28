import pandas as pd
import random
import os

random.seed(42)
os.makedirs("data", exist_ok=True)

print("Loading Kaggle dataset...")
df_raw = pd.read_csv("data/customer_support_tickets.csv")
print(f"Loaded {len(df_raw)} rows")
print(f"Columns: {list(df_raw.columns)}")
print(f"\nOriginal label distribution:")
print(df_raw["Ticket Type"].value_counts())

label_map = {
    "Product inquiry"    : "Inquiry",
    "Billing inquiry"    : "Inquiry",
    "Technical issue"    : "Complaint",
    "Refund request"     : "Complaint",
    "Cancellation request": "Complaint",
}

df_raw["category"] = df_raw["Ticket Type"].map(label_map)
df_mapped = df_raw[["Ticket Description", "category"]].copy()
df_mapped.columns = ["email_text", "category"]

print(f"\nAfter mapping:")
print(df_mapped["category"].value_counts())

feedback_emails = [
    "I just wanted to say that your customer support team was incredibly helpful today. The issue was resolved quickly.",
    "The new dashboard update is fantastic and has made my workflow much smoother. Great improvement.",
    "I have been using your service for over a year and it has been a great experience overall. Very reliable.",
    "The onboarding process was very well designed and easy to follow. Helped me get started quickly.",
    "I would suggest adding a dark mode option to improve the overall user experience on the platform.",
    "Your recent blog post on data security was very informative and well written. Keep it up.",
    "The mobile app works really well and the interface is clean and intuitive. Excellent design.",
    "I appreciate how quickly your team resolved my issue last week. Truly impressive response time.",
    "The product quality is excellent and far exceeded my expectations. Very happy with the purchase.",
    "I would love to see a bulk export feature added in a future update. That would save a lot of time.",
    "The webinar you hosted last week was very educational and well organized. Please do more of these.",
    "Thank you for the personalized recommendations. They have been very useful for my workflow.",
    "I think adding keyboard shortcuts would make the platform even more efficient for power users.",
    "The checkout process is very smooth and the payment confirmation was instant. Great experience.",
    "Your pricing is very competitive compared to similar services in the market. Good value for money.",
    "I really enjoy the weekly newsletter. The content is relevant and well curated. Keep it coming.",
    "The search functionality works perfectly and saves me a lot of time every day. Well implemented.",
    "I would recommend your service to anyone looking for a reliable and affordable solution.",
    "The recent performance improvements have made the platform noticeably faster. Big difference.",
    "I love the new collaboration features added in the latest version. Makes teamwork much easier.",
    "The tutorial videos are very clear and helped me get started quickly without needing support.",
    "Your team went above and beyond to help me set up my account properly. Truly great service.",
    "I think a progress tracker for long running tasks would be a great addition to the platform.",
    "The analytics dashboard gives me exactly the insights I need for my business decisions.",
    "Overall I am very satisfied with the service and will definitely continue using it long term.",
    "Just wanted to share that the latest update fixed all the issues I was experiencing before.",
    "The documentation is very thorough and well organized. Made integration much easier for me.",
    "I appreciate that you offer a free trial. It gave me confidence before committing to a plan.",
    "The notification system works perfectly and keeps me updated without being too intrusive.",
    "Your platform is by far the best I have used in this category. Well done to the entire team.",
    "I wanted to suggest adding multi-language support as it would help international users like me.",
    "The customer portal is very user friendly and easy to navigate. Everything is where I expect it.",
    "I love that you release regular updates with meaningful improvements. Shows commitment to quality.",
    "The integration with third party tools works seamlessly. Saved me hours of manual work.",
    "I was impressed by how fast the support team responded even on a weekend. Really appreciated.",
    "The product has helped our team increase productivity by at least 30 percent. Excellent tool.",
    "I appreciate the transparency in your pricing page. No hidden fees makes the decision easy.",
    "Your community forum is very active and helpful. Found answers to most of my questions there.",
    "The export feature works great and the file formats supported cover all my needs perfectly.",
    "I have referred three colleagues to your platform already. Everyone has been very happy so far.",
    "The loading speed has improved dramatically since the last update. Very noticeable difference.",
    "I wanted to say thank you for adding the feature I requested six months ago. Works perfectly.",
    "The user interface is clean modern and professional. Makes daily use a pleasant experience.",
    "I think adding a calendar integration would be a great feature for project management users.",
    "Your platform consistently delivers on what it promises. That kind of reliability is rare.",
    "The backup and restore feature saved me when I accidentally deleted important data. Lifesaver.",
    "I appreciate that accessibility features are built in. Makes the platform usable for everyone.",
    "The recent changes to the notification preferences were exactly what I had been hoping for.",
    "Your support articles are very detailed and usually answer my questions without needing to call.",
    "I have been very impressed with the consistency of service quality over the past two years.",
    "The new report templates are very professional and save me significant time every month.",
    "I like that you provide a detailed changelog with every update. Keeps users well informed.",
    "The two factor authentication setup was straightforward and gives me confidence in security.",
    "Your platform scales really well as our team has grown from 5 to 50 users without any issues.",
    "I wanted to highlight that your onboarding emails were very helpful during my first week.",
    "The custom dashboard feature is exactly what I needed to monitor my key metrics daily.",
    "I am glad you added the option to schedule reports. It has automated a big part of my workflow.",
    "The product comparison feature on your website helped me choose the right plan easily.",
    "I think the platform is excellent value and I regularly recommend it to other business owners.",
    "The data visualization options are comprehensive and produce professional looking charts.",
    "Your team clearly listens to user feedback. Many features I requested have been implemented.",
    "The performance on mobile devices is excellent. Works just as well as the desktop version.",
]

df_feedback = pd.DataFrame({
    "email_text": feedback_emails,
    "category"  : "Feedback"
})

print(f"\nAdding {len(df_feedback)} synthetic Feedback emails...")

n_inquiry   = len(df_mapped[df_mapped["category"] == "Inquiry"])
n_complaint = len(df_mapped[df_mapped["category"] == "Complaint"])
n_feedback  = len(df_feedback)

print(f"\nBefore balancing:")
print(f"  Inquiry   : {n_inquiry}")
print(f"  Complaint : {n_complaint}")
print(f"  Feedback  : {n_feedback}")

sample_size = min(n_inquiry, n_complaint, 500) 

df_inquiry   = df_mapped[df_mapped["category"] == "Inquiry"].sample(
    sample_size, random_state=42)
df_complaint = df_mapped[df_mapped["category"] == "Complaint"].sample(
    sample_size, random_state=42)

df_feedback_up = df_feedback.sample(
    sample_size, replace=True, random_state=42)

df_final = pd.concat(
    [df_inquiry, df_complaint, df_feedback_up],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal balanced dataset:")
print(df_final["category"].value_counts())
print(f"Total rows: {len(df_final)}")

df_final.to_csv("data/emails_clean.csv", index=False)
print("\n✅ Saved to data/emails_clean.csv")