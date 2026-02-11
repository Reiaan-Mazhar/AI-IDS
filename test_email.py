import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

USER = os.getenv("MAIL_USERNAME")
PASSWORD = os.getenv("MAIL_PASSWORD")

print(f"Testing email...")
print(f"User: {USER}")
print(f"Password length: {len(PASSWORD) if PASSWORD else 0}")

msg = EmailMessage()
msg.set_content("This is a test email from AI-IDS system.")
msg['Subject'] = "AI-IDS Test Alert"
msg['From'] = USER
msg['To'] = USER # Send to self

try:
    print("Connecting to smtp.gmail.com...")
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    print("Logging in...")
    server.login(USER, PASSWORD)
    print("Sending email...")
    server.send_message(msg)
    server.quit()
    print("✅ Success! Email sent.")
except Exception as e:
    print(f"❌ Failed: {e}")
