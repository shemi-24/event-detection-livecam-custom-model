import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(to_email, subject, message):
    sender_email = "support@zoftcares.in"  # Replace with your email
    sender_password = "U_Dr*0R($q&?"       # Replace with your email password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP_SSL("mail.zoftcares.in", 465)  # Use SSL instead of SMTP
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully!")
        return True  # Indicate success
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False  # Indicate failure

# Test email sending
email_sent = False  # Initialize flag

if send_email("hashirkp13@gmail.com", "Fall Detected!", "A person has fallen. Immediate attention needed!"):
    email_sent = True  # Set flag to True if email was sent successfully
