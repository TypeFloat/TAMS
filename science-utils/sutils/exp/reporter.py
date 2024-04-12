import smtplib
from email.mime.text import MIMEText
from typing import Optional


class Reporter:
    def __init__(self, host: str, user: str, password: str) -> None:
        self.server = smtplib.SMTP()
        self.server.connect(host, 25)
        self.server.login(user, password)

    def __del__(self):
        self.server.quit()

    def send_mail(
        self, title: str, content: str, from_email: str, to_email: Optional[str] = None
    ) -> None:
        message = MIMEText(content, 'plain', 'utf-8')
        message['Subject'] = title
        message['From'] = from_email
        message['To'] = to_email if to_email else from_email

        self.server.sendmail(message['From'], message['To'], message.as_string())
