from fastapi import APIRouter, Form
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import EmailStr
import os

load_dotenv()

class MailSettings(BaseSettings):
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: EmailStr
    MAIL_PORT: int = 587
    MAIL_SERVER: str
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    USE_CREDENTIALS: bool = True

    class Config:
        env_file = ".env"

settings = MailSettings()

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS=True,          
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=False         
)

router = APIRouter()

@router.post("/feedback")
async def send_feedback(
    message: str = Form(...),
):
    fm = FastMail(conf)

    subject = "New Feedback from OCT & Angio Image AI Platform"
    body = f"""New feedback received:

{message}

---
This is an automated message from the OCT & Angio AI Platform.
"""

    message_obj = MessageSchema(
        subject=subject,
        recipients=[
            "phongphu.07072001@gmail.com",
            "lapthai03@gmail.com",
            "trandinhson3086@gmail.com"
        ],
        body=body,
        subtype="plain"
    )

    await fm.send_message(message_obj)
    return {"status": "success", "message": "Feedback sent!"}
