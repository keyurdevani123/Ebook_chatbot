import os
import time
import jwt
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..database.users import Users

router = APIRouter(prefix="/auth", tags=["Auth"])

JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key-change-in-prod")

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str

class LoginRequest(BaseModel):
    email: str
    password: str

def create_jwt_token(user_id: str, email: str):
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": time.time() + 86400 * 7 # 7 days expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

@router.post("/register")
def register(req: RegisterRequest):
    from ..server import users_db
    user = users_db.create_user(req.email, req.password, req.name)
    if not user:
        raise HTTPException(status_code=400, detail="User already exists")
    token = create_jwt_token(user["_id"], user["email"])
    return {"status": True, "token": token, "user": user}

@router.post("/login")
def login(req: LoginRequest):
    from ..server import users_db
    user = users_db.verify_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_jwt_token(user["_id"], user["email"])
    return {"status": True, "token": token, "user": user}
