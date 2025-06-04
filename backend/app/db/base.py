# app/db/base.py
from sqlalchemy.ext.declarative import declarative_base

# Declarative base shared por todos los modelos
Base = declarative_base()