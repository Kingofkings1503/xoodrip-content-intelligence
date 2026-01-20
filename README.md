# Xoodrip Content Intelligence Service

This repository contains a **multimodal content intelligence microservice** for Xoodrip.
It dynamically categorizes posts (text, images, videos, polls) without predefined categories
and exposes the functionality via secured APIs.

---

## 🚀 Key Features
- Dynamic category creation using embeddings & clustering
- Supports text, images, videos, tags, and polls
- CLIP-based multimodal embedding pipeline
- API key–secured FastAPI service
- Designed as a plug-and-play microservice for Xoodrip backend

---

## 🏗️ Architecture Overview

Xoodrip Backend  
→ calls secured API  
→ Content Intelligence Service  
→ returns category + metadata

---

## 🔐 Security
- API key authentication for service-to-service communication
- No ML logic exposed directly to clients

---

## 🧰 Tech Stack
- Python
- FastAPI
- PyTorch
- CLIP
- NumPy, scikit-learn

---

## 📦 Project Structure

