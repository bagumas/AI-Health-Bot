# AI-Health-Bot
GitHub readme

# 🧠 HealthBot – AI-Powered Medical Assistant

[![Hugging Face Spaces](https://img.shields.io/badge/HuggingFace-HealthBot-blue?logo=huggingface)](https://huggingface.co/spaces/sbaguma/HealthBot)

**HealthBot** is a conversational AI assistant that helps users check symptoms, assess potential health risks, receive precautionary advice, and explore common medical conditions — all through natural language interaction.

🔗 **Live Demo:** [https://huggingface.co/spaces/sbaguma/HealthBot](https://huggingface.co/spaces/sbaguma/HealthBot)

---

## 📌 Overview

This project showcases the use of transformer-based models, LangChain agents, and vector search to deliver a responsive and informative chatbot for healthcare applications.

HealthBot supports multi-turn conversations and uses a modular tool-based architecture to simulate a virtual health assistant.

---

## 🎯 Features

- ✅ Symptom Checker: Match free-text symptoms to potential conditions
- ✅ Risk Estimator: Evaluate severity levels based on symptom patterns
- ✅ Precaution Advisor: Recommend safety measures and actions
- ✅ FAQ Agent: Retrieve health-related information and common questions
- ✅ Conversational Memory: Maintain chat context across turns

---

## 🛠️ Tech Stack

| Category | Tools & Frameworks |
|---------------------|-----------------------------------------------|
| Language Models | Hugging Face Transformers, OpenAI API |
| Orchestration | LangChain (multi-agent tool-based setup) |
| Vector Search | FAISS (for symptom/FAQ similarity search) |
| UI/Frontend | Gradio (hosted on Hugging Face Spaces) |
| Language | Python |

---

## 📂 Project Structure

```bash
├── app.py # Main Gradio app
├── agents/
│ ├── symptom_checker.py # Symptom matching tool
│ ├── risk_estimator.py # Severity risk scoring
│ ├── precaution_advisor.py# Precaution advice generator
│ └── faq_agent.py # Disease FAQ agent
├── vector_store/ # FAISS index and embedding store
├── prompts/ # Prompt templates for each tool
├── requirements.txt # Dependencies
└── README.md # Project documentation
