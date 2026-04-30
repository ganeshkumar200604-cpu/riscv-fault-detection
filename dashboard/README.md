# 🔬 AI-Based Fault Detection in RISC-V Processor

## 📌 Overview
This project implements an AI-based system to detect faults in a RISC-V processor using machine learning. The system uses QEMU simulation to generate execution data and a Random Forest model to classify faults.

---

## ⚙️ Workflow
1. RISC-V programs are compiled into `.elf` files  
2. Executed in QEMU simulator  
3. Execution metrics collected:
   - cycles
   - instructions
   - CPI
   - stack pointer (sp)
   - return address (ra)
   - exception flag  
4. Data is processed into dataset  
5. Machine learning model trained  
6. Streamlit dashboard used for visualization  

---

## 🧠 Machine Learning Model
- Algorithm: Random Forest  
- Features: 6  
- Dataset: ~4000 rows  
- Accuracy: 100%  

---

## 💻 Web Application
Built using Streamlit with:
- Interactive dashboard  
- Live simulation  
- Manual prediction  
- Data visualization  

---

## 🚀 How to Run
```bash
streamlit run app.py
