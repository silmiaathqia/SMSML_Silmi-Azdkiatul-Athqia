# 🤖 Sistem Machine Learning End-to-End
### Worker Productivity Classification — MLOps Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12.7-blue?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange?style=for-the-badge&logo=mlflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?style=for-the-badge&logo=grafana&logoColor=white)

**Proyek Akhir Kelas Membangun Sistem Machine Learning — Dicoding**

*oleh Silmi Azdkiatul Athqia (silmiathqia)*

</div>

---

## 📋 Deskripsi Proyek

Proyek ini membangun sistem machine learning **end-to-end** yang siap produksi untuk mengklasifikasikan tingkat produktivitas pekerja remote menjadi tiga kelas: **High**, **Medium**, dan **Low**. Pipeline mencakup seluruh tahapan MLOps mulai dari eksperimen data, pelatihan model, CI/CD otomatis, hingga monitoring dan alerting aktif.

---

## 🏗️ Arsitektur Sistem

```
Dataset Raw
    │
    ▼
┌─────────────────────┐
│  K1: Eksperimen     │  EDA + Preprocessing + GitHub Actions
│  & Preprocessing    │  → automate_silmi.py
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  K2: Training Model │  MLPClassifier + MLflow Tracking
│  dengan MLflow      │  → DagsHub (Advance)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  K3: CI Workflow    │  GitHub Actions + mlflow run .
│  MLflow Project     │  → Docker Hub
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  K4: Monitoring     │  Flask API + Prometheus + Grafana
│  & Logging          │  → Alerting via Email
└─────────────────────┘
```

---

## 📂 Struktur Repository

```
SMSML_Silmi-Azdkiatul-Athqia/
│
├── 📄 Eksperimen_SML_Silmi-Azdkiatul-Athqia.txt    # Link repo K1
├── 📄 Workflow-CI.txt                               # Link repo K3
│
├── 📁 Membangun_model/
│   ├── 🐍 modelling.py                              # Basic: MLflow autolog
│   ├── 🐍 modelling_tuning.py                       # Advance: Manual logging + DagsHub
│   ├── 📁 remote_worker_productivity_preprocessing/ # Dataset preprocessed
│   ├── 🖼️ screenshoot_dashboard.jpg                 # MLflow dashboard
│   ├── 🖼️ screenshoot_artifak.jpg                   # MLflow artifacts
│   ├── 📄 requirements.txt
│   └── 📄 DagsHub.txt                               # Link DagsHub tracking
│
└── 📁 Monitoring_dan_Logging/
    ├── 🐍 7.Inference.py                            # Flask API + Prometheus metrics
    ├── 🐍 prometheus_exporter.py                    # Custom metrics exporter
    ├── ⚙️ prometheus.yml                            # Prometheus config
    ├── ⚙️ alert_rules.yml                           # Grafana alert rules
    ├── 📊 grafana_dashboard.json                    # Dashboard export
    ├── 📁 1. bukti_serving/                         # Bukti model serving
    ├── 📁 4. bukti monitoring Prometheus/           # Screenshot Prometheus
    ├── 📁 5. bukti monitoring Grafana/              # Screenshot Grafana
    └── 📁 6. bukti alerting Grafana/                # Screenshot alerting
```

---

## 🚀 Kriteria & Pencapaian

| Kriteria | Deskripsi | Tier |
|:---:|---|:---:|
| **K1** | Eksperimen & Preprocessing Dataset | 🏆 Advance |
| **K2** | Membangun Model dengan MLflow | 🏆 Advance |
| **K3** | Workflow CI dengan GitHub Actions | 🏆 Advance |
| **K4** | Monitoring & Logging | 🏆 Advance |

---

## 🧠 Model Machine Learning

**Algoritma:** MLP Classifier (Multi-Layer Perceptron)

| Parameter | Nilai |
|---|---|
| Hidden Layers | (128, 64, 32) |
| Activation | ReLU |
| Solver | Adam |
| Learning Rate | 0.001 |
| Max Iterations | 500 |

**Hasil Evaluasi:**

| Metrik | Nilai |
|---|---|
| Accuracy | **97.5%** |
| Precision | **97.6%** |
| Recall | **97.5%** |
| F1-Score | **97.5%** |

---

## 📊 Monitoring Metrics

Dashboard Grafana **Dashboard-silmiathqia** memantau 10+ metrik:

```
System Metrics:          ML Model Metrics:
├── CPU Usage (%)        ├── Model Accuracy
├── Memory Usage (GB)    ├── Prediction Rate
├── Disk Usage (%)       ├── Prediction Latency
└── HTTP Request Rate    ├── Prediction Confidence
                         ├── Total Predictions
                         ├── Model Load Time
                         └── Active Users
```

**Alert Rules:**
- 🔴 `CPU Usage High` — trigger jika CPU > 80%
- 🔴 `Memory Usage High` — trigger jika memory > 3.8GB
- 🔴 `Disk Usage High` — trigger jika disk > 85%

---

## 🔗 Links

| Resource | Link |
|---|---|
| 📦 Repo Eksperimen (K1) | [Eksperimen_SML_Silmi-Azdkiatul-Athqia](https://github.com/silmiaathqia/Eksperimen_SML_Silmi-Azdkiatul-Athqia) |
| ⚙️ Repo Workflow CI (K3) | [Workflow-CI](https://github.com/silmiaathqia/Workflow-CI) |
| 📈 MLflow Tracking | [DagsHub](https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow) |
| 🐳 Docker Image | [Docker Hub](https://hub.docker.com/r/silmiathqia/worker-productivity-mlp) |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12.7 |
| ML Framework | Scikit-Learn 1.5.2 |
| Experiment Tracking | MLflow 2.19.0 + DagsHub |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Serving | Flask |
| Monitoring | Prometheus + Grafana |
| Alerting | Grafana + Email |

---

## 👩‍💻 Author

**Silmi Azdkiatul Athqia**
- Dicoding: `silmiathqia`
- GitHub: [@silmiaathqia](https://github.com/silmiaathqia)
- Program: Laskar AI 2025 Cohort — Mahasiswa & Fresh Graduate

---

<div align="center">
<i>Proyek Akhir — Membangun Sistem Machine Learning (MSML) — Dicoding 2025</i>
</div>
