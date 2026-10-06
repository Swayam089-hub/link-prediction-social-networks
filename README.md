# Link Prediction in Social Networks

A Flask web application for predicting potential future connections in a social network using the **K-Nearest Neighbors (KNN)** machine learning algorithm.

## Project Overview

This project represents users as nodes and existing relationships as edges. It uses graph-based features with KNN to estimate the likelihood of potential connections between users who are not already connected.

## Algorithm Used

### K-Nearest Neighbors (KNN)

KNN is a supervised machine learning algorithm that predicts a class by comparing a new data point with its nearest training examples.

In this project, KNN is used to classify user pairs as:

- **1** — Existing connection
- **0** — Non-existing connection

The model then calculates a link probability for candidate connections and ranks them from highest to lowest probability.

## Features Used

The KNN model uses the following features for each pair of users:

1. Degree of User 1
2. Degree of User 2
3. Number of Common Neighbors
4. Neighborhood Union Size
5. Jaccard Similarity
6. Degree Product

The features are standardized before being given to the KNN model.

## Network Data

The predefined social network contains:

- **12 users**
- **20 existing connections**
- **66 possible user pairs**

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend and prediction logic |
| Flask | Web application framework |
| Scikit-learn | KNN machine learning model |
| HTML5 | Web page structure |
| CSS3 | User interface styling |
| SVG | Social network visualization |
| Gunicorn | Production web server |

## Project Structure

```text
link-prediction-social-networks/
│
├── app.py
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

## How the System Works

```text
Social Network
      ↓
Select User
      ↓
Select KNN
      ↓
Extract Graph Features
      ↓
Standardize Features
      ↓
Train KNN Model
      ↓
Calculate Link Probability
      ↓
Rank Candidate Users
      ↓
Display Prediction Results
      ↓
Show Predicted Connections
```

## Main Features

- Select a user from the social network
- Use KNN as the link prediction algorithm
- Select the number of recommendations
- Calculate potential connections
- Display KNN link probabilities
- Show common neighbors for recommended users
- Visualize predicted connections using green dashed lines
- Reset the prediction and return to the original network

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Swayam089-hub/link-prediction-social-networks.git
cd link-prediction-social-networks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:5000
```

## Render Deployment

For deployment on Render:

**Build Command**

```text
pip install -r requirements.txt
```

**Start Command**

```text
gunicorn app:app
```

No environment variables are required for this project.

## Prediction Workflow

1. Select a user.
2. Select **K-Nearest Neighbors (KNN)**.
3. Enter the required number of recommendations.
4. Click **Predict Connections**.
5. The Flask backend extracts graph-based features.
6. The KNN model calculates link probabilities.
7. Candidate users are ranked by probability.
8. Results are displayed in the prediction table.
9. Predicted connections are highlighted on the network graph.

## Project Objective

The objective of this project is to demonstrate how machine learning and graph-based features can be used to identify potential future connections in a social network.

## Future Scope

The project can be extended by:

- Using larger real-world social network datasets
- Connecting the application to a database
- Adding additional graph features
- Comparing multiple machine learning models
- Adding model evaluation metrics
- Supporting dynamic network data

## Author

**Swayam Bobade**

**Project:** Link Prediction in Social Networks
