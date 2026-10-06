from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# =========================================================
# SOCIAL NETWORK DATA
# =========================================================

users = [
    "Aarav", "Diya", "Kabir", "Meera",
    "Rohan", "Anaya", "Vivaan", "Isha",
    "Arjun", "Sara", "Neel", "Tara"
]

edges = [
    ("Aarav", "Diya"),
    ("Aarav", "Kabir"),
    ("Aarav", "Rohan"),
    ("Diya", "Kabir"),
    ("Diya", "Meera"),
    ("Kabir", "Rohan"),
    ("Kabir", "Anaya"),
    ("Meera", "Rohan"),
    ("Meera", "Isha"),
    ("Rohan", "Vivaan"),
    ("Anaya", "Vivaan"),
    ("Anaya", "Arjun"),
    ("Isha", "Arjun"),
    ("Isha", "Sara"),
    ("Vivaan", "Arjun"),
    ("Vivaan", "Tara"),
    ("Sara", "Tara"),
    ("Neel", "Sara"),
    ("Neel", "Tara"),
    ("Kabir", "Neel")
]

positions = {
    "Aarav": (70, 80),
    "Diya": (200, 40),
    "Kabir": (340, 70),
    "Meera": (190, 170),
    "Rohan": (350, 180),
    "Anaya": (500, 60),
    "Vivaan": (570, 190),
    "Isha": (70, 300),
    "Arjun": (310, 300),
    "Sara": (480, 350),
    "Neel": (680, 300),
    "Tara": (670, 130)
}


# =========================================================
# GRAPH FUNCTIONS
# =========================================================

def neighbors(user):
    result = set()

    for user1, user2 in edges:
        if user1 == user:
            result.add(user2)
        elif user2 == user:
            result.add(user1)

    return result


def connected(user1, user2):
    return (
        (user1, user2) in edges
        or (user2, user1) in edges
    )


def common_neighbors(user1, user2):
    return neighbors(user1).intersection(neighbors(user2))


# =========================================================
# KNN FEATURES
# =========================================================

def pair_features(user1, user2):
    """
    Numerical features used by KNN:
    degree of user 1,
    degree of user 2,
    common neighbors,
    neighborhood union,
    Jaccard similarity,
    degree product.
    """

    n1 = neighbors(user1)
    n2 = neighbors(user2)

    common = n1.intersection(n2)
    union = n1.union(n2)

    degree_1 = len(n1)
    degree_2 = len(n2)
    common_count = len(common)
    union_count = len(union)

    jaccard = (
        common_count / union_count
        if union_count else 0.0
    )

    degree_product = degree_1 * degree_2

    return [
        degree_1,
        degree_2,
        common_count,
        union_count,
        jaccard,
        degree_product
    ]


# =========================================================
# KNN MODEL
# =========================================================

def build_knn_model(selected_user):
    """
    Train KNN on all user pairs that do not contain
    the selected user. Existing links are class 1 and
    non-existing links are class 0.
    """

    X = []
    y = []

    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1 = users[i]
            user2 = users[j]

            if user1 == selected_user or user2 == selected_user:
                continue

            X.append(pair_features(user1, user2))
            y.append(1 if connected(user1, user2) else 0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    knn.fit(X_scaled, y)

    return knn, scaler


def generate_predictions(selected_user, recommendation_count):
    knn, scaler = build_knn_model(selected_user)

    candidates = []
    candidate_features = []

    for user in users:
        if user == selected_user:
            continue

        if connected(selected_user, user):
            continue

        candidates.append(user)
        candidate_features.append(
            pair_features(selected_user, user)
        )

    if not candidates:
        return []

    candidate_scaled = scaler.transform(candidate_features)
    probabilities = knn.predict_proba(candidate_scaled)

    class_one_index = list(knn.classes_).index(1)

    results = []

    for user, probability in zip(
        candidates,
        probabilities[:, class_one_index]
    ):
        results.append({
            "user": user,
            "score": float(probability),
            "common": sorted(
                common_neighbors(selected_user, user)
            )
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:recommendation_count]


# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        users=users,
        edges=edges,
        positions=positions,
        predictions=[],
        selected_user="Aarav",
        recommendation_count=5,
        predicted=False
    )


@app.route("/predict", methods=["POST"])
def predict():
    selected_user = request.form.get("user", "Aarav")

    try:
        recommendation_count = int(
            request.form.get("recommendations", 5)
        )
    except ValueError:
        recommendation_count = 5

    recommendation_count = max(
        1,
        min(recommendation_count, 10)
    )

    predictions = generate_predictions(
        selected_user,
        recommendation_count
    )

    return render_template(
        "index.html",
        users=users,
        edges=edges,
        positions=positions,
        predictions=predictions,
        selected_user=selected_user,
        recommendation_count=recommendation_count,
        predicted=True
    )


if __name__ == "__main__":
    app.run(debug=True)
