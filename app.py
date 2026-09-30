from flask import Flask, render_template, request
import math

app = Flask(__name__)

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


def neighbors(user):
    result = set()

    for a, b in edges:
        if a == user:
            result.add(b)
        elif b == user:
            result.add(a)

    return result


def connected(user1, user2):
    return (
        (user1, user2) in edges or
        (user2, user1) in edges
    )


def common_neighbors(user1, user2):
    return neighbors(user1).intersection(neighbors(user2))


def common_neighbors_score(user1, user2):
    return len(common_neighbors(user1, user2))


def jaccard_score(user1, user2):
    n1 = neighbors(user1)
    n2 = neighbors(user2)

    union = n1.union(n2)

    if len(union) == 0:
        return 0

    return len(n1.intersection(n2)) / len(union)


def adamic_adar_score(user1, user2):
    common = common_neighbors(user1, user2)

    score = 0

    for person in common:
        degree = len(neighbors(person))

        if degree > 1:
            score += 1 / math.log(degree)

    return score


def calculate_score(user1, user2, algorithm):

    if algorithm == "Common Neighbors":
        return common_neighbors_score(user1, user2)

    elif algorithm == "Jaccard Similarity":
        return jaccard_score(user1, user2)

    elif algorithm == "Adamic-Adar":
        return adamic_adar_score(user1, user2)

    return 0


def generate_predictions(selected_user, algorithm, recommendation_count):

    results = []

    for user in users:

        if user == selected_user:
            continue

        if connected(selected_user, user):
            continue

        score = calculate_score(
            selected_user,
            user,
            algorithm
        )

        results.append({
            "user": user,
            "score": score,
            "common": list(
                common_neighbors(selected_user, user)
            )
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:recommendation_count]


@app.route("/")
def home():

    return render_template(
        "index.html",
        users=users,
        edges=edges,
        positions=positions,
        predictions=[],
        selected_user="",
        algorithm="Common Neighbors",
        recommendation_count=5,
        predicted=False
    )


@app.route("/predict", methods=["POST"])
def predict():

    selected_user = request.form.get(
        "user",
        "Aarav"
    )

    algorithm = request.form.get(
        "algorithm",
        "Common Neighbors"
    )

    try:
        recommendation_count = int(
            request.form.get(
                "recommendations",
                5
            )
        )
    except ValueError:
        recommendation_count = 5

    recommendation_count = max(
        1,
        min(recommendation_count, 10)
    )

    predictions = generate_predictions(
        selected_user,
        algorithm,
        recommendation_count
    )

    return render_template(
        "index.html",
        users=users,
        edges=edges,
        positions=positions,
        predictions=predictions,
        selected_user=selected_user,
        algorithm=algorithm,
        recommendation_count=recommendation_count,
        predicted=True
    )


if __name__ == "__main__":
    app.run(debug=True)
