import os
import json
from typing import List, Dict, Any

# -----------------------------
# 0) 데이터 로드
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
json_path = os.path.join(project_root, "yelp", "restaurants.json")
with open(json_path, "r", encoding="utf-8") as f:
    restaurants: List[Dict[str, Any]] = json.load(f)

# -----------------------------
# 1) 임베딩 모델
# -----------------------------
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# -----------------------------
# 2) Qdrant 클라이언트
# -----------------------------
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "restaurants"

# -----------------------------
# 3) 문서 텍스트 빌드 (reviews 제외)
# -----------------------------
def build_doc_text(item: Dict[str, Any]) -> str:
    name = item.get("name", "")
    addr = f'{item.get("address","")}, {item.get("city","")}, {item.get("state","")}, {item.get("postal_code","")}'
    cats = ", ".join(item.get("categories", []) or [])
    ambs = ", ".join(item.get("ambiences", []) or [])
    meals = ", ".join(item.get("good_for_meals", []) or [])
    tips = " | ".join(item.get("tips", [])[:10])

    flags = []
    if item.get("good_for_kids"): flags.append("good for kids")
    if item.get("dogs_allowed"): flags.append("dogs allowed")
    if item.get("wifi"): flags.append("wifi")
    if item.get("happy_hour"): flags.append("happy hour")
    if not item.get("has_tv", True): flags.append("no TV")
    flags_txt = ", ".join(flags)

    text = (
        f"Restaurant: {name}\n"
        f"Address: {addr}\n"
        f"Categories: {cats}\n"
        f"Ambiences: {ambs}\n"
        f"Good for meals: {meals}\n"
        f"Rating: {item.get('stars','')} stars, {item.get('review_count','')} reviews\n"
        f"Features: {flags_txt}\n"
        f"Tips: {tips}\n"
    )
    return text

# -----------------------------
# 4) 컬렉션 생성/재생성
# -----------------------------
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=model.get_sentence_embedding_dimension(),
        distance=Distance.COSINE
    ),
)

# -----------------------------
# 5) 포인트 생성 (reviews 제거)
# -----------------------------
def make_points(items: List[Dict[str, Any]]) -> List[PointStruct]:
    texts = [build_doc_text(x) for x in items]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()

    points: List[PointStruct] = []
    for idx, (x, v, doc_text) in enumerate(zip(items, vectors, texts), start=1):
        loc = x.get("location") or {}
        lat, lon = loc.get("lat"), loc.get("lon")

        payload = {
            "source_id": x.get("id"),
            "name": x.get("name"),
            "address": x.get("address"),
            "city": x.get("city"),
            "state": x.get("state"),
            "postal_code": x.get("postal_code"),
            "categories": x.get("categories"),
            "ambiences": x.get("ambiences"),
            "good_for_kids": x.get("good_for_kids"),
            "has_tv": x.get("has_tv"),
            "good_for_meals": x.get("good_for_meals"),
            "dogs_allowed": x.get("dogs_allowed"),
            "happy_hour": x.get("happy_hour"),
            "parkings": x.get("parkings"),
            "wifi": x.get("wifi"),
            "stars": float(x.get("stars")) if x.get("stars") is not None else None,
            "review_count": int(x.get("review_count")) if x.get("review_count") is not None else None,
            "tips": x.get("tips"),   # reviews는 제거
            "location": {"lat": lat, "lon": lon} if (lat is not None and lon is not None) else None,
            "doc_text": doc_text,
        }

        points.append(
            PointStruct(
                id=idx,
                vector=v,
                payload=payload
            )
        )
    return points

points = make_points(restaurants)

# -----------------------------
# 6) 업서트
# -----------------------------
client.upsert(collection_name=collection_name, points=points)

print(f"Upserted {len(points)} restaurants into '{collection_name}' without reviews.")
