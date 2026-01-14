from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


BUNDLE_PATH = Path("models/titanic_lgbm_bundle.joblib")
bundle = None


# ---- 入力は“生データ寄り”にする（API側で特徴量生成する）----
class TitanicRequest(BaseModel):
    Pclass: int
    Sex: Literal["male", "female"]
    Age: Optional[float] = None
    SibSp: int = 0
    Parch: int = 0
    Fare: Optional[float] = None
    Embarked: Literal["S", "C", "Q"]

    # 特徴量生成に必要
    Name: Optional[str] = None  # Title用
    Cabin: Optional[str] = None  # HasCabin用
    Ticket: Optional[str] = None  # TicketCount用


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if not BUNDLE_PATH.exists():
        raise RuntimeError(f"Model bundle not found: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Hello World"}


# ---- Title抽出（あなたの学習時ロジックに合わせて調整可）----
def extract_title(name: Optional[str]) -> str:
    if not name:
        return "Unknown"
    # TitanicのNameは "Surname, Title. Firstname" 形式が多い
    import re

    m = re.search(r",\s*([^\.]+)\.", name)
    if not m:
        return "Unknown"
    t = m.group(1).strip()

    # よくある正規化（学習時にやっているなら合わせる）
    # 例: Mlle/Ms -> Miss, Mme -> Mrs, Rareまとめ など
    t = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}.get(t, t)
    rare = {
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    }
    if t in rare:
        return "Rare"

    return t


def preprocess_one(req: TitanicRequest) -> pd.DataFrame:
    """リクエスト1件→学習時の13特徴量DataFrameに変換"""
    # まず生の列
    df = pd.DataFrame([req.model_dump()])

    # ---- 欠損補完（Jupyterと同じ統計）----
    age_median = bundle["age_median"]
    embarked_mode = bundle["embarked_mode"]
    fare_median = bundle["fare_median"]

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(age_median)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(fare_median)

    # ---- 特徴量生成（あなたの列一覧に合わせる）----
    # HasCabin: Cabinがあれば1
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Title: Nameから抽出
    df["Title"] = df["Name"].apply(extract_title)

    # FamilySize: SibSp + Parch + 1
    df["FamilySize"] = df["SibSp"].astype(int) + df["Parch"].astype(int) + 1

    # IsAlone: FamilySize==1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # TicketCount: 同じTicketの人数
    # 1件APIでは “データセット全体でのTicket頻度” が分からないので、
    # 学習時に「train+testでTicketCountを事前計算」していた場合は要注意。
    # ここでは最低限「単発は1」とする。
    ticket_count_map = bundle["ticket_count_map"]

    ticket = df.loc[0, "Ticket"] if "Ticket" in df.columns else None
    if ticket is None or (isinstance(ticket, float) and np.isnan(ticket)):
        df["TicketCount"] = 1
    else:
        # train-only map に無ければ 1
        df["TicketCount"] = int(ticket_count_map.get(ticket, 1))

    # Fare_log
    df["Fare_log"] = np.log1p(df["Fare"])

    # ---- category dtype保証 + 学習時カテゴリ集合に揃える ----
    cat_cols = bundle["cat_cols"]
    cat_categories = bundle["cat_categories"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
            df[c] = df[c].cat.set_categories(cat_categories[c])

    # ---- 学習時の列順に揃える（これがあなたの13列）----
    feature_cols = bundle["feature_cols"]
    X = df.reindex(columns=feature_cols)

    return X


@app.post("/predict")
def predict(req: TitanicRequest):
    model = bundle["model"]
    threshold = bundle["threshold"]

    X = preprocess_one(req)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)

    return {
        "survived_pred": pred,
        "survived_proba": proba,
        "threshold": float(threshold),
    }
