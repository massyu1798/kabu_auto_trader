"""
共有ユニバース定義

シンプル順張り戦略で使用する銘柄ユニバース。
"""

from __future__ import annotations

# ユニバース定義（36銘柄、Yahoo Finance ticker形式）
UNIVERSE: dict[str, dict[str, str]] = {
    "9984.T": {"name": "ソフトバンクG",        "sector": "通信"},
    "8306.T": {"name": "三菱UFJ FG",           "sector": "銀行"},
    "5401.T": {"name": "日本製鉄",              "sector": "鉄鋼"},
    "7203.T": {"name": "トヨタ自動車",          "sector": "自動車"},
    "6758.T": {"name": "ソニーG",               "sector": "電機"},
    "6501.T": {"name": "日立製作所",            "sector": "電機"},
    "8316.T": {"name": "三井住友FG",            "sector": "銀行"},
    "6762.T": {"name": "TDK",                   "sector": "電子部品"},
    "6857.T": {"name": "アドバンテスト",        "sector": "半導体"},
    "6981.T": {"name": "村田製作所",            "sector": "電子部品"},
    "7267.T": {"name": "ホンダ",                "sector": "自動車"},
    "6702.T": {"name": "富士通",                "sector": "電機"},
    "8058.T": {"name": "三菱商事",              "sector": "商社"},
    "4063.T": {"name": "信越化学",              "sector": "化学"},
    "6752.T": {"name": "パナソニックHD",        "sector": "電機"},
    "6098.T": {"name": "リクルートHD",          "sector": "サービス"},
    "7974.T": {"name": "任天堂",                "sector": "ゲーム"},
    "6503.T": {"name": "三菱電機",              "sector": "電機"},
    "4307.T": {"name": "野村総研",              "sector": "IT"},
    "8801.T": {"name": "三井不動産",            "sector": "不動産"},
    "6954.T": {"name": "ファナック",            "sector": "FA"},
    "6902.T": {"name": "デンソー",              "sector": "自動車部品"},
    "4568.T": {"name": "第一三共",              "sector": "医薬"},
    "3382.T": {"name": "セブン＆アイ",          "sector": "小売"},
    "9433.T": {"name": "KDDI",                  "sector": "通信"},
    "8766.T": {"name": "東京海上HD",            "sector": "保険"},
    "8035.T": {"name": "東京エレクトロン",      "sector": "半導体"},
    "4661.T": {"name": "オリエンタルランド",    "sector": "サービス"},
    "3659.T": {"name": "ネクソン",              "sector": "ゲーム"},
    "4502.T": {"name": "武田薬品",              "sector": "医薬"},
    "2914.T": {"name": "JT",                    "sector": "食品"},
    "4519.T": {"name": "中外製薬",              "sector": "医薬"},
    "7751.T": {"name": "キヤノン",              "sector": "電機"},
    "9022.T": {"name": "JR東海",               "sector": "運輸"},
    "7741.T": {"name": "HOYA",                  "sector": "精密"},
    "6367.T": {"name": "ダイキン工業",          "sector": "機械"},
}
