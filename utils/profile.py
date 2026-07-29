"""用户画像管理器 —— JSON 文件按 tenant_id 物理隔离

Profile 结构:
  devices:         [{"model": "Z3 Ultra", "purchased": "2026-07",
                     "usage": {frequency, primary_area, floor_type, has_pets},
                     "issues": [{"problem": "", "status": "未解决/已解决", "attempted_solutions": []}],
                     "consumables": {"hepa_filter": {"last_replaced": "2026-06"}, ...}}]
  current_location: "深圳"
  locations:       ["北京", "深圳"]  (历史)
  preferences:     {"tech_level": "进阶", "reply_style": "详细步骤"}
  purchase_intent: [{"product": "Z3 Ultra", "level": "高"}]
  service_history: [{"date": "", "type": "", "device": "", "description": ""}]
  question_history: [{"date": "", "category": "", "device": "", "problem": "", "query_summary": "", "resolved": false}]  (FIFO, max 100)

读取: <1ms 文件 I/O，写入: <1ms JSON dump
merge() 纯代码逻辑，不调 LLM，确定性行为
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from utils.logger_handler import logger


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class ProfileManager:
    def __init__(self):
        self._dir = getattr(config, "PROFILE_DIR", os.path.join(config.DATA_DIR, "profiles"))
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, tenant_id: str) -> str:
        return os.path.join(self._dir, f"{tenant_id}.json")

    # ---------- 读取 ----------

    def _empty_profile(self) -> dict:
        return {
            "devices": [],
            "locations": [],
            "current_location": "",
            "preferences": {},
            "purchase_intent": [],
            "service_history": [],
            "question_history": [],
        }

    def _ensure_fields(self, profile: dict) -> dict:
        """补齐缺失字段 + 迁移旧格式 issues（string→结构化）"""
        for key, val in self._empty_profile().items():
            if key not in profile:
                profile[key] = val
        for d in profile.get("devices", []):
            issues = d.get("issues", [])
            if issues and isinstance(issues[0], str):
                d["issues"] = [{"problem": s, "status": "未解决"} for s in issues]
        return profile

    def load(self, tenant_id: str) -> dict:
        """读取用户画像，文件不存在返回空结构"""
        path = self._path(tenant_id)
        if not os.path.exists(path):
            return self._empty_profile()
        try:
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            return self._ensure_fields(profile)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("画像读取失败 %s: %s", tenant_id, e)
            return self._empty_profile()

    # ---------- 写入 ----------

    def save(self, tenant_id: str, profile: dict):
        """全量覆盖写"""
        path = self._path(tenant_id)
        profile["updated_at"] = _now()
        if "created_at" not in profile:
            profile["created_at"] = _now()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            logger.debug("画像已保存: %s (devices=%d)", tenant_id, len(profile.get("devices", [])))
        except OSError as e:
            logger.warning("画像写入失败 %s: %s", tenant_id, e)

    # ---------- 合并 ----------

    def _normalize_device(self, dev: dict) -> dict:
        """补全设备字段默认值，迁移旧格式 issues"""
        d = {"model": dev.get("model", ""), "issues": [], "consumables": {}}
        if "purchased" in dev:
            d["purchased"] = dev["purchased"]
        if "usage" in dev and dev["usage"]:
            d["usage"] = dev["usage"]
        issues = dev.get("issues", [])
        if issues:
            if isinstance(issues[0], str):
                d["issues"] = [{"problem": s, "status": "未解决"} for s in issues]
            else:
                d["issues"] = issues
        if "consumables" in dev and dev["consumables"]:
            d["consumables"] = dev["consumables"]
        return d

    def _merge_device(self, existing: dict, update: dict):
        """增量合并 update 到已有设备"""
        if "purchased" in update and update["purchased"]:
            existing["purchased"] = update["purchased"]

        # 迁移旧格式 issues（string 列表 → 结构化）
        if existing.get("issues") and isinstance(existing["issues"][0], str):
            existing["issues"] = [{"problem": s, "status": "未解决"} for s in existing["issues"]]

        if "issues" in update and update["issues"]:
            existing.setdefault("issues", [])
            existing_problems = {i.get("problem", ""): i for i in existing["issues"]}
            for issue in update["issues"]:
                problem = issue.get("problem", "")
                if not problem:
                    continue
                if problem in existing_problems:
                    ext = existing_problems[problem]
                    if "status" in issue and issue["status"]:
                        ext["status"] = issue["status"]
                    if "attempted_solutions" in issue:
                        existing_sols = set(ext.get("attempted_solutions", []))
                        new_sols = set(issue.get("attempted_solutions", []))
                        ext["attempted_solutions"] = sorted(existing_sols | new_sols)
                    ext["last_mentioned"] = _now()
                else:
                    issue.setdefault("status", "未解决")
                    issue["last_mentioned"] = _now()
                    existing["issues"].append(issue)
                    existing_problems[problem] = issue

        if "usage" in update and update["usage"]:
            existing.setdefault("usage", {}).update(update["usage"])

        if "consumables" in update and update["consumables"]:
            existing.setdefault("consumables", {})
            for key, val in update["consumables"].items():
                if key in existing["consumables"]:
                    existing["consumables"][key].update(val)
                else:
                    existing["consumables"][key] = val

    def merge(self, tenant_id: str, update: dict) -> dict:
        """增量合并 profile_update 到已有画像

        规则:
          - devices: 按 model 去重，issues/usage/consumables 结构化合并
          - current_location: 覆盖（搬家则更新）
          - locations: 去重追加（历史记录）
          - preferences: 浅合并
          - purchase_intent: 按 product 合并
          - service_history: 追加
        """
        if not update:
            return self.load(tenant_id)

        profile = self.load(tenant_id)

        # --- devices ---
        if "devices" in update and update["devices"]:
            existing_models = {d.get("model", "") for d in profile.get("devices", [])}
            for dev in update["devices"]:
                model = dev.get("model", "")
                if not model:
                    continue
                if model in existing_models:
                    for d in profile["devices"]:
                        if d.get("model") == model:
                            self._merge_device(d, dev)
                            break
                else:
                    profile["devices"].append(self._normalize_device(dev))
                    existing_models.add(model)

        # --- current_location ---
        if "current_location" in update and update["current_location"]:
            profile["current_location"] = update["current_location"]
            loc = update["current_location"]
            if loc not in set(profile.get("locations", [])):
                profile.setdefault("locations", []).append(loc)

        # --- locations (历史) ---
        if "locations" in update and update["locations"]:
            existing = set(profile.get("locations", []))
            for loc in update["locations"]:
                if loc and loc not in existing:
                    profile["locations"].append(loc)
                    existing.add(loc)

        # --- preferences ---
        if "preferences" in update and update["preferences"]:
            profile.setdefault("preferences", {}).update(update["preferences"])

        # --- purchase_intent ---
        if "purchase_intent" in update and update["purchase_intent"]:
            existing_intents = {p.get("product", ""): p for p in profile.get("purchase_intent", [])}
            for intent in update["purchase_intent"]:
                product = intent.get("product", "")
                if not product:
                    continue
                if product in existing_intents:
                    existing_intents[product].update(intent)
                    existing_intents[product]["last_asked"] = _now()
                else:
                    intent["last_asked"] = _now()
                    profile["purchase_intent"].append(intent)
                    existing_intents[product] = intent

        # --- service_history ---
        if "service_history" in update and update["service_history"]:
            profile.setdefault("service_history", [])
            for entry in update["service_history"]:
                if "date" not in entry:
                    entry["date"] = _now()[:10]
                profile["service_history"].append(entry)

        # --- question_history ---
        if "question_entry" in update and update["question_entry"]:
            entry = update["question_entry"]
            entry.setdefault("date", _now())
            profile.setdefault("question_history", [])
            profile["question_history"].append(entry)
            # FIFO 淘汰：超出 100 条踢最旧的
            if len(profile["question_history"]) > 100:
                profile["question_history"] = profile["question_history"][-100:]

        self.save(tenant_id, profile)
        logger.info(
            "画像已更新: %s (devices=%d, locations=%d, questions=%d)",
            tenant_id,
            len(profile.get("devices", [])),
            len(profile.get("locations", [])),
            len(profile.get("question_history", [])),
        )
        return profile
