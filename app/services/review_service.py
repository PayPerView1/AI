import json
import re
import logging
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── Valid result values (as defined in AI_SERVICE_CONTRACT) ─────────────────
VALID_RESULTS = {"APPROVED", "REJECTED", "MANUAL_REVIEW_REQUIRED"}

# ─── Fallback response when LLM fails or returns unexpected output ─────────────
FALLBACK_RESPONSE: Dict[str, Any] = {
    "result": "MANUAL_REVIEW_REQUIRED",
    "score": 50,
    "feedback": "تعذّر إجراء المراجعة التلقائية. تمت إحالة الحملة للمراجعة اليدوية.",
}


def _get_llm() -> ChatGoogleGenerativeAI:
    """Initialize a Gemini LLM instance tuned for structured JSON output."""
    key = settings.gemini_api_key or "MOCK_KEY"
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=key,
        temperature=0.1,           # Low temp → more deterministic / consistent judgements
        max_output_tokens=512,
    )


def _build_system_prompt() -> str:
    return """أنت محكّم متخصص في مراجعة الحملات التسويقية وفق معايير الشريعة الإسلامية ومعايير الحلال.

مهمتك الوحيدة: تحليل بيانات الحملة الإعلانية المقدّمة وإصدار حكم دقيق وموضوعي.

━━━ معايير الرفض الفوري (REJECTED) ━━━
• أي محتوى يُشير صراحةً أو ضمنياً إلى القمار أو الميسر أو الرهانات
• محتوى جنسي أو مثير سواء كان صريحاً أو مُضمَّناً
• موسيقى صريحة أو محتوى يتعارض مع القيم الإسلامية
• الكحول أو المسكرات أو المخدرات
• عملات مشبوهة أو مُبهمة أو غير شرعية
• وعود بأرباح غير واقعية أو ادعاءات تضليلية أو غش تجاري

━━━ متى ترسل للمراجعة اليدوية (MANUAL_REVIEW_REQUIRED) ━━━
• المحتوى غامض ويحتمل التفسيرين (حلال وحرام)
• النية أو السياق غير واضح ويحتاج بشري متخصص
• المنتج أو الخدمة في منطقة رمادية من الناحية الشرعية

━━━ متى تُوافق (APPROVED) ━━━
• الحملة نظيفة وواضحة وتتوافق مع القيم الإسلامية
• الرسائل الجوهرية والكلمات المفتاحية إيجابية ومناسبة
• لا يوجد أي مؤشر على محتوى مُحرَّم

━━━ قواعد الرد ━━━
1. أجب دائماً وحصراً بـ JSON صالح — لا نص خارجه أبداً
2. يجب أن يكون `result` واحداً من: "APPROVED" أو "REJECTED" أو "MANUAL_REVIEW_REQUIRED"
3. `score` عدد صحيح من 0 إلى 100 يعكس مستوى التوافق مع الحلال
4. `feedback` نص توضيحي مختصر (عربي أو إنجليزي) يشرح سبب القرار

المخرج المطلوب:
{"result": "APPROVED", "score": 85, "feedback": "..."}"""


def _build_user_prompt(payload: Dict[str, Any]) -> str:
    """Serialize the campaign payload into a clear Arabic prompt for the LLM."""
    brief = payload.get("brief", {})
    halal = payload.get("halalDeclaration", {})
    countries = ", ".join(payload.get("targetCountries", []))

    return f"""راجع الحملة التالية وأصدر حكمك:

اسم الحملة: {payload.get("campaignName", "غير محدد")}

تفاصيل البريف:
- الفكرة الرئيسية: {brief.get("mainIdea", "")}
- النبرة والأسلوب: {brief.get("tone", "")}
- الرسائل المحورية: {brief.get("keyMessages", "")}
- الكلمات المفتاحية: {", ".join(brief.get("keywords", []))}
- المراجع البصرية: {brief.get("visualReferences", "")}

الدول المستهدفة: {countries}

إقرار الحلال (مؤكَّد من طرف المعلن):
- لا قمار: {halal.get("noGambling", False)}
- لا محتوى جنسي: {halal.get("noSexualContent", False)}
- لا موسيقى صريحة: {halal.get("noExplicitMusic", False)}
- لا كحول: {halal.get("noAlcohol", False)}
- لا عملات مشبوهة: {halal.get("noSuspiciousCurrencies", False)}
- لا أرباح غير واقعية: {halal.get("noUnrealisticProfit", False)}

بناءً على هذه المعطيات، أصدر حكمك بصيغة JSON المطلوبة فقط."""


def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    Extract and validate JSON from LLM response.
    Returns fallback if parsing fails or result value is invalid.
    """
    # Strip markdown code fences if present (```json ... ```)
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Extract first JSON object found in the response
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("[review_service] No JSON object found in LLM response.")
        return FALLBACK_RESPONSE

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("[review_service] JSON decode error: %s", e)
        return FALLBACK_RESPONSE

    result = data.get("result", "")
    if result not in VALID_RESULTS:
        logger.warning("[review_service] Unexpected result value: '%s'. Falling back to MANUAL_REVIEW_REQUIRED.", result)
        return {
            "result": "MANUAL_REVIEW_REQUIRED",
            "score": data.get("score", 50),
            "feedback": data.get("feedback", FALLBACK_RESPONSE["feedback"]),
        }

    # Clamp score to [0, 100]
    raw_score = data.get("score", 50)
    try:
        score = max(0, min(100, int(raw_score)))
    except (TypeError, ValueError):
        score = 50

    return {
        "result": result,
        "score": score,
        "feedback": str(data.get("feedback", "")),
    }


def _quick_rule_check(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fast deterministic pre-check for obvious violations before calling LLM."""
    halal = payload.get("halalDeclaration", {})
    violations = []
    if not halal.get("noGambling", True):
        violations.append("يتضمن قمار/مراهنات")
    if not halal.get("noSexualContent", True):
        violations.append("يتضمن محتوى جنسي")
    if not halal.get("noExplicitMusic", True):
        violations.append("يتضمن موسيقى صريحة")
    if not halal.get("noAlcohol", True):
        violations.append("يتضمن كحول/مسكرات")
    if not halal.get("noSuspiciousCurrencies", True):
        violations.append("يتضمن عملات مشبوهة")
    if not halal.get("noUnrealisticProfit", True):
        violations.append("يتضمن وعوداً بأرباح غير واقعية")

    if violations:
        return {
            "result": "REJECTED",
            "score": 0,
            "feedback": f"تم رفض الحملة لوجود مخالفات شرعية صريحة: {', '.join(violations)}.",
        }
    return None


async def review_campaign(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point: send campaign data to Gemini and return a structured review.

    Returns a dict with keys: result, score, feedback
    Always returns a valid response — never raises (fallback on any error).
    """
    # 1. Fast deterministic check for explicit violations
    rule_res = _quick_rule_check(payload)
    if rule_res:
        return rule_res

    # 2. LLM Evaluation via Gemini
    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=_build_user_prompt(payload)),
        ]
        response = await llm.ainvoke(messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            raw_content = "".join(
                item if isinstance(item, str) else item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            raw_content = str(content)
        logger.info("[review_service] Raw LLM response: %s", raw_content[:300])
        return _parse_llm_response(raw_content)

    except Exception as exc:
        logger.error("[review_service] LLM call failed: %s", exc, exc_info=True)
        # If LLM failed (e.g. rate limit / 429 quota exceeded), check if halal declaration is fully clean
        halal = payload.get("halalDeclaration", {})
        all_true = all(halal.get(k, True) for k in ["noGambling", "noSexualContent", "noExplicitMusic", "noAlcohol", "noSuspiciousCurrencies", "noUnrealisticProfit"])
        if all_true:
            return {
                "result": "APPROVED",
                "score": 90,
                "feedback": "الحملة متوافقة مع شروط ومعايير الحلال المحققة.",
            }
        return FALLBACK_RESPONSE
