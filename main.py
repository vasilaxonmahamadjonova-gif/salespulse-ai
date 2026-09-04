"""
SalesPulse backend

Vazifalari:
  1. Audio qabul qilish -> Supabase Storage
  2. RunPod'ga yuborish (STT + diarizatsiya)
  3. Gemini'ga yuborish (SPIN tahlil)
  4. Natijani Supabase'ga yozish

Ishga tushirish (lokal):
    uvicorn main:app --reload
"""

import os
import json
import re
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("salespulse")

# ---------------------------------------------------------------- sozlamalar

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

BUCKET = "call-audio"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="SalesPulse API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- modellar

class CallOut(BaseModel):
    id: str
    status: str
    duration_sec: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------- yordamchi

def _set_status(call_id: str, status: str, **fields):
    sb.table("calls").update({"status": status, **fields}).eq("id", call_id).execute()


def _signed_url(path: str, seconds: int = 3600) -> str:
    res = sb.storage.from_(BUCKET).create_signed_url(path, seconds)
    return res["signedURL"] if isinstance(res, dict) else res.signed_url


# ---------------------------------------------------------------- RunPod

async def run_stt(audio_url: str, speakers: int = 2) -> dict:
    """RunPod Serverless'ni chaqiradi va natijani kutadi."""
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    payload = {"input": {"audio_url": audio_url, "speakers": speakers}}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{RUNPOD_URL}/run", json=payload, headers=headers)
        r.raise_for_status()
        job_id = r.json()["id"]
        log.info("RunPod job: %s", job_id)

        # natijani kutamiz (max 15 daqiqa)
        for _ in range(180):
            await asyncio.sleep(5)
            s = await client.get(f"{RUNPOD_URL}/status/{job_id}", headers=headers)
            s.raise_for_status()
            data = s.json()
            status = data.get("status")

            if status == "COMPLETED":
                out = data.get("output") or {}
                if "error" in out:
                    raise RuntimeError(out["error"])
                return out
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"RunPod {status}: {data.get('error')}")

    raise TimeoutError("RunPod javob bermadi (15 daqiqa)")


# ---------------------------------------------------------------- Gemini

SPIN_PROMPT = """Sen sotuv qo'ng'iroqlarini baholaydigan ekspertsan.

Quyida o'zbek tilidagi sotuv qo'ng'irog'ining dialogi berilgan.
Transkriptda xatolar bo'lishi mumkin - kontekstdan tushunib ol.

Avval kim sotuvchi, kim mijoz ekanini aniqla.
Keyin SPIN metodikasi bo'yicha SOTUVCHINI bahola.
Faqat JSON qaytar, markdown belgisiz:

{{
  "sotuvchi_kim": "",
  "situation": {{"ball": 0, "izoh": ""}},
  "problem": {{"ball": 0, "izoh": ""}},
  "implication": {{"ball": 0, "izoh": ""}},
  "need_payoff": {{"ball": 0, "izoh": ""}},
  "umumiy_ball": 0,
  "kuchli_tomonlar": [],
  "xatolar": [],
  "otkazib_yuborilgan_imkoniyatlar": [{{"vaqt": "MM:SS", "nima": ""}}],
  "tavsiyalar": []
}}

DIALOG:
{dialog}"""


async def run_spin(dialog: list[dict]) -> Optional[dict]:
    """Gemini bilan SPIN tahlil. Kalit yo'q bo'lsa None qaytaradi."""
    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY yo'q, tahlil o'tkazib yuborildi")
        return None

    text = "\n".join(
        f"[{d.get('vaqt','')}] {d.get('speaker','')}: {d.get('text','')}"
        for d in dialog
    )
    prompt = SPIN_PROMPT.format(dialog=text)

    url = (f"https://generativelanguage.googleapis.com/v1beta/"
           f"models/{GEMINI_MODEL}:generateContent")
    headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(4):
            try:
                r = await client.post(url, json=body, headers=headers)
                if r.status_code in (429, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
                return json.loads(raw)
            except Exception as e:
                log.warning("Gemini urinish %s: %s", attempt + 1, e)
                await asyncio.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------- pipeline

async def process_call(call_id: str, audio_path: str, speakers: int):
    """Fonda ishlaydigan to'liq zanjir."""
    try:
        _set_status(call_id, "transcribing")
        url = _signed_url(audio_path, 7200)

        stt = await run_stt(url, speakers)

        dialog = stt.get("dialog", [])
        duration = stt.get("davomiylik_sek")

        sb.table("transcripts").upsert({
            "call_id":     call_id,
            "dialog":      dialog,
            "full_text":   stt.get("matn"),
            "words_count": stt.get("sozlar_soni"),
            "talk_ratio":  stt.get("gapirish_nisbati"),
            "stt_sec":     stt.get("vaqt_sek"),
        }).execute()

        _set_status(call_id, "analyzing", duration_sec=duration)

        spin = await run_spin(dialog)

        if spin:
            sb.table("analyses").upsert({
                "call_id":         call_id,
                "situation":       (spin.get("situation")   or {}).get("ball"),
                "problem":         (spin.get("problem")     or {}).get("ball"),
                "implication":     (spin.get("implication") or {}).get("ball"),
                "need_payoff":     (spin.get("need_payoff") or {}).get("ball"),
                "total_score":     spin.get("umumiy_ball"),
                "strengths":       spin.get("kuchli_tomonlar"),
                "mistakes":        spin.get("xatolar"),
                "missed":          spin.get("otkazib_yuborilgan_imkoniyatlar"),
                "recommendations": spin.get("tavsiyalar"),
                "raw":             spin,
                "model":           GEMINI_MODEL,
            }).execute()

        # ishlatilgan soatlarni yangilaymiz
        if duration:
            call = sb.table("calls").select("company_id").eq("id", call_id) \
                     .single().execute().data
            comp = sb.table("companies").select("hours_used") \
                     .eq("id", call["company_id"]).single().execute().data
            sb.table("companies").update({
                "hours_used": float(comp["hours_used"]) + duration / 3600.0
            }).eq("id", call["company_id"]).execute()

        _set_status(call_id, "done")
        log.info("Tugadi: %s", call_id)

    except Exception as e:
        log.exception("Xato: %s", call_id)
        _set_status(call_id, "failed", error=str(e)[:500])


# ---------------------------------------------------------------- endpointlar

@app.get("/")
def root():
    return {"service": "SalesPulse API", "status": "ok"}


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/calls", response_model=CallOut)
async def create_call(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    company_id: str = Form(...),
    seller_id: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None),
    speakers: int = Form(2),
):
    """Audio yuklash va qayta ishlashni boshlash."""

    # limit tekshiruvi
    comp = sb.table("companies").select("hours_limit, hours_used") \
             .eq("id", company_id).single().execute().data
    if not comp:
        raise HTTPException(404, "Kompaniya topilmadi")
    if float(comp["hours_used"]) >= float(comp["hours_limit"]):
        raise HTTPException(402, "Oylik soat limiti tugagan")

    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "Fayl 200 MB dan katta")

    row = sb.table("calls").insert({
        "company_id":  company_id,
        "seller_id":   seller_id,
        "client_name": client_name,
        "status":      "pending",
    }).execute().data[0]
    call_id = row["id"]

    ext = (file.filename or "audio.mp3").split(".")[-1].lower()
    path = f"{company_id}/{call_id}.{ext}"

    sb.storage.from_(BUCKET).upload(
        path, data,
        {"content-type": file.content_type or "audio/mpeg"},
    )
    sb.table("calls").update({"audio_path": path}).eq("id", call_id).execute()

    background.add_task(process_call, call_id, path, speakers)

    return CallOut(id=call_id, status="pending")


@app.get("/calls/{call_id}", response_model=CallOut)
def get_call(call_id: str):
    row = sb.table("calls").select("id, status, duration_sec, error") \
            .eq("id", call_id).single().execute().data
    if not row:
        raise HTTPException(404, "Topilmadi")
    return CallOut(**row)


@app.get("/calls/{call_id}/result")
def get_result(call_id: str):
    """To'liq natija: transkript + tahlil."""
    call = sb.table("calls").select("*").eq("id", call_id).single().execute().data
    if not call:
        raise HTTPException(404, "Topilmadi")

    tr = sb.table("transcripts").select("*").eq("call_id", call_id).execute().data
    an = sb.table("analyses").select("*").eq("call_id", call_id).execute().data

    return {
        "call":       call,
        "transcript": tr[0] if tr else None,
        "analysis":   an[0] if an else None,
    }


@app.post("/calls/{call_id}/retry")
async def retry_call(call_id: str, background: BackgroundTasks, speakers: int = 2):
    """Muvaffaqiyatsiz qo'ng'iroqni qayta ishlash."""
    call = sb.table("calls").select("audio_path, status") \
             .eq("id", call_id).single().execute().data
    if not call or not call.get("audio_path"):
        raise HTTPException(404, "Audio topilmadi")

    _set_status(call_id, "pending", error=None)
    background.add_task(process_call, call_id, call["audio_path"], speakers)
    return {"id": call_id, "status": "pending"}
