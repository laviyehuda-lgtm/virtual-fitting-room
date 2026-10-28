"""
חדר הלבשה - שרת

שני מודלים דרך fal.ai:
  1. FASHN Try-On v1.6   : תמונת המשתמש + תמונת בגד -> המשתמש לובש את הבגד
  2. Nano Banana Edit    : עריכה בהנחיה על התמונה (צבע, בד, אורך, קו עליון, זווית)

מפתח ה-API נשמר רק בשרת (משתנה סביבה FAL_KEY) ולא מגיע לדפדפן.
התמונות לא נשמרות בשרת: הן עוברות ל-fal ומשם חוזרות לדפדפן.

משתני סביבה:
  FAL_KEY        מפתח fal.ai (בלעדיו השרת רץ במצב הדגמה ומחזיר את התמונה כמו שהיא)
  APP_PASSWORD   סיסמה לכניסה לאפליקציה, כדי שזרים לא ינצלו את הקרדיט שלך
"""
import base64
import hmac
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

FAL_KEY = os.getenv("FAL_KEY", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
TRYON_MODEL = os.getenv("TRYON_MODEL", "fal-ai/fashn/tryon/v1.6")
EDIT_MODEL = os.getenv("EDIT_MODEL", "fal-ai/nano-banana/edit")
PRO_MODEL = os.getenv("PRO_MODEL", "fal-ai/nano-banana-pro/edit")
FAL_BASE = os.getenv("FAL_BASE", "https://fal.run").rstrip("/")
MAX_IMAGE_CHARS = 12_000_000  # data URL של כ-9MB
STATIC = Path(__file__).parent

app = FastAPI(title="Fitting Room")
http = httpx.AsyncClient(timeout=httpx.Timeout(240, connect=20))

# ---------- הגנה בסיסית ----------
_hits: dict[str, deque] = defaultdict(deque)


def guard(request: Request, x_app_password: str = Header(default="")):
    if APP_PASSWORD and not hmac.compare_digest(x_app_password, APP_PASSWORD):
        raise HTTPException(401, "סיסמה שגויה")
    # עד 20 בקשות לשעה לכל כתובת: מונע שריפת קרדיט בטעות או בניצול
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    q, now = _hits[ip], time.time()
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= 20:
        raise HTTPException(429, "הגעת למגבלה של 20 פעולות בשעה. נסה שוב מאוחר יותר.")
    q.append(now)


# ---------- קריאה ל-fal ----------
async def fal(model: str, payload: dict) -> str:
    r = await http.post(
        f"{FAL_BASE}/{model}",
        headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
        json=payload,
    )
    if r.status_code == 401 or r.status_code == 403:
        raise HTTPException(502, "מפתח ה-fal לא תקין או שאין לו הרשאה. בדוק את FAL_KEY ב-Render.")
    if r.status_code >= 400 and (r.status_code == 402 or "balance" in r.text.lower()):
        raise HTTPException(502, "נגמר הקרדיט בחשבון fal. אפשר להטעין ב-fal.ai/dashboard/billing.")
    if r.status_code == 422:
        raise HTTPException(422, f"המודל דחה את התמונה: {_detail(r)}")
    if r.status_code >= 400:
        raise HTTPException(502, f"שגיאה מהמודל ({r.status_code}): {_detail(r)}")
    images = r.json().get("images") or []
    if not images:
        raise HTTPException(502, "המודל לא החזיר תמונה. נסה תמונה אחרת או הנחיה אחרת.")
    return images[0]["url"]


def _detail(r: httpx.Response) -> str:
    try:
        d = r.json().get("detail")
        if isinstance(d, list) and d:
            return str(d[0].get("msg", d[0]))
        return str(d or r.text)[:200]
    except Exception:  # noqa: BLE001
        return r.text[:200]


async def to_data_url(url: str) -> str:
    """מחזירים לדפדפן data URL ולא קישור: כך העריכה הבאה לא תלויה בתוקף הקישור."""
    if url.startswith("data:"):
        return url
    r = await http.get(url)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "image/png").split(";")[0]
    return f"data:{ctype};base64,{base64.b64encode(r.content).decode()}"


def check_image(data_url: str):
    if not data_url.startswith("data:image/"):
        raise HTTPException(400, "יש לשלוח תמונה")
    if len(data_url) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "התמונה גדולה מדי")


# ---------- הנחיות עריכה ----------
KEEP = ("Keep the person's face, identity, skin tone, hair, body shape, pose, "
        "the background and the lighting exactly the same. Photorealistic result.")

COLORS = {
    "black": "black", "white": "white", "red": "deep red", "burgundy": "burgundy",
    "pink": "soft pink", "orange": "orange", "yellow": "mustard yellow", "green": "emerald green",
    "olive": "olive green", "teal": "teal", "blue": "royal blue", "navy": "navy blue",
    "purple": "purple", "beige": "beige", "brown": "chocolate brown", "grey": "grey",
}
FABRICS = {
    "cotton": "matte cotton", "linen": "natural linen with visible weave", "silk": "flowing silk with soft sheen",
    "satin": "glossy satin", "denim": "blue denim", "velvet": "rich velvet", "lace": "delicate lace",
    "knit": "chunky knit", "leather": "leather", "sequins": "sparkly sequins",
}
PATTERNS = {
    "plain": "a solid plain color with no pattern", "stripes": "vertical stripes", "polka": "polka dots",
    "floral": "a small floral print", "checks": "a checked plaid pattern", "leopard": "a leopard print",
}
LENGTHS = {
    "mini": "a mini length ending at mid-thigh", "knee": "a length ending just at the knee",
    "midi": "a midi length ending at mid-calf", "maxi": "a maxi length reaching the ankles",
}
NECKLINES = {
    "straps": "thin shoulder straps", "off_shoulder": "an off-the-shoulder neckline that exposes both shoulders",
    "strapless": "a strapless neckline", "short_sleeves": "short sleeves", "long_sleeves": "long sleeves",
    "v_neck": "a V-neckline", "high_neck": "a high neckline",
}
VIEWS = {
    "back": "Show the exact same person wearing the exact same outfit, photographed from behind (back view), "
            "standing in the same place. Keep the hair, body shape and the garment design consistent.",
    "side": "Show the exact same person wearing the exact same outfit, photographed from the side (profile view), "
            "standing in the same place. Keep the hair, body shape and the garment design consistent.",
    "front": "Show the exact same person wearing the exact same outfit, photographed from the front, "
             "standing in the same place. Keep the face, hair, body shape and garment design consistent.",
}


class TryOnIn(BaseModel):
    person: str
    garment: str
    category: str = Field("auto", pattern="^(auto|tops|bottoms|one-pieces)$")
    garment_photo_type: str = Field("auto", pattern="^(auto|model|flat-lay)$")
    engine: str = Field("pro", pattern="^(pro|fast)$")


PRO_TRYON = (
    "Image 1 shows a person. Image 2 shows a garment, possibly worn by a different model. "
    "Dress the person from image 1 in the exact garment from image 2. "
    "Reproduce the garment's design faithfully: its exact silhouette and cut, neckline and shoulders "
    "(for example one-shoulder or off-the-shoulder), sleeves or lack of sleeves, ruffles, bows, drapes and "
    "sculptural volumes, length, fabric, sheen and color. "
    "Completely remove everything the person is currently wearing in the area the new garment covers or "
    "replaces, including jackets, sleeves and trousers; show bare skin where the new garment leaves it bare. "
    "{extra}"
    "Keep the person's face, identity, hair, skin tone, body shape, glasses and accessories, pose, background "
    "and lighting exactly as in image 1. Ignore the model, face and body in image 2. Photorealistic photo."
)
PRO_EXTRA = {
    "auto": "",
    "one-pieces": "The garment is a one-piece (dress or jumpsuit) that replaces the whole outfit, top and bottom. ",
    "tops": "The garment is a top; keep the person's current bottom garment unchanged. ",
    "bottoms": "The garment is a bottom; keep the person's current top garment unchanged. ",
}


class EditIn(BaseModel):
    image: str
    kind: str = Field(pattern="^(color|hex|fabric|pattern|length|neckline|view|free)$")
    value: str = Field(max_length=400)


def build_prompt(kind: str, value: str) -> str:
    table = {"color": COLORS, "fabric": FABRICS, "pattern": PATTERNS, "length": LENGTHS, "neckline": NECKLINES, "view": VIEWS}
    if kind == "hex":
        if not (len(value) == 7 and value.startswith("#")):
            raise HTTPException(400, "צבע לא תקין")
        return f"Change only the color of the garment the person is wearing to the exact color {value}. {KEEP}"
    if kind == "free":
        if len(value.strip()) < 3:
            raise HTTPException(400, "כתוב מה לשנות")
        return f"Edit only the clothing the person is wearing: {value.strip()}. {KEEP}"
    if value not in table[kind]:
        raise HTTPException(400, "אפשרות לא מוכרת")
    v = table[kind][value]
    return {
        "color": f"Change only the color of the garment the person is wearing to {v}. Keep its cut, fabric texture and folds. {KEEP}",
        "fabric": f"Change only the fabric of the garment the person is wearing to {v}, keeping the same cut and color. {KEEP}",
        "pattern": f"Change only the print of the garment the person is wearing to {v}, keeping the same cut and base color. {KEEP}",
        "length": f"Alter the garment the person is wearing so it has {v}. Keep the same fabric, color and top part. {KEEP}",
        "neckline": f"Alter the top of the garment the person is wearing so it has {v}. Keep the same fabric, color and length. {KEEP}",
        "view": f"{v} Photorealistic result.",
    }[kind]


# ---------- נקודות קצה ----------
@app.get("/api/status")
async def status():
    return {"live": bool(FAL_KEY), "password": bool(APP_PASSWORD)}


@app.post("/api/login", dependencies=[Depends(guard)])
async def login():
    return {"ok": True}


@app.post("/api/tryon", dependencies=[Depends(guard)])
async def tryon(body: TryOnIn):
    check_image(body.person)
    check_image(body.garment)
    if not FAL_KEY:
        return {"image": body.person, "demo": True}
    if body.engine == "pro":
        url = await fal(PRO_MODEL, {
            "prompt": PRO_TRYON.format(extra=PRO_EXTRA[body.category]),
            "image_urls": [body.person, body.garment],
            "aspect_ratio": "auto",
            "resolution": "2K",
            "output_format": "jpeg",
        })
    else:
        url = await fal(TRYON_MODEL, {
            "model_image": body.person,
            "garment_image": body.garment,
            "category": body.category,
            "garment_photo_type": body.garment_photo_type,
            "segmentation_free": False,  # מסיר את הבגד הקודם לגמרי (למשל שרוולי בלייזר)
            "mode": "quality",
            "moderation_level": "permissive",
            "output_format": "jpeg",
        })
    return {"image": await to_data_url(url)}


@app.post("/api/edit", dependencies=[Depends(guard)])
async def edit(body: EditIn):
    check_image(body.image)
    prompt = build_prompt(body.kind, body.value)
    if not FAL_KEY:
        return {"image": body.image, "demo": True, "prompt": prompt}
    url = await fal(EDIT_MODEL, {
        "prompt": prompt,
        "image_urls": [body.image],
        "output_format": "jpeg",
        "aspect_ratio": "auto",
    })
    return {"image": await to_data_url(url)}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

