import os
import time
import json
import asyncio
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Deque
from collections import deque

from dotenv import load_dotenv

from telegram import Update
from telegram.constants import ChatType
from telegram.error import TimedOut, NetworkError
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ============================================================
# Load env
# ============================================================
load_dotenv()

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("annoyotron")

# ============================================================
# Env vars
# ============================================================
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN. Put it in .env")

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
USE_OPENAI = bool(OPENAI_API_KEY)

DB_PATH = (os.getenv("BOT_DB_PATH") or "bot_state.sqlite3").strip()

# ============================================================
# Optional OpenAI (Responses API)
# ============================================================
oai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI enabled (Responses API).")
    except Exception as e:
        logger.warning("OpenAI disabled: %s", e)
        USE_OPENAI = False
        oai_client = None

# ============================================================
# Behavior knobs (tune these)
# ============================================================
CHAT_REPLY_COOLDOWN_SEC = 20
CHAT_MAX_REPLIES_PER_10MIN = 12
USER_REPLY_COOLDOWN_SEC = 60

QUIET_HOUR_START = 1   # 1am
QUIET_HOUR_END = 9     # 9am

MAX_MEMORY_MESSAGES = 20

MUTE_KEYWORDS = ("stop", "stfu", "shut up", "mute bot", "no bot", "quiet bot")
WAKE_WORDS = ("bot", "annoyotron", "oi bot")

# AutoPoke: bot speaks on its own
AUTOPOKE_DEFAULT_INTERVAL_MIN = 180   # every 3 hours (when enabled)
AUTOPOKE_MIN_SILENCE_MIN = 60         # only speak if chat quiet for >= 60 min

# ============================================================
# DB
# ============================================================
_db_lock = asyncio.Lock()

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_def: str):
    col_name = col_def.split()[0].strip()
    cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if col_name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")

def init_db():
    conn = _db()
    c = conn.cursor()

    # Base schema
    c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        chat_id INTEGER PRIMARY KEY,
        title TEXT,
        chaos_enabled INTEGER DEFAULT 0,
        roast_level INTEGER DEFAULT 2,
        autopoke_enabled INTEGER DEFAULT 0,
        autopoke_interval_min INTEGER DEFAULT 180,
        last_activity_ts INTEGER DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS members (
        chat_id INTEGER,
        user_id INTEGER,
        username TEXT,
        first_name TEXT,
        roast_opt_in INTEGER DEFAULT 0,
        last_message_ts INTEGER DEFAULT 0,
        last_snippet TEXT DEFAULT '',
        PRIMARY KEY(chat_id, user_id)
    )
    """)

    # Migrations for older DBs (idempotent)
    _add_column_if_missing(conn, "chats", "chaos_enabled INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "chats", "roast_level INTEGER DEFAULT 2")
    _add_column_if_missing(conn, "chats", "autopoke_enabled INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "chats", "autopoke_interval_min INTEGER DEFAULT 180")
    _add_column_if_missing(conn, "chats", "last_activity_ts INTEGER DEFAULT 0")

    _add_column_if_missing(conn, "members", "roast_opt_in INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "members", "last_message_ts INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "members", "last_snippet TEXT DEFAULT ''")

    conn.commit()
    conn.close()

async def db_exec(sql: str, params: Tuple = ()):
    async with _db_lock:
        def run():
            conn = _db()
            conn.execute(sql, params)
            conn.commit()
            conn.close()
        await asyncio.to_thread(run)

async def db_fetchone(sql: str, params: Tuple = ()):
    async with _db_lock:
        def run():
            conn = _db()
            cur = conn.execute(sql, params)
            row = cur.fetchone()
            conn.close()
            return row
        return await asyncio.to_thread(run)

async def db_fetchall(sql: str, params: Tuple = ()):
    async with _db_lock:
        def run():
            conn = _db()
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            conn.close()
            return rows
        return await asyncio.to_thread(run)

# ============================================================
# In-memory rate limit + memory
# ============================================================
chat_last_reply_ts: Dict[int, int] = {}
chat_reply_times: Dict[int, Deque[int]] = {}
user_last_reply_ts: Dict[Tuple[int, int], int] = {}
chat_memory: Dict[int, Deque[Dict[str, str]]] = {}

def now_ts() -> int:
    return int(time.time())

def clip(text: str, n: int = 180) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t[:n] + ("…" if len(t) > n else "")

def in_quiet_hours() -> bool:
    h = datetime.now().hour
    if QUIET_HOUR_START < QUIET_HOUR_END:
        return QUIET_HOUR_START <= h < QUIET_HOUR_END
    return h >= QUIET_HOUR_START or h < QUIET_HOUR_END

def should_rate_limit(chat_id: int, user_id: int) -> bool:
    now = now_ts()

    last_chat = chat_last_reply_ts.get(chat_id, 0)
    if now - last_chat < CHAT_REPLY_COOLDOWN_SEC:
        return True

    key = (chat_id, user_id)
    last_user = user_last_reply_ts.get(key, 0)
    if now - last_user < USER_REPLY_COOLDOWN_SEC:
        return True

    dq = chat_reply_times.setdefault(chat_id, deque())
    cutoff = now - 600
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= CHAT_MAX_REPLIES_PER_10MIN:
        return True

    return False

def mark_replied(chat_id: int, user_id: int):
    now = now_ts()
    chat_last_reply_ts[chat_id] = now
    user_last_reply_ts[(chat_id, user_id)] = now
    dq = chat_reply_times.setdefault(chat_id, deque())
    dq.append(now)

def remember(chat_id: int, role: str, content: str):
    mem = chat_memory.setdefault(chat_id, deque(maxlen=MAX_MEMORY_MESSAGES))
    mem.append({"role": role, "content": content})

def get_memory(chat_id: int) -> List[Dict[str, str]]:
    return list(chat_memory.get(chat_id, []))

# ============================================================
# Telegram send safety (prevents crash on timeouts)
# ============================================================
async def safe_reply(msg, text: str):
    try:
        return await msg.reply_text(text)
    except (TimedOut, NetworkError) as e:
        logger.warning("Telegram reply failed (network): %s", e)
        return None
    except Exception as e:
        logger.warning("Telegram reply failed: %s", e)
        return None

async def safe_send(bot, chat_id: int, text: str):
    try:
        return await bot.send_message(chat_id=chat_id, text=text)
    except (TimedOut, NetworkError) as e:
        logger.warning("Telegram send failed (network): %s", e)
        return None
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)
        return None

# ============================================================
# OpenAI: generate roasty-but-safe group-member replies
# ============================================================
def build_persona(roast_level: int) -> str:
    # NOTE: no f-string braces inside besides roast_level variable
    return f"""
You are ANNOYOTRON, a chaotic roasty group-chat bro with Singapore/NUS vibes ("bro", "lah", "eh").
Roast level: {roast_level}/5.

Style:
- Short, punchy, witty. 1–3 sentences.
- Light teasing, mock disbelief, dramatic reactions, "skill issue" energy.
- Use Singlish particles sometimes (lah, sia, leh), but not every line.

Rules:
- No slurs, no identity-based insults, no threats.
- No humiliation or sustained targeting of one person.
- If someone says stop/mute/chill, back off immediately.

Goals:
- Reply like a normal group member, not a helpdesk.
- Move the convo forward with banter and a bit of chaos.
- Occasionally remind about /help, /autopoke_on, /chaos_on.
"""

async def llm_reply(chat_id: int, roast_level: int, user_name: str, text: str, memory: List[Dict[str, str]]) -> Optional[str]:
    if not USE_OPENAI or not oai_client:
        return None

    system = build_persona(roast_level)
    items = [{"role": "system", "content": system}]
    items.extend(memory[-MAX_MEMORY_MESSAGES:])
    items.append({"role": "user", "content": f"{user_name}: {text}"})

    def _call():
        resp = oai_client.responses.create(
            model="gpt-4.1-mini",
            input=items,
            max_output_tokens=160,  # must be >= 16
        )
        out = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        out.append(c.text)
        return " ".join(out).strip() or None

    try:
        return await asyncio.to_thread(_call)
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None

# ============================================================
# Commands
# ============================================================
HELP_TEXT = (
    "Annoyotron commands:\n"
    "/chaos_on – I start chatting like a real group member\n"
    "/chaos_off – I shut up (mostly)\n"
    "/roast_level <1-5> – set roast intensity\n"
    "/roast_optin – you consent to slightly spicier teasing\n"
    "/roast_optout – opt out\n"
    "/autopoke_on [minutes] – I will message on my own schedule\n"
    "/autopoke_off – stop autopoke\n"
    "/openai_test – test OpenAI from Telegram\n"
    "/status – show current settings\n"
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(
        update.effective_message,
        "✅ Annoyotron online.\n"
        "For normal group-chat replies, disable Bot Privacy:\n"
        "BotFather → /setprivacy → Disable\n\n"
        + HELP_TEXT
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update.effective_message, HELP_TEXT)

async def ensure_chat_row(chat_id: int, title: str):
    await db_exec("""
        INSERT INTO chats(chat_id, title, last_activity_ts)
        VALUES(?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET title=excluded.title
    """, (chat_id, title or "", now_ts()))

async def touch_chat_activity(chat_id: int):
    await db_exec("UPDATE chats SET last_activity_ts=? WHERE chat_id=?", (now_ts(), chat_id))

async def ensure_member_row(chat_id: int, user):
    await db_exec("""
        INSERT INTO members(chat_id, user_id, username, first_name)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(chat_id, user_id) DO UPDATE SET
            username=excluded.username,
            first_name=excluded.first_name
    """, (chat_id, user.id, user.username, user.first_name or ""))

async def chaos_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET chaos_enabled=1 WHERE chat_id=?", (chat.id,))
    await safe_reply(update.effective_message, "😈 Chaos mode ON. I will talk like a group member (rate-limited).")

async def chaos_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET chaos_enabled=0 WHERE chat_id=?", (chat.id,))
    await safe_reply(update.effective_message, "😇 Chaos mode OFF. I’ll mostly only respond to /commands.")

async def roast_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    if not context.args:
        await safe_reply(update.effective_message, "Usage: /roast_level 1-5")
        return
    try:
        lvl = int(context.args[0])
        lvl = max(1, min(5, lvl))
    except ValueError:
        await safe_reply(update.effective_message, "Give a number 1-5.")
        return
    await db_exec("UPDATE chats SET roast_level=? WHERE chat_id=?", (lvl, chat.id))
    await safe_reply(update.effective_message, f"🔥 Roast level set to {lvl}/5.")

async def roast_optin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    await ensure_chat_row(chat.id, chat.title or "")
    await ensure_member_row(chat.id, user)
    await db_exec("UPDATE members SET roast_opt_in=1 WHERE chat_id=? AND user_id=?", (chat.id, user.id))
    await safe_reply(update.effective_message, "✅ Opted into spicier teasing.")

async def roast_optout(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    await ensure_chat_row(chat.id, chat.title or "")
    await ensure_member_row(chat.id, user)
    await db_exec("UPDATE members SET roast_opt_in=0 WHERE chat_id=? AND user_id=?", (chat.id, user.id))
    await safe_reply(update.effective_message, "✅ Opted out. I’ll keep it calmer with you.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    row = await db_fetchone("SELECT * FROM chats WHERE chat_id=?", (chat.id,))
    if not row:
        await safe_reply(update.effective_message, "No settings yet.")
        return
    await safe_reply(
        update.effective_message,
        f"Settings:\n"
        f"- chaos_enabled: {int(row['chaos_enabled'])}\n"
        f"- roast_level: {int(row['roast_level'])}/5\n"
        f"- openai_enabled: {USE_OPENAI}\n"
        f"- autopoke_enabled: {int(row['autopoke_enabled'])}\n"
        f"- autopoke_interval_min: {int(row['autopoke_interval_min'])}\n"
        f"- quiet_hours: {QUIET_HOUR_START}:00–{QUIET_HOUR_END}:00\n"
        f"- cooldown: chat {CHAT_REPLY_COOLDOWN_SEC}s, user {USER_REPLY_COOLDOWN_SEC}s\n"
        f"- cap: {CHAT_MAX_REPLIES_PER_10MIN} replies/10min"
    )

async def openai_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not USE_OPENAI or not oai_client:
        await safe_reply(
            update.effective_message,
            "OpenAI not available. Check:\n"
            "- pip install -U openai\n"
            "- OPENAI_API_KEY in .env\n"
        )
        return

    def _call():
        resp = oai_client.responses.create(
            model="gpt-4.1-mini",
            input="Reply with exactly: OK",
            max_output_tokens=16,
        )
        out = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        out.append(c.text)
        return " ".join(out).strip()

    try:
        txt = await asyncio.to_thread(_call)
        await safe_reply(update.effective_message, f"OpenAI says: {txt}")
    except Exception as e:
        await safe_reply(update.effective_message, f"OpenAI test failed: {e}")

# ============================================================
# AutoPoke scheduling (JobQueue)
# ============================================================
_autopoke_jobs: Dict[int, object] = {}

async def autopoke_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    if in_quiet_hours():
        return

    row = await db_fetchone("SELECT * FROM chats WHERE chat_id=?", (chat_id,))
    if not row or int(row["autopoke_enabled"] or 0) != 1:
        return

    last_act = int(row["last_activity_ts"] or 0)
    if last_act > 0:
        silence_min = (now_ts() - last_act) // 60
        if silence_min < AUTOPOKE_MIN_SILENCE_MIN:
            return

    roast_lvl = int(row["roast_level"] or 2)

    # Rate-limit as "user_id=0"
    if should_rate_limit(chat_id, 0):
        return

    memory = get_memory(chat_id)
    prompt = "The group chat has been quiet. Say ONE short message to revive the chat. Do not target individuals."
    reply = await llm_reply(chat_id, roast_lvl, "system", prompt, memory)

    if not reply:
        reply = "Wah this chat went offline ah 😭 anyone alive?"

    await safe_send(context.bot, chat_id, reply)
    remember(chat_id, "assistant", reply)
    mark_replied(chat_id, 0)

def _schedule_autopoke(application: Application, chat_id: int, interval_min: int):
    old = _autopoke_jobs.get(chat_id)
    if old is not None:
        try:
            old.schedule_removal()
        except Exception:
            pass
        _autopoke_jobs.pop(chat_id, None)

    job = application.job_queue.run_repeating(
        autopoke_job,
        interval=interval_min * 60,
        first=30,
        chat_id=chat_id,
        name=f"autopoke_{chat_id}",
    )
    _autopoke_jobs[chat_id] = job

async def autopoke_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")

    interval_min = AUTOPOKE_DEFAULT_INTERVAL_MIN
    if context.args:
        try:
            interval_min = int(context.args[0])
            interval_min = max(5, min(24 * 60, interval_min))
        except ValueError:
            await safe_reply(update.effective_message, "Usage: /autopoke_on [minutes] (e.g. /autopoke_on 180)")
            return

    await db_exec(
        "UPDATE chats SET autopoke_enabled=1, autopoke_interval_min=? WHERE chat_id=?",
        (interval_min, chat.id)
    )

    _schedule_autopoke(context.application, chat.id, interval_min)
    await safe_reply(update.effective_message, f"🕒 AutoPoke ON. I’ll poke every ~{interval_min} min (only if chat is quiet).")

async def autopoke_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    await ensure_chat_row(chat.id, chat.title or "")
    await db_exec("UPDATE chats SET autopoke_enabled=0 WHERE chat_id=?", (chat.id,))

    old = _autopoke_jobs.get(chat.id)
    if old is not None:
        try:
            old.schedule_removal()
        except Exception:
            pass
        _autopoke_jobs.pop(chat.id, None)

    await safe_reply(update.effective_message, "🛑 AutoPoke OFF.")

async def restore_jobs(application: Application):
    rows = await db_fetchall(
        "SELECT chat_id, autopoke_enabled, autopoke_interval_min FROM chats WHERE autopoke_enabled=1"
    )
    for r in rows:
        chat_id = int(r["chat_id"])
        interval_min = int(r["autopoke_interval_min"] or AUTOPOKE_DEFAULT_INTERVAL_MIN)
        _schedule_autopoke(application, chat_id, interval_min)
    logger.info("Restored %d autopoke jobs", len(rows))

# ============================================================
# Error handler
# ============================================================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)

# ============================================================
# Main message handler: LLM-style replies
# ============================================================
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if not msg or not chat or not user or not msg.text:
        return

    text = msg.text.strip()
    text_l = text.lower()

    # Track activity for groups (used by autopoke)
    if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        await ensure_chat_row(chat.id, chat.title or "")
        await touch_chat_activity(chat.id)
        await ensure_member_row(chat.id, user)

    # Remember for LLM context
    if chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        remember(chat.id, "user", f"{user.first_name or user.username or 'someone'}: {clip(text)}")
    else:
        remember(chat.id, "user", f"{clip(text)}")

    # Private chat: always respond
    if chat.type == ChatType.PRIVATE:
        if should_rate_limit(chat.id, user.id):
            return
        if any(k in text_l for k in MUTE_KEYWORDS):
            await safe_reply(msg, "ok ok, I’ll chill 😇")
            mark_replied(chat.id, user.id)
            return

        roast_lvl = 2
        reply = await llm_reply(chat.id, roast_lvl, user.first_name or "User", clip(text, 500), get_memory(chat.id))
        if not reply:
            reply = "👀 I’m here. Try /help."
        await safe_reply(msg, reply)
        remember(chat.id, "assistant", reply)
        mark_replied(chat.id, user.id)
        return

    # Group / supergroup only beyond this point
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return

    settings = await db_fetchone("SELECT * FROM chats WHERE chat_id=?", (chat.id,))
    chaos_enabled = int(settings["chaos_enabled"] or 0) if settings else 0
    roast_lvl = int(settings["roast_level"] or 2) if settings else 2

    if not chaos_enabled:
        return

    if in_quiet_hours():
        return

    if any(k in text_l for k in MUTE_KEYWORDS):
        if should_rate_limit(chat.id, user.id):
            return
        await safe_reply(msg, "ok ok ok… I’ll shut up for a bit 😇")
        mark_replied(chat.id, user.id)
        return

    # Decide when to respond
    bot_username = (context.bot.username or "").lower()
    mentioned = bot_username and (f"@{bot_username}" in text_l)

    replied_to_bot = False
    if msg.reply_to_message and msg.reply_to_message.from_user:
        replied_to_bot = (msg.reply_to_message.from_user.id == context.bot.id)

    wake = any(text_l.startswith(w + " ") for w in WAKE_WORDS)
    interject = (hash(text + str(msg.message_id)) % 12 == 0)

    if not (mentioned or replied_to_bot or wake or interject):
        return

    if should_rate_limit(chat.id, user.id):
        return

    # Respect per-user opt-in/out (opt-out reduces roast slightly)
    mrow = await db_fetchone("SELECT roast_opt_in FROM members WHERE chat_id=? AND user_id=?", (chat.id, user.id))
    user_opt_in = int(mrow["roast_opt_in"] or 0) if mrow else 0
    effective_roast = roast_lvl if user_opt_in else max(1, roast_lvl - 1)

    reply = await llm_reply(
        chat.id,
        effective_roast,
        user.first_name or user.username or "someone",
        clip(text, 500),
        get_memory(chat.id),
        )

    if not reply:
        reply = "👀 I’m here. Say my name or use /help."

    await safe_reply(msg, reply)
    remember(chat.id, "assistant", reply)
    mark_replied(chat.id, user.id)

# ============================================================
# Entry
# ============================================================
def main():
    init_db()

    # Increase Telegram API timeouts (fixes TimedOut on slow networks)
    request = HTTPXRequest(
        connect_timeout=20.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .request(request)
        .post_init(restore_jobs)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("chaos_on", chaos_on))
    app.add_handler(CommandHandler("chaos_off", chaos_off))
    app.add_handler(CommandHandler("roast_level", roast_level))
    app.add_handler(CommandHandler("roast_optin", roast_optin))
    app.add_handler(CommandHandler("roast_optout", roast_optout))
    app.add_handler(CommandHandler("autopoke_on", autopoke_on))
    app.add_handler(CommandHandler("autopoke_off", autopoke_off))
    app.add_handler(CommandHandler("openai_test", openai_test))
    app.add_handler(CommandHandler("status", status))

    # Text handler (LLM-style)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Annoyotron running (polling).")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
