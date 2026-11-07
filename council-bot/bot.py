import collections
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

import config
import council_lookup
import db
import rag

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("council-bot")

LOCATION, PHOTO = range(2)
END = ConversationHandler.END


class RateLimiter:
    """Simple per-user sliding window so nobody can run up the API bill."""

    def __init__(self, limit: int, window: int = 60):
        self.limit, self.window = limit, window
        self.hits = collections.defaultdict(collections.deque)

    def allow(self, uid: int) -> bool:
        now, q = time.monotonic(), self.hits[uid]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


limiter = RateLimiter(config.RATE_LIMIT_PER_MIN)

WELCOME = (
    f"Hello! I'm **EstateKaki**, your virtual assistant for Singapore Town Councils 🇸🇬\n\n"
    "• **Ask me about estate services** (bulky waste collection, permits, opening hours...)\n"
    "• **Report estate problems** (broken streetlights, potholes, corridor clutter, rubbish...) — just share your 6-digit postal code!\n"
    "• `/status <number>` to check a report you made\n\n"
    "How can I help you today?"
)


async def get_or_create_topic(bot, council: str):
    if not council or not config.STAFF_CHAT_ID:
        return None
    tid = db.get_topic(council)
    if tid:
        return tid
    try:
        slug = council.lower().replace(" town council", "").replace(" ", "-")
        topic = await bot.create_forum_topic(chat_id=config.STAFF_CHAT_ID, name=slug[:64])
        tid = topic.message_thread_id
        db.save_topic(council, tid)
        return tid
    except Exception as e:
        log.info("Could not create forum topic for %s: %s", council, e)
        return None


async def notify_staff(context, text, photo_id=None, town_council=None):
    if not config.STAFF_CHAT_ID:
        return
    thread_id = None
    if town_council:
        thread_id = await get_or_create_topic(context.bot, town_council)

    kwargs = {"message_thread_id": thread_id} if thread_id else {}
    try:
        if photo_id:
            await context.bot.send_photo(config.STAFF_CHAT_ID, photo_id, caption=text, **kwargs)
        else:
            await context.bot.send_message(config.STAFF_CHAT_ID, text, **kwargs)
    except Exception:
        try:
            if photo_id:
                await context.bot.send_photo(config.STAFF_CHAT_ID, photo_id, caption=text)
            else:
                await context.bot.send_message(config.STAFF_CHAT_ID, text)
        except Exception:
            log.exception("Could not notify staff")


# ---------- Resident: free text (entry point of the report conversation) ----------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg, user = update.message, update.effective_user
    if not limiter.allow(user.id):
        await msg.reply_text("You're sending messages very quickly. Please wait a minute and try again.")
        return END

    text = msg.text.strip()
    if config.EMERGENCY_RE.search(text):  # keyword check first: fast, free, no LLM needed
        await msg.reply_text(config.EMERGENCY_TEXT, parse_mode="Markdown")
        return END

    # Smart Statutory Agency check (LTA, NParks, SFA, HDB)
    agency_guidance = council_lookup.check_statutory_agency(text)
    if agency_guidance:
        await msg.reply_text(agency_guidance, parse_mode="Markdown")
        return END

    await msg.chat.send_action("typing")
    info = await rag.classify(text)

    if info["intent"] == "emergency":
        await msg.reply_text(config.EMERGENCY_TEXT, parse_mode="Markdown")
        return END

    if info["intent"] == "other":
        if re.search(r"\b(hi|hello|hey|good morning|afternoon|evening)\b", text, re.IGNORECASE) and len(text) < 25:
            await msg.reply_text("Hello! I'm EstateKaki. How can I assist you with your Town Council estate matters today?")
        else:
            await msg.reply_text(
                "EstateKaki assists with Singapore Town Council matters (such as estate cleanliness, "
                "faulty public amenities, corridor maintenance, lifts, and town council policies).\n\n"
                "I am unable to assist with personal disputes, private affairs, or matters outside town council jurisdiction.\n"
                "If you have an estate fault to report or a council question, please let me know!"
            )
        return END

    if info["intent"] == "report":
        context.user_data["draft"] = {"description": text, "category": info["category"]}
        await msg.reply_text(
            "Sorry to hear about that issue, I'll help log it for the council.\n\n"
            "📍 Where is it? Please send your 6-digit postal code (e.g. 560123), "
            "block/street address, or share a Telegram location pin so we can route it "
            "to the right Town Council.\n\n"
            "(/cancel to stop)"
        )
        return LOCATION

    await handle_question(update, context, text)
    return END


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    msg, user = update.message, update.effective_user
    result = await rag.answer(text)

    if result["answer"] is None:
        log_id = db.log_qa(user.id, text, None, None, False)
        await msg.reply_text(
            "I'm not sure about that and I don't want to guess. "
            "I've passed your question to council staff, who will get back to you here."
        )
        who = f"@{user.username}" if user.username else user.full_name
        await notify_staff(
            context,
            f"❓ Unanswered question (log #{log_id}) from {who} [id {user.id}]:\n{text}\n\n"
            f"Reply with: /reply {user.id} your message",
        )
        return

    sources = "; ".join(f"{s['title']} (updated {s['updated']})" for s in result["sources"])
    log_id = db.log_qa(user.id, text, result["answer"], sources, True)
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("👍", callback_data=f"fb:1:{log_id}"),
          InlineKeyboardButton("👎", callback_data=f"fb:0:{log_id}")]]
    )
    await msg.reply_text(f"{result['answer']}\n\nSource: {sources}", reply_markup=keyboard)


async def on_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    _, value, log_id = q.data.split(":")
    db.set_feedback(int(log_id), int(value))
    await q.answer("Thanks for the feedback!")
    await q.edit_message_reply_markup(reply_markup=None)


# ---------- Resident: report flow ----------

async def got_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    lat, lon = None, None
    if msg.location:
        lat, lon = msg.location.latitude, msg.location.longitude
        raw_loc = f"{lat:.5f}, {lon:.5f}"
    else:
        raw_loc = msg.text.strip()

    # Automatically resolve address and map to Singapore Town Council & coordinates
    resolved_loc, council, out_lat, out_lng = council_lookup.resolve_town_council(raw_loc, lat=lat, lon=lon)
    context.user_data["draft"]["location"] = resolved_loc
    context.user_data["draft"]["town_council"] = council
    context.user_data["draft"]["lat"] = out_lat
    context.user_data["draft"]["lng"] = out_lng

    # Smart Duplicate Ticket Check
    category = context.user_data["draft"].get("category", "other")
    dup = db.find_duplicate_ticket(category, resolved_loc)
    if dup:
        db.subscribe_to_ticket(dup["id"], update.effective_user.id)
        context.user_data.pop("draft", None)
        await msg.reply_text(
            f"📍 Location: **{resolved_loc}** ({council})\n\n"
            f"💡 **Existing Ticket Found:** An identical issue for `{category}` was already reported by a neighbour "
            f"as **Ticket #{dup['id']}**.\n\n"
            f"We have linked your account to Ticket #{dup['id']}! You will automatically get live alerts when "
            f"our maintenance crew resolves it. No duplicate dispatch needed! 🎉",
            parse_mode="Markdown",
        )
        return END

    await msg.reply_text(
        f"📍 Location noted: {resolved_loc}\n"
        f"🏛️ Routed to: **{council}**\n\n"
        "Send a photo for instant AI damage inspection, or send /skip.",
        parse_mode="Markdown",
    )
    return PHOTO


async def got_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    photo = msg.photo[-1]
    diag_summary = None
    severity = "normal"
    try:
        photo_file = await context.bot.get_file(photo.file_id)
        img_bytes = await photo_file.download_as_bytearray()
        import vision
        diag = await asyncio.to_thread(vision.analyze_fault_photo, bytes(img_bytes))
        diag_summary = f"{diag['fault']} (Assigned: {diag['crew']})"
        severity = diag.get("severity", "Medium")
        await msg.reply_text(
            f"🔍 **AI Vision Assessment:**\n"
            f"• **Detected:** {diag['fault']}\n"
            f"• **Hazard Severity:** {severity}\n"
            f"• **Suggested Crew:** {diag['crew']}\n\n"
            f"Logging ticket now...",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.info("Photo analysis skipped: %s", e)

    context.user_data["draft"]["ai_diagnosis"] = diag_summary
    context.user_data["draft"]["severity"] = severity
    return await finish_report(update, context, photo.file_id)


async def skip_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await finish_report(update, context, None)


async def finish_report(update: Update, context: ContextTypes.DEFAULT_TYPE, photo_id):
    user = update.effective_user
    d = context.user_data.pop("draft")
    council = d.get("town_council") or config.COUNCIL_NAME
    lat = d.get("lat")
    lng = d.get("lng")
    ai_diag = d.get("ai_diagnosis")
    severity = d.get("severity", "normal")

    tid = db.create_ticket(
        user.id,
        d["category"],
        d["description"],
        d.get("location"),
        photo_id,
        town_council=council,
        lat=lat,
        lng=lng,
        ai_diagnosis=ai_diag,
        severity=severity,
    )
    await update.message.reply_text(
        f"✅ Report #{tid} created and tagged for **{council}**.\n\n"
        f"Check it any time with `/status {tid}`. I'll message you when there's an update.",
        parse_mode="Markdown",
    )
    who = f"@{user.username}" if user.username else user.full_name
    note = (
        f"🆕 Ticket #{tid} [{d['category']}] from {who}\n"
        f"🏛️ Council: {council}\n"
        f"⚠️ Severity: {severity.upper()}\n"
        f"📍 {d.get('location')}\n"
        + (f"🤖 AI Diagnosis: {ai_diag}\n" if ai_diag else "")
        + f"📝 {d['description']}\n\n"
        f"/progress {tid} note   |   /resolve {tid} note"
    )
    await notify_staff(
        context,
        note,
        photo_id=photo_id,
        town_council=council,
    )
    return END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("draft", None)
    await update.message.reply_text("Cancelled.")
    return END


# ---------- Commands & Interactive Menu ----------

MAIN_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("⚡ Report a Fault", callback_data="act:report"),
     InlineKeyboardButton("📦 Bulky Removal", callback_data="act:bulky")],
    [InlineKeyboardButton("❓ Ask Estate FAQ", callback_data="act:faq"),
     InlineKeyboardButton("🚨 Emergency", callback_data="act:emergency")],
])


async def on_action_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data

    if data == "act:report":
        await q.message.reply_text(
            "To report an estate fault (e.g. broken streetlight, dirty void deck, pothole, lift fault):\n\n"
            "Simply send a message describing the issue and include your **6-digit postal code** (e.g. `560123`) or a location pin!"
        )
    elif data == "act:bulky":
        await q.message.reply_text(
            "📦 **Bulky Item Removal Service**\n\n"
            "• **Free Quota:** Up to 3 movable household items per month for HDB residents.\n"
            "• **Eligible Items:** Sofas, mattresses, wardrobes, refrigerators, washing machines.\n"
            "• **Excluded:** Renovation hacking debris, commercial waste, pianos/safes.\n\n"
            "To book, please call your respective Town Council branch office or submit via the Town Council website 3 working days in advance."
        )
    elif data == "act:faq":
        await q.message.reply_text(
            "❓ **Frequently Asked Questions**\n\n"
            "You can type any question directly! For example:\n"
            "• *'When are bulky items collected?'*\n"
            "• *'How much are S&CC charges?'*\n"
            "• *'What are the allowed renovation hours?'*\n"
            "• *'What is the corridor fire safety 1.2m rule?'*\n"
            "• *'How do I book the void deck for a wedding or wake?'*"
        )
    elif data == "act:emergency":
        await q.message.reply_text(config.EMERGENCY_TEXT, parse_mode="Markdown")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME, parse_mode="Markdown", reply_markup=MAIN_KEYBOARD)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        t = db.get_ticket(int(context.args[0]))
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /status 12")
        return
    if not t or t["user_id"] != update.effective_user.id:  # residents only see their own tickets
        await update.message.reply_text("I couldn't find a report with that number under your account.")
        return
    note = f"\nNote from staff: {t['staff_note']}" if t["staff_note"] else ""
    council_tag = f" [{t['town_council']}]" if ("town_council" in t.keys() and t["town_council"]) else ""
    await update.message.reply_text(f"Report #{t['id']} ({t['category']}){council_tag}: {t['status']}{note}")


async def cmd_chatid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Chat id: {update.effective_chat.id}")


# ---------- Staff (only works inside the staff group) ----------

def is_staff(update: Update) -> bool:
    return config.STAFF_CHAT_ID != 0 and update.effective_chat.id == config.STAFF_CHAT_ID


async def _set_status(update: Update, context: ContextTypes.DEFAULT_TYPE, status: str, label: str):
    if not is_staff(update):
        return
    try:
        tid = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text(f"Usage: /{label} <ticket id> [note]")
        return
    t = db.get_ticket(tid)
    if not t:
        await update.message.reply_text("No such ticket.")
        return
    note = " ".join(context.args[1:]) or None
    db.set_status(tid, status, note)
    extra = f"\nNote: {note}" if note else ""
    text = f"Update on your report #{tid}: {status}.{extra}"

    subs = []
    try:
        subs = json.loads(t["subscribers"] or "[]") if "subscribers" in t.keys() else []
    except Exception:
        pass
    if t["user_id"] not in subs:
        subs.append(t["user_id"])

    notified = 0
    for uid in subs:
        try:
            await context.bot.send_message(uid, text)
            notified += 1
        except Exception:
            pass

    await update.message.reply_text(f"Ticket #{tid} → {status} ✅ ({notified} resident(s) notified)")


async def cmd_resolve(update, context):
    await _set_status(update, context, "resolved", "resolve")


async def cmd_progress(update, context):
    await _set_status(update, context, "in progress", "progress")


async def cmd_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Staff answer to an unanswered question: /reply <user_id> text"""
    if not is_staff(update):
        return
    try:
        uid = int(context.args[0])
        text = " ".join(context.args[1:])
        assert text
    except (IndexError, ValueError, AssertionError):
        await update.message.reply_text("Usage: /reply <user id> <message>")
        return
    await context.bot.send_message(uid, f"Message from {config.COUNCIL_NAME}:\n{text}")
    await update.message.reply_text("Sent ✅")


def main():
    if not config.TELEGRAM_TOKEN:
        raise SystemExit("Set TELEGRAM_TOKEN in .env")
    db.init()
    app = Application.builder().token(config.TELEGRAM_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, on_text)],
        states={
            LOCATION: [MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.LOCATION, got_location)],
            PHOTO: [MessageHandler(filters.PHOTO, got_photo), CommandHandler("skip", skip_photo)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(conv)
    app.add_handler(CommandHandler(["start", "help"], cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("chatid", cmd_chatid))
    app.add_handler(CommandHandler("resolve", cmd_resolve))
    app.add_handler(CommandHandler("progress", cmd_progress))
    app.add_handler(CommandHandler("reply", cmd_reply))
    app.add_handler(CallbackQueryHandler(on_feedback, pattern=r"^fb:"))
    app.add_handler(CallbackQueryHandler(on_action_button, pattern=r"^act:"))

    log.info("Bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
